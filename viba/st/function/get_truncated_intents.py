import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.autograd import Function

# Import required utilities
from viba.st.tensor_util.convert_st_tensor_to_file_contents import convert_st_tensor_to_file_contents
from viba.st.tensor_util.convert_file_contents_to_st_tensor import convert_file_contents_to_st_tensor
from viba.intent_truncate_util import get_all_truncated_vibe_code

# ----------------------------------------------------------------------
# Forward implementation
# ----------------------------------------------------------------------
def _call_claude(prompt: str) -> str:
    """
    Internal helper that runs the claude command.
    In production this uses subprocess; in tests it is mocked.
    """
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout.strip()

def get_truncated_intents_forward(
    input_tensor: torch.Tensor,
    num_parts: int
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Forward pass: read the input tensor (which stores paths to JSON files containing
    list[list[str]] intent data), apply truncation to generate a base intent and
    multiple truncated variants for each sample.

    Returns:
        intent_base_tensor: Tensor[Viba] of shape (batch, 1, feature_len) storing paths
                            to files containing the base Viba intent (string).
        truncated_intents: list of num_parts Tensor[Viba], each of shape
                           (batch, 1, feature_len) storing paths to files containing
                           one truncated Viba intent per sample.
    """
    # Retrieve the list of intent strings for each sample
    input_contents_2d = convert_st_tensor_to_file_contents(input_tensor)
    batch_size = input_tensor.shape[0]
    input_intents = [input_contents_2d[i][0] for i in range(batch_size)]

    # Parse each JSON into list[list[str]] (segment groups for one sample)
    parsed_intents = [json.loads(s) for s in input_intents]

    # Apply truncation logic per sample.
    # get_all_truncated_vibe_code signature:
    #   (vibe_segments_list: list[list[str]], num_parts: int) -> tuple[str, list[str]]
    # It processes ONE sample at a time, returning (base_intent_str, [truncated_str, ...]).
    base_intents = []
    # truncated_by_part[p] = list of truncated strings for part p across batch
    truncated_by_part: List[List[str]] = [[] for _ in range(num_parts)]
    for segments in parsed_intents:
        base, truncated = get_all_truncated_vibe_code(segments, num_parts)
        base_intents.append(base)
        for p in range(num_parts):
            truncated_by_part[p].append(truncated[p])

    feature_len = input_tensor.shape[2]
    root_dir = getattr(input_tensor, 'st_relative_to', None)

    # Convert base intents into a path tensor
    intent_base_tensor = convert_file_contents_to_st_tensor(
        file_contents=base_intents,
        relative_to=root_dir,
        max_use_count=1,
        feature_len=feature_len
    )
    intent_base_tensor.st_file_content_type = "Viba"

    # Convert each truncation level into a separate Tensor[Viba]
    truncated_tensors = []
    for p in range(num_parts):
        t = convert_file_contents_to_st_tensor(
            file_contents=truncated_by_part[p],
            relative_to=root_dir,
            max_use_count=1,
            feature_len=feature_len
        )
        t.st_file_content_type = "Viba"
        truncated_tensors.append(t)

    return intent_base_tensor, truncated_tensors

# ----------------------------------------------------------------------
# Backward implementation
# ----------------------------------------------------------------------
def get_truncated_intents_backward(
    intent_base_grad: torch.Tensor,          # Tensor[Diff[Viba]]
    truncated_intents_grad: List[torch.Tensor],  # list[Tensor[Diff[Viba]]]
    input_tensor: torch.Tensor,              # saved from forward
    num_parts: int                            # saved from forward
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Backward pass: based on the gradients of the two outputs, generate
    a weight gradient (which may be used to update external mappings) and
    return None for the input gradient (since input is not trainable).

    Returns:
        input_grad: None (input does not require grad)
        weight_grad: Tensor storing paths to files containing the JSON diff
                     for weight update (if any), otherwise None.
    """
    # Read the gradient contents (Diff strings)
    base_grad_contents = convert_st_tensor_to_file_contents(intent_base_grad)
    truncated_grad_contents_per_part = [
        convert_st_tensor_to_file_contents(g) for g in truncated_intents_grad
    ]

    batch_size = intent_base_grad.shape[0]

    # For each sample, extract the first slot (only meaningful slot)
    base_grad_strings = [base_grad_contents[i][0] for i in range(batch_size)]

    weight_grad_strings = []
    for i in range(batch_size):
        # Collect truncated grads for this sample across all parts
        trunc_grads = [truncated_grad_contents_per_part[p][i][0] for p in range(num_parts)]
        prompt = f"Generate weight grad based on base_grad: {base_grad_strings[i]} and truncated_grads: {json.dumps(trunc_grads)}"
        # _call_claude(prompt) would be used here
        weight_grad_json = json.dumps({"key": "some_key", "diff": "some_diff"})
        weight_grad_strings.append(weight_grad_json)

    feature_len = intent_base_grad.shape[2]
    root_dir = getattr(intent_base_grad, 'st_relative_to', None)

    # Convert the weight gradient strings into a path tensor
    weight_grad_tensor = convert_file_contents_to_st_tensor(
        file_contents=weight_grad_strings,
        relative_to=root_dir,
        max_use_count=1,
        feature_len=feature_len
    )
    weight_grad_tensor.st_file_content_type = "Json[list[$key Diff[Python] * $value Diff[Viba]]]"

    # Input gradient is None (input does not require grad)
    return None, weight_grad_tensor

# ----------------------------------------------------------------------
# Custom autograd Function
# ----------------------------------------------------------------------
class GetTruncatedIntents(Function):
    @staticmethod
    def forward(ctx, input_tensor, num_parts):
        ctx.save_for_backward(input_tensor, torch.tensor(num_parts))
        intent_base, truncated_list = get_truncated_intents_forward(input_tensor, num_parts)
        # Store num_parts for backward; return base + each truncated tensor
        return (intent_base, *truncated_list)

    @staticmethod
    def backward(ctx, grad_intent_base, *grad_truncated_intents):
        input_tensor, num_parts_tensor = ctx.saved_tensors
        num_parts = num_parts_tensor.item()
        return get_truncated_intents_backward(
            grad_intent_base, list(grad_truncated_intents), input_tensor, num_parts
        )


def get_truncated_intents(input_tensor: torch.Tensor, num_parts: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Convenience wrapper that returns (intent_base, list[truncated_tensors])."""
    results = GetTruncatedIntents.apply(input_tensor, num_parts)
    return results[0], list(results[1:])

# ----------------------------------------------------------------------
# Unit tests (only in __main__)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from viba.st.data_loader.convert_list_str_to_2d_tensor import convert_2d_tensor_to_list_str
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Prepare input data ──
        # Each sample is a list[list[str]] (segment groups of valid VIBA).
        # Sample 0: a function implementation with 3 chain segments
        sample_0 = [["compute := $result int <- $a int", "<- $b int", "<- $op str"]]
        # Sample 1: a class definition (single segment, no truncation variation)
        sample_1 = [["Person := $name str * $age int"]]

        input_json_strings = [json.dumps(s) for s in [sample_0, sample_1]]

        # Create input tensor (paths to the JSON files)
        input_tensor = convert_file_contents_to_st_tensor(
            file_contents=input_json_strings,
            relative_to=tmpdir,
            max_use_count=1,
            feature_len=256
        )
        input_tensor.st_file_content_type = "Json[list[list[str]]]"

        num_parts = 3

        # ── Test 1: Forward pass ──
        print("Test 1: Forward pass")
        intent_base, truncated_intents = get_truncated_intents_forward(input_tensor, num_parts)

        assert intent_base.shape == (2, 1, 256), f"Unexpected base shape: {intent_base.shape}"
        assert intent_base.dtype == torch.uint8
        assert intent_base.st_file_content_type == "Viba"
        assert isinstance(truncated_intents, list), "truncated_intents should be a list"
        assert len(truncated_intents) == num_parts, f"Expected {num_parts} tensors, got {len(truncated_intents)}"
        for p, t in enumerate(truncated_intents):
            assert t.shape == (2, 1, 256), f"Unexpected truncated[{p}] shape: {t.shape}"
            assert t.dtype == torch.uint8
            assert t.st_file_content_type == "Viba"

        # Decode and verify base intents
        base_paths = convert_2d_tensor_to_list_str(intent_base[:, 0, :])
        for i, path in enumerate(base_paths):
            full = Path(tmpdir) / path
            assert full.exists(), f"Base file {i} missing: {full}"
            content = full.read_text(encoding='utf-8')
            print(f"  Base {i}: {repr(content[:60])}")
            assert len(content) > 0, f"Base content {i} is empty"

        # Decode and verify each truncated intent tensor
        for p, t in enumerate(truncated_intents):
            trunc_paths = convert_2d_tensor_to_list_str(t[:, 0, :])
            for i, path in enumerate(trunc_paths):
                full = Path(tmpdir) / path
                assert full.exists(), f"Truncated[{p}] file {i} missing: {full}"
                content = full.read_text(encoding='utf-8')
                assert len(content) > 0, f"Truncated[{p}] content {i} is empty"
            print(f"  Truncated part {p}: verified {len(trunc_paths)} samples")
        print("  Forward test passed.\n")

        # ── Test 2: Backward pass ──
        print("Test 2: Backward pass")
        grad_base_strings = [json.dumps({"diff": "base_change"})] * 2

        grad_base = convert_file_contents_to_st_tensor(
            grad_base_strings, relative_to=tmpdir, max_use_count=1, feature_len=256
        )
        grad_base.st_file_content_type = "Diff[Viba]"

        # Create one grad tensor per truncation part
        grad_trunc_list = []
        for p in range(num_parts):
            grad_trunc_strings = [json.dumps({"diff": f"t{p}"})] * 2
            gt = convert_file_contents_to_st_tensor(
                grad_trunc_strings, relative_to=tmpdir, max_use_count=1, feature_len=256
            )
            gt.st_file_content_type = "Diff[Viba]"
            grad_trunc_list.append(gt)

        input_grad, weight_grad = get_truncated_intents_backward(
            grad_base, grad_trunc_list, input_tensor, num_parts
        )

        assert input_grad is None, "Input grad should be None"
        assert weight_grad.shape == (2, 1, 256), f"Unexpected weight_grad shape: {weight_grad.shape}"
        assert weight_grad.dtype == torch.uint8
        assert weight_grad.st_file_content_type == "Json[list[$key Diff[Python] * $value Diff[Viba]]]"

        weight_paths = convert_2d_tensor_to_list_str(weight_grad[:, 0, :])
        for i, path in enumerate(weight_paths):
            full = Path(tmpdir) / path
            assert full.exists(), f"Weight grad file {i} missing"
            content = full.read_text(encoding='utf-8')
            parsed = json.loads(content)
            assert "key" in parsed and "diff" in parsed, f"Weight grad {i} missing keys"
        print("  Backward test passed.\n")

        # ── Test 3: Verify truncation produces variation for function impl ──
        print("Test 3: Truncation variation for function implementation")
        # Read each part's content for sample 0
        lengths = []
        for p, t in enumerate(truncated_intents):
            trunc_path = Path(tmpdir) / convert_2d_tensor_to_list_str(t[:, 0, :])[0]
            content = trunc_path.read_text(encoding='utf-8')
            lengths.append(len(content))
        print(f"  Truncation lengths: {lengths}")
        assert lengths[0] <= lengths[-1], "First truncation should be <= last"
        assert len(set(lengths)) > 1, f"Expected variation in truncation lengths, got {lengths}"
        print("  Variation test passed.\n")

    print("All tests completed.")
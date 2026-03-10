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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass: read the input tensor (which stores paths to JSON files containing
    list[list[str]] intent data), apply truncation to generate a base intent and
    multiple truncated variants for each sample.

    Returns:
        intent_base_tensor: Tensor of shape (batch, 1, feature_len) storing paths to files
                            containing the base Viba intent (string).
        truncated_intents_tensor: Tensor of shape (batch, 1, feature_len) storing paths
                                  to files containing JSON lists of truncated Viba intents.
    """
    # Retrieve the list of intent strings for each sample
    # input_tensor stores paths to files containing JSON list of list of strings
    input_contents_2d = convert_st_tensor_to_file_contents(input_tensor)          # list of list of str, shape (batch, max_use_count)
    # Take the first slot (index 0) as the actual intent list
    batch_size = input_tensor.shape[0]
    input_intents = [input_contents_2d[i][0] for i in range(batch_size)]   # list of JSON strings

    # Parse each JSON into list[list[str]] (segment groups for one sample)
    parsed_intents = [json.loads(s) for s in input_intents]     # list[list[list[str]]]

    # Apply truncation logic per sample.
    # get_all_truncated_vibe_code signature:
    #   (vibe_segments_list: list[list[str]], num_parts: int) -> tuple[str, list[str]]
    # It processes ONE sample at a time, returning (base_intent_str, [truncated_str, ...]).
    base_intents = []
    truncated_json_strings = []
    for segments in parsed_intents:
        base, truncated = get_all_truncated_vibe_code(segments, num_parts)
        base_intents.append(base)
        truncated_json_strings.append(json.dumps(truncated))

    feature_len = input_tensor.shape[2]
    root_dir = getattr(input_tensor, 'st_relative_to', None)

    # Convert both result lists into path tensors (each batch element stored as a file)
    # We set max_use_count = 1 because each output is a single string per sample.
    intent_base_tensor = convert_file_contents_to_st_tensor(
        file_contents=base_intents,
        relative_to=root_dir,
        max_use_count=1,
        feature_len=feature_len
    )
    intent_base_tensor.st_file_content_type = "Viba"

    truncated_intents_tensor = convert_file_contents_to_st_tensor(
        file_contents=truncated_json_strings,
        relative_to=root_dir,
        max_use_count=1,
        feature_len=feature_len
    )
    truncated_intents_tensor.st_file_content_type = "Json[list[Viba]]"

    return intent_base_tensor, truncated_intents_tensor

# ----------------------------------------------------------------------
# Backward implementation
# ----------------------------------------------------------------------
def get_truncated_intents_backward(
    intent_base_grad: torch.Tensor,          # Tensor[Diff[Viba]]
    truncated_intents_grad: torch.Tensor,    # Tensor[Json[list[Diff[Viba]]]]
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
    base_grad_contents = convert_st_tensor_to_file_contents(intent_base_grad)          # list of list of str
    truncated_grad_contents = convert_st_tensor_to_file_contents(truncated_intents_grad)  # list of list of str

    batch_size = intent_base_grad.shape[0]

    # For each sample, extract the first slot (only meaningful slot)
    base_grad_strings = [base_grad_contents[i][0] for i in range(batch_size)]
    truncated_grad_strings = [truncated_grad_contents[i][0] for i in range(batch_size)]

    # In a real implementation, we might combine these gradients and call an LLM
    # to produce a weight update. Here we mock by constructing a single JSON diff.
    # According to the DSL, we generate a weight gradient JSON via LLM.
    # For simplicity, we create a dummy diff JSON for each sample.
    weight_grad_strings = []
    for i in range(batch_size):
        # Construct a prompt that includes the gradient information (mocked)
        # In production, you would call _call_claude with an appropriate prompt.
        # For the mock, we return a fixed JSON.
        prompt = f"Generate weight grad based on base_grad: {base_grad_strings[i]} and truncated_grad: {truncated_grad_strings[i]}"
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
        ctx.save_for_backward(input_tensor, torch.tensor(num_parts))  # save num_parts as tensor
        return get_truncated_intents_forward(input_tensor, num_parts)

    @staticmethod
    def backward(ctx, grad_intent_base, grad_truncated_intents):
        input_tensor, num_parts_tensor = ctx.saved_tensors
        num_parts = num_parts_tensor.item()
        return get_truncated_intents_backward(
            grad_intent_base, grad_truncated_intents, input_tensor, num_parts
        )

# Convenience alias
get_truncated_intents = GetTruncatedIntents.apply

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
        assert truncated_intents.shape == (2, 1, 256), f"Unexpected truncated shape: {truncated_intents.shape}"
        assert truncated_intents.dtype == torch.uint8
        assert truncated_intents.st_file_content_type == "Json[list[Viba]]"

        # Decode and verify base intents
        base_paths = convert_2d_tensor_to_list_str(intent_base[:, 0, :])
        for i, path in enumerate(base_paths):
            full = Path(tmpdir) / path
            assert full.exists(), f"Base file {i} missing: {full}"
            content = full.read_text(encoding='utf-8')
            print(f"  Base {i}: {repr(content[:60])}")
            assert len(content) > 0, f"Base content {i} is empty"

        # Decode and verify truncated intents (JSON lists)
        truncated_paths = convert_2d_tensor_to_list_str(truncated_intents[:, 0, :])
        for i, path in enumerate(truncated_paths):
            full = Path(tmpdir) / path
            assert full.exists(), f"Truncated file {i} missing: {full}"
            content = full.read_text(encoding='utf-8')
            parsed = json.loads(content)
            assert isinstance(parsed, list), f"Truncated {i} is not a list"
            assert len(parsed) == num_parts, f"Truncated {i} has {len(parsed)} parts, expected {num_parts}"
            print(f"  Truncated {i}: {len(parsed)} parts, first={repr(parsed[0][:40])}")
        print("  Forward test passed.\n")

        # ── Test 2: Backward pass ──
        print("Test 2: Backward pass")
        grad_base_strings = [json.dumps({"diff": "base_change"})] * 2
        grad_trunc_strings = [json.dumps([{"diff": "t1"}, {"diff": "t2"}])] * 2

        grad_base = convert_file_contents_to_st_tensor(
            grad_base_strings, relative_to=tmpdir, max_use_count=1, feature_len=256
        )
        grad_base.st_file_content_type = "Diff[Viba]"

        grad_trunc = convert_file_contents_to_st_tensor(
            grad_trunc_strings, relative_to=tmpdir, max_use_count=1, feature_len=256
        )
        grad_trunc.st_file_content_type = "Json[list[Diff[Viba]]]"

        input_grad, weight_grad = get_truncated_intents_backward(
            grad_base, grad_trunc, input_tensor, num_parts
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
        trunc_path_0 = Path(tmpdir) / truncated_paths[0]
        trunc_list_0 = json.loads(trunc_path_0.read_text(encoding='utf-8'))
        # Function impl should have increasing lengths across truncation levels
        lengths = [len(t) for t in trunc_list_0]
        print(f"  Truncation lengths: {lengths}")
        assert lengths[0] <= lengths[-1], "First truncation should be <= last"
        # At least some variation expected for a 3-segment function
        assert len(set(lengths)) > 1, f"Expected variation in truncation lengths, got {lengths}"
        print("  Variation test passed.\n")

    print("All tests completed.")
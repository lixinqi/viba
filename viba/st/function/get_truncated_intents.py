import os
import json
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
def get_truncated_intents_forward(
    input_tensor: torch.Tensor,
    num_parts: int
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Forward pass: read the input tensor (Tensor[Viba] storing paths to files
    containing raw viba code), apply truncation to generate a base intent and
    multiple truncated variants for each sample.

    Returns:
        intent_base_tensor: Tensor[Viba] of shape (batch, 1, feature_len) storing paths
                            to files containing the base Viba intent.
        truncated_intents: list of num_parts Tensor[Viba], each of shape
                           (batch, 1, feature_len) storing paths to files containing
                           one truncated Viba intent per sample.
    """
    # Retrieve the raw viba code strings for each sample
    input_contents_2d = convert_st_tensor_to_file_contents(input_tensor)
    batch_size = input_tensor.shape[0]
    input_viba_codes = [input_contents_2d[i][0] for i in range(batch_size)]

    # Apply truncation logic per sample.
    # get_all_truncated_vibe_code(raw_viba_code: str, num_parts: int) -> (str, list[str])
    base_intents = []
    truncated_by_part: List[List[str]] = [[] for _ in range(num_parts)]
    for viba_code in input_viba_codes:
        base, truncated = get_all_truncated_vibe_code(viba_code, num_parts)
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
) -> torch.Tensor:
    """
    Backward pass: combine the gradients from intent_base and truncated_intents
    to produce the input gradient Tensor[Diff[Viba]].

    Returns:
        input_grad: Tensor[Diff[Viba]] of shape (batch, 1, feature_len)
    """
    # Read the gradient contents (Diff strings)
    base_grad_contents = convert_st_tensor_to_file_contents(intent_base_grad)
    truncated_grad_contents_per_part = [
        convert_st_tensor_to_file_contents(g) for g in truncated_intents_grad
    ]

    batch_size = intent_base_grad.shape[0]

    # Combine base grad and truncated grads into a single input grad per sample
    input_grad_strings = []
    for i in range(batch_size):
        base_grad = base_grad_contents[i][0]
        trunc_grads = [truncated_grad_contents_per_part[p][i][0] for p in range(num_parts)]
        # Combine all grads into a single diff string
        combined = base_grad + "\n" + "\n".join(trunc_grads)
        input_grad_strings.append(combined)

    feature_len = intent_base_grad.shape[2]
    root_dir = getattr(intent_base_grad, 'st_relative_to', None)

    input_grad_tensor = convert_file_contents_to_st_tensor(
        file_contents=input_grad_strings,
        relative_to=root_dir,
        max_use_count=1,
        feature_len=feature_len
    )
    input_grad_tensor.st_file_content_type = "Diff[Viba]"

    return input_grad_tensor

# ----------------------------------------------------------------------
# Custom autograd Function
# ----------------------------------------------------------------------
class GetTruncatedIntents(Function):
    @staticmethod
    def forward(ctx, input_tensor, num_parts):
        ctx.save_for_backward(input_tensor, torch.tensor(num_parts))
        intent_base, truncated_list = get_truncated_intents_forward(input_tensor, num_parts)
        return (intent_base, *truncated_list)

    @staticmethod
    def backward(ctx, grad_intent_base, *grad_truncated_intents):
        input_tensor, num_parts_tensor = ctx.saved_tensors
        num_parts = num_parts_tensor.item()
        input_grad = get_truncated_intents_backward(
            grad_intent_base, list(grad_truncated_intents), input_tensor, num_parts
        )
        # Return grads for (input_tensor, num_parts) — num_parts has no grad
        return input_grad, None


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
        # Each sample is raw viba code (Tensor[Viba])
        # Sample 0: a function implementation with 4 exponent chain args
        sample_0 = "compute := $result int <- $a int <- $b int <- $op str"
        # Sample 1: a class definition (no truncation variation)
        sample_1 = "Person := $name str * $age int"

        # Create input tensor (paths to files containing viba code)
        input_tensor = convert_file_contents_to_st_tensor(
            file_contents=[sample_0, sample_1],
            relative_to=tmpdir,
            max_use_count=1,
            feature_len=256
        )
        input_tensor.st_file_content_type = "Viba"

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
            print(f"  Base {i}: {repr(content[:80])}")
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
        grad_base_strings = ["diff: base_change"] * 2

        grad_base = convert_file_contents_to_st_tensor(
            grad_base_strings, relative_to=tmpdir, max_use_count=1, feature_len=256
        )
        grad_base.st_file_content_type = "Diff[Viba]"

        # Create one grad tensor per truncation part
        grad_trunc_list = []
        for p in range(num_parts):
            grad_trunc_strings = [f"diff: truncated_part_{p}"] * 2
            gt = convert_file_contents_to_st_tensor(
                grad_trunc_strings, relative_to=tmpdir, max_use_count=1, feature_len=256
            )
            gt.st_file_content_type = "Diff[Viba]"
            grad_trunc_list.append(gt)

        input_grad = get_truncated_intents_backward(
            grad_base, grad_trunc_list, input_tensor, num_parts
        )

        assert input_grad.shape == (2, 1, 256), f"Unexpected input_grad shape: {input_grad.shape}"
        assert input_grad.dtype == torch.uint8
        assert input_grad.st_file_content_type == "Diff[Viba]"

        grad_paths = convert_2d_tensor_to_list_str(input_grad[:, 0, :])
        for i, path in enumerate(grad_paths):
            full = Path(tmpdir) / path
            assert full.exists(), f"Input grad file {i} missing"
            content = full.read_text(encoding='utf-8')
            assert "base_change" in content, f"Input grad {i} missing base grad content"
            assert "truncated_part_0" in content, f"Input grad {i} missing truncated grad content"
        print("  Backward test passed.\n")

        # ── Test 3: Verify truncation produces variation for function impl ──
        print("Test 3: Truncation variation for function implementation")
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

import os
import subprocess
import tempfile
from typing import List, Tuple

import torch
from torch.autograd import Function

from symbolic_tensor.tensor_util.convert_st_tensor_to_file_contents import convert_st_tensor_to_file_contents
from symbolic_tensor.tensor_util.convert_file_contents_to_st_tensor import convert_file_contents_to_st_tensor

# ----------------------------------------------------------------------
# Helper: run diff CLI on two strings, return the diff output
# ----------------------------------------------------------------------
def _get_diff(actual_content: str, expected_content: str) -> str:
    """
    Run the diff CLI on two content strings via temporary files.
    Returns the unified diff output (empty string if files are identical).
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.actual', delete=False) as f_actual, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.expected', delete=False) as f_expected:
        f_actual.write(actual_content)
        f_actual.flush()
        f_expected.write(expected_content)
        f_expected.flush()

        result = subprocess.run(
            ["diff", "-u", f_expected.name, f_actual.name],
            capture_output=True,
            text=True,
        )
        # diff returns 0 if identical, 1 if different, 2 on error
        diff_output = result.stdout

        os.unlink(f_actual.name)
        os.unlink(f_expected.name)

    return diff_output


def _num_lines(content: str) -> int:
    """Count the number of lines in a string. Empty string has 0 lines."""
    if not content:
        return 0
    return len(content.splitlines())


# ----------------------------------------------------------------------
# Forward implementation
# ----------------------------------------------------------------------
def get_diff_ratio_forward(
    actual_tensor: torch.Tensor,
    expected_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Forward pass: for each sample in the batch, compute the diff ratio between
    the actual and expected file contents.

    diff_ratio = num_lines(diff_file) / num_lines(expected_file)

    Returns a float32 tensor of shape (batch_size,).
    """
    actual_contents_2d = convert_st_tensor_to_file_contents(actual_tensor)
    expected_contents_2d = convert_st_tensor_to_file_contents(expected_tensor)

    batch_size = actual_tensor.shape[0]
    ratios = []

    for i in range(batch_size):
        actual_content = actual_contents_2d[i][0]
        expected_content = expected_contents_2d[i][0]

        diff_output = _get_diff(actual_content, expected_content)

        diff_lines = _num_lines(diff_output)
        expected_lines = _num_lines(expected_content)

        if expected_lines == 0:
            ratio = 0.0
        else:
            ratio = diff_lines / expected_lines

        ratios.append(ratio)

    return torch.tensor(ratios, dtype=torch.float32)


# ----------------------------------------------------------------------
# Backward implementation
# ----------------------------------------------------------------------
def get_diff_ratio_backward(
    grad_output_tensor: torch.Tensor,
    actual_tensor: torch.Tensor,
    expected_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Backward pass: compute the diff between actual and expected, store the diff
    text as the actual gradient. The output_grad scalar is attached as
    st_diff_coefficient_tensor for downstream use.

    Returns actual_grad (Tensor[Diff]) with shape copied from actual_tensor.
    """
    actual_contents_2d = convert_st_tensor_to_file_contents(actual_tensor)
    expected_contents_2d = convert_st_tensor_to_file_contents(expected_tensor)

    batch_size = actual_tensor.shape[0]
    diff_strings = []

    for i in range(batch_size):
        actual_content = actual_contents_2d[i][0]
        expected_content = expected_contents_2d[i][0]

        diff_output = _get_diff(actual_content, expected_content)
        diff_strings.append(diff_output)

    feature_len = actual_tensor.shape[2]
    root_dir = getattr(actual_tensor, 'st_relative_to', None)
    max_use_count = actual_tensor.shape[1]

    actual_grad = convert_file_contents_to_st_tensor(
        file_contents=diff_strings,
        relative_to=root_dir,
        max_use_count=max_use_count,
        feature_len=feature_len
    )
    actual_grad.st_file_content_type = "Diff"
    actual_grad.st_diff_coefficient_tensor = grad_output_tensor.clone()

    return actual_grad


# ----------------------------------------------------------------------
# Custom autograd Function
# ----------------------------------------------------------------------
class GetDiffRatio(Function):
    @staticmethod
    def forward(ctx, actual_tensor, expected_tensor):
        ctx.save_for_backward(actual_tensor, expected_tensor)
        return get_diff_ratio_forward(actual_tensor, expected_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        actual_tensor, expected_tensor = ctx.saved_tensors
        actual_grad = get_diff_ratio_backward(
            grad_output, actual_tensor, expected_tensor
        )
        # actual_tensor gets grad; expected does not require grad
        return actual_grad, None


def get_diff_ratio(actual_tensor, expected_tensor):
    """Convenience wrapper for GetDiffRatio autograd Function."""
    return GetDiffRatio.apply(actual_tensor, expected_tensor)

# ----------------------------------------------------------------------
# Unit tests (only in __main__)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from symbolic_tensor.data_loader.sole_file_batch_data_loader import SoleFileBatchDataLoader
    from symbolic_tensor.data_loader.convert_list_str_to_2d_tensor import convert_2d_tensor_to_list_str
    from symbolic_tensor.tensor_util.convert_st_tensor_to_file_contents import convert_st_tensor_to_file_contents
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create actual and expected file contents, store them as path tensors
        actual_contents = [
            "line1\nline2\nline3\n",
            "alpha\nbeta\ngamma\n",
        ]
        expected_contents = [
            "line1\nline2\nline3\n",       # identical to actual[0]
            "alpha\nBETA\ngamma\ndelta\n", # differs from actual[1]
        ]

        # Build path tensors using convert_file_contents_to_st_tensor
        actual_tensor = convert_file_contents_to_st_tensor(
            file_contents=actual_contents,
            relative_to=tmpdir,
            max_use_count=1,
            feature_len=256,
        )
        actual_tensor.st_file_content_type = "T"

        expected_tensor = convert_file_contents_to_st_tensor(
            file_contents=expected_contents,
            relative_to=tmpdir,
            max_use_count=1,
            feature_len=256,
        )
        expected_tensor.st_file_content_type = "T"

        # -------------------- Test 1: Forward pass --------------------
        print("Test 1: Forward pass")
        out = get_diff_ratio(actual_tensor, expected_tensor)
        assert out.shape == (2,), f"Unexpected shape: {out.shape}"
        assert out.dtype == torch.float32, f"Unexpected dtype: {out.dtype}"

        # Sample 0: identical files -> diff_ratio should be 0.0
        assert out[0].item() == 0.0, f"Expected 0.0 for identical files, got {out[0].item()}"
        # Sample 1: different files -> diff_ratio should be > 0.0
        assert out[1].item() > 0.0, f"Expected > 0.0 for different files, got {out[1].item()}"
        print(f"  Ratios: {out.tolist()}")
        print("  Forward test passed.\n")

        # -------------------- Test 2: Backward pass --------------------
        print("Test 2: Backward pass")
        dummy_grad = torch.tensor([1.0, 0.5], dtype=torch.float32)
        actual_grad = get_diff_ratio_backward(dummy_grad, actual_tensor, expected_tensor)

        assert actual_grad.shape == (2, 1, 256), f"Unexpected actual_grad shape: {actual_grad.shape}"
        assert actual_grad.dtype == torch.bfloat16
        assert actual_grad.st_file_content_type == "Diff"
        assert hasattr(actual_grad, 'st_diff_coefficient_tensor'), "Missing st_diff_coefficient_tensor"
        assert torch.equal(actual_grad.st_diff_coefficient_tensor, dummy_grad), \
            f"st_diff_coefficient_tensor mismatch"

        # Verify diff contents: sample 0 should be empty diff, sample 1 non-empty
        grad_contents = convert_st_tensor_to_file_contents(actual_grad)
        assert grad_contents[0][0] == "", f"Expected empty diff for identical files, got {repr(grad_contents[0][0])}"
        assert len(grad_contents[1][0]) > 0, "Expected non-empty diff for different files"
        print(f"  Diff content sample 1 (first 80 chars): {repr(grad_contents[1][0][:80])}")
        print("  Backward test passed.\n")

        # -------------------- Test 3: Autograd Function returns (actual_grad, None) --------------------
        print("Test 3: Autograd Function returns correct grad tuple")
        class MockCtx:
            saved_tensors = (actual_tensor, expected_tensor)
        mock_ctx = MockCtx()
        result = GetDiffRatio.backward(mock_ctx, dummy_grad)
        assert isinstance(result, tuple) and len(result) == 2, f"Expected tuple of 2, got {type(result)}"
        assert result[0] is not None, "actual_grad should not be None"
        assert result[1] is None, "expected_grad should be None"
        assert result[0].st_file_content_type == "Diff"
        assert hasattr(result[0], 'st_diff_coefficient_tensor')
        print("  Autograd Function test passed.\n")

        # -------------------- Test 4: loss.sum().backward() works --------------------
        print("Test 4: loss.sum().backward()")
        # Recreate actual_tensor with requires_grad=True (bfloat16 supports autograd)
        actual_tensor_grad = convert_file_contents_to_st_tensor(
            file_contents=actual_contents,
            relative_to=tmpdir,
            max_use_count=1,
            feature_len=256,
        )
        actual_tensor_grad.st_file_content_type = "T"
        actual_tensor_grad.requires_grad_(True)
        loss = get_diff_ratio(actual_tensor_grad, expected_tensor)
        loss.sum().backward()
        print(f"  loss = {loss.tolist()}, backward() completed.")
        print("  Backward autograd test passed.\n")

    print("All tests completed.")

import os
import tempfile
from pathlib import Path

import torch
from torch.autograd import Function

from symbolic_tensor.tensor_util.convert_st_tensor_to_file_contents import convert_st_tensor_to_file_contents
from symbolic_tensor.tensor_util.convert_file_contents_to_st_tensor import convert_file_contents_to_st_tensor


def copy_impl(input_tensor: torch.Tensor, dst_relative_to: str) -> torch.Tensor:
    """
    Copy a symbolic tensor's file contents to a new destination directory.

    Reads the files referenced by input_tensor, then writes them under
    dst_relative_to with the same metadata (feature_len, max_use_count,
    st_file_content_type).

    Args:
        input_tensor: A symbolic tensor of shape (batch, max_use_count, feature_len).
        dst_relative_to: Destination root directory for the copied files.

    Returns:
        A new symbolic tensor pointing to the copied files under dst_relative_to.
    """
    contents_2d = convert_st_tensor_to_file_contents(input_tensor)
    # Flatten: take the first slot per sample
    contents = [sample[0] for sample in contents_2d]

    feature_len = input_tensor.shape[2]
    max_use_count = input_tensor.shape[1]
    st_file_content_type = getattr(input_tensor, 'st_file_content_type', 'Any')

    output_tensor = convert_file_contents_to_st_tensor(
        file_contents=contents,
        relative_to=dst_relative_to,
        max_use_count=max_use_count,
        feature_len=feature_len,
    )
    output_tensor.st_file_content_type = st_file_content_type
    return output_tensor


class Copy(Function):
    @staticmethod
    def forward(ctx, input_tensor, dst_relative_to):
        ctx.save_for_backward(input_tensor)
        return copy_impl(input_tensor, dst_relative_to)

    @staticmethod
    def backward(ctx, grad_output):
        (input_tensor,) = ctx.saved_tensors
        input_st_relative_to = getattr(input_tensor, 'st_relative_to', None)
        grad_copied = copy_impl(grad_output, input_st_relative_to)
        return grad_copied, None


copy = Copy.apply


if __name__ == "__main__":
    from symbolic_tensor.data_loader.sole_file_batch_data_loader import SoleFileBatchDataLoader
    from symbolic_tensor.data_loader.convert_list_str_to_2d_tensor import convert_2d_tensor_to_list_str

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  PASS {name}")
        else:
            print(f"  FAIL {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # ---- Test 1: Forward pass ----
    print("Test 1: Forward pass")
    with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
        # Create source files
        src_files = {
            "a.txt": "Hello from file A",
            "b.txt": "Hello from file B",
        }
        for name, content in src_files.items():
            full = os.path.join(src_dir, name)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)

        loader = SoleFileBatchDataLoader(
            root_dir=src_dir,
            file_content_type="T",
            extension=".txt",
            batch_size=2,
            max_use_count=1,
            feature_len=256,
        )
        input_tensor = next(iter(loader))

        output_tensor = copy_impl(input_tensor, dst_dir)

        run_test("shape matches", output_tensor.shape == input_tensor.shape,
                 input_tensor.shape, output_tensor.shape)
        run_test("dtype bfloat16", output_tensor.dtype == torch.bfloat16)
        run_test("content_type matches",
                 output_tensor.st_file_content_type == input_tensor.st_file_content_type,
                 input_tensor.st_file_content_type, output_tensor.st_file_content_type)

        # Verify file contents at destination match source
        out_contents = convert_st_tensor_to_file_contents(output_tensor)
        in_contents = convert_st_tensor_to_file_contents(input_tensor)
        run_test("file contents identical", out_contents == in_contents,
                 in_contents, out_contents)
    print()

    # ---- Test 2: Backward pass (copy_impl on a grad tensor) ----
    print("Test 2: Backward pass (copy_impl)")
    with tempfile.TemporaryDirectory() as grad_dir, tempfile.TemporaryDirectory() as back_dir:
        grad_strings = [
            "diff: change type A",
            "diff: change type B",
        ]
        grad_tensor = convert_file_contents_to_st_tensor(
            grad_strings,
            relative_to=grad_dir,
            max_use_count=1,
            feature_len=256,
        )
        grad_tensor.st_file_content_type = "Diff"

        copied_grad = copy_impl(grad_tensor, back_dir)

        run_test("shape matches", copied_grad.shape == grad_tensor.shape,
                 grad_tensor.shape, copied_grad.shape)
        run_test("dtype bfloat16", copied_grad.dtype == torch.bfloat16)
        run_test("content_type matches",
                 copied_grad.st_file_content_type == grad_tensor.st_file_content_type,
                 grad_tensor.st_file_content_type, copied_grad.st_file_content_type)

        grad_out_contents = convert_st_tensor_to_file_contents(copied_grad)
        grad_in_contents = convert_st_tensor_to_file_contents(grad_tensor)
        run_test("file contents identical", grad_out_contents == grad_in_contents,
                 grad_in_contents, grad_out_contents)
    print()

    # ---- Test 3: Autograd Function backward returns (grad, None) ----
    print("Test 3: Copy.backward returns (grad_tensor, None)")
    with tempfile.TemporaryDirectory() as src_dir, \
         tempfile.TemporaryDirectory() as dst_dir, \
         tempfile.TemporaryDirectory() as grad_dir:
        # Create a source file and build input tensor
        src_path = os.path.join(src_dir, "x.txt")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write("source content")

        loader = SoleFileBatchDataLoader(
            root_dir=src_dir,
            file_content_type="T",
            extension=".txt",
            batch_size=1,
            max_use_count=1,
            feature_len=256,
        )
        input_tensor = next(iter(loader))

        # Build a fake grad_output tensor
        grad_out = convert_file_contents_to_st_tensor(
            ["grad content"],
            relative_to=grad_dir,
            max_use_count=1,
            feature_len=256,
        )
        grad_out.st_file_content_type = "Diff"

        # Simulate what autograd would do: call forward to populate ctx, then backward
        ctx = type('MockCtx', (), {})()
        ctx.save_for_backward = lambda *ts: setattr(ctx, 'saved_tensors', ts)
        Copy.forward(ctx, input_tensor, dst_dir)
        result = Copy.backward(ctx, grad_out)

        run_test("backward returns tuple of length 2", len(result) == 2, 2, len(result))
        run_test("second element is None", result[1] is None)
        run_test("first element is a tensor", isinstance(result[0], torch.Tensor))
        # Verify the grad was copied back to input's st_relative_to
        run_test("grad st_relative_to matches input",
                 result[0].st_relative_to == input_tensor.st_relative_to,
                 input_tensor.st_relative_to, result[0].st_relative_to)

    print("\nAll tests completed.")

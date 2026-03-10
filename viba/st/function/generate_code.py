import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.autograd import Function

# Import the existing convert_st_tensor_to_file_contents implementation from the correct module.
from viba.st.tensor_util.convert_st_tensor_to_file_contents import convert_st_tensor_to_file_contents
# Import the new tensor builder
from viba.st.tensor_util.convert_file_contents_to_st_tensor import convert_file_contents_to_st_tensor

# ----------------------------------------------------------------------
# Helper: call claude via subprocess
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

# ----------------------------------------------------------------------
# Forward implementation
# ----------------------------------------------------------------------
def generate_code_forward(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    output_file_content_type: str
) -> torch.Tensor:
    """
    Forward pass: for each sample in the batch, invoke claude to generate
    code (type T) from Viba intent and a Viba→T mapping.

    The output tensor has the same batch size and second dimension (max_use_count)
    as the input tensor, with data placed in the first layer (index 0). The remaining
    layers are zero-filled to allow for future gradient accumulation.
    """
    input_contents_2d = convert_st_tensor_to_file_contents(input_tensor)
    weight_contents_2d = convert_st_tensor_to_file_contents(weight_tensor)

    batch_size = input_tensor.shape[0]
    output_strings = []

    for i in range(batch_size):
        input_content = input_contents_2d[i][0]
        weight_content = weight_contents_2d[i][0]

        prompt = (
            "Given the following Viba intent code and the existing Viba→T mapping "
            "(output type: " + output_file_content_type + "), "
            "generate the target code. "
            "Viba intent:\n" + input_content +
            "\nMapping:\n" + weight_content
        )

        result_code = _call_claude(prompt)
        output_strings.append(result_code)

    feature_len = input_tensor.shape[2]
    root_dir = getattr(input_tensor, 'st_relative_to', None)
    max_use_count = input_tensor.shape[1]   # second dimension from input

    # Use the tensor builder to store the outputs as files and return a path tensor
    output_tensor = convert_file_contents_to_st_tensor(
        file_contents=output_strings,
        relative_to=root_dir,
        max_use_count=max_use_count,        # preserve input's second dimension
        feature_len=feature_len
    )
    # Set the content type from the explicit parameter.
    output_tensor.st_file_content_type = output_file_content_type
    return output_tensor

# ----------------------------------------------------------------------
# Backward implementation
# ----------------------------------------------------------------------
def generate_code_backward(
    grad_output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass: based on the gradient of the output (Diff strings), produce
    both input gradient and weight gradient.

    Returns:
        A tuple of (input_grad, weight_grad):
        - input_grad: Tensor[Diff[Viba]] with shape matching input tensor
        - weight_grad: Tensor[Json[list[$key Diff[Viba] * $value Diff[T]]]] with shape matching weight tensor
    """
    input_contents_2d = convert_st_tensor_to_file_contents(input_tensor)
    weight_contents_2d = convert_st_tensor_to_file_contents(weight_tensor)
    grad_contents_2d = convert_st_tensor_to_file_contents(grad_output_tensor)

    batch_size = input_tensor.shape[0]
    input_grad_strings = []
    weight_grad_strings = []

    for i in range(batch_size):
        input_content = input_contents_2d[i][0]
        weight_content = weight_contents_2d[i][0]
        grad_content = grad_contents_2d[i][0]

        # Compute input gradient
        input_grad_prompt = (
            "Given the following Viba intent code, the current Viba→T mapping, "
            "and the gradient of the output (as a Diff string), suggest modifications "
            "to the input Viba intent to reduce loss. Output a Diff[Viba] string. "
            "Viba intent:\n" + input_content +
            "\nCurrent mapping:\n" + weight_content +
            "\nOutput gradient (Diff):\n" + grad_content
        )

        input_grad_result = _call_claude(input_grad_prompt)
        input_grad_strings.append(input_grad_result)

        # Compute weight gradient
        weight_grad_prompt = (
            "Given the following Viba intent code, the current Viba→T mapping, "
            "and the gradient of the output (as a Diff string), suggest modifications "
            "to the mapping to reduce loss. Output a JSON object with keys "
            "'key' (Viba fragment diff) and 'diff' (T diff). "
            "Viba intent:\n" + input_content +
            "\nCurrent mapping:\n" + weight_content +
            "\nOutput gradient (Diff):\n" + grad_content
        )

        weight_grad_json = _call_claude(weight_grad_prompt)
        weight_grad_strings.append(weight_grad_json)

    feature_len = input_tensor.shape[2]
    input_root_dir = getattr(input_tensor, 'st_relative_to', None)
    weight_root_dir = getattr(weight_tensor, 'st_relative_to', None)
    input_max_use_count = input_tensor.shape[1]

    # Build input gradient tensor
    input_grad_tensor = convert_file_contents_to_st_tensor(
        file_contents=input_grad_strings,
        relative_to=input_root_dir,
        max_use_count=input_max_use_count,
        feature_len=feature_len
    )
    input_grad_tensor.st_file_content_type = "Diff[Viba]"

    # Build weight gradient tensor
    weight_grad_tensor = convert_file_contents_to_st_tensor(
        file_contents=weight_grad_strings,
        relative_to=weight_root_dir,
        max_use_count=input_max_use_count,
        feature_len=feature_len
    )
    weight_grad_tensor.st_file_content_type = "Json[list[$key Diff[Viba] * $value Diff[T]]]"

    return input_grad_tensor, weight_grad_tensor

# ----------------------------------------------------------------------
# Custom autograd Function
# ----------------------------------------------------------------------
class GenerateCode(Function):
    @staticmethod
    def forward(ctx, input_tensor, weight_tensor, output_file_content_type):
        ctx.save_for_backward(input_tensor, weight_tensor)
        return generate_code_forward(input_tensor, weight_tensor, output_file_content_type)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight_tensor = ctx.saved_tensors
        input_grad, weight_grad = generate_code_backward(
            grad_output, input_tensor, weight_tensor
        )
        # Both input and weight receive gradients; no grad for output_file_content_type
        return input_grad, weight_grad, None

# Convenience alias
generate_code = GenerateCode.apply

# ----------------------------------------------------------------------
# Unit tests (only in __main__)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from viba.st.data_loader.sole_file_batch_data_loader import SoleFileBatchDataLoader
    from viba.st.data_loader.convert_list_str_to_2d_tensor import convert_2d_tensor_to_list_str
    from viba.st.tensor_util.convert_st_tensor_to_file_contents import convert_st_tensor_to_file_contents
    from unittest.mock import patch
    import tempfile

    def mock_claude_response(prompt: str) -> str:
        if "generate the target code" in prompt:  # forward prompt heuristic
            return "def compute(a: int, b: int) -> int:\n    return a + b"
        elif "suggest modifications to the input Viba intent" in prompt:  # input grad prompt
            return "diff: update intent argument types"
        else:  # weight grad prompt
            return json.dumps({"key": "some_viba_key", "diff": "some_t_diff"})

    # Correctly patch the _call_claude function inside __main__
    with patch('__main__._call_claude', side_effect=mock_claude_response):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input Viba intent files
            input_files = {
                "intent1.viba": "compute := $ret int <- $a int <- $b int",
                "intent2.viba": "greet := $ret str <- $name str",
            }
            for name, content in input_files.items():
                full = os.path.join(tmpdir, name)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)

            # Create weight JSON files (Viba→T mappings)
            weight_files = {
                "weight1.json": json.dumps({"viba_key1": "code_val1", "viba_key2": "code_val2"}),
                "weight2.json": json.dumps({"viba_keyA": "code_valA", "viba_keyB": "code_valB"}),
            }
            for name, content in weight_files.items():
                full = os.path.join(tmpdir, name)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)

            # Input tensor (batch=2, max_use_count=1, feature_len=256)
            input_loader = SoleFileBatchDataLoader(
                root_dir=tmpdir,
                file_content_type="Viba",
                extension=".viba",
                batch_size=2,
                max_use_count=1,
                feature_len=256,
            )
            input_tensor = next(iter(input_loader))

            # Weight tensor (batch=2, max_use_count=1, feature_len=256)
            weight_loader = SoleFileBatchDataLoader(
                root_dir=tmpdir,
                file_content_type="Json[list[$key Viba * $value T]]",
                extension=".json",
                batch_size=2,
                max_use_count=1,
                feature_len=256,
            )
            weight_tensor = next(iter(weight_loader))

            # -------------------- Test 1: Forward pass --------------------
            print("Test 1: Forward pass")
            out = generate_code(input_tensor, weight_tensor, "Python")
            # Expect shape (2, 1, 256) because input second dim is 1
            assert out.shape == (2, 1, 256), f"Unexpected shape: {out.shape}"
            assert out.dtype == torch.uint8
            assert out.st_file_content_type == "Python"

            stored_paths = convert_2d_tensor_to_list_str(out[:, 0, :])
            for i, path in enumerate(stored_paths):
                full_path = Path(tmpdir) / path
                print(f"Sample {i} path: {path}")
                assert full_path.exists(), f"File not found: {full_path}"
                written = full_path.read_text(encoding='utf-8')
                print(f"File content (repr): {repr(written)}")
                assert written == "def compute(a: int, b: int) -> int:\n    return a + b", f"Content mismatch for sample {i}"
            print("  Forward test passed.\n")

            # -------------------- Test 2: Backward pass --------------------
            print("Test 2: Backward pass")
            dummy_grad_strings = [
                "diff: change return type",
                "diff: change argument type",
            ]
            grad_out = convert_file_contents_to_st_tensor(
                dummy_grad_strings,
                relative_to=tmpdir,
                max_use_count=1,
                feature_len=256
            )
            grad_out.st_file_content_type = "Diff[T]"

            input_grad, weight_grad = generate_code_backward(grad_out, input_tensor, weight_tensor)

            # Verify input_grad
            assert input_grad.shape == (2, 1, 256), f"Unexpected input_grad shape: {input_grad.shape}"
            assert input_grad.dtype == torch.uint8
            assert input_grad.st_file_content_type == "Diff[Viba]"

            stored_input_grad_paths = convert_2d_tensor_to_list_str(input_grad[:, 0, :])
            for i, path in enumerate(stored_input_grad_paths):
                full_path = Path(tmpdir) / path
                print(f"Input grad {i} path: {path}")
                assert full_path.exists(), f"Input grad file not found: {full_path}"
                written = full_path.read_text(encoding='utf-8')
                print(f"Input grad content: {repr(written)}")
                assert written == "diff: update intent argument types", f"Input grad content mismatch for sample {i}"

            # Verify weight_grad
            assert weight_grad.shape == (2, 1, 256), f"Unexpected weight_grad shape: {weight_grad.shape}"
            assert weight_grad.dtype == torch.uint8
            assert weight_grad.st_file_content_type == "Json[list[$key Diff[Viba] * $value Diff[T]]]"

            stored_weight_grad_paths = convert_2d_tensor_to_list_str(weight_grad[:, 0, :])
            for i, path in enumerate(stored_weight_grad_paths):
                full_path = Path(tmpdir) / path
                print(f"Weight grad {i} path: {path}")
                assert full_path.exists(), f"Weight grad file not found: {full_path}"
                written = full_path.read_text(encoding='utf-8')
                print(f"Weight grad content: {repr(written)}")
                expected = json.dumps({"key": "some_viba_key", "diff": "some_t_diff"})
                assert json.loads(written) == json.loads(expected), f"Weight grad content mismatch for sample {i}"
            print("  Backward test passed.\n")

            # -------------------- Test 3: Autograd Function returns both grads --------------------
            print("Test 3: Autograd Function returns both grads")
            # Verify GenerateCode.backward returns (input_grad, weight_grad, None)
            class MockCtx:
                saved_tensors = (input_tensor, weight_tensor)
            mock_ctx = MockCtx()
            result = GenerateCode.backward(mock_ctx, grad_out)
            assert isinstance(result, tuple) and len(result) == 3, f"Expected tuple of 3, got {type(result)}"
            assert result[0] is not None, "input_grad should not be None"
            assert result[1] is not None, "weight_grad should not be None"
            assert result[2] is None, "output_file_content_type grad should be None"
            assert result[0].st_file_content_type == "Diff[Viba]"
            assert result[1].st_file_content_type == "Json[list[$key Diff[Viba] * $value Diff[T]]]"
            print("  Autograd Function test passed.\n")

    print("All tests completed.")

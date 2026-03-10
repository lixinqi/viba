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
# Helper: encode a list of strings into a 3D uint8 tensor (batch, 1, feature_len)
# ----------------------------------------------------------------------
def encode_strings_to_tensor(
    strings: List[str],
    root_dir: Optional[str] = None,
    content_type: str = "",
    feature_len: int = 4096,
) -> torch.Tensor:
    """Convert a list of strings to a 3D tensor of shape (len(strings), 1, feature_len)."""
    batch = len(strings)
    two_dim = torch.zeros((batch, feature_len), dtype=torch.uint8)
    for i, s in enumerate(strings):
        b = s.encode('utf-8')[:feature_len]
        if b:
            two_dim[i, :len(b)] = torch.tensor(list(b), dtype=torch.uint8)
    three_dim = two_dim.unsqueeze(1)  # shape (batch, 1, feature_len)
    three_dim.st_relative_to = root_dir
    three_dim.st_file_content_type = content_type
    return three_dim

# ----------------------------------------------------------------------
# Forward implementation (calls claude via subprocess)
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

def generate_raw_viba_intent_forward(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Forward pass: for each sample in the batch, invoke claude to generate
    Viba intent code (7 segments per ExponentChain).

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
            "Given the following source code and the existing T→Viba mapping, "
            "generate the Viba intent code. "
            "Source code:\n" + input_content +
            "\nMapping:\n" + weight_content
        )

        result = _call_claude(prompt)
        output_strings.append(result)

    feature_len = input_tensor.shape[2]
    root_dir = getattr(input_tensor, 'st_relative_to', None)
    max_use_count = input_tensor.shape[1]   # second dimension from input, for consistency

    # Use the new tensor builder to store the outputs as files and return a path tensor
    output_tensor = convert_file_contents_to_st_tensor(
        file_contents=output_strings,
        relative_to=root_dir,
        max_use_count=max_use_count,        # preserve input's second dimension
        feature_len=feature_len
    )
    # The builder already sets st_relative_to and st_file_content_type.
    # Override the content type to the expected one.
    output_tensor.st_file_content_type = "Viba"
    return output_tensor

# ----------------------------------------------------------------------
# Backward implementation
# ----------------------------------------------------------------------
def generate_raw_viba_intent_backward(
    grad_output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Backward pass: based on the gradient of the output (Diff strings), produce
    weight gradient (Diff JSON for each weight sample).

    The returned weight gradient tensor has the same shape as the weight tensor
    (batch, max_use_count, feature_len), with data placed in the first layer.
    """
    input_contents_2d = convert_st_tensor_to_file_contents(input_tensor)
    weight_contents_2d = convert_st_tensor_to_file_contents(weight_tensor)
    grad_contents_2d = convert_st_tensor_to_file_contents(grad_output_tensor)

    batch_size = weight_tensor.shape[0]
    weight_grad_strings = []

    for i in range(batch_size):
        input_content = input_contents_2d[i][0]
        weight_content = weight_contents_2d[i][0]
        grad_content = grad_contents_2d[i][0]

        prompt = (
            "Given the following source code, the current T→Viba mapping, "
            "and the gradient of the output (as a Diff string), suggest modifications "
            "to the mapping to reduce loss. Output a JSON object with keys "
            "'key' (T fragment diff) and 'diff' (Viba diff). "
            "Source code:\n" + input_content +
            "\nCurrent mapping:\n" + weight_content +
            "\nOutput gradient (Diff):\n" + grad_content
        )

        weight_grad_json = _call_claude(prompt)
        weight_grad_strings.append(weight_grad_json)

    feature_len = weight_tensor.shape[2]
    root_dir = getattr(weight_tensor, 'st_relative_to', None)
    max_use_count = input_tensor.shape[1]   # second dimension from weight tensor

    # Use the new tensor builder for the weight gradient as well
    grad_tensor = convert_file_contents_to_st_tensor(
        file_contents=weight_grad_strings,
        relative_to=root_dir,
        max_use_count=max_use_count,        # preserve weight's second dimension
        feature_len=feature_len
    )
    grad_tensor.st_file_content_type = "Json[list[$key Diff[T] * $value Diff[Viba]]]"
    return grad_tensor

# ----------------------------------------------------------------------
# Custom autograd Function
# ----------------------------------------------------------------------
class GenerateRawVibaIntent(Function):
    @staticmethod
    def forward(ctx, input_tensor, weight_tensor):
        ctx.save_for_backward(input_tensor, weight_tensor)
        return generate_raw_viba_intent_forward(input_tensor, weight_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight_tensor = ctx.saved_tensors
        weight_grad = generate_raw_viba_intent_backward(
            grad_output, input_tensor, weight_tensor
        )
        # input does not require grad
        return None, weight_grad

# Convenience alias
generate_raw_viba_intent = GenerateRawVibaIntent.apply

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
        if "generate the Viba intent code" in prompt:  # forward prompt heuristic
            return "compute := $ret int <- $a int <- $b int"
        else:  # backward prompt
            return json.dumps({"key": "some_key", "diff": "some_diff"})

    # Correctly patch the _call_claude function inside __main__
    with patch('__main__._call_claude', side_effect=mock_claude_response):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input Python files
            input_files = {
                "sample1.py": "print('hello 1')",
                "sample2.py": "print('hello 2')",
            }
            for name, content in input_files.items():
                full = os.path.join(tmpdir, name)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)

            # Create weight JSON files
            weight_files = {
                "weight1.json": json.dumps({"py_key1": "viba_val1", "py_key2": "viba_val2"}),
                "weight2.json": json.dumps({"py_keyA": "viba_valA", "py_keyB": "viba_valB"}),
            }
            for name, content in weight_files.items():
                full = os.path.join(tmpdir, name)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)

            # Input tensor (batch=2, max_use_count=1, feature_len=256)
            input_loader = SoleFileBatchDataLoader(
                root_dir=tmpdir,
                file_content_type="T",
                extension=".py",
                batch_size=2,
                max_use_count=1,
                feature_len=256,
            )
            input_tensor = next(iter(input_loader))

            # Weight tensor (batch=2, max_use_count=1, feature_len=256)
            weight_loader = SoleFileBatchDataLoader(
                root_dir=tmpdir,
                file_content_type="Json[list[$key T * $value Viba]]",
                extension=".json",
                batch_size=2,
                max_use_count=1,
                feature_len=256,
            )
            weight_tensor = next(iter(weight_loader))

            # -------------------- Test 1: Forward pass --------------------
            print("Test 1: Forward pass")
            out = generate_raw_viba_intent(input_tensor, weight_tensor)
            # Expect shape (2, 1, 256) because input second dim is 1
            assert out.shape == (2, 1, 256), f"Unexpected shape: {out.shape}"
            assert out.dtype == torch.uint8
            assert out.st_file_content_type == "Viba"

            stored_paths = convert_2d_tensor_to_list_str(out[:, 0, :])
            for i, path in enumerate(stored_paths):
                full_path = Path(tmpdir) / path
                print(f"Sample {i} path: {path}")
                assert full_path.exists(), f"File not found: {full_path}"
                written = full_path.read_text(encoding='utf-8')
                print(f"File content (repr): {repr(written)}")
                assert written == "compute := $ret int <- $a int <- $b int", f"Content mismatch for sample {i}"
            print("  Forward test passed.\n")

            # -------------------- Test 2: Backward pass --------------------
            print("Test 2: Backward pass")
            dummy_grad_strings = [
                "diff: change result type",
                "diff: change argument type",
            ]
            grad_out = convert_file_contents_to_st_tensor(
                dummy_grad_strings,
                relative_to=tmpdir,
                max_use_count=1,
                feature_len=256
            )
            grad_out.st_file_content_type = "Diff[Viba]"

            weight_grad = generate_raw_viba_intent_backward(grad_out, input_tensor, weight_tensor)

            assert weight_grad.shape == (2, 1, 256), f"Unexpected weight_grad shape: {weight_grad.shape}"
            assert weight_grad.dtype == torch.uint8
            assert weight_grad.st_file_content_type == "Json[list[$key Diff[T] * $value Diff[Viba]]]"

            stored_weight_grad_paths = convert_2d_tensor_to_list_str(weight_grad[:, 0, :])
            for i, path in enumerate(stored_weight_grad_paths):
                full_path = Path(tmpdir) / path
                print(f"Weight grad {i} path: {path}")
                assert full_path.exists(), f"Weight grad file not found: {full_path}"
                written = full_path.read_text(encoding='utf-8')
                print(f"Weight grad content: {repr(written)}")
                expected = json.dumps({"key": "some_key", "diff": "some_diff"})
                assert json.loads(written) == json.loads(expected), f"Weight grad content mismatch for sample {i}"
            print("  Backward test passed.\n")

            # -------------------- Test 3: Input gradient is None (not applicable) --------------------
            print("Test 3: No autograd dependency")
            print("  (Input gradient concept is not applicable with uint8 tensors.)\n")

    print("All tests completed.")

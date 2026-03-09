import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.autograd import Function

# Import the existing get_file_content implementation from the correct module.
from viba.st.tensor_util.get_file_content import get_file_content

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
    a JSON list of 6 intent segments.
    """
    input_contents_2d = get_file_content(input_tensor)
    weight_contents_2d = get_file_content(weight_tensor)

    batch_size = input_tensor.shape[0]
    output_strings = []

    for i in range(batch_size):
        input_content = input_contents_2d[i][0]
        weight_content = weight_contents_2d[i][0]

        prompt = (
            "Given the following Python code and the existing Python→Viba mapping, "
            "generate exactly 6 Viba intent segments (as a JSON list of strings). "
            "Python code:\n" + input_content +
            "\nMapping:\n" + weight_content
        )

        result_json = _call_claude(prompt)
        output_strings.append(result_json)

    feature_len = input_tensor.shape[2]
    root_dir = getattr(input_tensor, 'st_relative_to', None)
    output_tensor = encode_strings_to_tensor(
        output_strings,
        root_dir=root_dir,
        content_type="Json[list[list[str]]]",
        feature_len=feature_len
    )
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
    """
    input_contents_2d = get_file_content(input_tensor)
    weight_contents_2d = get_file_content(weight_tensor)
    grad_contents_2d = get_file_content(grad_output_tensor)

    batch_size = weight_tensor.shape[0]
    weight_grad_strings = []

    for i in range(batch_size):
        input_content = input_contents_2d[i][0]
        weight_content = weight_contents_2d[i][0]
        grad_content = grad_contents_2d[i][0]

        prompt = (
            "Given the following Python code, the current Python→Viba mapping, "
            "and the gradient of the output (as a Diff string), suggest modifications "
            "to the mapping to reduce loss. Output a JSON object with keys "
            "'key' (Python fragment) and 'diff' (Viba diff). "
            "Python code:\n" + input_content +
            "\nCurrent mapping:\n" + weight_content +
            "\nOutput gradient (Diff):\n" + grad_content
        )

        weight_grad_json = _call_claude(prompt)
        weight_grad_strings.append(weight_grad_json)

    feature_len = weight_tensor.shape[2]
    root_dir = getattr(weight_tensor, 'st_relative_to', None)
    grad_tensor = encode_strings_to_tensor(
        weight_grad_strings,
        root_dir=root_dir,
        content_type="Json[list[$key Diff[Python] * $value Diff[Viba]]]",
        feature_len=feature_len
    )
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
    from viba.st.data_loader.sole_file_batch_data_loader import (
        SoleFileBatchDataLoader,
    )
    from viba.st.data_loader.convert_list_str_to_2d_tensor import (
        convert_2d_tensor_to_list_str,
    )
    from unittest.mock import patch

    def mock_claude_response(prompt: str) -> str:
        if "Viba intent segments" in prompt:  # forward
            return json.dumps(["intent1", "intent2", "intent3", "intent4", "intent5", "intent6"])
        else:  # backward
            return json.dumps({"key": "some_key", "diff": "some_diff"})

    with patch(__name__ + "._call_claude", side_effect=mock_claude_response):
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

            # Input tensor (batch=2)
            input_loader = SoleFileBatchDataLoader(
                root_dir=tmpdir,
                file_content_type="Python",
                extension=".py",
                batch_size=2,
                max_use_count=1,
                feature_len=128,
            )
            input_tensor = next(iter(input_loader))

            # Weight tensor (batch=2)
            weight_loader = SoleFileBatchDataLoader(
                root_dir=tmpdir,
                file_content_type="Json[list[$key Python * $value Viba]]",
                extension=".json",
                batch_size=2,
                max_use_count=1,
                feature_len=128,
            )
            weight_tensor = next(iter(weight_loader))

            # Test 1: Forward pass
            print("Test 1: Forward pass")
            out = generate_raw_viba_intent(input_tensor, weight_tensor)
            assert out.shape == (2, 1, 128)
            assert out.dtype == torch.uint8
            assert out.st_file_content_type == "Json[list[list[str]]]"
            first_layer = out[:, 0, :]
            decoded = convert_2d_tensor_to_list_str(first_layer)
            for json_str in decoded:
                data = json.loads(json_str)
                assert isinstance(data, list) and len(data) == 6
            print("  Forward test passed.\n")

            # Test 2: Backward pass (direct function call, not through autograd)
            print("Test 2: Backward pass")
            # Create a dummy grad_output tensor
            dummy_grad_strings = [
                json.dumps([{"diff": "change1"}, {"diff": "change2"}]),
                json.dumps([{"diff": "changeA"}, {"diff": "changeB"}]),
            ]
            grad_out = encode_strings_to_tensor(
                dummy_grad_strings,
                root_dir=tmpdir,
                content_type="Json[list[list[Diff[str]]]]",
                feature_len=128
            )

            # Call backward function directly
            weight_grad = generate_raw_viba_intent_backward(grad_out, input_tensor, weight_tensor)

            assert weight_grad.shape == weight_tensor.shape
            assert weight_grad.dtype == torch.uint8
            assert weight_grad.st_file_content_type == "Json[list[$key Diff[Python] * $value Diff[Viba]]]"
            print("  Backward test passed.\n")

            # Test 3: Check that input gradient is None (if we had autograd, but we skip it)
            print("Test 3: No autograd dependency")
            # In our design, input does not require grad, so we don't need to check anything.
            print("  (Input gradient concept is not applicable with uint8 tensors.)\n")

    print("All tests completed.")

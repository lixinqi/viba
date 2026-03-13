import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from symbolic_tensor.function.generate_code import generate_code


class CodeGenerator(nn.Module):
    """
    A PyTorch module that holds references to Viba→T mapping files as a persistent buffer.

    During initialization, all JSON mapping files located under `weight_dir` are enumerated,
    and their relative paths are stored in a 3D bfloat16 tensor of shape (1, num_mappings, feature_len).
    This tensor is registered as a buffer, with the attribute `st_relative_to` set to `weight_dir`.

    In the forward pass, the input tensor must also contain file paths (with its own `st_relative_to`).
    The module expands the weight buffer to match the batch size and passes both tensors to the
    imported autograd Function `generate_code`, which internally uses `convert_st_tensor_to_file_contents`
    to read the actual content from the files. The result is a tensor of shape (batch, max_use_count, feature_len)
    containing generated code per sample.

    Args:
        weight_dir (str): Path to a directory containing .json files. Each file must contain
            a valid JSON object mapping Viba fragments (keys) to T fragments (values).
        feature_len (int): Fixed byte length for encoding each relative path. Default: 4096.
    """

    def __init__(self, weight_dir: str, output_file_content_type: str = "T", feature_len: int = 4096):
        super().__init__()
        self.feature_len = feature_len
        self.weight_dir = weight_dir
        self.output_file_content_type = output_file_content_type

        weight_tensor = self._build_path_tensor(weight_dir, feature_len)
        weight_tensor.st_relative_to = weight_dir
        self.register_buffer('weight', weight_tensor)   # shape: (1, num_files, feature_len)

    def _build_path_tensor(self, weight_dir: str, feature_len: int) -> torch.Tensor:
        """
        Recursively collect all .json files under `weight_dir`, compute their relative paths,
        and encode each relative path as a zero-padded UTF-8 byte sequence of length `feature_len`.

        Returns:
            A bfloat16 tensor of shape (1, num_files, feature_len).
        """
        json_files = sorted(Path(weight_dir).glob('**/*.json'))
        if not json_files:
            raise RuntimeError(f"No JSON mapping files found in {weight_dir}")

        encoded_rows = []
        for path in json_files:
            rel_path = str(path.relative_to(weight_dir))
            b = rel_path.encode('utf-8')[:feature_len]
            row = torch.zeros(feature_len, dtype=torch.bfloat16)
            if b:
                row[:len(b)] = torch.tensor(list(b), dtype=torch.bfloat16)
            encoded_rows.append(row)

        # Stack rows to (num_files, feature_len) and add a batch dimension
        return torch.stack(encoded_rows, dim=0).unsqueeze(0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Execute one forward pass.

        Args:
            input_tensor (torch.Tensor): bfloat16 tensor of shape (batch, max_use_count, feature_len)
                containing UTF-8 encoded relative paths to Viba intent files. Must have the
                attribute `st_relative_to` pointing to the root directory of those files.

        Returns:
            torch.Tensor: bfloat16 tensor of shape (batch, max_use_count, feature_len) containing
                generated code (type T).
        """
        batch_size = input_tensor.shape[0]
        expanded_weight = self.weight.expand(batch_size, -1, -1)
        # Expand creates a view; preserve the st_relative_to attribute.
        if hasattr(self.weight, 'st_relative_to'):
            expanded_weight.st_relative_to = self.weight.st_relative_to
        return generate_code(input_tensor, expanded_weight, self.output_file_content_type)


if __name__ == "__main__":
    # Imports allowed only in __main__
    from symbolic_tensor.data_loader.convert_list_str_to_2d_tensor import (
        convert_2d_tensor_to_list_str,
    )
    from unittest.mock import patch
    import tempfile

    # ------------------------------------------------------------------
    # Mock the internal _call_claude function used by generate_code.
    # ------------------------------------------------------------------
    def mock_claude_response(prompt: str) -> str:
        if "generate the target code" in prompt:
            return "def compute(a: int, b: int) -> int:\n    return a + b"
        else:
            return json.dumps({"key": "some_viba_key", "diff": "some_t_diff"})

    target_patch = 'symbolic_tensor.function.generate_code._call_claude'

    with patch(target_patch, side_effect=mock_claude_response):
        with tempfile.TemporaryDirectory() as tmpdir:
            # ------------------------------------------------------------------
            # Prepare input Viba intent files.
            # ------------------------------------------------------------------
            input_files = {
                "intent1.viba": "compute := $ret int <- $a int <- $b int",
                "intent2.viba": "greet := $ret str <- $name str",
            }
            for name, content in input_files.items():
                full = os.path.join(tmpdir, name)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)

            # ------------------------------------------------------------------
            # Prepare weight JSON files (Viba → T mappings).
            # ------------------------------------------------------------------
            weight_files = {
                "weight1.json": json.dumps({"viba_key1": "code_val1", "viba_key2": "code_val2"}),
                "weight2.json": json.dumps({"viba_keyA": "code_valA", "viba_keyB": "code_valB"}),
            }
            for name, content in weight_files.items():
                full = os.path.join(tmpdir, name)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)

            # ------------------------------------------------------------------
            # Instantiate the module under test. It will build the weight buffer
            # containing paths to the JSON files.
            # ------------------------------------------------------------------
            model = CodeGenerator(tmpdir, output_file_content_type="Python", feature_len=256)

            # ------------------------------------------------------------------
            # Create an input tensor that contains paths to the Viba files,
            # and set its st_relative_to attribute.
            # ------------------------------------------------------------------
            input_paths = [["intent1.viba"], ["intent2.viba"]]   # batch=2, max_use_count=1
            feature_len = 256
            encoded_rows = []
            for sample in input_paths:
                for path in sample:
                    b = path.encode('utf-8')[:feature_len]
                    row = torch.zeros(feature_len, dtype=torch.bfloat16)
                    if b:
                        row[:len(b)] = torch.tensor(list(b), dtype=torch.bfloat16)
                    encoded_rows.append(row)
            input_tensor = torch.stack(encoded_rows, dim=0).view(2, 1, feature_len)
            input_tensor.st_relative_to = tmpdir

            # ------------------------------------------------------------------
            # Test 1: Forward pass shape and content.
            # ------------------------------------------------------------------
            print("Test 1: Forward pass")
            output = model(input_tensor)
            assert output.shape == (2, 1, 256), f"Unexpected output shape: {output.shape}"
            assert output.dtype == torch.bfloat16
            assert output.st_file_content_type == "Python"

            # Decode the output and verify content.
            first_layer = output[:, 0, :]
            decoded = convert_2d_tensor_to_list_str(first_layer)
            for path_str in decoded:
                full_path = Path(tmpdir) / path_str
                assert full_path.exists(), f"Output file not found: {full_path}"
                written = full_path.read_text(encoding='utf-8')
                assert written == "def compute(a: int, b: int) -> int:\n    return a + b"
            print("  Forward test passed.\n")

            # ------------------------------------------------------------------
            # Test 2: Weight buffer shape and content.
            # ------------------------------------------------------------------
            print("Test 2: Module weight buffer")
            assert model.weight.shape == (1, 2, 256)   # two mapping files
            weight_paths = convert_2d_tensor_to_list_str(model.weight[0])  # shape (2,)
            expected_paths = ["weight1.json", "weight2.json"]
            assert weight_paths == expected_paths, f"Expected {expected_paths}, got {weight_paths}"
            print(f"  Weight paths: {weight_paths}")
            print(f"  Weight shape: {model.weight.shape}\n")

            print("All tests completed.")

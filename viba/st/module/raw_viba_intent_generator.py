import os
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from viba.st.function.generate_raw_viba_intent import generate_raw_viba_intent


class RawVibaIntentGenerator(nn.Module):
    """
    A PyTorch module that holds references to Python→Viba mapping files as a persistent buffer.

    During initialization, all JSON mapping files located under `weight_dir` are enumerated,
    and their relative paths are stored in a 3D uint8 tensor of shape (1, num_mappings, feature_len).
    This tensor is registered as a buffer, with the attribute `st_relative_to` set to `weight_dir`.

    In the forward pass, the input tensor must also contain file paths (with its own `st_relative_to`).
    The module expands the weight buffer to match the batch size and passes both tensors to the
    imported autograd Function `generate_raw_viba_intent`, which internally uses `get_file_content`
    to read the actual content from the files. The result is a tensor of shape (batch, 1, feature_len)
    containing JSON lists of six Viba intent segments per sample.

    Args:
        weight_dir (str): Path to a directory containing .json files. Each file must contain
            a valid JSON object mapping Python fragments (keys) to Viba fragments (values).
        feature_len (int): Fixed byte length for encoding each relative path. Default: 4096.
    """

    def __init__(self, weight_dir: str, feature_len: int = 4096):
        super().__init__()
        self.feature_len = feature_len
        self.weight_dir = weight_dir

        weight_tensor = self._build_path_tensor(weight_dir, feature_len)
        weight_tensor.st_relative_to = weight_dir
        self.register_buffer('weight', weight_tensor)   # shape: (1, num_files, feature_len)

    def _build_path_tensor(self, weight_dir: str, feature_len: int) -> torch.Tensor:
        """
        Recursively collect all .json files under `weight_dir`, compute their relative paths,
        and encode each relative path as a zero‑padded UTF‑8 byte sequence of length `feature_len`.

        Returns:
            A uint8 tensor of shape (1, num_files, feature_len).
        """
        json_files = sorted(Path(weight_dir).glob('**/*.json'))
        if not json_files:
            raise RuntimeError(f"No JSON mapping files found in {weight_dir}")

        encoded_rows = []
        for path in json_files:
            rel_path = str(path.relative_to(weight_dir))
            b = rel_path.encode('utf-8')[:feature_len]
            row = torch.zeros(feature_len, dtype=torch.uint8)
            if b:
                row[:len(b)] = torch.tensor(list(b), dtype=torch.uint8)
            encoded_rows.append(row)

        # Stack rows to (num_files, feature_len) and add a batch dimension
        return torch.stack(encoded_rows, dim=0).unsqueeze(0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Execute one forward pass.

        Args:
            input_tensor (torch.Tensor): uint8 tensor of shape (batch, max_use_count, feature_len)
                containing UTF‑8 encoded relative paths to Python source files. Must have the
                attribute `st_relative_to` pointing to the root directory of those files.

        Returns:
            torch.Tensor: uint8 tensor of shape (batch, 1, feature_len) containing
                JSON‑encoded lists of six Viba intent segments.
        """
        batch_size = input_tensor.shape[0]
        expanded_weight = self.weight.expand(batch_size, -1, -1)
        # Expand creates a view; preserve the st_relative_to attribute.
        if hasattr(self.weight, 'st_relative_to'):
            expanded_weight.st_relative_to = self.weight.st_relative_to
        return generate_raw_viba_intent(input_tensor, expanded_weight)


if __name__ == "__main__":
    # Imports allowed only in __main__
    from viba.st.data_loader.convert_list_str_to_2d_tensor import (
        convert_2d_tensor_to_list_str,
    )
    from unittest.mock import patch
    import tempfile

    # ------------------------------------------------------------------
    # Mock the internal _call_claude function used by generate_raw_viba_intent.
    # ------------------------------------------------------------------
    def mock_claude_response(prompt: str) -> str:
        if "Viba intent segments" in prompt:
            return json.dumps(["intent1", "intent2", "intent3", "intent4", "intent5", "intent6"])
        else:
            return json.dumps({"key": "some_key", "diff": "some_diff"})

    target_patch = 'viba.st.function.generate_raw_viba_intent._call_claude'

    with patch(target_patch, side_effect=mock_claude_response):
        with tempfile.TemporaryDirectory() as tmpdir:
            # ------------------------------------------------------------------
            # Prepare input Python files.
            # ------------------------------------------------------------------
            input_files = {
                "sample1.py": "print('hello 1')",
                "sample2.py": "print('hello 2')",
            }
            for name, content in input_files.items():
                full = os.path.join(tmpdir, name)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)

            # ------------------------------------------------------------------
            # Prepare weight JSON files (Python → Viba mappings).
            # ------------------------------------------------------------------
            weight_files = {
                "weight1.json": json.dumps({"py_key1": "viba_val1", "py_key2": "viba_val2"}),
                "weight2.json": json.dumps({"py_keyA": "viba_valA", "py_keyB": "viba_valB"}),
            }
            for name, content in weight_files.items():
                full = os.path.join(tmpdir, name)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(content)

            # ------------------------------------------------------------------
            # Instantiate the module under test. It will build the weight buffer
            # containing paths to the JSON files.
            # ------------------------------------------------------------------
            model = RawVibaIntentGenerator(tmpdir, feature_len=128)

            # ------------------------------------------------------------------
            # Create an input tensor that contains paths to the Python files,
            # and set its st_relative_to attribute.
            # ------------------------------------------------------------------
            input_paths = [["sample1.py"], ["sample2.py"]]   # batch=2, max_use_count=1
            feature_len = 128
            encoded_rows = []
            for sample in input_paths:
                for path in sample:
                    b = path.encode('utf-8')[:feature_len]
                    row = torch.zeros(feature_len, dtype=torch.uint8)
                    if b:
                        row[:len(b)] = torch.tensor(list(b), dtype=torch.uint8)
                    encoded_rows.append(row)
            input_tensor = torch.stack(encoded_rows, dim=0).view(2, 1, feature_len)
            input_tensor.st_relative_to = tmpdir

            # ------------------------------------------------------------------
            # Test 1: Forward pass shape and content.
            # ------------------------------------------------------------------
            print("Test 1: Forward pass")
            output = model(input_tensor)
            assert output.shape == (2, 1, 128), f"Unexpected output shape: {output.shape}"
            assert output.dtype == torch.uint8
            assert output.st_file_content_type == "Json[list[list[str]]]"

            # Decode the output and verify JSON structure.
            first_layer = output[:, 0, :]
            decoded = convert_2d_tensor_to_list_str(first_layer)
            for json_str in decoded:
                data = json.loads(json_str)
                assert isinstance(data, list) and len(data) == 6
            print("  Forward test passed.\n")

            # ------------------------------------------------------------------
            # Test 2: Weight buffer shape and content.
            # ------------------------------------------------------------------
            print("Test 2: Module weight buffer")
            assert model.weight.shape == (1, 2, 128)   # two mapping files
            weight_paths = convert_2d_tensor_to_list_str(model.weight[0])  # shape (2,)
            expected_paths = ["weight1.json", "weight2.json"]
            assert weight_paths == expected_paths, f"Expected {expected_paths}, got {weight_paths}"
            print(f"  Weight paths: {weight_paths}")
            print(f"  Weight shape: {model.weight.shape}\n")

            print("All tests completed.")
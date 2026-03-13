from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

from symbolic_tensor.function.get_truncated_intents import get_truncated_intents


class IntentTruncator(nn.Module):
    """
    A PyTorch module that applies truncation to Viba intent segments.

    During initialization, `num_parts` is stored as the number of truncation levels
    to produce in the forward pass.

    In the forward pass, the input tensor (Tensor[Viba]) must contain file paths
    (with `st_relative_to`) pointing to files containing raw viba source code.
    The module delegates to the autograd Function `get_truncated_intents`, which
    internally uses `convert_st_tensor_to_file_contents` to read the actual content,
    applies `get_all_truncated_vibe_code` per sample, and returns:
      - intent_base: Tensor[Viba] of shape (batch, 1, feature_len)
      - truncated_intents: list of num_parts Tensor[Viba], each (batch, 1, feature_len)

    Args:
        num_parts (int): Number of truncation levels to generate. Default: 3.
    """

    def __init__(self, num_parts: int = 3):
        super().__init__()
        self.num_parts = num_parts

    def forward(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Execute one forward pass.

        Args:
            input_tensor (torch.Tensor): Tensor[Viba] - bfloat16 tensor of shape
                (batch, max_use_count, feature_len) containing UTF-8 encoded relative
                paths to files with raw viba source code. Must have the attribute
                `st_relative_to` pointing to the root directory.

        Returns:
            intent_base (torch.Tensor): bfloat16 tensor of shape (batch, 1, feature_len)
                storing paths to files containing the base Viba intent.
            truncated_intents (list[torch.Tensor]): list of num_parts bfloat16 tensors,
                each of shape (batch, 1, feature_len), storing paths to files containing
                one truncated Viba intent per sample.
        """
        return get_truncated_intents(input_tensor, self.num_parts)


if __name__ == "__main__":
    from symbolic_tensor.data_loader.convert_list_str_to_2d_tensor import (
        convert_2d_tensor_to_list_str,
    )
    from symbolic_tensor.tensor_util.convert_file_contents_to_st_tensor import (
        convert_file_contents_to_st_tensor,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # --------------------------------------------------------------
        # Prepare input data: two samples of raw viba code (Tensor[Viba]).
        # --------------------------------------------------------------
        sample_0 = "compute := $result int <- $a int <- $b int <- $op str"
        sample_1 = "Person := $name str * $age int"

        input_tensor = convert_file_contents_to_st_tensor(
            file_contents=[sample_0, sample_1],
            relative_to=tmpdir,
            max_use_count=1,
            feature_len=256,
        )
        input_tensor.st_file_content_type = "Viba"

        num_parts = 3

        # --------------------------------------------------------------
        # Test 1: Forward pass via module.
        # --------------------------------------------------------------
        print("Test 1: Forward pass via IntentTruncator module")
        model = IntentTruncator(num_parts=num_parts)
        intent_base, truncated_intents = model(input_tensor)

        assert intent_base.shape == (2, 1, 256), f"Unexpected base shape: {intent_base.shape}"
        assert intent_base.dtype == torch.bfloat16
        assert intent_base.st_file_content_type == "Viba"
        assert isinstance(truncated_intents, list), "truncated_intents should be a list"
        assert len(truncated_intents) == num_parts, f"Expected {num_parts} tensors, got {len(truncated_intents)}"
        for p, t in enumerate(truncated_intents):
            assert t.shape == (2, 1, 256), f"Unexpected truncated[{p}] shape: {t.shape}"
            assert t.dtype == torch.bfloat16
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

        # --------------------------------------------------------------
        # Test 2: num_parts parameter is respected.
        # --------------------------------------------------------------
        print("Test 2: num_parts parameter")
        assert model.num_parts == 3
        model2 = IntentTruncator(num_parts=5)
        assert model2.num_parts == 5
        intent_base2, truncated2 = model2(input_tensor)
        assert len(truncated2) == 5, f"Expected 5 tensors, got {len(truncated2)}"
        for p, t in enumerate(truncated2):
            assert t.shape == (2, 1, 256), f"Unexpected truncated2[{p}] shape: {t.shape}"
            assert t.st_file_content_type == "Viba"
        print(f"  num_parts=5 produced {len(truncated2)} Tensor[Viba] tensors.")
        print("  num_parts test passed.\n")

    print("All tests completed.")

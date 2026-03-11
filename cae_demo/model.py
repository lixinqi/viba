import os
from pathlib import Path

import torch
import torch.nn as nn

from viba.st.function.generate_raw_viba_intent import generate_raw_viba_intent
from viba.st.function.generate_code import generate_code


class DemoModel(nn.Module):
    """
    Code AutoEncoder demo model.

    Forward: input (T) → RawVibaIntentGenerator → Viba intent → CodeGenerator → output (T)

    Weight files are .viba semantic mappings under weight_dir.
    """

    def __init__(self, weight_dir: str, output_file_content_type: str = "Python", feature_len: int = 4096):
        super().__init__()
        self.feature_len = feature_len
        self.weight_dir = weight_dir
        self.output_file_content_type = output_file_content_type

        weight_tensor = self._build_path_tensor(weight_dir, feature_len)
        weight_tensor.st_relative_to = weight_dir
        weight_tensor.st_file_content_type = "Json[list[$key T * $value Viba]]"
        self.register_buffer('weight', weight_tensor)

    def _build_path_tensor(self, weight_dir: str, feature_len: int) -> torch.Tensor:
        viba_files = sorted(Path(weight_dir).glob('**/*.viba'))
        if not viba_files:
            raise RuntimeError(f"No .viba mapping files found in {weight_dir}")

        encoded_rows = []
        for path in viba_files:
            rel_path = str(path.relative_to(weight_dir))
            b = rel_path.encode('utf-8')[:feature_len]
            row = torch.zeros(feature_len, dtype=torch.uint8)
            if b:
                row[:len(b)] = torch.tensor(list(b), dtype=torch.uint8)
            encoded_rows.append(row)

        return torch.stack(encoded_rows, dim=0).unsqueeze(0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = input_tensor.shape[0]

        # Encoder: T → Viba intent
        enc_w = self.weight.expand(batch_size, -1, -1)
        enc_w.st_relative_to = self.weight_dir
        enc_w.st_file_content_type = "Json[list[$key T * $value Viba]]"
        viba_intent = generate_raw_viba_intent(input_tensor, enc_w)

        # Decoder: Viba intent → T
        dec_w = self.weight.expand(batch_size, -1, -1)
        dec_w.st_relative_to = self.weight_dir
        dec_w.st_file_content_type = "Json[list[$key Viba * $value T]]"
        output = generate_code(viba_intent, dec_w, self.output_file_content_type)

        return output

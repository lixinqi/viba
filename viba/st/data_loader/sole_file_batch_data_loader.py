import os
import torch
from torch.utils.data import IterableDataset

# Import previously defined helper functions
from viba.st.data_loader.get_all_relative_file_paths import get_all_relative_file_paths
from viba.st.data_loader.convert_list_str_to_2d_tensor import convert_list_str_to_2d_tensor

def convert_2d_tensor_to_3d_tensor(two_dim_tensor: torch.Tensor, max_use_count: int) -> torch.Tensor:
    """
    Convert a 2D tensor (batch, feature_len) into a 3D tensor (batch, max_use_count, feature_len).
    The original data is placed in the first layer (index 0); remaining layers are zero.
    This extra dimension is reserved for gradient accumulation via concatenation,
    avoiding addition on uint8 tensors.

    Args:
        two_dim_tensor: Input tensor of shape (batch, feature_len), dtype=torch.uint8.
        max_use_count: Size of the second dimension (for future accumulation).

    Returns:
        A tensor of shape (batch, max_use_count, feature_len), dtype=torch.uint8.
    """
    batch, feature_len = two_dim_tensor.shape
    three_dim = torch.zeros((batch, max_use_count, feature_len), dtype=torch.uint8)
    three_dim[:, 0, :] = two_dim_tensor  # first slot holds the original data
    return three_dim


class SoleFileDataset(IterableDataset):
    """
    An iterable dataset that recursively walks a directory and yields relative file paths as strings.
    """
    def __init__(self, root_dir: str, extension: str = None):
        self.root_dir = root_dir
        self.extension = extension
        self.file_paths = get_all_relative_file_paths(root_dir, extension)

    def __iter__(self):
        for rel_path in self.file_paths:
            yield rel_path


class SoleFileBatchDataLoader:
    """
    A DataLoader-like iterator that yields 3D uint8 tensors from a directory of files.

    Args:
        root_dir: Root directory to scan for files.
        file_content_type: Semantic type of the file content (e.g., "python_code", "text").
        extension: File extension filter (None = all files). (Note: Viba DSL uses "extention", kept as "extension" for consistency.)
        batch_size: Number of files per batch.
        max_use_count: Size of the second dimension (for gradient accumulation).
        feature_len: Fixed byte length for encoding each relative file path.
    """
    def __init__(self, root_dir: str, file_content_type: str, extension: str = None,
                 batch_size: int = 1, max_use_count: int = 64,
                 feature_len: int = 4096):
        self.root_dir = root_dir
        self.file_content_type = file_content_type
        self.extension = extension
        self.batch_size = batch_size
        self.max_use_count = max_use_count
        self.feature_len = feature_len
        self.dataset = SoleFileDataset(root_dir, extension)

    def __iter__(self):
        batch_paths = []
        for rel_path in self.dataset:
            batch_paths.append(rel_path)
            if len(batch_paths) == self.batch_size:
                two_dim = convert_list_str_to_2d_tensor(batch_paths, self.feature_len)
                three_dim = convert_2d_tensor_to_3d_tensor(two_dim, self.max_use_count)
                # Attach metadata required by Viba DSL
                three_dim.st_relative_to = self.root_dir
                three_dim.st_file_content_type = self.file_content_type
                yield three_dim
                batch_paths = []
        # Yield the last incomplete batch if any
        if batch_paths:
            two_dim = convert_list_str_to_2d_tensor(batch_paths, self.feature_len)
            three_dim = convert_2d_tensor_to_3d_tensor(two_dim, self.max_use_count)
            three_dim.st_relative_to = self.root_dir
            three_dim.st_file_content_type = self.file_content_type
            yield three_dim

    def __len__(self):
        total_files = len(self.dataset.file_paths)
        return (total_files + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    # Quick test with a temporary directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        files = {
            "a.txt": "Hello, world!",
            "sub/b.txt": "This is a test.",
            "c.py": "print('hello')",  # will be ignored if extension=".txt"
        }
        for rel_path, content in files.items():
            full = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(content)

        loader = SoleFileBatchDataLoader(
            root_dir=tmpdir,
            file_content_type="text",
            extension=".txt",
            batch_size=2,
            feature_len=20
        )
        print("Number of batches:", len(loader))
        for i, batch in enumerate(loader):
            print(f"Batch {i} shape: {batch.shape}")
            # Check attached metadata
            print(f"  st_relative_to: {batch.st_relative_to}")
            print(f"  st_file_content_type: {batch.st_file_content_type}")
            # Decode the first sample for verification — should be a relative file path
            first_row = batch[0, 0, :]  # first sample, first accumulation slot
            bytes_data = bytes(first_row.tolist())
            zero_pos = bytes_data.find(b'\x00')
            if zero_pos != -1:
                bytes_data = bytes_data[:zero_pos]
            decoded_path = bytes_data.decode('utf-8', errors='replace')
            print("  Decoded first sample (relative path):", decoded_path)
            # Verify the file exists under root_dir
            full_path = os.path.join(tmpdir, decoded_path)
            assert os.path.exists(full_path), f"File not found: {full_path}"
            print("  File exists: True")
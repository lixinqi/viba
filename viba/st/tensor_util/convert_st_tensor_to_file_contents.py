import os
import torch
import tempfile
from pathlib import Path
from typing import List, Optional

# SoleFileBatchDataLoader is imported only in __main__ as required.


def _get_file_path(tensor: torch.Tensor) -> List[List[str]]:
    """
    Decode each slot of a symbolic tensor (bfloat16, shape B x M x F) into a string,
    interpreted as a relative file path (UTF-8, zero‑padded).

    Args:
        tensor: A torch.Tensor of shape (batch, max_use_count, feature_len), dtype=torch.bfloat16.

    Returns:
        A list of length batch, each element is a list of length max_use_count
        containing the decoded strings (relative paths).
    """
    batch, max_use_count, _ = tensor.shape
    result = []
    for i in range(batch):
        sample_paths = []
        for j in range(max_use_count):
            row = tensor[i, j, :]
            bytes_data = bytes(row.to(torch.uint8).tolist())
            # Truncate at first zero byte
            zero_pos = bytes_data.find(b'\x00')
            if zero_pos != -1:
                bytes_data = bytes_data[:zero_pos]
            s = bytes_data.decode('utf-8', errors='replace')
            sample_paths.append(s)
        result.append(sample_paths)
    return result


def convert_st_tensor_to_file_contents(tensor: torch.Tensor) -> List[List[str]]:
    """
    Convert a symbolic tensor (containing relative file paths) into a 2D list of file contents.
    The tensor must have an attribute `st_relative_to` (root directory path). For each slot,
    the decoded relative path is combined with the root directory, and the file content is read
    (UTF‑8 text). If a file does not exist or cannot be read, an empty string is returned.

    Args:
        tensor: A torch.Tensor of shape (batch, max_use_count, feature_len), dtype=torch.bfloat16,
                with attribute `st_relative_to`.

    Returns:
        A list of length batch, each element is a list of length max_use_count
        containing the file contents (strings).
    """
    # Access required attribute
    root_dir: Optional[str] = getattr(tensor, 'st_relative_to', None)
    if root_dir is None:
        raise AttributeError("Tensor must have 'st_relative_to' attribute")

    # Obtain the relative paths from the tensor
    rel_paths_2d = _get_file_path(tensor)

    # Convert root_dir to a Path object (as required by the DSL inline)
    root_path = Path(root_dir)

    result = []
    for sample_paths in rel_paths_2d:
        sample_contents = []
        for rel_path in sample_paths:
            if not rel_path:  # empty path -> empty content
                sample_contents.append("")
                continue
            full_path = root_path / rel_path
            try:
                # Use pathlib.Path.read_text as required
                content = full_path.read_text(encoding='utf-8')
            except Exception:
                content = ""  # fallback on read error
            sample_contents.append(content)
        result.append(sample_contents)

    return result


if __name__ == "__main__":
    # Imports allowed only in __main__
    from viba.st.data_loader.sole_file_batch_data_loader import (
        SoleFileBatchDataLoader,
    )
    from viba.st.data_loader.convert_list_str_to_2d_tensor import (
        convert_list_str_to_2d_tensor,
    )

    # Helper for test reporting
    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            if expected is not None and actual is not None:
                print(f"  expected: {expected}")
                print(f"  actual:   {actual}")

    # Create temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write some files (these will be read back to verify content)
        files = {
            "hello.txt": "Hello, world!",
            "sub/greet.txt": "This is a test.",
            "chinese.txt": "中文测试",
            "empty.txt": "",
        }
        for rel_path, content in files.items():
            full = os.path.join(tmpdir, rel_path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)

        # Prepare a list of relative paths to encode into a tensor.
        # We create two samples, each with max_use_count=3 slots.
        # Sample 0: three paths (hello.txt, sub/greet.txt, empty.txt)
        # Sample 1: two valid paths and one empty slot (represented by empty string)
        paths_batch = [
            ["hello.txt", "sub/greet.txt", "empty.txt"],
            ["chinese.txt", "hello.txt", ""],
        ]

        # Encode the paths into a 3D tensor.
        feature_len = 64  # enough for our short paths
        two_dim = convert_list_str_to_2d_tensor(
            [p for sample in paths_batch for p in sample], feature_len
        )
        batch_size = len(paths_batch)
        max_use_count = len(paths_batch[0])
        three_dim = two_dim.view(batch_size, max_use_count, feature_len).clone()

        # Attach the required metadata
        three_dim.st_relative_to = tmpdir

        # Test 1: basic decoding of paths via _get_file_path
        print("\nTest 1: _get_file_path (path decoding)")
        decoded_paths = _get_file_path(three_dim)
        expected_paths = paths_batch
        run_test("Path decoding matches", decoded_paths == expected_paths,
                 expected_paths, decoded_paths)

        # Test 2: convert_st_tensor_to_file_contents reading actual files and comparing with original content
        print("\nTest 2: convert_st_tensor_to_file_contents (file reading)")
        contents = convert_st_tensor_to_file_contents(three_dim)
        expected_contents = [
            [files["hello.txt"], files["sub/greet.txt"], files["empty.txt"]],
            [files["chinese.txt"], files["hello.txt"], ""],
        ]
        run_test("File contents match", contents == expected_contents,
                 expected_contents, contents)

        # Test 3: Single sample, single slot, non‑existent file (should return empty string)
        print("\nTest 3: Non‑existent file handling")
        bad_paths = [["nonexistent.txt"]]
        two_dim_bad = convert_list_str_to_2d_tensor(
            [bad_paths[0][0]], feature_len
        ).view(1, 1, feature_len)
        two_dim_bad.st_relative_to = tmpdir
        bad_contents = convert_st_tensor_to_file_contents(two_dim_bad)
        run_test("Non‑existent file returns empty string", bad_contents == [[""]],
                 [[""]], bad_contents)

        # Additional verification: ensure that the read content exactly matches the original file data.
        # This is already covered by test 2, but we add a direct check for the first file.
        print("\nTest 4: Direct content comparison via pathlib")
        first_path = os.path.join(tmpdir, "hello.txt")
        original = Path(first_path).read_text(encoding='utf-8')
        # In test 2, we already compared the result with files["hello.txt"], which is the same.
        run_test("Direct read matches stored content", original == files["hello.txt"])

        # (Optional) Show that SoleFileBatchDataLoader was imported – not used here because
        # we are testing path tensors, not content tensors.
        print("\n(SoleFileBatchDataLoader imported successfully)")

        print("\nAll tests completed.")
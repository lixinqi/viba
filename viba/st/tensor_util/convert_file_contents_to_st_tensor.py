import os
import hashlib
import torch
from pathlib import Path
from typing import List


def _get_hashed_tensor_name(file_contents: List[str]) -> str:
    """
    Compute a unique name for a tensor based on the concatenated content of all files.

    Args:
        file_contents: List of strings to be stored.

    Returns:
        SHA‑256 hex digest of the concatenated string.
    """
    joint = "".join(file_contents)
    return hashlib.sha256(joint.encode('utf-8')).hexdigest()


def _get_hashed_file_path(file_content: str, tensor_name: str) -> str:
    """
    Generate a relative file path for a single content string using a hash‑based directory structure.

    The path is of the form: <tensor_name>/<first two chars of hash>/<next two chars>/<full hash>

    Args:
        file_content: The actual content string.
        tensor_name: The tensor name (hash of the whole batch).

    Returns:
        A relative path (using forward slashes) where the file should be stored.
    """
    file_hash = hashlib.sha256(file_content.encode('utf-8')).hexdigest()
    return f"{tensor_name}/{file_hash[0:2]}/{file_hash[2:4]}/{file_hash}"


def _convert_2d_tensor_to_3d_tensor(two_dim: torch.Tensor, max_use_count: int) -> torch.Tensor:
    """
    Convert a 2D tensor (batch, feature_len) into a 3D tensor (batch, max_use_count, feature_len).

    The original data is placed in the first layer (index 0); the remaining layers are zero‑filled.

    Args:
        two_dim: Input tensor of shape (batch, feature_len) with dtype torch.bfloat16.
        max_use_count: Size of the new second dimension.

    Returns:
        A 3D tensor of shape (batch, max_use_count, feature_len).
    """
    batch, feat = two_dim.shape
    three_dim = torch.zeros((batch, max_use_count, feat), dtype=torch.bfloat16)
    three_dim[:, 0, :] = two_dim
    return three_dim


def convert_file_contents_to_st_tensor(
    file_contents: List[str],
    relative_to: str,
    max_use_count: int,
    feature_len: int
) -> torch.Tensor:
    """
    Convert a list of file content strings into a symbolic tensor (3D bfloat16).

    For each string:
      - Write it to disk under the `relative_to` directory using a hash‑based path.
      - Encode the relative path as a fixed‑length UTF‑8 byte sequence (truncated/padded to feature_len).
      - Collect all encoded paths.

    All encoded paths are stacked into a 2D tensor and then expanded to 3D by placing them into the first layer.
    The resulting tensor carries two attributes:
        - st_relative_to: the root directory (relative_to)
        - st_file_content_type: "Any" (can be overridden later)

    Args:
        file_contents: List of strings to be stored as files.
        relative_to: Root directory under which the files will be written.
        max_use_count: Size of the second dimension (reserved for accumulation).
        feature_len: Fixed byte length for encoding each relative path.

    Returns:
        A bfloat16 tensor of shape (batch, max_use_count, feature_len).
    """
    tensor_name = _get_hashed_tensor_name(file_contents)

    encoded_paths = []
    root = Path(relative_to)

    for content in file_contents:
        rel_path = _get_hashed_file_path(content, tensor_name)
        full_path = root / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding='utf-8')

        # Encode the relative path as bytes, truncate/pad to feature_len
        b = rel_path.encode('utf-8')[:feature_len]
        row = torch.zeros(feature_len, dtype=torch.bfloat16)
        if b:
            row[:len(b)] = torch.tensor(list(b), dtype=torch.bfloat16)
        encoded_paths.append(row)

    two_dim = torch.stack(encoded_paths, dim=0)                     # (batch, feature_len)
    three_dim = _convert_2d_tensor_to_3d_tensor(two_dim, max_use_count)  # (batch, max_use_count, feature_len)

    # Attach metadata required by the symbolic tensor framework
    three_dim.st_relative_to = relative_to
    three_dim.st_file_content_type = "Any"

    return three_dim


if __name__ == "__main__":
    import tempfile

    print("Running tests for convert_file_contents_to_st_tensor...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        """Simple test helper that prints ✓ or ✗."""
        if condition:
            print(f"✓ {name}")
        else:
            print(f"✗ {name}")
            if expected is not None and actual is not None:
                print(f"  expected: {expected}")
                print(f"  actual:   {actual}")

    # ------------------------------------------------------------------
    # Test 1: Basic conversion of a single file content
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        content = "print('hello')"
        tensor = convert_file_contents_to_st_tensor(
            [content], tmpdir, max_use_count=3, feature_len=256
        )

        run_test("Test1: Output shape", tensor.shape == (1, 3, 256))
        run_test("Test1: dtype bfloat16", tensor.dtype == torch.bfloat16)
        run_test("Test1: st_relative_to set", tensor.st_relative_to == tmpdir)

        # Decode the stored path from the tensor and verify the file exists with correct content.
        row = tensor[0, 0, :].to(torch.uint8).tolist()
        bytes_data = bytes(row)
        zero_pos = bytes_data.find(b'\x00')
        if zero_pos != -1:
            bytes_data = bytes_data[:zero_pos]
        stored_path = bytes_data.decode('utf-8', errors='replace')
        full_path = Path(tmpdir) / stored_path
        run_test("Test1: File exists", full_path.exists())
        if full_path.exists():
            written = full_path.read_text(encoding='utf-8')
            run_test("Test1: File content matches", written == content)

    # ------------------------------------------------------------------
    # Test 2: Multiple file contents – batch dimension and hashing consistency
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        contents = ["def foo(): pass", "class Bar: pass"]
        tensor = convert_file_contents_to_st_tensor(
            contents, tmpdir, max_use_count=2, feature_len=256
        )

        run_test("Test2: Batch size 2", tensor.shape[0] == 2)
        run_test("Test2: Second dimension 2", tensor.shape[1] == 2)

        # The tensor name should be the hash of the concatenated contents.
        joint = "".join(contents)
        expected_name = hashlib.sha256(joint.encode('utf-8')).hexdigest()

        for i, content in enumerate(contents):
            row = tensor[i, 0, :].to(torch.uint8).tolist()
            bytes_data = bytes(row)
            zero_pos = bytes_data.find(b'\x00')
            if zero_pos != -1:
                bytes_data = bytes_data[:zero_pos]
            stored_path = bytes_data.decode('utf-8')
            run_test(f"Test2: Path for sample {i} starts with tensor name",
                     stored_path.startswith(expected_name + "/"))
            full_path = Path(tmpdir) / stored_path
            run_test(f"Test2: File {i} exists", full_path.exists())
            if full_path.exists():
                written = full_path.read_text(encoding='utf-8')
                run_test(f"Test2: File {i} content matches", written == content)

    # ------------------------------------------------------------------
    # Test 3: Edge cases – empty string and very long content
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        long_str = "a" * 200   # longer than feature_len (256, but fine)
        contents = ["", long_str]
        tensor = convert_file_contents_to_st_tensor(
            contents, tmpdir, max_use_count=1, feature_len=256
        )

        # Check empty content
        row0 = tensor[0, 0, :].to(torch.uint8).tolist()
        bytes0 = bytes(row0)
        zero_pos0 = bytes0.find(b'\x00')
        if zero_pos0 != -1:
            bytes0 = bytes0[:zero_pos0]
        path0 = bytes0.decode('utf-8')
        full_path0 = Path(tmpdir) / path0
        run_test("Test3: Empty file exists", full_path0.exists())
        if full_path0.exists():
            written0 = full_path0.read_text(encoding='utf-8')
            run_test("Test3: Empty file content", written0 == "")

        # Check long content
        row1 = tensor[1, 0, :].to(torch.uint8).tolist()
        bytes1 = bytes(row1)
        zero_pos1 = bytes1.find(b'\x00')
        if zero_pos1 != -1:
            bytes1 = bytes1[:zero_pos1]
        path1 = bytes1.decode('utf-8')
        full_path1 = Path(tmpdir) / path1
        run_test("Test3: Long file exists", full_path1.exists())
        if full_path1.exists():
            written1 = full_path1.read_text(encoding='utf-8')
            run_test("Test3: Long file content preserved", written1 == long_str)

        # Ensure the first byte of the long row is non‑zero (the path is not empty).
        run_test("Test3: First byte of long row is non-zero", row1[0] != 0)

    print("\nAll tests completed.")
import torch

def convert_list_str_to_2d_tensor(strs: list[str], feature_len: int = 4096) -> torch.Tensor:
    """
    Encode a list of strings into a 2D bfloat16 tensor (CPU).
    Each string is UTF-8 encoded and truncated/padded to fixed length.

    Args:
        strs: List of input strings.
        feature_len: Number of bytes per string.

    Returns:
        Tensor of shape (len(strs), feature_len), dtype=torch.bfloat16, on CPU.
    """
    batch_size = len(strs)
    tensor = torch.zeros((batch_size, feature_len), dtype=torch.bfloat16)
    for i, s in enumerate(strs):
        # Encode to bytes and truncate
        b = s.encode('utf-8')[:feature_len]
        # Place bytes into tensor row (directly convert list of ints to tensor)
        if b:  # non-empty bytes
            tensor[i, :len(b)] = torch.tensor(list(b), dtype=torch.bfloat16)
    return tensor


def convert_2d_tensor_to_list_str(tensor: torch.Tensor) -> list[str]:
    """
    Decode a 2D bfloat16 tensor (CPU) back to a list of strings.
    Each row is interpreted as a UTF-8 byte sequence, stopping at the first zero byte.
    Invalid UTF-8 sequences are handled by replacing errors.

    Args:
        tensor: Tensor of shape (batch_size, feature_len), dtype=torch.bfloat16, on CPU.

    Returns:
        List of decoded strings.
    """
    batch_size, feature_len = tensor.shape
    result = []
    for i in range(batch_size):
        row = tensor[i].to(torch.uint8).numpy()  # cast bfloat16 back to uint8 for byte extraction
        # Find first zero byte (padding)
        zero_pos = (row == 0).argmax() if (row == 0).any() else feature_len
        bytes_data = bytes(row[:zero_pos])
        result.append(bytes_data.decode('utf-8', errors='replace'))
    return result


if __name__ == "__main__":
    print("Testing identity: encode → decode\n")

    test_strs = [
        "hello",
        "world",
        "a b c",
        "",
        "这是一个测试",
        "a" * 5000,
        "🌟 Unicode emoji",
    ]
    feature_len = 4096

    encoded = convert_list_str_to_2d_tensor(test_strs, feature_len)
    decoded = convert_2d_tensor_to_list_str(encoded)

    all_match = True
    for i, (orig, dec) in enumerate(zip(test_strs, decoded)):
        if orig == dec:
            print(f"✓ [{i}] Match: {repr(orig[:50])}")
        else:
            all_match = False
            print(f"✗ [{i}] Mismatch:")
            print(f"   Original: {repr(orig[:100])}")
            print(f"   Decoded:  {repr(dec[:100])}")

    # Byte‑level check for truncated long string
    long_orig = test_strs[-2]
    long_enc = long_orig.encode('utf-8')[:feature_len]
    long_dec = decoded[-2].encode('utf-8')[:feature_len]
    if long_enc == long_dec:
        print("\n✓ Long string correctly truncated (byte‑wise match).")
    else:
        print("\n✗ Long string truncation mismatch.")

    if all_match:
        print("\nAll reversible strings passed identity test.")
    else:
        print("\nSome strings did not match – expected due to truncation.")
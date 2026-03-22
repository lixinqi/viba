import os
import subprocess
import tempfile

import torch
from torch.autograd import Function

from symbolic_tensor.tensor_util.make_tensor import make_tensor


def _read_storage(tensor: torch.Tensor, flat_index: int) -> str:
    """Read the text content at a given flat storage index."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    with open(path) as f:
        return f.read()


def _get_diff(actual_content: str, expected_content: str) -> str:
    """Run diff -u on two content strings via temp files. Returns unified diff output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".expected", delete=False) as f_exp, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".actual", delete=False) as f_act:
        f_exp.write(expected_content)
        f_exp.flush()
        f_act.write(actual_content)
        f_act.flush()

        result = subprocess.run(
            ["diff", "-u", f_exp.name, f_act.name],
            capture_output=True,
            text=True,
        )
        diff_output = result.stdout

        os.unlink(f_exp.name)
        os.unlink(f_act.name)

    return diff_output


def _num_lines(content: str) -> int:
    """Count lines in a string. Empty string has 0 lines."""
    if not content:
        return 0
    return len(content.splitlines())


def get_diff_ratio_impl(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> torch.Tensor:
    """
    Forward: for each flat element, compute diff ratio between actual and expected text.
    diff_ratio = num_lines(diff_output) / num_lines(expected_text)
    Returns a float32 tensor with the same shape as actual.
    """
    numel = actual.numel()
    ratios = []

    for i in range(numel):
        actual_text = _read_storage(actual, i)
        expected_text = _read_storage(expected, i)

        diff_output = _get_diff(actual_text, expected_text)
        diff_lines = _num_lines(diff_output)
        expected_lines = _num_lines(expected_text)

        if expected_lines == 0:
            ratio = 0.0
        else:
            ratio = diff_lines / expected_lines

        ratios.append(ratio)

    return torch.tensor(ratios, dtype=torch.float32).reshape(actual.shape)


def get_diff_ratio_backward_impl(
    grad_output: torch.Tensor,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> torch.Tensor:
    """
    Backward: compute diff text for each element, store as symbolic gradient.
    The grad coefficient channel carries grad_output values.
    Returns a symbolic tensor of diff texts.
    """
    numel = actual.numel()
    diff_texts = []

    for i in range(numel):
        actual_text = _read_storage(actual, i)
        expected_text = _read_storage(expected, i)
        diff_output = _get_diff(actual_text, expected_text)
        diff_texts.append(diff_output)

    # Unflatten diff_texts into nested list matching actual.shape
    def _unflatten(flat, shape):
        if len(shape) == 0:
            return flat[0]
        if len(shape) == 1:
            return flat[:shape[0]]
        chunk = 1
        for s in shape[1:]:
            chunk *= s
        return [_unflatten(flat[i * chunk:(i + 1) * chunk], shape[1:]) for i in range(shape[0])]

    nested = _unflatten(diff_texts, list(actual.shape))
    actual_grad = make_tensor(nested, actual.st_relative_to)
    # Set coefficient channel from grad_output
    actual_grad.data = grad_output.broadcast_to(actual.shape).clone().to(actual_grad.dtype)
    return actual_grad


class GetDiffRatio(Function):
    @staticmethod
    def forward(ctx, actual, expected):
        ctx.save_for_backward(actual, expected)
        return get_diff_ratio_impl(actual, expected)

    @staticmethod
    def backward(ctx, grad_output):
        actual, expected = ctx.saved_tensors
        actual_grad = get_diff_ratio_backward_impl(grad_output, actual, expected)
        return actual_grad, None


get_diff_ratio = GetDiffRatio.apply


if __name__ == "__main__":
    from symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running get_diff_ratio tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    def read_storage(tensor, flat_index):
        digits = list(str(flat_index))
        path = os.path.join(
            tensor.st_relative_to,
            tensor.st_tensor_uid,
            "storage",
            os.path.join(*digits),
            "data",
        )
        with open(path) as f:
            return f.read()

    # Test 1: Forward - identical texts give ratio 0.0
    print("Test 1: Forward - identical texts")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["line1\nline2\nline3"], tmpdir)
        expected_t = make_tensor(["line1\nline2\nline3"], tmpdir)

        out = get_diff_ratio_impl(actual_t, expected_t)
        run_test("Shape matches", list(out.shape) == [1])
        run_test("dtype float32", out.dtype == torch.float32)
        run_test("Identical => ratio 0.0", out[0].item() == 0.0, 0.0, out[0].item())

    # Test 2: Forward - different texts give ratio > 0
    print("Test 2: Forward - different texts")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["alpha\nbeta\ngamma"], tmpdir)
        expected_t = make_tensor(["alpha\nBETA\ngamma\ndelta"], tmpdir)

        out = get_diff_ratio_impl(actual_t, expected_t)
        run_test("Ratio > 0.0", out[0].item() > 0.0)
        print(f"    ratio = {out[0].item():.4f}")

    # Test 3: Forward - 2D batch
    print("Test 3: Forward - 2D batch")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor([["same", "differs"]], tmpdir)
        expected_t = make_tensor([["same", "original"]], tmpdir)

        out = get_diff_ratio_impl(actual_t, expected_t)
        run_test("Shape [1, 2]", list(out.shape) == [1, 2])
        run_test("Identical element => 0.0", out[0, 0].item() == 0.0)
        run_test("Different element => > 0.0", out[0, 1].item() > 0.0)

    # Test 4: Backward produces symbolic gradient
    print("Test 4: Backward - symbolic gradient")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["hello world"], tmpdir)
        expected_t = make_tensor(["hello earth"], tmpdir)
        grad_out = torch.tensor([1.0], dtype=torch.float32)

        actual_grad = get_diff_ratio_backward_impl(grad_out, actual_t, expected_t)
        run_test("Grad has st_tensor_uid", hasattr(actual_grad, "st_tensor_uid"))
        run_test("Grad has st_relative_to", hasattr(actual_grad, "st_relative_to"))
        run_test("Grad shape matches actual", list(actual_grad.shape) == list(actual_t.shape))
        diff_text = read_storage(actual_grad, 0)
        run_test("Diff text non-empty", len(diff_text) > 0)
        print(f"    diff (first 80): {repr(diff_text[:80])}")

    # Test 5: Backward - identical texts produce empty diff
    print("Test 5: Backward - identical texts => empty diff")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["same content"], tmpdir)
        expected_t = make_tensor(["same content"], tmpdir)
        grad_out = torch.tensor([1.0], dtype=torch.float32)

        actual_grad = get_diff_ratio_backward_impl(grad_out, actual_t, expected_t)
        diff_text = read_storage(actual_grad, 0)
        run_test("Identical => empty diff", diff_text == "")

    # Test 6: GetDiffRatio autograd Function
    print("Test 6: GetDiffRatio autograd Function")
    with tempfile.TemporaryDirectory() as tmpdir:
        actual_t = make_tensor(["foo\nbar"], tmpdir)
        expected_t = make_tensor(["foo\nbaz"], tmpdir)

        ctx = type("MockCtx", (), {})()
        ctx.save_for_backward = lambda *ts: setattr(ctx, "saved_tensors", ts)
        fwd = GetDiffRatio.forward(ctx, actual_t, expected_t)
        run_test("Forward returns float32", fwd.dtype == torch.float32)

        grad_out = torch.tensor([0.5], dtype=torch.float32)
        result = GetDiffRatio.backward(ctx, grad_out)
        run_test("Returns tuple of 2", len(result) == 2)
        run_test("Second is None", result[1] is None)
        run_test("First has st_tensor_uid", hasattr(result[0], "st_tensor_uid"))

    print("\nAll tests completed.")

import os
import subprocess
import tempfile
import torch
from typing import Any, Tuple

from symbolic_tensor.function.symbolic_transform_forward import symbolic_transform_forward
from symbolic_tensor.function.symbolic_transform_backward import symbolic_transform_backward
from symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like


class SymbolicTransform(torch.autograd.Function):
    """
    torch.autograd.Function wrapping symbolic_transform_forward and
    symbolic_transform_backward.

    Usage:
        output, selected_indexes = SymbolicTransform.apply(
            input, experience, forward_prompt, topk
        )

    forward(ctx, input, experience, forward_prompt="", topk=16)
        -> (output, selected_experience_qkv_indexes_list)

    backward(ctx, grad_output, grad_selected_indexes=None)
        grad_output has two channels:
          a) coefficient (float) — semi-differentiable, flows through autograd
          b) text diff (str) — symbolic gradient stored in tensor files
        -> (grad_input, grad_experience, None, None)

    selected_experience_qkv_indexes_list contains detached index tensors (no grad).
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        experience: torch.Tensor,
        forward_prompt: str = "",
        topk: int = 16,
    ) -> Tuple[torch.Tensor, Any]:
        output, selected_experience_qkv_indexes_list = symbolic_transform_forward(
            input, experience, forward_prompt, topk
        )

        # Save tensors for backward
        ctx.save_for_backward(input, output, experience)
        # Save non-tensor state
        ctx.selected_experience_qkv_indexes_list = selected_experience_qkv_indexes_list
        ctx.forward_prompt = forward_prompt
        ctx.topk = topk

        return output, selected_experience_qkv_indexes_list

    @staticmethod
    def backward(ctx, grad_output, grad_selected_indexes=None):
        input, output, experience = ctx.saved_tensors

        # If grad_output is a plain tensor (from autograd, e.g. loss.backward()),
        # wrap it as a symbolic tensor so backward can process it.
        # The coefficient channel carries the autograd values;
        # the text diff channel gets "TODO" (no upstream symbolic gradient).
        if not hasattr(grad_output, "st_relative_to"):
            symbolic_grad_output = todo_tensor_like(output)
            symbolic_grad_output.data.copy_(grad_output.data)
            grad_output = symbolic_grad_output

        grad_input, grad_experience = symbolic_transform_backward(
            grad_output,
            input,
            output,
            experience,
            ctx.selected_experience_qkv_indexes_list,
            ctx.forward_prompt,
            ctx.topk,
        )

        # Return grads for (input, experience, forward_prompt, topk)
        return grad_input, grad_experience, None, None


symbolic_transform = SymbolicTransform.apply


if __name__ == "__main__":
    from symbolic_tensor.tensor_util.make_tensor import make_tensor

    # Source anthropic env vars
    result = subprocess.run(
        ["bash", "-c", "source ~/.anthropic.sh && env"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            os.environ[key] = val
    os.environ.pop("CLAUDECODE", None)

    print("Running SymbolicTransform (autograd.Function) tests...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  \u2713 {name}")
        else:
            print(f"  \u2717 {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    # Test 1: Forward pass via .apply()
    print("Test 1: Forward pass via SymbolicTransform.apply")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["Hello world in English"]
        input_tensor = make_tensor(input_data, tmpdir)
        input_tensor.requires_grad_(True)

        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)
        experience_tensor.requires_grad_(True)

        output, selected_indexes = SymbolicTransform.apply(
            input_tensor, experience_tensor,
            "Translate the English text to French.",
            2,
        )

        run_test("Output shape matches input", list(output.shape) == list(input_tensor.shape))
        run_test("Selected indexes returned", selected_indexes is not None)
        run_test("Output requires grad", output.requires_grad)

        # Check output content
        root = os.path.join(tmpdir, output.st_tensor_uid, "storage")
        path = os.path.join(root, "0", "data")
        if os.path.isfile(path):
            with open(path) as f:
                content = f.read()
            run_test("Output not TODO", "TODO" not in content)
            print(f"  Output: {repr(content[:120])}")

    # Test 2: Forward + backward (direct call, no loss.backward)
    print("\nTest 2: Forward + backward (direct call)")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tensor = make_tensor(["Hello world in English"], tmpdir)

        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)

        output, selected_indexes = symbolic_transform_forward(
            input_tensor, experience_tensor,
            forward_prompt="Translate the English text to French.",
            topk=2,
        )

        run_test("Output has st attrs", hasattr(output, "st_relative_to"))

        # Construct a symbolic grad_output with text diff
        grad_output = make_tensor(
            ["The translation should use formal French: 'Bonjour le monde' -> 'Bonjour au monde'"],
            tmpdir,
        )
        grad_output.data.fill_(1.0)

        grad_input, grad_experience = symbolic_transform_backward(
            grad_output, input_tensor, output, experience_tensor,
            selected_experience_qkv_indexes_list=selected_indexes,
            forward_prompt="Translate the English text to French.",
            topk=2,
        )

        run_test("grad_input shape matches", list(grad_input.shape) == list(input_tensor.shape))
        run_test("grad_experience shape matches", list(grad_experience.shape) == list(experience_tensor.shape))

        # Check grad_input text diff
        root = os.path.join(tmpdir, grad_input.st_tensor_uid, "storage")
        path = os.path.join(root, "0", "data")
        if os.path.isfile(path):
            with open(path) as f:
                content = f.read()
            run_test("grad_input text not TODO", "TODO" not in content)
            print(f"  grad_input text: {repr(content[:120])}")

        # Check grad_experience text diffs
        root = os.path.join(tmpdir, grad_experience.st_tensor_uid, "storage")
        for i in range(grad_experience.numel()):
            digits = list(str(i))
            path = os.path.join(root, os.path.join(*digits), "data")
            if os.path.isfile(path):
                with open(path) as f:
                    content = f.read()
                print(f"  grad_experience[{i}]: TODO={'TODO' == content.strip()} {repr(content[:80])}")

        # Numeric channel
        run_test("grad_input coeff == 1.0", grad_input.data[0].item() == 1.0)
        run_test("grad_experience coeff accumulated", grad_experience.data[0, 0].item() == 1.0)

    print("\nAll tests completed.")

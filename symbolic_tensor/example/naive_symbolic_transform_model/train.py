"""
Training script for NaiveModel: Translate Python to Viba.

5 iterations, validate loss convergence.
Save loss of each iteration to /tmp/loss.log.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from symbolic_tensor.tensor_util.make_tensor import make_tensor
from symbolic_tensor.function.symbolic_transform_forward import symbolic_transform_forward
from symbolic_tensor.function.symbolic_transform_backward import symbolic_transform_backward
from symbolic_tensor.function.get_edit_distance_ratio import get_edit_distance_ratio_impl
from symbolic_tensor.optimizer.symbolic_sgd import SymbolicSGD
from symbolic_tensor.example.naive_symbolic_transform_model.model import NaiveModel

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


EXAMPLE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(EXAMPLE_DIR, "dataset")
LOSS_LOG = "/tmp/loss.log"
NUM_ITERATIONS = 5
FORWARD_PROMPT = "Translate Python To Viba"


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


def main():
    print("=" * 60)
    print("NaiveModel Training: Translate Python To Viba")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Load dataset ──
        # Input: Python files, Expected: Viba files
        pairs = [
            # ("seq.py", "seq.viba"),
            ("branch.py", "branch.viba"),
            ("loop.py", "loop.viba"),
        ]

        py_paths = []
        viba_contents = []
        for py_name, viba_name in pairs:
            py_paths.append(Path(os.path.join(DATASET_DIR, py_name)))
            with open(os.path.join(DATASET_DIR, viba_name)) as f:
                viba_contents.append(f.read())

        # Create input tensor (symlinks to .py files)
        input_tensor = make_tensor(
            [Path(p) for p in py_paths],
            tmpdir,
            symlink=True,
        )

        # Create expected output tensor (.viba content)
        expected_tensor = make_tensor(viba_contents, tmpdir)

        print(f"\nDataset: {len(pairs)} pairs")
        for i, (py_name, viba_name) in enumerate(pairs):
            print(f"  [{i}] {py_name} -> {viba_name}")

        # ── Build experience ──
        # Experience: [query_keywords, key_python, value_viba] per entry
        experience_entries = []
        for i, (py_name, viba_name) in enumerate(pairs):
            py_content = read_storage(input_tensor, i)
            viba_content = viba_contents[i]
            # Extract keywords from python code for query
            keywords = py_name.replace(".py", "") + "\n" + "python\nviba\ntranslate"
            experience_entries.append([keywords, py_content, viba_content])

        experience_tensor = make_tensor(experience_entries, tmpdir)

        # ── Create model and optimizer ──
        model = NaiveModel(forward_prompt=FORWARD_PROMPT, topk=len(pairs))
        model.load_experience(experience_tensor)

        optimizer = SymbolicSGD(
            model.parameters(),
            lr=1.0,
            step_prompt="You are updating Python-to-Viba translation experience entries "
                        "(query keywords, key python code, value viba code).",
        )

        print(f"\nExperience shape: {list(experience_tensor.shape)}")
        print(f"Input shape: {list(input_tensor.shape)}")
        print(f"Expected shape: {list(expected_tensor.shape)}")

        # ── Training loop ──
        losses = []

        for iteration in range(1, NUM_ITERATIONS + 1):
            print(f"\n{'─' * 60}")
            print(f"Iteration {iteration}/{NUM_ITERATIONS}")
            print(f"{'─' * 60}")

            # Forward
            print("\n  [Forward]")
            output, selected_indexes = symbolic_transform_forward(
                input_tensor,
                model.transform.experience,
                forward_prompt=FORWARD_PROMPT,
                topk=len(pairs),
            )

            for i in range(output.numel()):
                out_text = read_storage(output, i)
                print(f"    output[{i}] (first 80): {repr(out_text[:80])}")

            # Compute loss: edit distance ratio between output and expected
            print("\n  [Loss]")
            loss = get_edit_distance_ratio_impl(output, expected_tensor)
            mean_loss = loss.mean().item()
            losses.append(mean_loss)
            print(f"    Per-sample losses: {[f'{l:.4f}' for l in loss.tolist()]}")
            print(f"    Mean loss: {mean_loss:.4f}")

            # Backward
            print("\n  [Backward]")
            # Create gradient: diff between output and expected as text feedback
            grad_texts = []
            for i in range(output.numel()):
                out_text = read_storage(output, i)
                exp_text = read_storage(expected_tensor, i)
                if out_text.strip() == exp_text.strip():
                    grad_texts.append("No change needed. Output matches expected.")
                else:
                    grad_texts.append(
                        f"The output does not match expected.\n"
                        f"Expected output:\n{exp_text}\n\n"
                        f"Actual output:\n{out_text}\n\n"
                        f"Please update the experience so future translations "
                        f"produce output closer to the expected Viba code."
                    )

            grad_output = make_tensor(grad_texts, tmpdir)
            grad_output.data.fill_(1.0)

            grad_input, grad_experience = symbolic_transform_backward(
                grad_output,
                input_tensor,
                output,
                model.transform.experience,
                selected_experience_qkv_indexes_list=selected_indexes,
                forward_prompt=FORWARD_PROMPT,
                topk=len(pairs),
            )

            model.transform.experience.grad = grad_experience

            for i in range(min(grad_experience.numel(), 6)):
                gt = read_storage(grad_experience, i)
                print(f"    grad_exp[{i}]: {repr(gt[:60])}")

            # Optimizer step
            print("\n  [Optimizer Step]")
            exp_before = read_storage(model.transform.experience, 2)
            optimizer.step()
            exp_after = read_storage(model.transform.experience, 2)
            changed = exp_before != exp_after
            print(f"    Experience[0].value changed: {changed}")

            # Reset coefficients for next iteration
            model.transform.experience.data.fill_(1.0)

            print(f"\n  Iteration {iteration} loss: {mean_loss:.4f}")

        # ── Save losses ──
        print(f"\n{'=' * 60}")
        print("Training Complete")
        print(f"{'=' * 60}")
        print(f"\nLoss trajectory: {[f'{l:.4f}' for l in losses]}")

        with open(LOSS_LOG, "w") as f:
            for i, loss_val in enumerate(losses, 1):
                f.write(f"iteration {i}: {loss_val:.6f}\n")
            f.write(f"\nConverged: {losses[-1] < losses[0] if len(losses) > 1 else 'N/A'}\n")

        print(f"Losses saved to {LOSS_LOG}")

        # Validate convergence
        if len(losses) > 1 and losses[-1] < losses[0]:
            print("Loss CONVERGED (final < initial)")
        else:
            print("Loss did NOT converge (final >= initial)")


if __name__ == "__main__":
    main()

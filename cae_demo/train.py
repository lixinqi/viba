"""
CAE Demo training script.

Runs 5 training iterations of the DemoModel (Code AutoEncoder):
  1. Forward: input source code -> model -> reconstructed code
  2. Loss: get_diff_ratio(output, input) measures reconstruction quality
  3. Validation: convergence trend across iterations
"""

import os
import sys

import torch

from viba.st.data_loader.sole_file_batch_data_loader import SoleFileBatchDataLoader
from viba.st.function.get_diff_ratio import get_diff_ratio
from viba.st.function.copy import copy as copy_tensor
from cae_demo.model import DemoModel


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    weight_dir = os.path.join(script_dir, "weights")

    model = DemoModel(
        weight_dir=weight_dir,
        output_file_content_type="Python",
    )

    dataloader = SoleFileBatchDataLoader(
        root_dir=dataset_dir,
        file_content_type="Python",
        extension=".py",
        batch_size=1,
        max_use_count=1,
    )

    num_iters = 5
    loss_history = []

    for iteration in range(num_iters):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iters}")
        print(f"{'='*60}")

        epoch_losses = []
        for batch_idx, input_tensor in enumerate(dataloader):
            # Forward pass
            input_tensor = copy_tensor(input_tensor, "/tmp/tensor_data")
            print(f"before forward, {input_tensor.st_relative_to=}")
            output_tensor = model(input_tensor)
            print(f"after forward, {output_tensor.st_relative_to=}")

            # Loss: diff ratio between reconstructed output and original input
            loss = get_diff_ratio(output_tensor, input_tensor)
            batch_loss = loss.mean().item()
            epoch_losses.append(batch_loss)
            print(f"  Batch {batch_idx}: loss = {batch_loss:.4f}")

            # Backward pass
            loss.mean().backward()

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        loss_history.append(avg_loss)
        print(f"  Average loss: {avg_loss:.4f}")

    # Validation: convergence trend
    print(f"\n{'='*60}")
    print("Convergence Trend")
    print(f"{'='*60}")
    for i, loss_val in enumerate(loss_history):
        bar = '#' * int(loss_val * 10)
        print(f"  Iter {i+1}: {loss_val:.4f} {bar}")

    if len(loss_history) >= 2:
        if loss_history[-1] < loss_history[0]:
            print("\nResult: IMPROVING (loss decreased)")
        elif loss_history[-1] == loss_history[0]:
            print("\nResult: STABLE (loss unchanged)")
        else:
            print("\nResult: DIVERGING (loss increased)")


if __name__ == "__main__":
    main()

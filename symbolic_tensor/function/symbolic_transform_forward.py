import os
import asyncio
import tempfile
import itertools
import torch
from typing import Any, List, Tuple, Union

from symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from symbolic_tensor.tensor_util.slice_view import slice_view
from symbolic_tensor.tensor_util.dump_view import dump_view
from symbolic_tensor.function.get_input_query_tensor import get_input_query_tensor
from symbolic_tensor.function.select_qkv_indexes import select_qkv_indexes
from symbolic_tensor.llm_client.coding_agent_query import coding_agent_query


def _scalar_slice_indices(shape: torch.Size) -> List[List[int]]:
    """Generate all coordinate tuples for iterating over each scalar element."""
    ranges = [range(s) for s in shape]
    return [list(coord) for coord in itertools.product(*ranges)]


def _replace_last_tensor_with_full_slice(
    index_tensors: List[torch.Tensor],
    last_dim_size: int,
) -> List[Union[torch.Tensor, slice]]:
    """Replace the last index tensor with a full slice to keep all q/k/v."""
    result: List[Union[torch.Tensor, slice]] = list(index_tensors[:-1])
    result.append(slice(None))  # full slice on last dim (q, k, v)
    return result


def _read_file_content(tensor: torch.Tensor, flat_index: int) -> str:
    """Read the file content for a given flat index from tensor storage."""
    digits = list(str(flat_index))
    path = os.path.join(
        tensor.st_relative_to,
        tensor.st_tensor_uid,
        "storage",
        os.path.join(*digits),
        "data",
    )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_nested_result(flat_results: List[Any], shape: List[int]) -> Any:
    """Reshape a flat list of results into a nested list matching the given shape."""
    if not shape:
        return flat_results[0]
    if len(shape) == 1:
        return flat_results
    chunk_size = 1
    for s in shape[1:]:
        chunk_size *= s
    return [
        _build_nested_result(flat_results[i * chunk_size:(i + 1) * chunk_size], shape[1:])
        for i in range(shape[0])
    ]


def symbolic_transform_forward(
    input: torch.Tensor,
    experience: torch.Tensor,
    forward_prompt: str = "",
    topk: int = 16,
) -> Tuple[torch.Tensor, Any]:
    """
    Forward pass of the symbolic transform: translate input to output using
    experience (q/k/v mappings) and an LLM coding agent.

    For each scalar element of the input tensor:
      1. Generate query keywords from input
      2. Select top-k experience entries by Jaccard similarity
      3. Slice experience to get relevant q/k/v mappings
      4. Dump views of experience, input element, and TODO output element
      5. Ask LLM to translate input -> output guided by experience

    All batch elements are processed concurrently via asyncio.gather.

    Args:
        input: A symbolic tensor to translate.
        experience: An Experience tensor (last dim=3: query, key, value).
        forward_prompt: Optional additional prompt guidance for the LLM.
        topk: Number of top experience entries to use per element.

    Returns:
        A tuple of:
        - output: The translated symbolic tensor (same shape as input).
        - selected_experience_qkv_indexes_list: A nested list matching input's
          shape, where each leaf is a list[torch.Tensor[int]] of selected
          experience coordinates.
    """
    # Create TODO-filled output tensor
    output = todo_tensor_like(input)

    # Generate input query keywords
    input_query = get_input_query_tensor(input)

    # Iterate over each scalar element
    coords_list = _scalar_slice_indices(input.size())
    flat_selected_indexes: List[List[torch.Tensor]] = []

    async def _process_element(coords: List[int]):
        """Process a single scalar element asynchronously."""
        # Create scalar views for this element
        int_slices = [c for c in coords]
        scalar_input_view = slice_view(input, int_slices)
        scalar_output_view = slice_view(output, int_slices)

        # Read the input file content for this element
        stride = input.stride()
        flat_index = sum(c * s for c, s in zip(coords, stride))
        batch_input_file_content = _read_file_content(input_query, flat_index)
        query_key_words = [w for w in batch_input_file_content.strip().split("\n") if w.strip()]

        # Select top-k experience entries by similarity
        select_experience_query_indexes = select_qkv_indexes(
            experience, query_key_words, topk
        )

        # Record selected indexes (list of tensors, one per dim)
        flat_selected_indexes.append(select_experience_query_indexes)

        # Replace last index tensor with full slice to keep q/k/v together
        select_experience_indexes = _replace_last_tensor_with_full_slice(
            select_experience_query_indexes, experience.size()[-1]
        )

        # Slice experience to get relevant entries
        experience_sliced_view = slice_view(experience, select_experience_indexes)

        # Create workspace with dump views
        with tempfile.TemporaryDirectory() as workspace_dir:
            exp_view_dir = os.path.join(workspace_dir, "const_experiance_view")
            input_view_dir = os.path.join(workspace_dir, "const_input_view")
            output_view_dir = os.path.join(workspace_dir, "mutable_output_view")

            dump_view(experience_sliced_view, exp_view_dir, "txt")
            dump_view(scalar_input_view, input_view_dir, "txt")
            dump_view(scalar_output_view, output_view_dir, "txt")

            prompt = (
                "You are a semantic translator.\n\n"
                f"{forward_prompt}\n\n"
                "Experience mappings are defined as:\n"
                f"  1) file \"<root_dir>/<experience_coordinate>.../0/data.xxx\" means query file of <experience_coordinate>...\n"
                f"  2) file \"<root_dir>/<experience_coordinate>.../1/data.xxx\" means key file of <experience_coordinate>...\n"
                f"  3) file \"<root_dir>/<experience_coordinate>.../2/data.xxx\" means value file of <experience_coordinate>...\n\n"
                "You need read all the key => value pairs to get the experiences.\n\n"
                f"Conducted by \"{exp_view_dir}\",\n"
                f"please translate source semantic text \"{input_view_dir}\"\n"
                f"to target semantic text \"{output_view_dir}\".\n\n"
                f"Replace TODO in \"{output_view_dir}\" with target semantic text.\n"
            )

            # Ensure CLAUDECODE env var is unset
            env_backup = os.environ.pop("CLAUDECODE", None)
            try:
                async for _ in coding_agent_query(prompt=prompt, cwd=workspace_dir, allowed_tools=["Read", "Edit", "Write"]):
                    pass
            finally:
                if env_backup is not None:
                    os.environ["CLAUDECODE"] = env_backup

            # Copy results back from workspace view to tensor storage.
            # The LLM's Write tool may replace symlinks with regular files,
            # so we read from the view files and write to the actual storage.
            output_stride = output.stride()
            output_flat_index = sum(c * s for c, s in zip(coords, output_stride))
            output_digits = list(str(output_flat_index))
            output_storage_path = os.path.join(
                output.st_relative_to,
                output.st_tensor_uid,
                "storage",
                os.path.join(*output_digits),
                "data",
            )
            # Find the output view data file
            for root, _dirs, files in os.walk(output_view_dir):
                for fname in files:
                    view_file = os.path.join(root, fname)
                    with open(view_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    with open(output_storage_path, "w", encoding="utf-8") as f:
                        f.write(content)

    async def _run_all():
        await asyncio.gather(*[_process_element(coords) for coords in coords_list])

    asyncio.run(_run_all())

    # Build nested structure matching input shape
    selected_experience_qkv_indexes_list = _build_nested_result(
        flat_selected_indexes, list(input.size())
    )

    return output, selected_experience_qkv_indexes_list


if __name__ == "__main__":
    import subprocess
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

    print("Running symbolic_transform_forward test...\n")

    def run_test(name: str, condition: bool, expected=None, actual=None):
        if condition:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")
            if expected is not None and actual is not None:
                print(f"    expected: {expected}")
                print(f"    actual:   {actual}")

    print("Test 1: English to French translation")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = ["Hello world in English"]
        input_tensor = make_tensor(input_data, tmpdir)
        print(f"  Input shape: {list(input_tensor.shape)}")

        experience_data = [
            ["greeting\nhello\nworld", "Hello world in English", "Bonjour le monde en francais"],
            ["farewell\ngoodbye", "Goodbye in English", "Au revoir en francais"],
        ]
        experience_tensor = make_tensor(experience_data, tmpdir)
        print(f"  Experience shape: {list(experience_tensor.shape)}")

        output, selected_indexes = symbolic_transform_forward(
            input_tensor, experience_tensor,
            forward_prompt="Translate the English text to French.",
            topk=2,
        )

        run_test("Output shape matches input", list(output.shape) == list(input_tensor.shape))
        run_test("Selected indexes is a list", isinstance(selected_indexes, list))

        # Read output storage
        root = os.path.join(tmpdir, output.st_tensor_uid, "storage")
        for i in range(output.numel()):
            digits = list(str(i))
            path = os.path.join(root, os.path.join(*digits), "data")
            run_test(f"Output file {i} exists", os.path.isfile(path))
            if os.path.isfile(path):
                with open(path) as f:
                    content = f.read()
                run_test(f"Output {i} not TODO", "TODO" not in content)
                print(f"  Output {i}: {repr(content)}")

    print("\nAll tests completed.")

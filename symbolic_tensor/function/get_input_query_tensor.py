import os
import asyncio
import tempfile
import torch

from symbolic_tensor.tensor_util.todo_tensor_like import todo_tensor_like
from symbolic_tensor.tensor_util.dump_view import dump_view
from symbolic_tensor.llm_client.coding_agent_query import coding_agent_query


def get_input_query_tensor(input: torch.Tensor) -> torch.Tensor:
    """
    Generate a query keyword tensor from an input symbolic tensor.

    Creates a TODO-filled output tensor matching the input shape, dumps both
    as symlink views, then invokes a coding agent to replace each TODO file
    with grep/query keywords (one keyword per line).

    Args:
        input: A symbolic tensor with st_relative_to and st_tensor_uid attributes.

    Returns:
        The output symbolic tensor whose storage files now contain keywords.
    """
    # Create output tensor with same shape, filled with "TODO"
    output = todo_tensor_like(input)

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_view_dir = os.path.join(tmp_dir, "input_view")
        output_view_dir = os.path.join(tmp_dir, "output_view")

        # Dump symlink views for input and output
        dump_view(input, input_view_dir, "txt")
        dump_view(output, output_view_dir, "txt")

        # Build prompt for the coding agent
        prompt = (
            "You are a semantic grep keyword generator.\n"
            f"Given the symbolic tensor view in \"{input_view_dir}\",\n"
            f"please generate the query keywords into corresponding files in \"{output_view_dir}\".\n"
            "Each line in an output file should contain only a single keyword "
            "that would be used for grep/query-like operations.\n"
            "The keywords of files are used for calculating similarity between files.\n"
            "All \"TODO\" in output files should be replaced with keywords.\n"
        )

        # Ensure CLAUDECODE env var is unset to avoid conflicts with nested claude instances
        env_backup = os.environ.pop("CLAUDECODE", None)
        try:
            async def _run():
                async for _ in coding_agent_query(prompt=prompt, cwd=tmp_dir):
                    pass

            asyncio.run(_run())
        finally:
            if env_backup is not None:
                os.environ["CLAUDECODE"] = env_backup

    return output


if __name__ == "__main__":
    from symbolic_tensor.tensor_util.make_tensor import make_tensor

    print("Running get_input_query_tensor demo...\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Source the Anthropic env vars
        os.system("source ~/.anthropic.sh 2>/dev/null && env | grep ANTHROPIC")

        data = ["def hello():\n    print('hello world')", "class Foo:\n    pass"]
        t = make_tensor(data, tmpdir)
        print(f"Input shape: {list(t.shape)}")
        print(f"Input uid: {t.st_tensor_uid}")

        result = get_input_query_tensor(t)
        print(f"Output shape: {list(result.shape)}")
        print(f"Output uid: {result.st_tensor_uid}")

        # Read output storage files
        root = os.path.join(tmpdir, result.st_tensor_uid, "storage")
        for i in range(result.numel()):
            digits = list(str(i))
            path = os.path.join(root, os.path.join(*digits), "data")
            if os.path.isfile(path):
                with open(path) as f:
                    content = f.read()
                print(f"Output element {i}: {repr(content)}")

    print("\nDemo completed.")

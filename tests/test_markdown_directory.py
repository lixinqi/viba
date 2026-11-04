"""A plain directory written as one document: the document wins over the directory.

`tests/data/markdown_directory/` holds a directory of two ordinary modules twice:
as `report/` (one of them, `point.viba`) and as `report.viba.md` (both, as
sections). The document says `Point` is two integers; the directory's own
`point.viba` says two strings. Both are there at once on purpose: the document
wins, and nothing of the directory is read.

    python3 tests/test_markdown_directory.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Host, Checks, answer_of, is_ok, message_of
from viba import viba_ast
from viba.check_convertable_to_json_schema import check_convertable_to_json_schema
from viba.convert_to_json_schema import convert_to_json_schema
from viba.interpret import interpret
from viba.is_complete import is_complete
from viba.type import Ok

CASES = Path(__file__).resolve().parent / "data" / "markdown_directory"
checks = Checks("markdown_directory")
check = checks.check


def _answer(name: str):
    result = interpret(str(CASES / f"{name}.viba"), Host().environ(viba_path=str(CASES)))
    if not is_ok(result):
        return ("stop", message_of(result))
    return ("value", viba_ast.unparse_type(answer_of(result).data))


def run() -> None:
    check(_answer("use") == ("value", "$x int\n* $y int"),
          f"a directory's document is what its modules are: {_answer('use')!r}")
    check(_answer("missing")[0] == "stop",
          f"a name its document does not hold is not found: {_answer('missing')!r}")

    design = "import report.point as point\nX = point.Point\n"
    check(is_complete(design, viba_paths=[str(CASES)]),
          "a design through a document-written directory is complete")

    judged = check_convertable_to_json_schema([str(CASES)], ("type_names", ["report.point.Point"]))
    check(isinstance(judged, Ok), f"the judgment takes a type out of the document: {judged!r}")
    answer = convert_to_json_schema([str(CASES)], ("type_names", ["report.point.Point"]))
    check(isinstance(answer, Ok), f"the converter takes it too: {answer!r}")
    if isinstance(answer, Ok):
        properties = json.loads(answer.ok_value)["$defs"]["report.point.Point"]["properties"]
        check(properties == {"x": {"type": "integer"}, "y": {"type": "integer"}},
              f"the schema is the document's: {properties!r}")


def main() -> int:
    run()
    return checks.report()


if __name__ == "__main__":
    sys.exit(main())

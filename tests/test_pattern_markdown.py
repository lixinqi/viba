"""The demo generics, written as one Markdown document: the same answers.

A directory of viba files can be given as a single Markdown file: each section
names one file of it, and its first fenced code block is that file's text
(`viba/markdown.py`). Every short generic of `tests/data/pattern/demo/` is here
as a document — `tests/data/pattern_markdown/demo/*.md` — and this suite runs
the demo's own designs against both: the directory, and the document. The
answer must not change: the value, the type it extracted, or the stop.

    python3 tests/test_pattern_markdown.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Host, Checks, answer_of, is_ok, message_of
from test_pattern import EXTRACTED_CASES, RUNTIME_CASES

from viba import viba_ast
from viba.interpret import interpret
from viba.is_complete import is_complete

CASES = Path(__file__).resolve().parent / "data" / "pattern"
DOCUMENTS = Path(__file__).resolve().parent / "data" / "pattern_markdown"

checks = Checks("pattern_markdown")
check = checks.check

# The designs that use a generic, in both forms: every case of `test_pattern`'s
# two tables, and the three the tables leave to a function of their own.
EXTRA_CASES = ["call_it_answer", "arg_name_of_a", "tagged_again_a"]


def _answer(path: Path, root: Path):
    """What one design answers under this root: the value it computed, or why not."""
    result = interpret(str(path), Host().environ(viba_path=str(root)))
    if not is_ok(result):
        return ("stop", message_of(result))
    return ("value", viba_ast.unparse_type(answer_of(result).data))


# A design that reaches the generic through its pool: `is_complete` takes the
# files of a path, so the document must be read as the directory it stands for.
POOL_DESIGN = ("import demo.element_type_of as element_type_of\n"
               "Elem = element_type_of[list[int]].type\n")


def run(scratch: Path) -> None:
    for name, want, label in RUNTIME_CASES:
        _compare(name, label, scratch)
    for name, want, label in EXTRACTED_CASES:
        _compare(name, label, scratch)
    for name in EXTRA_CASES:
        _compare(name, name, scratch)
    _the_pool_takes_it()


def _the_pool_takes_it() -> None:
    """A layer that builds a pool from a path sees the document's files."""
    for label, root in (("the directory", CASES), ("the document", DOCUMENTS)):
        check(is_complete(POOL_DESIGN, viba_paths=[str(root)]),
              f"{label}: a design through the generic is complete")


def _compare(name: str, label: str, scratch: Path) -> None:
    """The two forms of one generic, through the design that uses it."""
    case = CASES / f"{name}.viba"
    directory = _answer(case, CASES)
    for beside in ("a_module.viba",):          # the modules the designs import
        (scratch / beside).write_text((CASES / beside).read_text(), encoding="utf-8")
    copy = scratch / f"{name}.viba"
    copy.write_text(case.read_text(), encoding="utf-8")
    document = _answer(copy, DOCUMENTS)
    check(document == directory,
          f"{label}: a document answers what the directory does "
          f"({document!r} against {directory!r})")
    check(document[0] == "value", f"{label}: and it answers a value ({document!r})")


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="viba-pattern-markdown-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
        scratch.rmdir()
    return checks.report()


if __name__ == "__main__":
    sys.exit(main())

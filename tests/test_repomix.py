"""repomix: a directory of viba code, packed into one Markdown document.

`tests/data/repomix/kb/` holds a little repo: a file with a comment above its
code, a generic directory (one file with a trailing comment, one with a comment
between its lines, and the marker), and a directory written as one document
(`box.viba.md`) with a `box/` beside it that the document shadows.

    python3 tests/test_repomix.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Host, answer_of, is_ok, message_of
from viba import markdown, viba_ast
from viba.interpret import interpret
from viba.repomix import viba_repomix

DATA = Path(__file__).resolve().parent / "data" / "repomix"
PATTERNS = Path(__file__).resolve().parent / "data" / "pattern"

PASS = 0
FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}\n  expected: {want!r}\n  got:      {got!r}")


def _ran(path: Path, root: Path):
    """What a run answered: the value's source form, or why it stopped."""
    result = interpret(str(path), Host().environ(viba_path=str(root)))
    if not is_ok(result):
        return ("stop", message_of(result))
    return ("value", viba_ast.unparse_type(answer_of(result).data))


def run() -> None:
    pack = viba_repomix(str(DATA / "kb"))
    sections = dict(markdown.files_of(pack))

    check("one section per file, its path the name",
          sorted(sections),
          ["box/a.viba", "box/b.viba", "g/100.viba", "g/200.viba",
           "g/__generic__.viba", "report.viba"])
    check("a file's code is the code block, its comment a comment",
          sections["report.viba"],
          "# The point and the note.\n\nPoint = Object * $x int * $y int\n")
    check("the comment above the code is prose",
          pack.index("The point and the note.") < pack.index("Point = Object"),
          True)
    check("the comment below the code is prose",
          pack.index("```viba g/100.viba") < pack.index("a trailing note"),
          True)
    check("a comment between two lines of code stays with them",
          sections["g/200.viba"],
          "pattern A\n\n# this one sits with the line, so it stays\nvalue = false\n")
    check("a directory's document is what is packed",
          sections["box/a.viba"], "A = int\n")
    check("and its directory is not",
          "box/Shadowed.viba" in sections, False)
    check("a file the document shadows is not packed",
          "Shadowed = int" in pack, False)
    check("a marker's comment is prose, and a comment in the file",
          sections["g/__generic__.viba"], "# __generic__.viba\n")
    check("no code, no notes", "notes.txt" in sections or "README.md" in sections,
          False)

    plain = DATA / "plain"
    check("a directory of files packs to its own document",
          markdown.files_of(viba_repomix(str(plain))),
          [(name, (plain / name).read_text()) for name in ("x.viba", "y.viba")])
    check("a path that is no directory packs nothing",
          viba_repomix(str(DATA / "nope")), "")

    with tempfile.TemporaryDirectory(prefix="viba-repomix-") as scratch:
        root = Path(scratch)
        (root / "plain.viba.md").write_text(viba_repomix(str(DATA / "plain")))
        (root / "use.viba").write_text(
            "import plain.x as px\n\n__decl__ = Any <- $env Env\n"
            "\nargs = __get_args__ << __decl__\n\n"
            "__impl__ = px << (args.env.sub_env << args.env << \"use\")\n")
        check("a directory's pack is the document the loader reads",
              _ran(root / "use.viba", root), ("value", "7"))

        (root / "demo").mkdir()
        (root / "demo" / "is_base_type.viba.md").write_text(
            viba_repomix(str(PATTERNS / "demo" / "is_base_type")))
        (root / "generic.viba").write_text(
            "import demo.is_base_type as g\n\n__decl__ = Any <- $env Env\n"
            "\n__impl__ = g[bool].value\n")
        check("a generic's pack is too", _ran(root / "generic.viba", root),
              ("value", "true"))


def main() -> int:
    global PASS
    run()
    print(f"repomix: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

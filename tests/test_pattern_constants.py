"""Pattern: a generic can match a constant.

`tests/data/pattern_constants/is_red.viba.md` is a generic written as one Markdown
document. Its first file's pattern is the literal type `"red"` — a literal is a
type, so a pattern with no parameter in it is judged by `is_sub_type` like any
other: the argument `"red"` falls in, `"blue"` and `int` do not. The second file's
`pattern T` catches everything else, so those answer `false`.

    python3 tests/test_pattern_constants.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host, answer_of, is_ok, value_of
from viba import viba_ast
from viba.interpret import interpret

CASES = Path(__file__).resolve().parent / "data" / "pattern_constants"

checks = Checks("pattern_constants")
check = checks.check


def _answer(name: str):
    """What one design answers: `is_red[...].value`, given the document's root."""
    return interpret(str(CASES / f"{name}.viba"),
                     Host().environ(viba_path=str(CASES)))


def run() -> None:
    for name, want, label in (("is_red_red", True, 'is_red["red"]'),
                              ("is_red_blue", False, 'is_red["blue"]'),
                              ("is_red_int", False, "is_red[int]")):
        result = _answer(name)
        check(is_ok(result) and value_of(result) is want,
              f"{label} -> {want}: {result!r}")

    # The literal's own type, taken through the pool: the document is read as the
    # directory it stands for, and the argument the first file's pattern matches
    # is a type like any other.
    from viba.is_complete import is_complete
    text = ('import is_red as is_red\n'
            'V = is_red["red"].value\n')
    check(is_complete(text, viba_paths=[str(CASES)]) is True,
          'is_complete reads is_red["red"].value through the document')

    return checks.report()


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())

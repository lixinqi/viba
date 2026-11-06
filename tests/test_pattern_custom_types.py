"""Pattern: a generic can match a named type — matching is structural.

`tests/data/pattern_custom_types/is_red.viba.md` is a generic written as one
Markdown document. Its first file's pattern is the imported type `types.colors.Red`:
the pattern file names it, so it is a type and not a parameter, and the matcher
tests the argument with `is_sub_type`. `types.colors.Red` falls in; `types.colors.Blue` and
`types.colors.Green` do not, because their tags differ — the three colors are three
types.

`is_unit_red.viba.md` is the other half: `types.units.Red` and `types.units.Blue` are both
`Object`, and two definitions with the same structure are the same type — so
asking for `types.units.Blue` answers true as well.

    python3 tests/test_pattern_custom_types.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host, answer_of, is_ok, value_of
from viba.interpret import interpret

CASES = Path(__file__).resolve().parent / "data" / "pattern_custom_types"

checks = Checks("pattern_custom_types")
check = checks.check


def _answer(name: str):
    """What one design answers under this root."""
    return interpret(str(CASES / f"{name}.viba"),
                     Host().environ(viba_path=str(CASES)))


def run() -> None:
    for name, want, label in (
            ("ask_red", True, "is_red[colors.Red]"),
            ("ask_blue", False, "is_red[colors.Blue]"),
            ("ask_green", False, "is_red[colors.Green]"),
            ("ask_unit_blue", True, "is_unit_red[units.Blue]: one unit type, three names"),
            ("ask_unit_green", True, "is_unit_red[units.Green]: the same one type")):
        result = _answer(name)
        check(is_ok(result) and value_of(result) is want,
              f"{label} -> {want}: {result!r}")
    return checks.report()


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())

"""A hint block is the product's unit, as a member and as a whole.

`A * {说明}` is `A`: the block has no members, so it is no member — the style
guide says so (`Assert` / `Hint` / `Appendix` are units), and the judgment has
to take it that way. The rule is recursive, it holds through names, and it holds
inside a tagged member. The pattern matcher counts members the same way, so
`pattern $A T` takes `$A int * {说明}`.

    python3 tests/test_hint_member.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host, is_ok, value_of
from viba.interpret import interpret
from viba.is_sub_type import is_sub_type
from viba.reflect import language_config
from viba.type import Ok, custom_module, entry_type

checks = Checks("hint_member")
check = checks.check


def _same(sub_text: str, sup_text: str, label: str, config=None,
          module=None) -> None:
    sub = entry_type(sub_text, module)
    sup = entry_type(sup_text, module)
    first = is_sub_type(sub, sup, config=config)
    second = is_sub_type(sup, sub, config=config)
    check(isinstance(first, Ok) and first.ok_value
          and isinstance(second, Ok) and second.ok_value,
          f"{label}: {first!r} / {second!r}")


def run() -> int:
    _same("int * {说明}", "int", "a hint member")
    _same("$a (str * {说明})", "$a str", "a tagged member whose body is a product")
    _same("$A int * {说明}", "$A int", "a tagged member beside a hint")
    _same("((int * {说明}) * {说明})", "int", "recursive")
    _same("{说明} * {说明}", "nil", "a product of units alone")
    _same("int * Assert[{a}] * Appendix[{b}]", "int",
          "the names a config calls the unit", config=language_config)

    module = custom_module("H = {说明}\n")
    _same("int * H", "int", "a name bound to a hint", module=module)

    with tempfile.TemporaryDirectory(prefix="viba-hint-member-") as scratch:
        root = Path(scratch)
        (root / "g").mkdir()
        (root / "g" / "__generic__.viba").write_text("# g — a generic\n")
        (root / "g" / "100.viba").write_text("pattern $A T\n\nvalue = true\n")
        (root / "g" / "900.viba").write_text("pattern T\n\nvalue = false\n")
        (root / "ask.viba").write_text(
            "import g as g\n\n__decl__ = Any <- $env Env\n\n"
            "__impl__ = g[$A int * {说明}].value\n")
        result = interpret(str(root / "ask.viba"),
                           Host().environ(viba_path=str(root)))
        check(is_ok(result) and value_of(result) is True,
              f"pattern $A T takes $A int * {{说明}}: {result!r}")
    return checks.report()


if __name__ == "__main__":
    sys.exit(run())

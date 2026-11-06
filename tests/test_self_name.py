"""`__type__`: a definition's own name, as the symbol of `tagged`.

`A = tagged[__type__, T]` is `A = tagged["A", T]` — the tagged type `$A T` — so a
definition can carry a tag of its own without spelling the name twice. A generic
uses the generic's own name, not the arguments it was applied to
(`X[T] = tagged[__type__, T]` makes `X[int]` the type `$X int`). Only the symbol
of a `tagged` application means this.

    python3 tests/test_self_name.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import viba_ast
from viba.is_sub_type import is_sub_type
from viba.type import Ok, custom_module, entry_type
from viba.viba_ast.nodes import Tagged

PASS = FAIL = 0


def check(ok: bool, label: str) -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def _both(a, b, label: str) -> None:
    """The two types are one: each falls into the other."""
    first = is_sub_type(a, b)
    second = is_sub_type(b, a)
    check(isinstance(first, Ok) and first.ok_value
          and isinstance(second, Ok) and second.ok_value,
          f"{label}: {first!r} / {second!r}")


def run() -> None:
    tree = viba_ast.parse("A = tagged[__type__, Object]\n")
    body = tree.body[0].body
    check(isinstance(body, Tagged) and body.tag == "$A",
          f"the body is the tagged type $A: {body!r}")

    _both(entry_type("tagged[__type__, Object]"), entry_type("$__entry__ Object"),
          "a definition's own name")

    module = custom_module("X[T] = tagged[__type__, T]\n")
    _both(entry_type("X[int]", module), entry_type("$X int"),
          "a generic's own name, not its arguments")

    _both(entry_type("tagged[__type__, int] * $x int"),
          entry_type("$__entry__ int * $x int"),
          "the tag sits where the application was")

    print(f"self_name: {PASS} passed, {FAIL} failed")


if __name__ == "__main__":
    run()
    sys.exit(1 if FAIL else 0)

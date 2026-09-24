"""Branch algebra: products select a branch, and sums drop never.

The host switches take the branch value and implement their choice through:

    never * a = never
    nil * a   = a
    never | a = a

    python3 tests/test_interpreter_branch.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, value_of, write

import branch
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import Ok
from viba import viba_ast

checks = Checks("interpreter_branch")
check = checks.check
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def host_for(a_value):
    """The case's own foo/ge; the two selectors come from branch.py itself."""
    def get_func(module_path, func_name):
        if func_name == "foo":
            return lambda env: a_value
        if func_name == "ge":
            return lambda env, x, y: x.value >= y.value
        return branch.get_func(module_path, func_name)
    return get_func


BRANCH_PROGRAM = """
import branch

foo =
    int <- $env Environment <- { the case's value }
ge =
    bool <- $env Environment <- $x int <- $y int <- { x >= y }

a = foo << $env environ
condition = ge << $env environ << $x a << $y 0
__ret__ =
  Oneof
  | (branch.nil_or_never << $env environ << $condition condition << $v a)
  | (branch.never_or_nil << $env environ << $condition condition << $v 0)
"""


def run(tmp: Path):
    _if_else_picks(tmp)
    _selectors_keep_or_eliminate_each_value_kind(tmp)
    _product_identities_and_absorption(tmp)
    _sum_identities_and_live_branches(tmp)


def _run(tmp, name, source, a_value=0):
    path = write(tmp, name, source)
    env = Environment(EnvironmentStorage("root", None, str(tmp)),
                      EnvironmentCompute(host_for(a_value)), str(REPOSITORY_ROOT))
    return interpret(path, env)


def _if_else_picks(tmp: Path):
    """a >= 0 hands back a; otherwise it hands back 0."""
    for a_value, want in ((7, 7), (1, 1), (0, 0), (-1, 0), (-8, 0)):
        result = _run(tmp, f"branch_{a_value}.viba", BRANCH_PROGRAM, a_value)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"if/else with a={a_value} answers {want}: {result!r}")


def _selectors_keep_or_eliminate_each_value_kind(tmp: Path):
    values = (("7", 7), ('"kept"', "kept"), ("false", False), ("nil", None))
    selectors = (
        ("nil_or_never", "true", True),
        ("nil_or_never", "false", False),
        ("never_or_nil", "true", False),
        ("never_or_nil", "false", True),
    )
    for selector, condition, keeps_value in selectors:
        for written, expected in values:
            source = f"""import branch
__ret__ = branch.{selector} << $env environ << $condition {condition} << $v {written}
"""
            result = _run(tmp, f"{selector}_{condition}_{written!r}.viba", source)
            if keeps_value:
                check(isinstance(result, Ok) and value_of(result) == expected,
                      f"{selector}({condition}, {written}) keeps {expected!r}: {result!r}")
            else:
                check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.Never),
                      f"{selector}({condition}, {written}) eliminates its branch: {result!r}")


def _product_identities_and_absorption(tmp: Path):
    identities = (
        ("nil * 7", 7),
        ("7 * nil", 7),
        ("Object * 7", 7),
        ("7 * Object", 7),
        ("void * 7", 7),
        ("7 * void", 7),
        ("None * 7", 7),
        ("7 * None", 7),
        ("nil * Object * 7", 7),
        ("7 * Object * nil", 7),
    )
    for index, (expression, expected) in enumerate(identities):
        result = _run(tmp, f"product_identity_{index}.viba", f"__ret__ = {expression}\n")
        check(isinstance(result, Ok) and value_of(result) == expected,
              f"{expression} is {expected}: {result!r}")

    absorptions = (
        "never * 7",
        "7 * never",
        "Object * never * 7",
        "7 * nil * never",
        "never * nil * Object",
    )
    for index, expression in enumerate(absorptions):
        result = _run(tmp, f"product_absorption_{index}.viba", f"__ret__ = {expression}\n")
        check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.Never),
              f"{expression} is never: {result!r}")


def _sum_identities_and_live_branches(tmp: Path):
    identities = (
        ("never | 7", 7),
        ("7 | never", 7),
        ("Oneof | 7", 7),
        ("7 | Oneof", 7),
        ("never | Oneof | 7", 7),
        ("7 | never | Oneof", 7),
    )
    for index, (expression, expected) in enumerate(identities):
        result = _run(tmp, f"sum_identity_{index}.viba", f"__ret__ = {expression}\n")
        check(isinstance(result, Ok) and value_of(result) == expected,
              f"{expression} is {expected}: {result!r}")

    for index, expression in enumerate(("never | never", "Oneof | never", "never | Oneof")):
        result = _run(tmp, f"all_never_sum_{index}.viba", f"__ret__ = {expression}\n")
        check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.Never),
              f"{expression} is never: {result!r}")

    live_sums = (("1 | 2", 2), ("never | 1 | 2", 2), ("1 | never | 2", 2),
                 ("1 | 2 | never", 2), ("1 | 2 | 3", 3))
    for index, (expression, count) in enumerate(live_sums):
        result = _run(tmp, f"live_sum_{index}.viba", f"__ret__ = {expression}\n")
        data = result.ok_value.data if isinstance(result, Ok) else None
        check(isinstance(data, viba_ast.SumChain) and len(data.elements) == count,
              f"{expression} keeps {count} live branches: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-branch-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

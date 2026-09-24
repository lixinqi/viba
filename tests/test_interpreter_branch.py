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


def comparison_if_program(threshold, when_true="a", when_false="0", reverse=False):
    """A complete comparison, selection and merge program."""
    branches = (
        f"  | (branch.id_or_never << $env environ << $condition condition << $v {when_true})\n"
        f"  | (branch.never_or_nil << $env environ << $condition condition << $v {when_false})"
    )
    if reverse:
        branches = (
            f"  | (branch.never_or_nil << $env environ << $condition condition << $v {when_false})\n"
            f"  | (branch.id_or_never << $env environ << $condition condition << $v {when_true})"
        )
    return f"""
import branch

foo =
    int <- $env Environment <- {{ the case's value }}
ge =
    bool <- $env Environment <- $x int <- $y int <- {{ x >= y }}

a = foo << $env environ
condition = ge << $env environ << $x a << $y {threshold}
__ret__ =
  Oneof
{branches}
"""


def literal_if_program(condition, when_true, when_false, reverse=False):
    """A complete branch program with a written boolean condition."""
    branches = (
        f"  | (branch.id_or_never << $env environ << $condition {condition} << $v {when_true})\n"
        f"  | (branch.never_or_nil << $env environ << $condition {condition} << $v {when_false})"
    )
    if reverse:
        branches = (
            f"  | (branch.never_or_nil << $env environ << $condition {condition} << $v {when_false})\n"
            f"  | (branch.id_or_never << $env environ << $condition {condition} << $v {when_true})"
        )
    return f"""import branch
__ret__ =
  Oneof
{branches}
"""


NESTED_BRANCH_PROGRAM = """
import branch

foo =
    int <- $env Environment <- { the case's value }
ge =
    bool <- $env Environment <- $x int <- $y int <- { x >= y }

a = foo << $env environ
high = ge << $env environ << $x a << $y 10
nonnegative = ge << $env environ << $x a << $y 0
nonnegative_or_negative =
  Oneof
  | (branch.id_or_never
      << $env environ << $condition nonnegative << $v "nonnegative")
  | (branch.never_or_nil
      << $env environ << $condition nonnegative << $v "negative")
__ret__ =
  Oneof
  | (branch.id_or_never << $env environ << $condition high << $v "high")
  | (branch.never_or_nil
      << $env environ << $condition high << $v nonnegative_or_negative)
"""


def run(tmp: Path):
    _comparison_driven_if_else(tmp)
    _literal_if_else_value_kinds_and_order(tmp)
    _nested_if_elif_else(tmp)
    _guarded_multi_branch_merge(tmp)
    _selectors_keep_or_eliminate_each_value_kind(tmp)
    _product_identities_and_absorption(tmp)
    _sum_identities_and_live_branches(tmp)


def _run(tmp, name, source, a_value=0):
    path = write(tmp, name, source)
    env = Environment(EnvironmentStorage("root", None, str(tmp)),
                      EnvironmentCompute(host_for(a_value)), str(REPOSITORY_ROOT))
    return interpret(path, env)


def _comparison_driven_if_else(tmp: Path):
    """Threshold boundaries run through comparison, both selectors and merge."""
    cases = (
        (0, -8, 0), (0, -1, 0), (0, 0, 0), (0, 1, 1), (0, 7, 7),
        (5, 0, 0), (5, 4, 0), (5, 5, 5), (5, 6, 6), (5, 20, 20),
        (10, 1, 0), (10, 9, 0), (10, 10, 10), (10, 11, 11), (10, 99, 99),
    )
    for index, (threshold, a_value, want) in enumerate(cases):
        source = comparison_if_program(threshold)
        result = _run(tmp, f"comparison_branch_{index}.viba", source, a_value)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"if a={a_value} >= {threshold} answers {want}: {result!r}")


def _literal_if_else_value_kinds_and_order(tmp: Path):
    """Both outcomes and both branch orders work for each scalar value kind."""
    cases = (
        ("true", "7", "8", 7),
        ("false", "7", "8", 8),
        ("true", '"left"', '"right"', "left"),
        ("false", '"left"', '"right"', "right"),
        ("true", "true", "false", True),
        ("false", "true", "false", False),
        ("true", "nil", "9", None),
        ("false", "9", "nil", None),
    )
    for reverse in (False, True):
        for index, (condition, when_true, when_false, want) in enumerate(cases):
            source = literal_if_program(condition, when_true, when_false, reverse)
            result = _run(tmp, f"literal_branch_{reverse}_{index}.viba", source)
            check(isinstance(result, Ok) and value_of(result) == want,
                  f"if {condition} with reverse={reverse} answers {want!r}: {result!r}")


def _nested_if_elif_else(tmp: Path):
    """A nested merge implements high / nonnegative / negative classification."""
    cases = ((25, "high"), (11, "high"), (10, "high"),
             (9, "nonnegative"), (1, "nonnegative"), (0, "nonnegative"),
             (-1, "negative"), (-20, "negative"))
    for index, (a_value, want) in enumerate(cases):
        result = _run(tmp, f"nested_branch_{index}.viba",
                      NESTED_BRANCH_PROGRAM, a_value)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"nested branch classifies {a_value} as {want}: {result!r}")


def _guarded_multi_branch_merge(tmp: Path):
    """Three independent guards preserve exactly their live branch set."""
    for mask in range(8):
        conditions = tuple("true" if mask & (1 << index) else "false"
                           for index in range(3))
        source = f"""import branch
__ret__ =
  Oneof
  | (branch.id_or_never << $env environ << $condition {conditions[0]} << $v 1)
  | (branch.id_or_never << $env environ << $condition {conditions[1]} << $v 2)
  | (branch.id_or_never << $env environ << $condition {conditions[2]} << $v 3)
"""
        result = _run(tmp, f"guarded_merge_{mask}.viba", source)
        live = [index + 1 for index in range(3) if mask & (1 << index)]
        if not live:
            valid = isinstance(result, Ok) and isinstance(result.ok_value.data,
                                                           viba_ast.Never)
        elif len(live) == 1:
            valid = isinstance(result, Ok) and value_of(result) == live[0]
        else:
            data = result.ok_value.data if isinstance(result, Ok) else None
            valid = isinstance(data, viba_ast.SumChain) and len(data.elements) == len(live)
        check(valid, f"guards {conditions} preserve branches {live}: {result!r}")


def _selectors_keep_or_eliminate_each_value_kind(tmp: Path):
    values = (("7", 7), ('"kept"', "kept"), ("false", False), ("nil", None))
    selectors = (
        ("id_or_never", "true", True),
        ("id_or_never", "false", False),
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

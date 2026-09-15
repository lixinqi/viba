"""Data-driven tests for viba.is_sub_type.

Layout:  tests/data/is_sub_type/{sub,sup}NNN.viba  +  expected.txt
Each .viba file is a module whose LAST definition is the entry
(earlier ones provide context: aliases, generics, recursion).
expected.txt maps NNN -> the expected judgment.

Run directly:  python3 tests/test_is_sub_type.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ast as py_ast

from viba import ast as viba_ast
from viba.type import (
    BUILTIN_MODULE,
    Err,
    Ok,
    RuleContainsPoisonError,
    custom_module,
    entry_type,
)
from viba.is_sub_type import is_sub_type

DATA = Path(__file__).resolve().parent / "data" / "is_sub_type"
PASS = FAIL = 0


def check(got: bool, expected: bool, label: str):
    global PASS, FAIL
    if got is expected:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}: expected {expected}, got {got}")


def _entry_node(text: str):
    tree = viba_ast.parse(text)
    defs = [n for n in tree.body if not isinstance(n, viba_ast.Import)]
    node = defs[-1]
    return node.body if isinstance(node, viba_ast.TypeDefinition) else node


def load_entry(text: str):
    """Last definition of the module, as the judgment's entry type."""
    from viba.type import AstNodeType
    module = custom_module(text)
    return AstNodeType(_entry_node(text), module)


def load_entry_as(text: str, module):
    """Re-parsed entry node pinned to an existing module context.

    Round-trip identity: unparse -> re-parse must denote the same type
    in the SAME module, or nominal constructors would differ by the
    fresh module's object identity alone.
    """
    from viba.type import AstNodeType
    return AstNodeType(_entry_node(text), module)


def expect_raise_poison(label: str, thunk):
    global PASS, FAIL
    try:
        thunk()
        print(f"FAIL: {label}: no RuleContainsPoisonError raised")
        FAIL += 1
    except RuleContainsPoisonError:
        PASS += 1


def has_poison(entry) -> bool:
    refs = viba_ast.walk(entry.ast_node)
    return any(isinstance(n, viba_ast.TypeRef) and n.name == "AssertionViolated" for n in refs)


def run_data_cases():
    expected = {}
    for line in (DATA / "expected.txt").read_text().splitlines():
        num, _, want = line.partition(" ")
        if num:
            expected[num] = want.strip().startswith("true")
    for sup_path in sorted(DATA.glob("sup*.viba")):
        num = sup_path.stem[3:]
        want = expected[num]
        sub_e = load_entry((DATA / f"sub{num}.viba").read_text())
        sup_e = load_entry(sup_path.read_text())
        check(is_sub_type(sub_e, sup_e), want, f"case {num}")
        if has_poison(sub_e) or has_poison(sup_e):
            continue  # reflexive checks would lint-raise on the poison
        check(is_sub_type(sub_e, sub_e), True, f"case {num} sub reflexive")
        check(is_sub_type(sup_e, sup_e), True, f"case {num} sup reflexive")
        canon_sub = viba_ast.unparse(viba_ast.parse((DATA / f"sub{num}.viba").read_text()))
        canon_sup = viba_ast.unparse(viba_ast.parse(sup_path.read_text()))
        rt_sub = load_entry_as(canon_sub, sub_e.container_module)
        rt_sup = load_entry_as(canon_sup, sup_e.container_module)
        check(is_sub_type(rt_sub, sub_e), True, f"case {num} sub roundtrip")
        check(is_sub_type(rt_sup, sup_e), True, f"case {num} sup roundtrip")


def run_py_side_cases():
    """What data files cannot express: poison raises, env generics."""
    check(is_sub_type(entry_type("42"), entry_type("int")), True, "42 <: int")
    expect_raise_poison(
        "poison on sup side raises",
        lambda: is_sub_type(entry_type("42"), entry_type("AssertionViolated")))
    expect_raise_poison(
        "poison nested in sup sum raises",
        lambda: is_sub_type(entry_type("$a 1"), entry_type("$a int | $b AssertionViolated")))
    g_target = custom_module("G[T] := $v T")
    g_source = custom_module("", environment=lambda name: Ok(g_target))
    g_local = custom_module("G[T] := $v T")
    check(is_sub_type(entry_type("G[int]", g_source), entry_type("G[int]", g_local)),
          False, "generic via env vs local: nominal, different modules -> False")


def _is_test_cases_assign(node) -> bool:
    if not isinstance(node, py_ast.Assign):
        return False
    return getattr(node.targets[0], "id", "") == "test_cases"


def _is_case_tuple(tup) -> bool:
    return isinstance(tup, py_ast.Tuple) and isinstance(tup.elts[0], py_ast.Constant)


def _suite_case_nodes():
    """Source strings of the parser suite's definition cases."""
    src = Path("viba/parser.py").read_text()
    for node in filter(_is_test_cases_assign, py_ast.walk(py_ast.parse(src))):
        for tup in filter(_is_case_tuple, node.value.elts):
            yield tup.elts[0].value


def run_suite_reflexivity():
    """Every parser test case must be reflexive."""
    count = 0
    for src in _suite_case_nodes():
        body = viba_ast.parse(src).body[0]
        if isinstance(body, viba_ast.Import):
            continue
        text = viba_ast.unparse(viba_ast.Module([body]))
        t = load_entry(text)
        check(is_sub_type(t, t), True, f"suite reflexive {src!r}")
        count += 1
    print(f"bulk reflexivity on {count} suite definitions")


run_data_cases()
run_py_side_cases()
run_suite_reflexivity()
print(f"\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)

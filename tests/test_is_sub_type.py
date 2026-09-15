"""Data-driven tests for viba.is_sub_type.

Layout:  tests/data/is_sub_type/{sub,sup}NNN.viba  +  expected.txt
Each .viba file is a module whose LAST definition is the entry
(earlier ones provide context: aliases, generics, recursion).
expected.txt maps NNN -> true | false | error (an Err result).

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


def check_result(result, want, label: str):
    """want: True / False as Ok verdicts, or "error" for an Err."""
    global PASS, FAIL
    if want == "error":
        ok, got = isinstance(result, Err), type(result).__name__
    elif isinstance(result, Ok):
        ok, got = result.value is want, repr(result.value)
    else:
        ok, got = False, f"Err({result.message!r})"
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}: expected {want}, got {got}")


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


def has_ref(entry, name: str) -> bool:
    refs = viba_ast.walk(entry.ast_node)
    return any(isinstance(n, viba_ast.TypeRef) and n.name == name for n in refs)


def parse_want(text: str):
    text = text.strip()
    if text.startswith("true"):
        return True
    if text.startswith("false"):
        return False
    return "error"


def run_data_cases():
    expected = {}
    for line in (DATA / "expected.txt").read_text().splitlines():
        num, _, want = line.partition(" ")
        if num:
            expected[num] = parse_want(want)
    for sup_path in sorted(DATA.glob("sup*.viba")):
        num = sup_path.stem[3:]
        want = expected[num]
        sub_e = load_entry((DATA / f"sub{num}.viba").read_text())
        sup_e = load_entry(sup_path.read_text())
        check_result(is_sub_type(sub_e, sup_e), want, f"case {num}")
        if want == "error" or has_ref(sub_e, "AssertionViolated"):
            continue  # reflexive checks are for clean judgments only
        check_result(is_sub_type(sub_e, sub_e), True, f"case {num} sub reflexive")
        check_result(is_sub_type(sup_e, sup_e), True, f"case {num} sup reflexive")
        canon_sub = viba_ast.unparse(viba_ast.parse((DATA / f"sub{num}.viba").read_text()))
        canon_sup = viba_ast.unparse(viba_ast.parse(sup_path.read_text()))
        rt_sub = load_entry_as(canon_sub, sub_e.container_module)
        rt_sup = load_entry_as(canon_sup, sup_e.container_module)
        check_result(is_sub_type(rt_sub, sub_e), True, f"case {num} sub roundtrip")
        check_result(is_sub_type(rt_sup, sup_e), True, f"case {num} sup roundtrip")


def run_py_side_cases():
    """What data files cannot express: lint verdicts, env behavior."""
    check_result(is_sub_type(entry_type("42"), entry_type("int")), True, "42 <: int")
    poisoned_sup = is_sub_type(entry_type("42"), entry_type("AssertionViolated"))
    check_result(poisoned_sup, "error", "poison on sup side -> Err")
    nested = is_sub_type(entry_type("$a 1"), entry_type("$a int | $b AssertionViolated"))
    check_result(nested, "error", "poison nested in sup sum -> Err")
    poison_sub = is_sub_type(entry_type("AssertionViolated"), entry_type("int"))
    check_result(poison_sub, False, "poison on sub side -> Ok(False)")
    sup_ellipsis = is_sub_type(entry_type("42"), entry_type("..."))
    check_result(sup_ellipsis, "error", "ellipsis on sup side -> Err")
    sub_ellipsis = is_sub_type(entry_type("..."), entry_type("int"))
    check_result(sub_ellipsis, "error", "ellipsis on sub side -> Err")


def _err_env(name):
    return Err(f"no module {name!r}")


def _env_serving(target, wanted: str):
    def env(name):
        return Ok(target) if name == wanted else Err("?")
    return env


def run_env_cases():
    """Module environment: Err paths and mixed resolution."""
    target = custom_module("Ext := $x int")
    hard = custom_module("", environment=_err_env)
    t_here = entry_type("T", hard)
    t_there = entry_type("T", custom_module(""))
    check_result(is_sub_type(t_here, t_here), True, "env Err: opaque atom reflexive")
    check_result(is_sub_type(t_here, t_there), True, "env Err: opaque is name-keyed")
    u_here = entry_type("U", hard)
    check_result(is_sub_type(t_here, u_here), False, "env Err: different names differ")
    mixed = custom_module("Local := $y str", environment=_env_serving(target, "Ext"))
    ext = entry_type("Ext", mixed)
    ext_again = entry_type("Ext", custom_module("Ext := $x int"))
    check_result(is_sub_type(ext, ext_again), True, "env Ok: resolves, structural")
    wrap = custom_module("Wrap := $w Missing", environment=_err_env)
    wrap_a = entry_type("Wrap", wrap)
    wrap_b = entry_type("Wrap", custom_module("Wrap := $w Missing"))
    check_result(is_sub_type(wrap_a, wrap_b), True, "env Err inside body: name-keyed")
    check_generic_env_case()


def check_generic_env_case():
    g_target = custom_module("G[T] := $v T")
    g_source = custom_module("", environment=_env_serving(g_target, "G"))
    g_local = custom_module("G[T] := $v T")
    sub = entry_type("G[int]", g_source)
    sup = entry_type("G[int]", g_local)
    check_result(is_sub_type(sub, sup), False, "generic env vs local: nominal")


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
    """Every parser test case must be reflexive (ellipsis cases skip)."""
    count = skipped = 0
    for src in _suite_case_nodes():
        body = viba_ast.parse(src).body[0]
        if isinstance(body, viba_ast.Import):
            continue
        text = viba_ast.unparse(viba_ast.Module([body]))
        if "..." in text:
            skipped += 1  # open types are lint errors, not judgments
            continue
        t = load_entry(text)
        check_result(is_sub_type(t, t), True, f"suite reflexive {src!r}")
        count += 1
    print(f"bulk reflexivity on {count} suite definitions ({skipped} ellipsis skipped)")


run_data_cases()
run_py_side_cases()
run_env_cases()
run_suite_reflexivity()
print(f"\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)

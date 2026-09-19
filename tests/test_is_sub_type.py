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

from viba import viba_ast
from viba.type import (
    BUILTIN_MODULE,
    AstNodeType,
    Err,
    IntType,
    Ok,
    StrType,
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
        ok, got = result.ok_value is want, repr(result.ok_value)
    else:
        ok, got = False, f"Err({result.err_msg!r})"
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
        if want == "error":
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
    check_result(is_sub_type(t_here, t_here), "error", "env Err: unresolved -> Err")
    check_result(is_sub_type(t_here, t_there), "error", "env Err: unresolved either side")
    u_here = entry_type("U", hard)
    check_result(is_sub_type(t_here, u_here), "error", "env Err: names differ, still Err")
    mixed = custom_module("Local := $y str", environment=_env_serving(target, "Ext"))
    ext = entry_type("Ext", mixed)
    ext_again = entry_type("Ext", custom_module("Ext := $x int"))
    check_result(is_sub_type(ext, ext_again), True, "env Ok: resolves, structural")
    wrap = custom_module("Wrap := $w Missing", environment=_err_env)
    wrap_a = entry_type("Wrap", wrap)
    wrap_b = entry_type("Wrap", custom_module("Wrap := $w Missing"))
    check_result(is_sub_type(wrap_a, wrap_b), "error", "env Err inside body: Err")
    check_generic_env_case()


def check_generic_env_case():
    g_target = custom_module("G[T] := $v T")
    g_source = custom_module("", environment=_env_serving(g_target, "G"))
    g_local = custom_module("G[T] := $v T")
    sub = entry_type("G[int]", g_source)
    sup = entry_type("G[int]", g_local)
    check_result(is_sub_type(sub, sup), False, "generic env vs local: nominal")


def _type_serving(target, wanted: str):
    def env(name):
        return Ok(target) if name == wanted else Err("?")
    return env


def entry_with_env(source, env_get, module=None):
    """An inline entry whose free names fall back to env_get."""
    module = module or custom_module("")
    node = entry_type(source, module).ast_node
    return AstNodeType(node, module, env_get)


def run_env_get_cases():
    """AstNodeType.env_get: free-name channel and module priority."""
    env_int = _type_serving(IntType(), "T")
    env_str = _type_serving(StrType(), "T")
    bound = entry_with_env("$x T", env_int)
    res = is_sub_type(bound, entry_type("$x int"))
    check_result(res, True, "env_get binds free name -> True")
    local = custom_module("T := $t str")
    over = entry_with_env("$x T", env_int, local)
    res = is_sub_type(over, entry_type("$x str", local))
    check_result(res, True, "module definition wins over env_get")
    res = is_sub_type(over, entry_type("$x int"))
    check_result(res, False, "module definition wins: not the env meaning")
    sub_t = entry_with_env("$x T", env_int)
    sup_t = entry_with_env("$x T", env_str)
    res = is_sub_type(sub_t, sup_t)
    check_result(res, False, "sides stay independent under the same free name")
    bare = entry_with_env("$x T", _err_env)
    res = is_sub_type(bare, entry_type("$x int"))
    check_result(res, "error", "env_get miss falls back to the module error")


def run_env_get_scope_cases():
    """env_get reaches into TypeApp args and survives exponent swaps."""
    env_int = _type_serving(IntType(), "T")
    elem = entry_with_env("list[T]", env_int)
    res = is_sub_type(elem, entry_type("list[int]"))
    check_result(res, True, "env_get reaches inside a TypeApp argument")
    swapped = entry_with_env("int <- $a int", None)
    sup_swapped = entry_with_env("int <- $a T", env_int)
    res = is_sub_type(swapped, sup_swapped)
    check_result(res, True, "exponent contravariance keeps the sup env")


def run_applied_generic_cases():
    """The driver: one-side TypeApp unfolds, params bound via env_get."""
    box_mod = custom_module("Box[T] := $v T")
    free = entry_with_env("Box[T]", _type_serving(IntType(), "T"), box_mod)
    res = is_sub_type(entry_type("$v 3"), free)
    check_result(res, True, "free actual resolves through env_get at unfold")
    pair_mod = custom_module("Pair[T] := $left T * $right T")
    pair = entry_type("Pair[int]", pair_mod)
    res = is_sub_type(entry_type("$left 3 * $right 4"), pair)
    check_result(res, True, "generic product body unfolds against a product witness")
    loop_mod = custom_module("Loop[T] := $l Loop[int]")
    looping = entry_type("Loop[int]", loop_mod)
    res = is_sub_type(looping, entry_type("$l 'x'"))
    check_result(res, True, "self-referencing application terminates")


def run_not_cases():
    """50 not[...] cases, one sub/sup .viba pair each (data/not)."""
    base = DATA / "not"
    expected = {}
    for line in (base / "expected.txt").read_text().splitlines():
        num, _, want = line.partition(" ")
        if num:
            expected[num] = parse_want(want)
    for sup_path in sorted(base.glob("sup*.viba")):
        num = sup_path.stem[3:]
        sub_e = load_entry((base / f"sub{num}.viba").read_text())
        sup_e = load_entry(sup_path.read_text())
        check_result(is_sub_type(sub_e, sup_e), expected[num], f"not case {num}")


def run_canonical_chain_cases():
    """主链压成链、支链留成分组：规范化只并一路 $left / $result。

    支链（$right 那侧的嵌套）原样算主链里的一个元素，它自己也递归成链，
    所以 A * (B * C) 与 A * B * C 不同形——指数的分组更是不能丢。
    """
    def canon(text: str) -> str:
        module = custom_module(f"__entry__ := {text}")
        body = {d.name: d for d in module.module.body}["__entry__"].body
        tree = viba_ast.Module([viba_ast.TypeDefinition(
            "x", viba_ast.convert_to_chain_style(body))])
        return viba_ast.unparse_module(tree)

    def judge(sub: str, sup: str):
        return is_sub_type(entry_type(sub), entry_type(sup))

    check(canon("int | str | bool") == canon("(int | str) | bool"), True,
          "canonical: 主链一路 $left，压成同一条")
    check(canon("int | str | bool") == canon("int | (str | bool)"), False,
          "canonical: 和类型的支链保留分组")
    check(canon("int * str * bool") == canon("int * (str * bool)"), False,
          "canonical: 积类型的支链保留分组")
    check(canon("int * (str * (bool * int))") == canon("int * str * bool * int"), False,
          "canonical: 支链的支链递归成深层链")
    check(canon("int <- str <- bool") == canon("(int <- str) <- bool"), True,
          "canonical: 指数的主链（$result 那侧）压成一条")
    check(canon("int <- str <- bool") == canon("int <- (str <- bool)"), False,
          "canonical: 指数的支链（$argument）不被并进主链")
    check_result(judge("int <- str <- bool", "(int <- str) <- bool"), True,
                 "judgment: 左嵌套的指数同一条主链")
    check_result(judge("int <- str <- bool", "int <- (str <- bool)"), False,
                 "judgment: 右嵌套的指数不是同一回事")
    check_result(judge("int * str * bool", "int * (str * bool)"), True,
                 "judgment: 积的结合律仍然成立（分组不同形，判定不受影响）")
    check_result(judge("int | str | bool", "int | (str | bool)"), True,
                 "judgment: 和的结合律仍然成立")


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
    """Every closed parser suite case must be reflexive (others skip)."""
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
        result = is_sub_type(t, t)
        if isinstance(result, Err):
            skipped += 1  # parser cases need not be closed judgments
            continue
        check(result.ok_value, True, f"suite reflexive {src!r}")
        count += 1
    print(f"bulk reflexivity on {count} suite definitions ({skipped} skipped)")


run_data_cases()
run_py_side_cases()
run_env_cases()
run_env_get_cases()
run_env_get_scope_cases()
run_applied_generic_cases()
run_not_cases()
run_canonical_chain_cases()
run_suite_reflexivity()
print(f"\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)

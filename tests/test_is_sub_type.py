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
from viba.check_tag_and_inline import check_tag_and_inline
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
from viba.reflect import Config
from viba.viba_type_descriptor import (empty_pool, parse_viba_file, pool_add_file,
                                       pool_find_definition)

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
    in the SAME module, or a name that module resolution has to settle
    (a bare generic, say) would be looked up in a fresh, empty one.
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


# 语料里本来就写错的：重标签（sub071 写在同层、sub112 摊进来、sup113 写在同层）
# 与内联成环（sub117、sub120-sub123、sub125）。语料按"对/错"标的是子类型判定，
# 不是写法；这两边恰好一致，所以能拿它当设计的审查。
MALFORMED_CORPUS = {"sub071", "sub112", "sup113", "sub117",
                    "sub120", "sub121", "sub122", "sub123", "sub125"}


def run_design_review():
    """每份语料都是一个设计：摊开以后 tag 不得重复，内联链要摊到底。"""
    wrong = []
    files = 0
    for path in sorted(DATA.glob("*.viba")):
        if not path.stem.startswith(("sub", "sup")):
            continue
        files += 1
        pool = empty_pool()
        parsed = parse_viba_file(pool, path.read_text(), path.name, path.stem)
        if not isinstance(parsed, Ok):
            reviewed = f"does not parse: {parsed.err_msg}"
        else:
            built = pool_add_file(pool, parsed.ok_value)
            # 连池子都建不起来（写在同一层的重标签）也是这一遍的事
            reviewed = built if not isinstance(built, Ok) else check_tag_and_inline(built.ok_value)
        want_clean = path.stem not in MALFORMED_CORPUS
        if isinstance(reviewed, Ok) is not want_clean:
            wrong.append(f"{path.name}: {reviewed}")
    for line in wrong:
        check(False, True, f"design review {line}")
    if not wrong:
        check(True, True, f"design review of {files} corpus files")


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
    check_result(is_sub_type(sub, sup), True, "generic env vs local: same body")


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


def run_literal_alias_cases():
    """A literal sub is structural: an alias of a container unfolds."""
    module = custom_module("""
AppendOnlyList[T] := list[T]
ReadOnlyList[T] := list[T]
AppendOnlyDict[K, V] := dict[K, V]
MaybeList[T] := list[T] | nil
Layer[T] := AppendOnlyList[T]
Box[T] := $v T
Loop[T] := Loop[T]
""")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    check_result(judge('ListLiteral["a"]', "AppendOnlyList[str]"), True,
                 "a literal container is a resident of an alias of list")
    check_result(judge("ListLiteral[1]", "AppendOnlyList[int]"), True,
                 "elements are checked against the bound parameter")
    check_result(judge('DictLiteral[("k", 1)]', "AppendOnlyDict[str, int]"), True,
                 "the same for dict, both parameters")
    check_result(judge("ListLiteral[1]", "AppendOnlyList[str]"), False,
                 "an element that does not fit the parameter is still False")
    check_result(judge('ListLiteral["a"]', "MaybeList[str]"), True,
                 "the alias body may be a sum with the container in it")
    check_result(judge('ListLiteral["a"]', "Layer[str]"), True,
                 "aliases of aliases unfold")
    check_result(judge("ListLiteral[]", "Loop[int]"), True,
                 "a self-referencing alias terminates (coinductive assumption)")
    check_result(judge("AppendOnlyList[str]", "list[str]"), True,
                 "an alias and its body are the same type")
    check_result(judge("ReadOnlyList[str]", "AppendOnlyList[str]"), True,
                 "two aliases of one body are each other's subtype")
    check_result(judge("ListLiteral[ListLiteral[1]]", "AppendOnlyList[list[int]]"), True,
                 "a literal container of literal containers, through an alias")
    check_result(judge("ListLiteral[1]", "MaybeList[int]"), True,
                 "the alias body is a sum: the container branch is enough")
    check_result(judge("ListLiteral[1]", "Box[int]"), False,
                 "an alias of something that is not a container seats no literal")
    check_result(judge("ListLiteral[1]", "ListLiteral[int]"), True,
                 "a literal against a literal: element by element")
    check_result(judge("ListLiteral[1]", "ListLiteral[str]"), False,
                 "the same, element that does not fit")
    check_result(judge('DictLiteral[("k", 1)]', "AppendOnlyList[str]"), False,
                 "container families do not mix")
    check_result(judge("SetLiteral[]", "SetLiteral[]"), True,
                 "an empty literal against itself")


def run_recursive_generic_cases():
    """Equi-recursive: a definition is its own unfolding, coinductively."""
    module = custom_module("""
Tree[T] := $leaf T * $kids list[Tree[T]]
Other[T] := $leaf T * $kids list[Other[T]]
MyList[T] := $head T * $tail MyList[T] | nil
List[T] := $head T * $tail List[T] | nil
A[T] := B[T]
B[T] := A[T]
Loop[T] := Loop[T]
Box[T] := $v T
H[T] := str <- $in H[T]
H2[T] := str <- $in H2[list[T]]
SumRec[T] := $a T | $b SumRec[T]
""")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    check_result(judge("Tree[int]", "Tree[int]"), True,
                 "a recursive generic is its own subtype (the assumption holds)")
    check_result(judge("Tree[int]", "Tree[str]"), False,
                 "the leaf still has to fit")
    check_result(judge("MyList[int]", "List[int]"), True,
                 "two names that unfold into one another are the same type")
    check_result(judge("A[int]", "B[int]"), True,
                 "mutual recursion is assumed true while it is in flight")
    check_result(judge("Loop[int]", "Loop[str]"), True,
                 "a definition that only reaches itself is the largest type")
    check_result(judge("Loop[int]", "Box[int]"), True,
                 "which is why it fits anything it is compared with")
    check_result(judge("Box[int]", "Loop[int]"), True,
                 "and anything fits it")
    check_result(judge("H[int]", "H[int]"), True,
                 "recursion through an exponent terminates")
    check_result(judge("H[int]", "H[str]"), True,
                 "and such a definition is likewise the largest type")
    check_result(judge("SumRec[int]", "SumRec[int]"), True,
                 "recursion through a sum branch terminates")
    check_result(judge("SumRec[int]", "SumRec[str]"), False,
                 "the branch that is not recursive still has to fit")
    check_result(judge("H2[int]", "H2[int]"), True,
                 "an actual that grows every turn still converges")


def run_structural_generic_cases():
    """No nominal generics: every name is an alias of its body."""
    module = custom_module("""
Num := int | float
Handler[T] := str <- $in T
DeepHandler[T] := Handler[T]
Putter[T] := $out T
Pair[K, V] := $key K * $value V
Alpha[T] := $v T
Beta[T] := $v T
Box[T] := $v T
AliasOfBox[T] := Box[T]
MissingAlias[T] := Missing
Metric[T] := $value T
Predicate[Cond, Code] := $assert_cond Cond * $assert_python_code Code
PredicationFailed[T, Msg] := $__assertion_failed_original_data T * $__assertion_failed_error_msg Msg
""")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    check_result(judge("Handler[Num]", "Handler[int]"), True,
                 "a contravariant parameter: Num <: int in the sup's place")
    check_result(judge("Handler[int]", "Handler[Num]"), False,
                 "and the other way round is False")
    check_result(judge("Putter[int]", "Putter[Num]"), True,
                 "a covariant parameter: int <: Num")
    check_result(judge("Putter[Num]", "Putter[int]"), False,
                 "covariant the other way is False")
    check_result(judge("Alpha[int]", "Beta[int]"), True,
                 "two names with one body are each other's subtype")
    check_result(judge("Pair[int]", "Pair[int]"), True,
                 "actuals that do not fit the parameters: no body, so name and args")
    check_result(judge("Pair[int]", "Pair[str]"), False,
                 "the same fallback still compares the actuals")
    check_result(judge("Pair[int, str]", "Pair[int]"), False,
                 "the fallback also compares how many actuals there are")
    check_result(judge("list[int]", "list[int, str]"), False,
                 "a builtin compares its actuals the same way")
    check_result(judge("Handler[Num]", "DeepHandler[int]"), True,
                 "variance survives an alias chain")
    check_result(judge("Handler[int]", "DeepHandler[Num]"), False,
                 "and the wrong direction is still False")
    check_result(judge("Box[int]", "AliasOfBox[int]"), True,
                 "an alias of an alias")
    check_result(judge("AliasOfBox[int]", "Box[int]"), True,
                 "the same, the other way round")
    check_result(judge("AliasOfBox[int]", "Box[str]"), False,
                 "and the actuals still count")
    check_result(judge("Box", "Box"), True,
                 "two bare generic names: nothing to unfold, the name matches")
    check_result(judge("Box", "AliasOfBox"), False,
                 "two bare names that differ do not")
    check_result(judge("Box[int]", "MissingAlias[int]"), "error",
                 "unfolding an alias whose body names nothing is an Err")
    check_result(judge("Box[int]", "Missing"), "error",
                 "the same when the missing name is the whole body")
    check_result(judge("Metric[int]", "Metric[Num]"), True,
                 "a rule-layer shaped generic is covariant in its parameter")
    check_result(judge("Metric[Num]", "Metric[int]"), False,
                 "and False the other way")
    check_result(judge("PredicationFailed[nil, str]", "Predicate[int, str]"), False,
                 "the poison seats in no Predicate: their tags differ")


def run_unit_alias_cases():
    """An alias of nil or never is that unit: the sub unfolds first."""
    module = custom_module("""
MyNil[T] := nil
MyNever[T] := never
Nil2 := nil
Never2 := never
Box[T] := $v T
Pair[K, V] := $key K * $value V
""")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    check_result(judge("MyNil[int]", "nil"), True,
                 "X[T] := nil against nil")
    check_result(judge("MyNil[int]", "Nil2"), True,
                 "the same against a plain alias of nil")
    check_result(judge("nil", "MyNil[int]"), True,
                 "and the other way round (the sup unfolds)")
    check_result(judge("MyNil[int]", "MyNil[str]"), True,
                 "two applications of it: the body has no parameter")
    check_result(judge("MyNil[int]", "Box[int]"), False,
                 "nil is not a product")
    check_result(judge("Box[int]", "MyNil[int]"), False,
                 "and a product is not nil")
    check_result(judge("MyNever[int]", "never"), True,
                 "X[T] := never against never")
    check_result(judge("MyNever[int]", "Never2"), True,
                 "the same against a plain alias of never")
    check_result(judge("never", "MyNever[int]"), True,
                 "never fits anything, alias or not")
    check_result(judge("list[int]", "nil"), False,
                 "a builtin container is not nil")
    check_result(judge("Pair[int]", "nil"), False,
                 "an application with no body to unfold is not nil either")


def run_inline_member_cases():
    """An untagged product member is an inline slot: a product hands its own
    members over (recursively), the product unit disappears, and the same tag
    twice — inlined or written — is malformed input."""
    module = custom_module("""
A := $x int * $y str
B := A * $z bool
C := B * $w float
Unit := Object
UBox := Unit * $a int
Pos := int * $a int
Inner := $p int * $q str
Wrap[T] := T * $a int
Applied := Wrap[Inner]
MyNil[T] := nil
MyUnit[T] := T
Dup := A * $x bool
Cycle := Cycle * $c int
AliasCycle := AliasTail * $x int
AliasTail := AliasCycle
Mutual := Other * $m int
Other := Mutual * $o int
Tagged[T] := $x T
TaggedBox := Tagged[int] * $y int
BareBox[T] := T * $x int
BareInt := BareBox[int] * $y int
Nested[T] := $x (Tagged[T])
NestedBox := Nested[int] * $z bool
First[T] := $f T
Second[T] := $s T
TwoParams := First[int] * Second[str] * $z bool
""")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    check_result(judge("B", "$x int * $y str * $z bool"), True,
                 "the inlined tags are B's own")
    check_result(judge("$x int * $y str * $z bool", "B"), True,
                 "and the judgment is the other way round too")
    check_result(judge("C", "B * $w float"), True,
                 "the chain inlines all the way down")
    check_result(judge("B", "$x int * $y str"), True,
                 "a tagged field of an inlined member is covered by width")
    check_result(judge("$x int * $y str", "B"), False,
                 "but the inline's own tag is still required")
    check_result(judge("UBox", "$a int"), True,
                 "the unit behind a name is no member")
    check_result(judge("$a int", "UBox"), True,
                 "in both directions")
    check_result(judge("Pos", "int * $a int"), True,
                 "a member that is no product keeps its position")
    check_result(judge("Applied", "$p int * $q str * $a int"), True,
                 "a generic actual that is a product is inlined")
    check_result(judge("MyNil[int] * $a int", "$a int"), True,
                 "an application whose body is the unit is no member")
    check_result(judge("$a int", "MyNil[int] * $a int"), True,
                 "in both directions")
    check_result(judge("MyUnit[Inner] * $a int", "$p int * $q str * $a int"), True,
                 "an application that lands on a product is inlined")
    check_result(judge("Wrap[Inner] * $z bool", "$p int * $q str * $a int * $z bool"), True,
                 "as an untagged member inside another product")
    check_result(judge("Cycle", "Cycle"), "error",
                 "a member that inlines itself has no expansion: an Err, not a hang")
    cycle = is_sub_type(entry_type("Cycle", module), entry_type("Cycle", module))
    check(isinstance(cycle, Err) and "comes back to 'Cycle'" in cycle.err_msg, True,
          "and the Err says which chain comes back")
    check_result(judge("AliasCycle", "AliasCycle"), "error",
                 "an alias can close the same loop")
    check_result(judge("Mutual", "Mutual"), "error",
                 "so can two definitions")
    check_result(judge("Cycle", "$c int"), "error",
                 "the cycle is malformed whatever it is judged against")
    check_result(judge("TaggedBox", "$x int * $y int"), True,
                 "an inlined generic carries its actual into a tagged member")
    check_result(judge("$x int * $y int", "TaggedBox"), True,
                 "in both directions")
    check_result(judge("TaggedBox", "$x str * $y int"), False,
                 "and the actual really is the member's type")
    check_result(judge("BareInt", "int * $x int * $y int"), True,
                 "a bare parameter member keeps its own actual")
    check_result(judge("NestedBox", "$x ($x int) * $z bool"), True,
                 "a nested application keeps its own bindings")
    check_result(judge("TwoParams", "$f int * $s str * $z bool"), True,
                 "two inlined generics do not share a parameter name")
    check_result(judge("TwoParams", "$f str * $s int * $z bool"), False,
                 "each member keeps the binding it was written under")
    check_result(judge("Dup", "$x int"), "error",
                 "the same tag twice through an inline -> Err")
    check_result(judge("$a int * $a str", "$a int"), "error",
                 "written twice at one level -> Err")
    dup = is_sub_type(entry_type("Dup", module), entry_type("$x int", module))
    check(isinstance(dup, Err) and "$x" in dup.err_msg, True,
          "the Err names the tag that repeats")


def run_inline_cycle_guard_cases():
    """成环的内联在每条路上都是 Err，不是崩：普通判定、带终止子的禁止链
    （禁止链要先问 sub 的字段，那条路以前会 RecursionError）。"""
    module = custom_module("""
H := int
notcrimes := never <- $not_operand ($h H | $a A)
A := A * $x int
Sub := A * $x int * $h H
Loop := Loop * $c int
""")
    terminators = frozenset({"PredicationFailed"})
    check_result(is_sub_type(entry_type("Loop", module), entry_type("Loop", module)),
                 "error", "an inline cycle is malformed input")
    judged = is_sub_type(entry_type("Sub", module), entry_type("notcrimes", module),
                         terminators=terminators)
    check_result(judged, "error", "the never-headed path says so too, instead of spinning")
    check(isinstance(judged, Err) and "comes back to 'A'" in judged.err_msg, True,
          "and names the chain that comes back")
    sums = custom_module("""
H := int
notcrimes := never <- $not_operand (S | $h H)
S := S | $a int
""")
    check_result(is_sub_type(entry_type("S", sums), entry_type("notcrimes", sums),
                             terminators=terminators), False,
                 "a sum cycle leaves the branches unsettled, and comes back at all")


def run_cross_module_inline_cases():
    """跨模块的内联：别的模块那个积的 tag 一样摊进来；重标签与成环也跨文件抓。
    判定这边没有池子，所以模块从池子里的定义取（那份容器模块）再比。"""
    pool = empty_pool()
    for file_name, module_name, source in (
            ("base.viba", "base", "A := $x int * $y str\nU := Object\n"),
            ("main.viba", "main",
             "import base\nB := base.A * $z bool\nUBox := base.U * $w int\n"
             "Dup := base.A * $x int\n"),
            ("other.viba", "other", "import ring\nCyc := ring.Ring * $c int\n"),
            ("ring.viba", "ring", "import other\nRing := other.Cyc * $r int\n"),
            ("mid.viba", "pkg.mod", "Mid := int\n"),
            ("deepest.viba", "pkg.mod.sub",
             "import pkg.mod\nDeep := Object * $m pkg.mod.Mid * $k int\n"),
            ("user.viba", "user",
             "import pkg.mod\nimport pkg.mod.sub\n"
             "Long := pkg.mod.sub.Deep\nShort := pkg.mod.Mid\nOld := mod.Mid\n")):
        parsed = parse_viba_file(pool, source, file_name, module_name)
        assert isinstance(parsed, Ok), parsed
        pool = pool_add_file(pool, parsed.ok_value).ok_value

    def judge(name, sup):
        definition = pool_find_definition(pool, name).ok_value
        module = definition.body.payload.resolvable_type.container_module
        here = AstNodeType(definition.body.payload.resolvable_type.ast_node, module)
        return is_sub_type(here, entry_type(sup, module))

    check_result(judge("main.B", "$x int * $y str * $z bool"), True,
                 "a product from another module is inlined")
    check_result(judge("main.B", "$x int * $y str"), True,
                 "its tags are the product's own")
    check_result(judge("main.UBox", "$w int"), True,
                 "a unit from another module is no member")
    check_result(judge("main.Dup", "$x int"), "error",
                 "a tag repeated across modules -> Err")
    check_result(judge("ring.Ring", "Ring"), "error",
                 "an inline ring across two files -> Err")
    ringed = judge("ring.Ring", "Ring")
    check(isinstance(ringed, Err) and "comes back to" in ringed.err_msg, True,
          "and it names the chain")
    check_result(judge("user.Long", "$m int * $k int"), True,
                 "import pkg.mod without as binds pkg.mod, so pkg.mod.sub.Deep resolves")
    check_result(judge("user.Short", "int"), True,
                 "and pkg.mod.Mid resolves by the shorter module too")
    check_result(judge("pkg.mod.sub.Deep", "$m int * $k int"), True,
                 "a module may reach another by its whole path")
    check_result(judge("user.Old", "int"), "error",
                 "the last word of a module path is not bound: mod.Mid resolves nowhere")


def run_function_chain_cases():
    """函数之间比函数：结果协变、参数逆变，按书写顺序逐位比（$arg0 对 $arg0）。
    先把 sub 弄到 sup 的长度（sup 不动）：sub 短了补 never，长了截掉；弄齐再逐位比。
    除 never 外，非函数一律 False。"""
    module = custom_module("")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    check_result(judge("int <- str <- float", "int <- str"), True,
                 "the longer chain is used where the shorter is expected")
    check_result(judge("int <- str", "int <- str <- float"), False,
                 "the other way round it is not: sub has nothing to offer for $arg1")
    check_result(judge("int <- str", "int <- str <- never"), True,
                 "shorter, the sub is padded with never, and never is what sup asks for")
    check_result(judge("int <- str <- float", "int <- float"), False,
                 "the shared prefix compares by position: $arg0 is str, not float")
    check_result(judge("int <- float <- str", "int <- float"), True,
                 "here $arg0 is float on both sides")
    check_result(judge("int <- str <- float", "int <- str <- nil"), False,
                 "equal lengths compare every argument: nil is no float")
    check_result(judge("int <- str <- nil", "int <- str"), True,
                 "a longer chain is not asked about the arguments it adds")
    check_result(judge("int <- str <- never", "int <- str"), True,
                 "cut, and the never it drops was never supplyable")
    check_result(judge("int <- str", "int <- str <- never"), True,
                 "with never padding the other way as well: the two are interchangeable")
    check_result(judge("int <- str", "int <- str <- nil"), False,
                 "a nil extra is not interchangeable: cut away one way, unanswered coming back")
    check_result(judge("int <- str <- float", "int <- str"), True,
                 "and an unused float extra is cut away just the same")
    check_result(judge("int <- float", "int <- float <- nil"), False,
                 "an extra nil argument is still an argument: arity is part of the shape")
    check_result(judge("int <- float", "int <- float <- Object"), False,
                 "Object is nil, and the argument is still asked for")
    check_result(judge("int <- float", "int <- float <- never"), True,
                 "an extra never is exactly what the padded sub can answer")
    check_result(judge("int <- str <- float <- bool", "int <- str"), True,
                 "however many it adds, they are all beyond the sup")
    check_result(judge("int <- str", "int <- str <- float <- never"), False,
                 "two short: the padded never meets float and that is no bool")
    check_result(judge("int <- str <- nil", "int <- nil"), False,
                 "$arg0 is str against nil")
    check_result(judge("int <- str", "int"), False,
                 "a function is not a plain type")
    check_result(judge("int", "int <- str"), False,
                 "a plain type is not a function")
    check_result(judge("str <- str", "int <- str"), False,
                 "the results compare covariantly")
    check_result(judge("int <- str", "(int | nil) <- str"), True,
                 "a wider result is a supertype: int <: int | nil")
    check_result(judge("{x} <- str", "int <- str"), False,
                 "a code block is no int either")
    check_result(judge("never", "int <- str"), True,
                 "the exception is never: the bottom fits a function")
    check_result(judge("never <- str", "int <- str"), True,
                 "a never-headed chain is a function like any other")
    check_result(judge("int <- never", "int <- float"), False,
                 "an argument compares contravariantly: float <: never is what is asked")
    check_result(judge("int <- float", "int <- never"), True,
                 "so the never argument is the wider one here")
    check_result(judge("int <- float", "int <- nil"), False,
                 "an argument slot: nil is no float, so the sub cannot take what sup passes")
    check_result(judge("int <- nil", "int <- float"), False,
                 "and float is no nil the other way")
    check_result(judge("int <- float", "int <- never"), True,
                 "never in the sup's slot asks for nothing: anything fits")
    check_result(judge("int <- never", "int <- float"), False,
                 "while a sub that asks for never can take only nothing")
    check_result(judge("int <- nil", "int <- Object"), True,
                 "Object is nil in an argument slot too")
    check_result(judge("int <- int", "int <- 42"), True,
                 "the sup takes only 42 and the sub takes any int: True")
    check_result(judge("int <- 42", "int <- int"), False,
                 "the other way the sub takes only 42 while sup may pass any int")
    check_result(judge("int <- (int | nil)", "int <- int"), True,
                 "a wider argument slot is a subtype: int <: int | nil")
    check_result(judge("int <- int", "int <- (int | nil)"), False,
                 "a narrower one is not: nil <: int fails")
    check_result(judge("int <- (int <- str <- never)", "int <- (int <- str <- float)"), True,
                 "a branch argument flips the inner direction: the result of contravariance twice")
    check_result(judge("int <- (int <- str <- float)", "int <- (int <- str <- never)"), False,
                 "and the other way round is false")
    check_result(judge("int <- (int <- never)", "int <- (int <- float <- str)"), True,
                 "a branch argument of a different arity: the inner sub is cut, never <: float")
    check_result(judge("int <- (int <- float <- str)", "int <- (int <- never)"), False,
                 "the other way the inner sub is padded, and float <: never fails")


def run_config_cases():
    """config：哪些写下来的名字算单位元，裸名与应用同名同权。规则层把说明块
    `Assert[{...}]` 说成单位，判定就该当单位读，而不是去解析那个名字。"""
    module = custom_module("""
Result[T] := Oneof | $ok T | $err str
JsonLike :=
    Oneof
  | nil
  | bool
  | int
  | float
  | str
  | list[JsonLike]
  | set[JsonLike]
  | dict[str, JsonLike]
MetricFuncInterface := Result[JsonLike] <- never
DemoPoint := ($x int * $y int)
GetDistance := Result[int] <- DemoPoint <- DemoPoint
MetricObject := Object * $__metric_object_yanatuttn__ nil  # yanatuttn = you_are_not_allowed_to_use_this_tag_name
Metric[CoreFunc] :=
    MetricObject
  * $func CoreFunc
  * Assert[{
      CoreFunc <: (Result[JsonLike] <- never)
    }]
""")
    config = Config(nil_eqv={"Assert", "Hint"}, never_eqv={"Oneof"})

    def judge(sub, sup, given=config):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module), config=given)

    check_result(judge("Metric[GetDistance]", "Metric[GetDistance]"), True,
                 "with the config the Assert block is a unit, so the metric compares")
    check_result(judge("Metric[GetDistance]", "Metric[GetDistance]", given=None), "error",
                 "without it the block is a name nothing defines, and the judgment Errs")
    check_result(judge("Assert[{x}]", "nil"), True,
                 "an applied unit name is the unit")
    check_result(judge("nil", "Hint[{y}]"), True,
                 "and the same read backwards")
    check_result(judge("Object * Assert[{x}] * $a int", "$a int"), True,
                 "an untagged unit member is no member")
    check_result(judge("Result[int] <- DemoPoint <- m.Hint[$python_code {x}]",
                       "Result[int] <- DemoPoint"), True,
                 "a unit named through an import counts as the unit")
    check_result(judge("Object * m.Assert[{x}] * $a int", "$a int"), True,
                 "the same for an inline member")
    check_result(judge("GetDistance", "MetricFuncInterface"), True,
                 "the metric's own interface holds")
    check_result(judge("GetDistance", "MetricFuncInterface", given=None), True,
                 "and it needs no config to hold")
    check_result(judge("Result[int] <- DemoPoint <- Assert[{x}]", "Result[int] <- DemoPoint"), True,
                 "documentation carries no position: a block handed to a unit name drops")
    check_result(judge("Result[int] <- {说明} <- DemoPoint", "Result[int] <- DemoPoint"), True,
                 "and a bare block drops on its own")
    check_result(judge("Result[int] <- Hint <- DemoPoint", "Result[int] <- DemoPoint"), False,
                 "a unit written as a plain name is a real argument")
    check_result(judge("Result[int] <- DemoPoint <- Hint[$python_code {def metric_func(): ...}]",
                       "Result[int] <- DemoPoint"), True,
                 "a block behind a tag is documentation too")
    check_result(judge("Result[int] <- DemoPoint",
                       "Result[int] <- DemoPoint <- Hint[$python_code {def metric_func(): ...}]"),
                 True, "and the same read backwards")
    check_result(judge("Result[int] <- DemoPoint",
                       "Result[int] <- DemoPoint <- Hint[str]"), False,
                 "but an applied unit with no block in it keeps its slot")
    check_result(judge("Result[int] <- DemoPoint", "Result[int] <- DemoPoint <- Assert[{x}]"), True,
                 "and the same read backwards")
    check_result(judge("Result[int] <- Assert[{x}] <- DemoPoint", "Result[int] <- DemoPoint"), True,
                 "written in the middle: dropping it leaves the argument order alone")
    check_result(judge("Result[int] <- $a DemoPoint <- Hint[{y}]", "Result[int] <- $a DemoPoint"), True,
                 "a tagged unit argument drops too")
    check_result(judge("Result[int] <- Hint[{y}]", "Result[int] <- Assert[{x}]"), True,
                 "a chain whose only argument is a unit is a chain of no arguments")
    check_result(judge("Result[int] <- Hint[{y}]", "Result[int]"), False,
                 "and a chain of no arguments is still a function, not the bare result")
    check_result(judge("Result[int] <- DemoPoint", "Result[int] <- DemoPoint <- str"), False,
                 "a real argument is still an argument")
    check_result(judge("Result[int] <- nil <- DemoPoint", "Result[int] <- DemoPoint"), False,
                 "a real nil keeps its position: the argument list is a tuple")
    check_result(judge("Result[int] <- $t nil <- DemoPoint", "Result[int] <- DemoPoint"), False,
                 "a tagged real nil keeps it too")
    check_result(judge("Result[int] <- Object <- DemoPoint", "Result[int] <- DemoPoint"), False,
                 "Object is that nil: a plain unit name is no documentation")
    check_result(judge("Result[int] <- Oneof <- DemoPoint", "Result[int] <- DemoPoint"), False,
                 "never is not nil: it keeps the position it was written in")
    check_result(judge("Result[int] <- DemoPoint", "Result[int] <- DemoPoint <- Assert[{x}]",
                       given=None), "error",
                 "without the config nothing is named, so nothing drops")


def run_apply_cases():
    """`<<`：部分计算。给出一个参数，剩下的就是函数；给全了就是结果本身。
    给函数没有的参数、或者给一个不是函数的东西，都是不合法输入（Err）。"""
    module = custom_module("""
A := int
B := str
C := bool
Num := int | str
Int := int
Wide := (A <- $b Num) << $b Int
Narrow := (A <- $b Int) << $b Num
Given := (A <- $b B <- $c C) << $b B
GivenTail := (A <- $b B <- $c C) << $c C
GivenAll := (A <- $b B <- $c C) << $c C << $b B
GivenAllOther := (A <- $b B <- $c C) << $b B << $c C
Named := Box << $b B
Box := A <- $b B
""")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    check_result(judge("Given", "A <- $c C"), True,
                 "giving the front argument leaves the rest")
    check_result(judge("A <- $c C", "Given"), True,
                 "and the two are the same type the other way round")
    check_result(judge("GivenTail", "A <- $b B"), True,
                 "giving the last one leaves the front")
    check_result(judge("GivenAll", "A"), True,
                 "giving every argument leaves the result")
    check_result(judge("GivenAllOther", "A"), True,
                 "in any order")
    check_result(judge("Named", "A"), True,
                 "the function may be written as a name")
    check_result(judge("(A <- $b B <- $c C) << $b B", "A <- $c C"), True,
                 "written inline, no definition needed")
    check_result(judge("$x ((A <- $b B) << $b B)", "$x A"), True,
                 "a member of a product is reduced too")
    check_result(judge("(A <- $b B) << $c C", "never"), "error",
                 "an argument the function does not have -> Err")
    check_result(judge("(A * B) << $b B", "never"), "error",
                 "an argument given to something that is no function -> Err")
    check_result(judge("Wide", "A"), True,
                 "the given argument has to fit the slot: Int <: Num holds")
    check_result(judge("Narrow", "A"), "error",
                 "giving Num to an Int slot is refused (Num <: Int does not hold)")


def run_any_cases():
    """Any 是所有类型的上界：T <: Any 恒真，反过来只有落在 Any（或等价于 Any 的
    形状，如 Any | int、Any * nil）上才真。"""
    module = custom_module("""
A := int
B := $x int * $y str
C := int | str
D := int <- $x str
E := (int, str)
Box[T] := $v T
""")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    for sub in ("int", "str", "bool", "float", "nil", "never", "A", "B", "C", "D", "E",
                "Box[int]", "7", '"x"', "true", "{code}", "Any | int", "Any * nil",
                "$x Any", "list[int]"):
        check_result(judge(sub, "Any"), True, f"{sub} <: Any")
    check_result(judge("never", "Any"), True, "bottom fits the top too")
    check_result(judge("Any", "Any"), True, "Any is a subtype of itself")
    check_result(judge("Any", "Any | int"), True, "a sum with an Any branch is Any")
    check_result(judge("Any", "Any * nil"), True, "a product whose rest is the unit is Any")
    for sup in ("int", "nil", "never", "A", "B", "C", "D", "Any <- $x int", "$x int",
                "list[int]"):
        check_result(judge("Any", sup), False, f"Any <: {sup}")
    check_result(judge("$x int", "$x Any"), True, "a member may widen to Any")
    check_result(judge("$x Any", "$x int"), False, "but not narrow")
    check_result(judge("Any", "$x Any"), True, "a bare Any fits the tagged field it is")


def run_never_head_cases():
    """Without terminators named, a never-headed chain is just an exponent."""
    module = custom_module("""
not[A] := never <- $not_operand A
P := int
""")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    check_result(judge("not[P]", "not[P]"), True,
                 "the core judges it as an exponent: reflexivity holds")
    check_result(judge("$a (never <- P)", "not[$a P]"), False,
                 "a product is not that exponent")
    check_result(judge("never <- $not_operand P", "never <- $not_operand P"), True,
                 "written as an exponent: reflexivity holds")
    check_result(judge("never <- $not_operand (P | str)", "never <- $not_operand P"), True,
                 "the argument is contravariant: a wider one is accepted")
    check_result(judge("never <- $not_operand P", "never <- $not_operand (P | str)"), False,
                 "a narrower one is not")


def run_code_block_cases():
    """A code block has no members: the unit is its resident."""
    module = custom_module("Guard := Object * $a {return 1}\n")

    def judge(sub, sup):
        return is_sub_type(entry_type(sub, module), entry_type(sup, module))

    check_result(judge("nil", "{return 1}"), True, "the unit resides in a code block")
    check_result(judge("{other code}", "{return 1}"), True,
                 "another code block is a unit too (the text is not reachable)")
    check_result(judge("7", "{return 1}"), False, "a literal does not")
    check_result(judge("{return 1}", "nil"), True, "a code block fits nil")
    check_result(judge("{return 1}", "never"), False, "and never fits nothing")


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
run_design_review()
run_py_side_cases()
run_env_cases()
run_env_get_cases()
run_env_get_scope_cases()
run_applied_generic_cases()
run_literal_alias_cases()
run_structural_generic_cases()
run_recursive_generic_cases()
run_unit_alias_cases()
run_inline_member_cases()
run_inline_cycle_guard_cases()
run_cross_module_inline_cases()
run_function_chain_cases()
run_config_cases()
run_never_head_cases()
run_code_block_cases()
run_canonical_chain_cases()
run_apply_cases()
run_any_cases()
run_suite_reflexivity()
print(f"\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)

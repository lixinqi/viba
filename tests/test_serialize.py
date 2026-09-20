"""viba.serialize 的验收：写出来、读得回来、是居民。

每个用例都做三件事：把材料写成源码；用同一个设计把源码当材料读回来，逐地址比
叶子；把写出来的体判成那个定义体的子类型。

    python3 tests/test_serialize.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_rule_reflect_api import _materials, _load, _definition_of

from viba import builder, serialize, viba_ast
from viba.type import AstNodeType, Err, Ok, custom_module
from viba.rule.is_compliant import is_compliant
from viba.viba_type_descriptor import (empty_pool, parse_viba_file,
                                        pool_add_file, pool_find_definition)
from viba.reflect import VibaData

PASS = FAIL = 0


def check(ok: bool, label: str):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def _entry_body(source: str):
    tree = viba_ast.parse(source)
    return [n for n in tree.body if getattr(n, "name", None) == "entry"][0].body


def _leaves(access, node):
    """{address: leaf}: every leaf reachable in this piece."""
    out = {}
    for reached in access._walk(node):
        given = access.leaf(reached)
        if isinstance(given, Ok):
            out[reached.path] = given.ok_value
    return out


def _round_trip(label, definition, access, node, design_source, design_name,
                resident=True, strict=True):
    written = serialize.serialize("entry", access, node)
    check(isinstance(written, Ok), f"{label}: writes ({written})")
    if not isinstance(written, Ok):
        return
    source = written.ok_value
    check(source.startswith("entry :="), f"{label}: the definition is named entry")
    check(source.endswith("\n"), f"{label}: one trailing newline")

    body = _entry_body(source)
    again = access.root(definition, VibaData(body))
    check(isinstance(again, Ok), f"{label}: the written body roots")
    if not isinstance(again, Ok):
        return
    before, after = _leaves(access, node), _leaves(access, again.ok_value)
    if strict:
        missing = {address: value for address, value in before.items()
                   if after.get(address, "<missing>") != value}
        check(not missing, f"{label}: every address reads the same leaf ({missing})")
    else:
        # 材料把和式那一支写全了，写出来只剩值：地址少一跳，叶子还是那些。
        check(sorted(map(repr, before.values())) == sorted(map(repr, after.values())),
              f"{label}: the same leaves under either spelling ({before} vs {after})")

    twice = serialize.serialize("entry", access, again.ok_value)
    check(isinstance(twice, Ok) and twice.ok_value == source,
          f"{label}: written again it is the same source ({twice})")

    canonical = viba_ast.unparse(viba_ast.parse(source)).rstrip("\n")
    check(canonical == source.rstrip("\n"),
          f"{label}: the source is already canonical viba ({canonical!r})")

    if not resident:
        return
    module = custom_module(f"{design_source}\n{source}\n")
    design = {d.name: d.body for d in viba_ast.parse(design_source).body}
    verdict = is_compliant(AstNodeType(body, module),
                           AstNodeType(design[design_name], module))
    check(isinstance(verdict, Ok) and verdict.ok_value is True,
          f"{label}: the written body is a resident of the definition ({verdict})")


def _design(pool_source: str, name: str):
    """A pool with this design, the definition, and its witness."""
    pool = empty_pool()
    parsed = parse_viba_file(pool, pool_source, "design.viba", "design")
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    return pool, _definition_of(pool, f"design.{name}")


def run_fixture_cases():
    m = _materials()
    demo_source = (Path(__file__).resolve().parent / "data"
                   / "rule_coding_style_check" / "demo.viba").read_text()
    _round_trip("demo", m.demo, m.demo_access, m.demo_node, demo_source, "DemoRule")
    _round_trip("node1", m.node1, m.node1_access, m.node1_node, """Node1 :=
  Object
  * $items list[int]
  * $seen set[str]
  * $table dict[str, int]
  * $maybe int
""", "Node1")


def run_empty_container_cases():
    source = """Box := Object * $xs list[int] * $ys set[str] * $zs dict[str, int]
"""
    pool, definition = _design(source, "Box")
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$xs", viba_ast.TypeApp("ListLiteral", [])),
        viba_ast.Tagged("$ys", viba_ast.TypeApp("SetLiteral", [])),
        viba_ast.Tagged("$zs", viba_ast.TypeApp("DictLiteral", [])),
    ])
    from viba.reflect import access
    node = access.root(definition, VibaData(body)).ok_value
    written = serialize.serialize("entry", access, node)
    check(isinstance(written, Ok), f"empty: writes ({written})")
    if isinstance(written, Ok):
        source_out = written.ok_value
        check("ListLiteral[]" in source_out and "SetLiteral[]" in source_out
              and "DictLiteral[]" in source_out, "empty: three empty literals")
        _round_trip("empty", definition, access, node, source, "Box")


def run_nil_slot_cases():
    source = """Maybe := Object * $a int * $b (int | nil)
"""
    pool, definition = _design(source, "Maybe")
    from viba.reflect import access
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$a", viba_ast.Constant(1)),
    ])
    node = access.root(definition, VibaData(body)).ok_value
    written = serialize.serialize("entry", access, node)
    check(isinstance(written, Ok), f"nil slot: writes ({written})")
    if isinstance(written, Ok):
        check("* $b nil" in written.ok_value, "nil slot: written as nil")
        _round_trip("nil slot", definition, access, node, source, "Maybe")


def run_set_order_cases():
    source = """Box := Object * $seen set[str]
"""
    pool, definition = _design(source, "Box")
    from viba.reflect import access
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$seen", viba_ast.TypeApp(
            "SetLiteral", [viba_ast.Constant("c"), viba_ast.Constant("a")])),
    ])
    node = access.root(definition, VibaData(body)).ok_value
    written = serialize.serialize("entry", access, node)
    check(isinstance(written, Ok), f"set order: writes ({written})")
    if isinstance(written, Ok):
        check('SetLiteral["c", "a"]' in written.ok_value,
              "set order: the material's order, not sorted")
        _round_trip("set order", definition, access, node, source, "Box")


def run_exponent_cases():
    """指数字段：按设计写 never <- $not_operand (...)，值从材料里拿。"""
    source = """Guard := Object * $no (never <- $not_operand Bad)
Bad := $kill int | $steal int
"""
    pool, definition = _design(source, "Guard")
    from viba.reflect import access
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$no", viba_ast.ExponentChain([
            viba_ast.Never(),
            viba_ast.Tagged("$not_operand", viba_ast.SumChain([
                viba_ast.Tagged("$kill", viba_ast.Constant(1)),
            ])),
        ])),
    ])
    node = access.root(definition, VibaData(body)).ok_value
    written = serialize.serialize("entry", access, node)
    check(isinstance(written, Ok), f"exponent: writes ({written})")
    if isinstance(written, Ok):
        out = written.ok_value
        check("<- $not_operand" in out and "never" in out,
              "exponent: written as never <- $not_operand (...)")
        # 这份材料只带一支，写出来也只有一支；判成居民与否是材料自己的事
        _round_trip("exponent", definition, access, node, source, "Guard",
                    resident=False)


def run_code_block_cases():
    """代码块没有格子取材料自己的文本，按约定写 nil。"""
    source = """Guard := Object * $code {return 1}
"""
    pool, definition = _design(source, "Guard")
    from viba.reflect import access
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$code", viba_ast.CodeBlock("return 1")),
    ])
    node = access.root(definition, VibaData(body)).ok_value
    written = serialize.serialize("entry", access, node)
    check(isinstance(written, Ok), f"code block: writes ({written})")
    if isinstance(written, Ok):
        check("* $code nil" in written.ok_value, "code block: written as nil")


def _product(*items):
    """Object * item * ... : the product chain a material is."""
    return viba_ast.ProductChain([viba_ast.TypeRef("Object"), *items])


def _tagged(tag, body):
    return viba_ast.Tagged(tag, body)


def _corner(label, source, name, body, expect=None, resident=True, strict=True):
    """同一个设计写一份材料：文本里有 expect，读得回来，还是居民。"""
    from viba.reflect import access
    pool, definition = _design(source, name)
    rooted = access.root(definition, VibaData(body))
    check(isinstance(rooted, Ok), f"{label}: the material roots ({rooted})")
    if not isinstance(rooted, Ok):
        return
    written = serialize.serialize("entry", access, rooted.ok_value)
    check(isinstance(written, Ok), f"{label}: writes ({written})")
    if not isinstance(written, Ok):
        return
    if expect is not None:
        check(expect in written.ok_value,
              f"{label}: {expect!r} is in\n{written.ok_value}")
    _round_trip(label, definition, access, rooted.ok_value, source, name,
                resident=resident, strict=strict)


def _gap(label, source, name, body, needle):
    """写不出来：Err，且说的是那件事。"""
    from viba.reflect import access
    pool, definition = _design(source, name)
    rooted = access.root(definition, VibaData(body))
    check(isinstance(rooted, Ok), f"{label}: the material roots ({rooted})")
    if not isinstance(rooted, Ok):
        return
    written = serialize.serialize("entry", access, rooted.ok_value)
    check(isinstance(written, Err) and needle in written.err_msg,
          f"{label}: an Err saying {needle!r} ({written})")


def run_shape_corner_cases():
    """形状的边角：深、宽、套、元组、别名、泛型、递归、无成员。"""
    depth_type, depth_body = "int", viba_ast.Constant(1)
    for _ in range(6):
        depth_type, depth_body = (f"list[{depth_type}]",
                                  viba_ast.TypeApp("ListLiteral", [depth_body]))
    _corner("depth 6", f"Box := Object * $a {depth_type}\n", "Box",
            _product(_tagged("$a", depth_body)), expect="ListLiteral[ListLiteral[")

    wide_source = ("Box := Object * "
                   + " * ".join(f"$f{i} int" for i in range(20)) + "\n")
    _corner("width 20", wide_source, "Box",
            _product(*[_tagged(f"$f{i}", viba_ast.Constant(i)) for i in range(20)]),
            expect="* $f19 19")

    _corner("list of products",
            "Box := Object * $a list[Inner]\nInner := Object * $x int\n", "Box",
            _product(_tagged("$a", viba_ast.TypeApp("ListLiteral", [
                _product(_tagged("$x", viba_ast.Constant(1))),
                _product(_tagged("$x", viba_ast.Constant(2)))]))),
            expect="ListLiteral[Object")
    _corner("nested product",
            "Outer := Object * $inner Inner\nInner := Object * $a int\n", "Outer",
            _product(_tagged("$inner", _product(_tagged("$a", viba_ast.Constant(1))))),
            expect="* $inner(Object")
    _corner("product member beside a unit",
            "Box := Object * $u Object * $a int\n", "Box",
            _product(_tagged("$u", viba_ast.Nil()), _tagged("$a", viba_ast.Constant(1))),
            expect="* $u nil")

    _corner("tuple of none", "T := Object * $t ()\n", "T",
            _product(_tagged("$t", viba_ast.Tuple([]))), expect="* $t ()")
    _corner("tuple of one", "T := Object * $t (int,)\n", "T",
            _product(_tagged("$t", viba_ast.Tuple([viba_ast.Constant(1)]))),
            expect="* $t (1,)")
    _corner("tuple nested", "T := Object * $t (int, (str, int))\n", "T",
            _product(_tagged("$t", viba_ast.Tuple([
                viba_ast.Constant(1),
                viba_ast.Tuple([viba_ast.Constant("x"), viba_ast.Constant(3)])]))),
            expect='* $t (1, ("x", 3))')

    _corner("aliases to the end", "A := B\nB := int\nBox := Object * $a A\n", "Box",
            _product(_tagged("$a", viba_ast.Constant(7))), expect="* $a 7")
    _corner("generic pair",
            "Pair[K, V] := Object * $fst K * $snd V\nBox := Object * $p Pair[int, str]\n",
            "Box",
            _product(_tagged("$p", _product(_tagged("$fst", viba_ast.Constant(1)),
                                            _tagged("$snd", viba_ast.Constant("x"))))),
            expect="* $fst 1")
    _corner("positional members beside tagged ones",
            "Box := Object * int * $a int * str\n", "Box",
            viba_ast.ProductChain([viba_ast.TypeRef("Object"), viba_ast.Constant(1),
                                   _tagged("$a", viba_ast.Constant(2)),
                                   viba_ast.Constant("three")]),
            expect='* 1\n  * $a 2\n  * "three"')

    _corner("zero-member product", "Only := Object\n", "Only", viba_ast.Nil(),
            expect="entry :=\n  nil\n")
    _corner("the head the material left out", "Box := Object * $a int\n", "Box",
            viba_ast.ProductChain([_tagged("$a", viba_ast.Constant(1))]),
            expect="entry :=\n  Object\n  * $a 1\n")
    _corner("a member the design has no address for", "Box := Object * $a int\n", "Box",
            _product(_tagged("$a", viba_ast.Constant(1)),
                     _tagged("$b", viba_ast.Constant(2))),
            expect="* $a 1\n")
    _corner("recursive product", "Loop := Object * $next (Loop | nil)\n", "Loop",
            _product(_tagged("$next", viba_ast.SumChain([
                _product(_tagged("$next", viba_ast.Nil()))]))),
            expect="* $next(Object")


def run_sum_corner_cases():
    """和式的边角：挑中的那一支、带标签的支、名字里的和、nil 支。"""
    _corner("sum: the int branch", "S := int | $a str\nBox := Object * $s S\n", "Box",
            _product(_tagged("$s", viba_ast.SumChain([viba_ast.Constant(3)]))),
            expect="* $s 3", strict=False)
    _corner("sum: the tagged branch", "S := int | $a str\nBox := Object * $s S\n", "Box",
            _product(_tagged("$s", viba_ast.SumChain([
                _tagged("$a", viba_ast.Constant("x"))]))),
            expect='* $s($a "x")')
    _corner("sum: reached through a name",
            "X := int | str\nS := X | $a bool\nBox := Object * $s S\n", "Box",
            _product(_tagged("$s", viba_ast.Constant("deep"))),
            expect='* $s "deep"')
    _corner("sum: the nil branch", "S := nil | int\nBox := Object * $s S\n", "Box",
            _product(_tagged("$s", viba_ast.Nil())), expect="* $s nil")
    _corner("sum: nil written last", "S := int | nil\nBox := Object * $s S\n", "Box",
            _product(_tagged("$s", viba_ast.Nil())), expect="* $s nil")


def run_leaf_corner_cases():
    """叶子的边角：转义、unicode、真假、小数、大整数、集合与字典的顺序。"""
    _corner("leaf: unicode string", "Box := Object * $s str\n", "Box",
            _product(_tagged("$s", viba_ast.Constant("中文🙂"))), expect='"中文🙂"')
    _corner("leaf: a string with a double quote", "Box := Object * $s str\n", "Box",
            _product(_tagged("$s", viba_ast.Constant('say "hi" now'))),
            expect="'say \"hi\" now'")
    _corner("leaf: a string with a newline and both quotes",
            "Box := Object * $s str\n", "Box",
            _product(_tagged("$s", viba_ast.Constant("a'b\"c\nd"))),
            expect="'''a'b\"c\nd'''")
    _corner("leaf: a string ending in a backslash", "Box := Object * $s str\n", "Box",
            _product(_tagged("$s", viba_ast.Constant("trail\\"))),
            expect="'''trail\\'''")
    _corner("leaf: false", "Box := Object * $b bool\n", "Box",
            _product(_tagged("$b", viba_ast.Constant(False))), expect="* $b false")
    _corner("leaf: a plain float", "Box := Object * $f float\n", "Box",
            _product(_tagged("$f", viba_ast.Constant(0.5))), expect="* $f 0.5")
    _corner("leaf: a big int", "Box := Object * $n int\n", "Box",
            _product(_tagged("$n", viba_ast.Constant(10 ** 30))),
            expect=f"* $n {10 ** 30}")

    _corner("set: the material's order, repeats and all",
            "Box := Object * $seen set[str]\n", "Box",
            _product(_tagged("$seen", viba_ast.TypeApp("SetLiteral", [
                viba_ast.Constant("b"), viba_ast.Constant("a"),
                viba_ast.Constant("b")]))),
            expect='SetLiteral["b", "a", "b"]')
    _corner("dict: odd keys, in the material's order",
            "Box := Object * $d dict[str, int]\n", "Box",
            _product(_tagged("$d", viba_ast.TypeApp("DictLiteral", [
                viba_ast.Tuple([viba_ast.Constant(""), viba_ast.Constant(1)]),
                viba_ast.Tuple([viba_ast.Constant('q"uote'), viba_ast.Constant(2)]),
                viba_ast.Tuple([viba_ast.Constant("中文"), viba_ast.Constant(3)])]))),
            expect="""("", 1), ('q"uote', 2), ("中文", 3)""")
    _corner("code block: nil where its text cannot be read",
            "Box := Object * $g list[{x}]\n", "Box",
            _product(_tagged("$g", viba_ast.TypeApp("ListLiteral", [
                viba_ast.CodeBlock("x")]))),
            expect="ListLiteral[nil]")


def run_exponent_corner_cases():
    """指数链的边角：多个实参、nil 头、非单位头、装在容器里。"""
    _corner("exponent: two arguments",
            "G := Object * $g (never <- $a int <- $b Bad)\nBad := $k int\n", "G",
            _product(_tagged("$g", viba_ast.ExponentChain([
                viba_ast.Never(), _tagged("$a", viba_ast.Constant(1)),
                _tagged("$b", viba_ast.SumChain([_tagged("$k", viba_ast.Constant(2))]))]))),
            expect="<- $a 1\n    <- $b($k 2)", resident=False)
    _corner("exponent: nil head", "G := Object * $g (nil <- $a int)\n", "G",
            _product(_tagged("$g", viba_ast.ExponentChain([
                viba_ast.Nil(), _tagged("$a", viba_ast.Constant(1))]))),
            expect="* $g(nil", resident=False)
    _corner("exponent: a value in the result", "G := Object * $g (int <- $a int)\n", "G",
            _product(_tagged("$g", viba_ast.ExponentChain([
                viba_ast.Constant(5), _tagged("$a", viba_ast.Constant(1))]))),
            expect="* $g(5", resident=False)
    _corner("exponent: inside a container",
            "G := Object * $g list[never <- $a int]\n", "G",
            _product(_tagged("$g", viba_ast.TypeApp("ListLiteral", [
                viba_ast.ExponentChain([viba_ast.Never(),
                                        _tagged("$a", viba_ast.Constant(1))])]))),
            expect="ListLiteral[never", resident=False)


def run_more_gap_cases():
    """更多写不出来的边角：数字、字符串、never、名字当值。"""
    _gap("gap: never inside a container", "Box := Object * $a list[never]\n", "Box",
         _product(_tagged("$a", viba_ast.TypeApp("ListLiteral", [viba_ast.Never()]))),
         "nothing resides in never")
    _gap("gap: a negative int", "Box := Object * $i int\n", "Box",
         _product(_tagged("$i", viba_ast.Constant(-1))), "no literal for this number")
    _gap("gap: a float with an exponent", "Box := Object * $f float\n", "Box",
         _product(_tagged("$f", viba_ast.Constant(1e30))), "no literal for this number")
    _gap("gap: a float that is not a number", "Box := Object * $f float\n", "Box",
         _product(_tagged("$f", viba_ast.Constant(float("nan")))),
         "no literal for this number")
    _gap("gap: a string no literal holds", "Box := Object * $s str\n", "Box",
         _product(_tagged("$s", viba_ast.Constant("a'''b\nd"))),
         "no viba string literal holds this text")
    _gap("gap: a name where a value goes", "Only := Object\n", "Only",
         viba_ast.TypeRef("Object"), "cannot write this piece out")


def run_never_and_key_cases():
    """never 没有居民；键不是 str 的 dict 没有写法。"""
    from viba.reflect import access
    pool, definition = _design("Guard := Object * $a never\n", "Guard")
    node = access.root(definition, VibaData(viba_ast.ProductChain([
        viba_ast.TypeRef("Object"), viba_ast.Tagged("$a", viba_ast.Never())]))).ok_value
    written = serialize.serialize("entry", access, node)
    check(isinstance(written, Err), f"never slot: an Err ({written})")

    pool2, definition2 = _design("Guard := Object * $t dict[int, str]\n", "Guard")
    node2 = access.root(definition2, VibaData(viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$t", viba_ast.TypeApp("DictLiteral", [
            viba_ast.Tuple([viba_ast.Constant(7), viba_ast.Constant("x")])]))]))).ok_value
    written2 = serialize.serialize("entry", access, node2)
    check(isinstance(written2, Err), f"int-keyed dict: an Err ({written2})")


def run_gap_cases():
    """写不出来给 Err，不硬写。"""
    source = """Box := Object * $a int * $b int
"""
    pool, definition = _design(source, "Box")
    from viba.reflect import access
    sparse = viba_ast.ProductChain([viba_ast.TypeRef("Object"),
                                    viba_ast.Tagged("$a", viba_ast.Constant(1))])
    node = access.root(definition, VibaData(sparse)).ok_value
    written = serialize.serialize("entry", access, node)
    check(isinstance(written, Err) and "no value here" in written.err_msg,
          f"gap: a tagged slot with no value and no nil is an Err ({written})")

    # 材料本身是空的：设计里的那个地址上什么都没有，就写成 gap，不硬编
    pool2, definition2 = _design("not[A] := never <- $not_operand A\n", "not")
    node2 = access.root(definition2, VibaData(viba_ast.Never())).ok_value
    written2 = serialize.serialize("entry", access, node2)
    check(isinstance(written2, Err), f"gap: an exponent is an Err ({written2})")


# ---------------------------------------------------------------------------
# 成批的边角：叶子逐个过、形状两两套、和式每一支、标签、链长、深浅、压力。
# ---------------------------------------------------------------------------

# 字符串这一列挑的是"引号、转义、空、换行、看起来像语言关键字"这些点：
# 每一种都得挑出一个装得下它的字面量，读回来还得一个字不差。
_STRINGS = [
    "", "a", "0", " ", "\n", "a\nb", "\r\n", "line\n", "\t", "\\", "a\\b",
    '"', '""', "'", "''", 'a"b', "a'b", 'a"b\'c', "中文", "🙂", "a b",
    "-1", "0.5", "nil", "never", "true", "Object", "$a", "$a.b", "a:b",
    "(A, B)", "}", "{", "#", "|", "*", "<-", ":=", "a'''b", "'''",
    "ab'\ncd", "ab'\n", "a''\nb", "x" * 200, "中\n文",
]
_INTS = [0, 1, 7, 42, 10 ** 9, 2 ** 63, 10 ** 40, 123456789012345678901234567890]
_FLOATS = [0.0, 0.5, 1.0, 2.0, 0.1, 0.0001, 3.141592653589793, 100.0, 1e15,
           123456.789, 1e-3]
# 语言里没有写法的：没有负号，也没有指数；nan / inf 更不是数。
_UNSPELLABLE_NUMBERS = [-1, -(10 ** 20), -0.5, -0.0, 1e30, 1e16, 1e-5,
                        float("inf"), float("-inf"), float("nan")]
# 三种引号都占上了的文本（跨行的还带上 '''），没有一个字面量装得下。
_UNSPELLABLE_STRINGS = ["a'''b\nc", "'''\n'''", "a'''b\"c'd"]


def run_leaf_matrix():
    """叶子逐个过：写得出、读回来一模一样、还是那个类型的居民。"""
    for index, text in enumerate(_STRINGS):
        _corner(f"leaf str[{index}]", "Box := Object * $s str\n", "Box",
                _product(_tagged("$s", viba_ast.Constant(text))))
    for value in _INTS:
        _corner(f"leaf int {value}", "Box := Object * $n int\n", "Box",
                _product(_tagged("$n", viba_ast.Constant(value))),
                expect=f"* $n {value}")
    for value in _FLOATS:
        _corner(f"leaf float {value}", "Box := Object * $f float\n", "Box",
                _product(_tagged("$f", viba_ast.Constant(value))),
                expect=f"* $f {value}")
    for value in (True, False):
        _corner(f"leaf bool {value}", "Box := Object * $b bool\n", "Box",
                _product(_tagged("$b", viba_ast.Constant(value))),
                expect=f"* $b {str(value).lower()}")


def run_number_and_string_gaps():
    """写不出来的数字与文本：Err，不是"写成别的"。"""
    for value in _UNSPELLABLE_NUMBERS:
        _gap(f"gap number {value}", "Box := Object * $f float\n", "Box",
             _product(_tagged("$f", viba_ast.Constant(value))),
             "no literal for this number")
    for index, text in enumerate(_UNSPELLABLE_STRINGS):
        _gap(f"gap string[{index}]", "Box := Object * $s str\n", "Box",
             _product(_tagged("$s", viba_ast.Constant(text))),
             "no viba string literal holds this text")


def run_tag_matrix():
    """标签名：普通、带数字、下划线、点分路径。"""
    for tag in ("$a", "$ab", "$a_b", "$a1", "$a_1_b", "$agent_name2", "$A",
                "$aB_c1", "$x.y", "$x.y.z", "$meta.id.hash", "$a.b.c.d.e"):
        _corner(f"tag {tag}", f"Box := Object * {tag} int\n", "Box",
                _product(_tagged(tag, viba_ast.Constant(1))),
                expect=f"* {tag} 1")


def run_depth_and_width_ladders():
    """深浅两级台阶：1..7 层容器，1..8 个成员。"""
    for depth in range(1, 8):
        type_text, body = "int", viba_ast.Constant(1)
        for _ in range(depth):
            type_text = f"list[{type_text}]"
            body = viba_ast.TypeApp("ListLiteral", [body])
        _corner(f"depth {depth}", f"Box := Object * $a {type_text}\n", "Box",
                _product(_tagged("$a", body)),
                expect="ListLiteral[" * min(depth, 2))
    for width in range(1, 9):
        source = ("Box := Object * "
                  + " * ".join(f"$f{i} int" for i in range(width)) + "\n")
        _corner(f"width {width}", source, "Box",
                _product(*[_tagged(f"$f{i}", viba_ast.Constant(i))
                           for i in range(width)]),
                expect=f"* $f{width - 1} {width - 1}")


def run_absent_member_positions():
    """nil 收得下的成员缺在头、中、尾三处，都写 nil。"""
    for position in range(3):
        members = ["$a int", "$b int", "$c int"]
        members[position] = members[position].split()[0] + " (int | nil)"
        present = [i for i in range(3) if i != position]
        _corner(f"absent member at {position}",
                "Box := Object * " + " * ".join(members) + "\n", "Box",
                _product(*[_tagged(f"${'abc'[i]}", viba_ast.Constant(i))
                           for i in present]),
                expect=f"* ${'abc'[position]} nil")


def run_alias_ladders():
    """别名链一路走到头：一层到五层。"""
    for length in range(1, 6):
        lines = ["A0 := int"]
        for step in range(1, length + 1):
            lines.append(f"A{step} := A{step - 1}")
        source = "\n".join(lines) + f"\nBox := Object * $a A{length}\n"
        _corner(f"alias chain {length}", source, "Box",
                _product(_tagged("$a", viba_ast.Constant(7))), expect="* $a 7")


def run_sum_ladders():
    """和式的每一支都挑一遍：支数 2..6，挑中的位置 0..n-1。"""
    branches = [("int", viba_ast.Constant(3)),
                ("$b str", _tagged("$b", viba_ast.Constant("x"))),
                ("$c bool", _tagged("$c", viba_ast.Constant(True))),
                ("$d float", _tagged("$d", viba_ast.Constant(0.5))),
                ("$e Object", _tagged("$e", viba_ast.Nil())),
                ("$f list[int]", _tagged("$f", viba_ast.TypeApp(
                    "ListLiteral", [viba_ast.Constant(1)])))]
    for count in range(2, len(branches) + 1):
        source = ("S := " + " | ".join(text for text, _ in branches[:count])
                  + "\nBox := Object * $s S\n")
        for chosen in range(count):
            _corner(f"sum {count} branches, branch {chosen}", source, "Box",
                    _product(_tagged("$s", viba_ast.SumChain(
                        [branches[chosen][1]]))),
                    strict=chosen != 0)
    # 第一支就带标签：支数 1..4
    tagged = [("$a int", _tagged("$a", viba_ast.Constant(1))),
              ("$b str", _tagged("$b", viba_ast.Constant("x"))),
              ("$c bool", _tagged("$c", viba_ast.Constant(True))),
              ("$d float", _tagged("$d", viba_ast.Constant(0.5)))]
    for count in range(1, len(tagged) + 1):
        source = ("S := " + " | ".join(text for text, _ in tagged[:count])
                  + "\nBox := Object * $s S\n")
        for chosen in range(count):
            _corner(f"tagged sum {count}, branch {chosen}", source, "Box",
                    _product(_tagged("$s", viba_ast.SumChain(
                        [tagged[chosen][1]]))))


_CONTEXT = ("Inner := Object * $x int\n"
            "A := int\n"
            "S2 := $k int | $j str\n"
            "W[V] := Object * $v V\n")
# 里面那一层：单位、字面量、产品、元组、别名、和式、容器、代码块、泛型应用。
_INNER_SHAPES = [
    ("unit", "Object", viba_ast.Nil()),
    ("literal", "int", viba_ast.Constant(1)),
    ("product", "Inner", _product(_tagged("$x", viba_ast.Constant(2)))),
    ("tuple", "(int, str)", viba_ast.Tuple([viba_ast.Constant(3),
                                            viba_ast.Constant("t")])),
    ("alias", "A", viba_ast.Constant(4)),
    ("sum", "S2", viba_ast.SumChain([_tagged("$k", viba_ast.Constant(5))])),
    ("container", "list[int]", viba_ast.TypeApp("ListLiteral",
                                                [viba_ast.Constant(6)])),
    ("code block", "{x}", viba_ast.CodeBlock("x")),
    ("generic", "W[int]", _product(_tagged("$v", viba_ast.Constant(7)))),
]
# 外面那一层：成员位、容器元素、元组位、和式的两支。
_OUTER_SHAPES = [
    ("member", lambda t: f"$m {t}", lambda m: _tagged("$m", m)),
    ("list", lambda t: f"$m list[{t}]",
     lambda m: _tagged("$m", viba_ast.TypeApp("ListLiteral", [m]))),
    ("set", lambda t: f"$m set[{t}]",
     lambda m: _tagged("$m", viba_ast.TypeApp("SetLiteral", [m]))),
    ("dict", lambda t: f"$m dict[str, {t}]",
     lambda m: _tagged("$m", viba_ast.TypeApp("DictLiteral", [
         viba_ast.Tuple([viba_ast.Constant("key"), m])]))),
    ("tuple", lambda t: f"$m ({t}, int)",
     lambda m: _tagged("$m", viba_ast.Tuple([m, viba_ast.Constant(9)]))),
    ("sum first", lambda t: f"$m ($w {t} | int)",
     lambda m: _tagged("$m", viba_ast.SumChain([_tagged("$w", m)]))),
    ("sum second", lambda t: f"$m (int | $w {t})",
     lambda m: _tagged("$m", viba_ast.SumChain([_tagged("$w", m)]))),
]


def run_shape_matrix():
    """形状两两套：9 种里层 × 7 种外层 = 63 个格子。"""
    for outer_name, outer_type, outer_body in _OUTER_SHAPES:
        for inner_name, inner_type, inner_body in _INNER_SHAPES:
            _corner(f"{outer_name} of {inner_name}",
                    _CONTEXT + "Box := Object * " + outer_type(inner_type) + "\n",
                    "Box", _product(outer_body(inner_body)))


def run_container_fills():
    """容器填满：12 个元素、40 个键、30 个集合成员，顺序照材料。"""
    twelve = [viba_ast.Constant(i) for i in range(12)]
    _corner("list of 12", "Box := Object * $a list[int]\n", "Box",
            _product(_tagged("$a", viba_ast.TypeApp("ListLiteral", twelve))),
            expect="ListLiteral[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]")
    _corner("set of 12, back to front",
            "Box := Object * $a set[int]\n", "Box",
            _product(_tagged("$a", viba_ast.TypeApp("SetLiteral",
                                                    list(reversed(twelve))))),
            expect="SetLiteral[11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]")
    keys = [f"k{i}" for i in range(40)]
    pairs = [viba_ast.Tuple([viba_ast.Constant(key), viba_ast.Constant(i)])
             for i, key in enumerate(keys)]
    _corner("dict of 40", "Box := Object * $a dict[str, int]\n", "Box",
            _product(_tagged("$a", viba_ast.TypeApp("DictLiteral", pairs))),
            expect='("k0", 0), ("k1", 1)')
    _corner("string of 4000", "Box := Object * $a str\n", "Box",
            _product(_tagged("$a", viba_ast.Constant("中" * 2000))))
    _corner("a product of 40 members",
            "Box := Object * " + " * ".join(f"$f{i} int" for i in range(40)) + "\n",
            "Box",
            _product(*[_tagged(f"$f{i}", viba_ast.Constant(i))
                       for i in range(40)]),
            expect="* $f39 39")


def run_never_positions():
    """never 出现在哪里都是 Err：容器里、元组里、别名背后。"""
    _gap("gap never in list", "Box := Object * $a list[never]\n", "Box",
         _product(_tagged("$a", viba_ast.TypeApp("ListLiteral",
                                                 [viba_ast.Never()]))),
         "nothing resides in never")
    _gap("gap never in set", "Box := Object * $a set[never]\n", "Box",
         _product(_tagged("$a", viba_ast.TypeApp("SetLiteral",
                                                 [viba_ast.Never()]))),
         "nothing resides in never")
    _gap("gap never in a dict value", "Box := Object * $a dict[str, never]\n", "Box",
         _product(_tagged("$a", viba_ast.TypeApp("DictLiteral", [
             viba_ast.Tuple([viba_ast.Constant("k"), viba_ast.Never()])]))),
         "nothing resides in never")
    _gap("gap never in a tuple", "Box := Object * $a (int, never)\n", "Box",
         _product(_tagged("$a", viba_ast.Tuple([viba_ast.Constant(1),
                                                viba_ast.Never()]))),
         "nothing resides in never")
    _gap("gap never behind an alias", "N := never\nBox := Object * $a list[N]\n", "Box",
         _product(_tagged("$a", viba_ast.TypeApp("ListLiteral",
                                                 [viba_ast.Never()]))),
         "nothing resides in never")
    _gap("gap never in a sum branch", "Box := Object * $a ($w never | int)\n", "Box",
         _product(_tagged("$a", viba_ast.SumChain([_tagged("$w", viba_ast.Never())]))),
         "nothing resides in never")


def run_dict_key_aliases():
    """键类型是 str 的别名或泛型：还是 str，照样写得出来。"""
    for label, source in (("an alias of str", "S := str\nBox := Object * $d dict[S, int]\n"),
                          ("a generic landing on str",
                           "G[V] := str\nBox := Object * $d dict[G[int], int]\n"),
                          ("a chain of aliases",
                           "S := str\nT := S\nBox := Object * $d dict[T, int]\n")):
        pool = _pool(("m.viba", "m", source))
        material = _product(_tagged("$d", viba_ast.TypeApp("DictLiteral", [
            viba_ast.Tuple([viba_ast.Constant("k"), viba_ast.Constant(1)])])))
        _corner_in(f"dict keyed by {label}", pool, "m.Box", material,
                   expect='("k", 1)',
                   resident_source="Box := Object * $d dict[str, int]\n")


def run_dict_key_gaps():
    """键不是 str 的 dict：int / float / bool / 容器键都没有写法。"""
    for key_type in ("int", "float", "bool", "list[int]"):
        _gap(f"gap dict keyed by {key_type}",
             f"Box := Object * $a dict[{key_type}, str]\n", "Box",
             _product(_tagged("$a", viba_ast.TypeApp("DictLiteral", [
                 viba_ast.Tuple([viba_ast.Constant("k"), viba_ast.Constant("v")])]))),
             "the protocol hands dict keys over as strings")


def run_positional_gaps():
    """材料缺了成员：头、中、尾，都不硬编。"""
    for position in range(3):
        members = ["$a int", "$b int", "$c int"]
        present = [i for i in range(3) if i != position]
        _gap(f"gap absent member at {position}",
             "Box := Object * " + " * ".join(members) + "\n", "Box",
             _product(*[_tagged(f"${'abc'[i]}", viba_ast.Constant(i))
                        for i in present]),
             f"no value here: ${'abc'[position]}")
    _gap("gap a wrong tag", "Box := Object * $a int\n", "Box",
         _product(_tagged("$z", viba_ast.Constant(9))), "no value here: $a")
    _gap("gap an empty product material", "Box := Object * $a int\n", "Box",
         viba_ast.ProductChain([]), "no value here: $a")
    _gap("gap never as the whole material", "Box := Object * $a int\n", "Box",
         viba_ast.Never(), "no value here: $a")
    _gap("gap a name where a container element goes",
         "Box := Object * $a list[int]\n", "Box",
         _product(_tagged("$a", viba_ast.TypeApp("ListLiteral",
                                                 [viba_ast.TypeRef("int")]))),
         "cannot write this piece out")


def run_unit_member_shapes():
    """单位成员：带标签、不带标签、在产品头后面。

    不带标签的单位元不是成员：不占位置，也不写出来（带标签的照写，只是没有值）。
    """
    _corner("a unit member with no tag", "Box := Object * Object * $a int\n", "Box",
            _product(viba_ast.Nil(), _tagged("$a", viba_ast.Constant(1))),
            expect="entry :=\n  Object\n  * $a 1\n")
    _corner("two unit members, one tagged",
            "Box := Object * Object * $u Object * $a int\n", "Box",
            _product(viba_ast.Nil(), _tagged("$u", viba_ast.Nil()),
                     _tagged("$a", viba_ast.Constant(1))),
            expect="* $u nil")
    _corner("a unit member behind an alias",
            "U := Object\nBox := Object * $u U\n", "Box",
            _product(_tagged("$u", viba_ast.Nil())), expect="* $u nil")


def run_code_block_positions():
    """代码块的每一处：成员位、容器元素、元组位、和式支、产品头。"""
    block = viba_ast.CodeBlock("x + 1")
    _corner("code block as a member", "Box := Object * $c {x}\n", "Box",
            _product(_tagged("$c", block)), expect="* $c nil")
    _corner("code block in a set", "Box := Object * $c set[{x}]\n", "Box",
            _product(_tagged("$c", viba_ast.TypeApp("SetLiteral", [block]))),
            expect="SetLiteral[nil]")
    _corner("code block in a dict value", "Box := Object * $c dict[str, {x}]\n", "Box",
            _product(_tagged("$c", viba_ast.TypeApp("DictLiteral", [
                viba_ast.Tuple([viba_ast.Constant("k"), block])]))),
            expect='("k", nil)')
    _corner("code block in a tuple", "Box := Object * $c ({x}, int)\n", "Box",
            _product(_tagged("$c", viba_ast.Tuple([block, viba_ast.Constant(1)]))),
            expect="(nil, 1)")
    _corner("code block as a sum branch", "Box := Object * $c ({x} | int)\n", "Box",
            _product(_tagged("$c", viba_ast.SumChain([block]))))
    _corner("code block behind an alias", "K := {x}\nBox := Object * $c list[K]\n",
            "Box",
            _product(_tagged("$c", viba_ast.TypeApp("ListLiteral", [block]))),
            expect="ListLiteral[nil]")


def run_exponent_batteries():
    """指数链成批：结果位、实参位、嵌在容器与元组里、参数带标签。"""
    chain = lambda *elements: viba_ast.ExponentChain(list(elements))
    _corner("exponent: three arguments",
            "G := Object * $g (never <- $a int <- $b int <- $c int)\n", "G",
            _product(_tagged("$g", chain(viba_ast.Never(),
                                         _tagged("$a", viba_ast.Constant(1)),
                                         _tagged("$b", viba_ast.Constant(2)),
                                         _tagged("$c", viba_ast.Constant(3))))),
            expect="<- $a 1\n    <- $b 2\n    <- $c 3", resident=False)
    _corner("exponent: an untagged argument",
            "G := Object * $g (never <- int)\n", "G",
            _product(_tagged("$g", chain(viba_ast.Never(), viba_ast.Constant(4)))),
            expect="<- 4", resident=False)
    _corner("exponent: nested behind a tagged argument",
            "G := Object * $g (never <- $b (never <- $a int))\n", "G",
            _product(_tagged("$g", chain(viba_ast.Never(), _tagged("$b", chain(
                viba_ast.Never(), _tagged("$a", viba_ast.Constant(5))))))),
            expect="<- $b(never", resident=False)
    _corner("exponent: inside a tuple",
            "G := Object * $g ((never <- $a int), int)\n", "G",
            _product(_tagged("$g", viba_ast.Tuple([
                chain(viba_ast.Never(), _tagged("$a", viba_ast.Constant(6))),
                viba_ast.Constant(7)]))),
            expect="<- $a 6", resident=False)
    _corner("exponent: inside a dict value",
            "G := Object * $g dict[str, (never <- $a int)]\n", "G",
            _product(_tagged("$g", viba_ast.TypeApp("DictLiteral", [
                viba_ast.Tuple([viba_ast.Constant("k"),
                                chain(viba_ast.Never(),
                                      _tagged("$a", viba_ast.Constant(8)))])]))),
            expect="<- $a 8", resident=False)
    _corner("exponent: behind an alias",
            "X := never <- $a int\nG := Object * $g X\n", "G",
            _product(_tagged("$g", chain(viba_ast.Never(),
                                         _tagged("$a", viba_ast.Constant(9))))),
            expect="* $g(never", resident=False)


def run_name_alias_shapes():
    """名字背后的形状：单位、nil、以及各种别名当成员类型。"""
    _corner("a unit behind an alias as the head",
            "H := Object\nBox := H * $a int\n", "Box",
            _product(_tagged("$a", viba_ast.Constant(1))),
            expect="entry :=\n  Object\n  * $a 1\n")
    _corner("a unit behind an alias as a member",
            "U := Object\nBox := Object * $u U * $a int\n", "Box",
            _product(_tagged("$u", viba_ast.Nil()),
                     _tagged("$a", viba_ast.Constant(1))),
            expect="* $u nil")
    _corner("a unit behind two names", "U := Object\nV := U\nBox := Object * $u V\n",
            "Box", _product(_tagged("$u", viba_ast.Nil())), expect="* $u nil")
    _corner("a generic landing on a unit",
            "U[V] := Object\nBox := Object * $u U[int]\n", "Box",
            _product(_tagged("$u", viba_ast.Nil())), expect="* $u nil")
    _corner("nil behind a name", "Z := nil\nBox := Object * $z Z\n", "Box",
            _product(_tagged("$z", viba_ast.Nil())), expect="* $z nil")
    _corner("a never head behind a name",
            "N := never\nG := Object * $g (N <- $a int)\n", "G",
            _product(_tagged("$g", viba_ast.ExponentChain([
                viba_ast.Never(), _tagged("$a", viba_ast.Constant(1))]))),
            expect="* $g(never", resident=False)

    _corner("an alias to a product", "P := Object * $x int\nBox := Object * $p P\n", "Box",
            _product(_tagged("$p", _product(_tagged("$x", viba_ast.Constant(1))))),
            expect="* $p(Object")
    _corner("an alias to a tuple", "T := (int, str)\nBox := Object * $t T\n", "Box",
            _product(_tagged("$t", viba_ast.Tuple([viba_ast.Constant(1),
                                                   viba_ast.Constant("s")]))),
            expect='* $t (1, "s")')
    _corner("an alias to a sum", "S := $k int | $j str\nBox := Object * $s S\n", "Box",
            _product(_tagged("$s", viba_ast.SumChain([
                _tagged("$k", viba_ast.Constant(1))]))),
            expect="* $s($k 1)")
    _corner("an alias to a list", "L := list[int]\nBox := Object * $l L\n", "Box",
            _product(_tagged("$l", viba_ast.TypeApp("ListLiteral",
                                                     [viba_ast.Constant(1)]))),
            expect="* $l ListLiteral[1]")
    _corner("a generic landing on a list",
            "G[V] := list[V]\nBox := Object * $g G[int]\n", "Box",
            _product(_tagged("$g", viba_ast.TypeApp("ListLiteral",
                                                     [viba_ast.Constant(1)]))),
            expect="* $g ListLiteral[1]")
    _corner("an alias to a code block", "K := {x}\nBox := Object * $k K\n", "Box",
            _product(_tagged("$k", viba_ast.CodeBlock("x"))), expect="* $k nil")


def run_name_gaps():
    """名字背后的 never：材料里放什么都是 Err。"""
    for label, material in (("a value", viba_ast.Constant(1)),
                            ("never", viba_ast.Never())):
        _gap(f"gap never behind a name, material {label}",
             "N := never\nBox := Object * $n N\n", "Box",
             _product(_tagged("$n", material)), "nothing resides in never")
    _gap("gap never behind two names", "M := never\nN := M\nBox := Object * $n N\n", "Box",
         _product(_tagged("$n", viba_ast.Constant(1))), "nothing resides in never")
    _gap("gap never behind a name in a container",
         "N := never\nBox := Object * $n list[N]\n", "Box",
         _product(_tagged("$n", viba_ast.TypeApp("ListLiteral",
                                                  [viba_ast.Constant(1)]))),
         "nothing resides in never")


class _HostileString(str):
    """一个把 __format__ 改掉的 str：写它等于让它往源码里塞东西。"""

    def __format__(self, spec):
        return 'x" * $b 1 * "y'


def run_subclass_leaf_cases():
    """叶子就是那四个内建类型本身，子类不是：写不出 Err，不会被带出去。"""
    _gap("gap a str subclass that formats elsewhere", "Box := Object * $s str\n",
         "Box", _product(_tagged("$s", viba_ast.Constant(_HostileString("plain")))),
         "no literal for")
    _gap("gap a plain str subclass", "Box := Object * $s str\n", "Box",
         _product(_tagged("$s", viba_ast.Constant(type("S", (str,), {})("plain")))),
         "no literal for")


def run_more_gap_corners():
    """剩下的边角：和式的 nil 支、省略号、字面量容器当设计类型。"""
    _gap("gap a nil branch spelled as an element",
         "S := nil | $a int\nBox := Object * $s S\n", "Box",
         _product(_tagged("$s", viba_ast.SumChain([viba_ast.Nil()]))),
         "no branch of this sum carries a value")
    _gap("gap an ellipsis material", "Box := Object * $a int\n", "Box",
         _product(_tagged("$a", viba_ast.Ellipsis())), "no value here: $a")
    _gap("gap an ellipsis design", "Box := Object * $a ...\n", "Box",
         _product(_tagged("$a", viba_ast.Ellipsis())), "no value here: $a")
    _gap("gap a literal container as the design type",
         "Box := Object * $a ListLiteral[int]\n", "Box",
         _product(_tagged("$a", viba_ast.TypeApp("ListLiteral",
                                                  [viba_ast.Constant(1)]))),
         "cannot write this piece out")



# ---------------------------------------------------------------------------
# 再一轮：跨模块、环、材料本身的形状、名字参数、和式的容器、指数字段实参、压力。
# ---------------------------------------------------------------------------


def _pool(*files):
    """把这几份文件编进一个池子：(文件名, 模块名, 源码)。"""
    pool = empty_pool()
    for file_name, module_name, source in files:
        parsed = parse_viba_file(pool, source, file_name, module_name)
        check(isinstance(parsed, Ok), f"pool: {file_name} parses ({parsed})")
        if not isinstance(parsed, Ok):
            return None
        pool = pool_add_file(pool, parsed.ok_value).ok_value
    return pool


def _corner_in(label, pool, full_name, body, expect=None, resident=True,
               strict=True, resident_source=None, resident_name=None):
    """同一个池子里的定义写一份材料；居民那一判用本地拼法的等价设计来问。"""
    from viba.reflect import access
    found = pool_find_definition(pool, full_name)
    check(isinstance(found, Ok), f"{label}: the definition is in the pool ({found})")
    if not isinstance(found, Ok):
        return
    rooted = access.root(found.ok_value, VibaData(body))
    check(isinstance(rooted, Ok), f"{label}: the material roots ({rooted})")
    if not isinstance(rooted, Ok):
        return
    written = serialize.serialize("entry", access, rooted.ok_value)
    check(isinstance(written, Ok), f"{label}: writes ({written})")
    if not isinstance(written, Ok):
        return
    if expect is not None:
        check(expect in written.ok_value,
              f"{label}: {expect!r} is in\n{written.ok_value}")
    _round_trip(label, found.ok_value, access, rooted.ok_value,
                resident_source if resident_source is not None else "",
                resident_name or full_name.split(".")[-1],
                resident=resident if resident_source is not None else False,
                strict=strict)


def _gap_in(label, pool, full_name, body, needle):
    from viba.reflect import access
    found = pool_find_definition(pool, full_name)
    check(isinstance(found, Ok), f"{label}: the definition is in the pool ({found})")
    if not isinstance(found, Ok):
        return
    rooted = access.root(found.ok_value, VibaData(body))
    check(isinstance(rooted, Ok), f"{label}: the material roots ({rooted})")
    if not isinstance(rooted, Ok):
        return
    written = serialize.serialize("entry", access, rooted.ok_value)
    check(isinstance(written, Err) and needle in written.err_msg,
          f"{label}: an Err saying {needle!r} ({written})")


_DEFS = ("defs.viba", "defs",
         "U := Object\nP := Object * $x int\nW[V] := Object * $v V\nN := never\n")


def _main(body):
    return ("main.viba", "main", "import defs as d\n" + body)


def run_cross_module_cases():
    """跨模块：import 前缀的名字也是名字，单位/积/泛型/never 一样展开。"""
    def pool_of(body):
        return _pool(_DEFS, _main(body))

    _corner_in("a unit from another module as a member",
               pool_of("Box := Object * $u d.U\n"), "main.Box",
               _product(_tagged("$u", viba_ast.Nil())), expect="* $u nil",
               resident_source="U := Object\nBox := Object * $u U\n")
    _corner_in("a unit from another module as the head",
               pool_of("Box := d.U * $a int\n"), "main.Box",
               _product(_tagged("$a", viba_ast.Constant(1))),
               expect="entry :=\n  Object\n  * $a 1\n",
               resident_source="U := Object\nBox := U * $a int\n")
    _corner_in("a product from another module",
               pool_of("Box := Object * $p d.P\n"), "main.Box",
               _product(_tagged("$p", _product(_tagged("$x", viba_ast.Constant(1))))),
               expect="* $p(Object",
               resident_source="P := Object * $x int\nBox := Object * $p P\n")
    _corner_in("a generic from another module",
               pool_of("Box := Object * $w d.W[int]\n"), "main.Box",
               _product(_tagged("$w", _product(_tagged("$v", viba_ast.Constant(1))))),
               expect="* $w(Object",
               resident_source="W[V] := Object * $v V\nBox := Object * $w W[int]\n")
    _corner_in("a never from another module",
               pool_of("Box := Object * $n (d.N | int)\n"), "main.Box",
               _product(_tagged("$n", viba_ast.SumChain([viba_ast.Constant(1)]))),
               expect="* $n 1",
               resident_source="N := never\nBox := Object * $n (N | int)\n")
    _corner_in("a module path used without an alias",
               _pool(("pkg/util.viba", "pkg.util", "T := str\n"),
                     ("main.viba", "main",
                      "import pkg.util\nBox := Object * $t util.T\n")),
               "main.Box",
               _product(_tagged("$t", viba_ast.Constant("x"))), expect='* $t "x"',
               resident_source="T := str\nBox := Object * $t T\n")
    _corner_in("an import of an import",
               _pool(_DEFS,
                     ("mid.viba", "mid", "import defs as d\nM := d.U\n"),
                     ("main.viba", "main",
                      "import mid as m\nBox := Object * $m m.M\n")),
               "main.Box", _product(_tagged("$m", viba_ast.Nil())), expect="* $m nil",
               resident_source="U := Object\nBox := Object * $m U\n")
    _gap_in("gap never from another module", pool_of("Box := Object * $n d.N\n"),
            "main.Box", _product(_tagged("$n", viba_ast.Constant(1))),
            "nothing resides in never")


def run_cycle_cases():
    """池子里的环：名字对、自名、自指的泛型，都不转圈。"""
    _corner_in("a pair of names that point at each other",
               _pool(("m.viba", "m", "A := B\nB := A\nBox := Object * $a A\n")),
               "m.Box", _product(_tagged("$a", viba_ast.Constant(1))),
               expect="* $a 1", resident=False)
    _corner_in("a name that points at itself",
               _pool(("m.viba", "m", "A := A\nBox := Object * $a A\n")),
               "m.Box", _product(_tagged("$a", viba_ast.Constant(1))),
               expect="* $a 1", resident=False)
    _corner_in("a generic that asks for itself",
               _pool(("m.viba", "m", "W[T] := W[T]\nBox := Object * $w W[int]\n")),
               "m.Box", _product(_tagged("$w", viba_ast.Constant(1))),
               expect="* $w 1")
    _corner_in("a generic whose body only grows",
               _pool(("m.viba", "m", "W[T] := W[list[T]]\nBox := Object * $w W[int]\n")),
               "m.Box", _product(_tagged("$w", viba_ast.Constant(1))),
               expect="* $w 1", resident=False)


def run_material_root_cases():
    """材料本身不是一个产品：叶子、单个标签、和式、空链、nil。"""
    def design():
        return _pool(("m.viba", "m", "Box := Object * $a int\n"))

    _corner_in("a leaf where the whole product is",
               design(), "m.Box", viba_ast.Constant(1), expect="entry :=\n  1\n",
               resident=False)
    _corner_in("a lone tag where the product is",
               design(), "m.Box", _tagged("$a", viba_ast.Constant(1)),
               expect="entry :=\n  Object\n  * $a 1\n")
    _corner_in("nil where the whole product is",
               design(), "m.Box", viba_ast.Nil(), expect="entry :=\n  nil\n",
               resident=False)
    _gap_in("gap a sum chain where the product is", design(), "m.Box",
            viba_ast.SumChain([viba_ast.Constant(1)]), "no value here: $a")
    _gap_in("gap an empty chain where the product is", design(), "m.Box",
            viba_ast.ProductChain([]), "no value here: $a")


def run_definition_name_cases():
    """serialize 的名字参数：普通名字写得出来，内建名与关键字写不出来。"""
    pool = _pool(("m.viba", "m", "Box := Object * $a int\n"))
    definition = pool_find_definition(pool, "m.Box").ok_value
    from viba.reflect import access
    node = access.root(definition, VibaData(
        _product(_tagged("$a", viba_ast.Constant(1))))).ok_value
    for name in ("entry", "Entry_2", "a", "中文"):
        written = serialize.serialize(name, access, node)
        check(isinstance(written, Ok) and written.ok_value.startswith(f"{name} :="),
              f"definition name {name!r}: writes ({written})")
    for name in ("", "a.b", "a-b", "nil", "never", "void", "None", "true",
                 "false", "import", "as", "list", "set", "dict", "ListLiteral"):
        written = serialize.serialize(name, access, node)
        check(isinstance(written, Err),
              f"definition name {name!r}: an Err ({written})")


def run_alias_of_definition_cases():
    """定义自己的别名：写出来的源码跟写原定义时一样。"""
    from viba.reflect import access
    pool = _pool(("m.viba", "m", "Box := Object * $a int\nAlias := Box\nDeeper := Alias\n"))
    body = _product(_tagged("$a", viba_ast.Constant(1)))
    written = []
    for full_name in ("m.Box", "m.Alias", "m.Deeper"):
        definition = pool_find_definition(pool, full_name).ok_value
        node = access.root(definition, VibaData(body)).ok_value
        got = serialize.serialize("entry", access, node)
        check(isinstance(got, Ok), f"{full_name}: writes ({got})")
        written.append(got.ok_value if isinstance(got, Ok) else None)
    check(written[0] == written[1] == written[2],
          f"the alias writes the same source ({written})")


def run_sums_in_containers():
    """和式装在容器与元组里，每一格挑不同的支。"""
    source = "S := int | $a str\nBox := Object * $xs list[S] * $ss set[S] * $t (S, S)\n"
    pool = _pool(("m.viba", "m", source))
    material = _product(
        _tagged("$xs", viba_ast.TypeApp("ListLiteral", [
            viba_ast.SumChain([viba_ast.Constant(1)]),
            viba_ast.SumChain([_tagged("$a", viba_ast.Constant("x"))]),
            viba_ast.SumChain([viba_ast.Constant(2)])])),
        _tagged("$ss", viba_ast.TypeApp("SetLiteral", [
            viba_ast.SumChain([_tagged("$a", viba_ast.Constant("y"))])])),
        _tagged("$t", viba_ast.Tuple([
            viba_ast.SumChain([viba_ast.Constant(3)]),
            viba_ast.SumChain([_tagged("$a", viba_ast.Constant("z"))])])))
    _corner_in("sums in a list, a set and a tuple", pool, "m.Box", material,
               expect='ListLiteral[1, $a "x", 2]', strict=False)


def run_exponent_argument_shapes():
    """指数链的实参本身是什么形状：容器、元组、产品、代码块、nil、和式。"""
    cases = [
        ("list[int]", viba_ast.TypeApp("ListLiteral", [viba_ast.Constant(1)]),
         "ListLiteral[1]"),
        ("(int, str)", viba_ast.Tuple([viba_ast.Constant(1),
                                       viba_ast.Constant("s")]), '(1, "s")'),
        ("(Object * $x int)",
         _product(_tagged("$x", viba_ast.Constant(2))), "* $x 2"),
        ("{x}", viba_ast.CodeBlock("x"), "nil"),
        ("nil", viba_ast.Nil(), "nil"),
        ("(int | $a str)", viba_ast.SumChain([_tagged("$a", viba_ast.Constant("v"))]),
         '$a "v"'),
    ]
    for index, (argument_type, argument, expect) in enumerate(cases):
        pool = _pool(("m.viba", "m",
                      f"G := Object * $g (never <- $b {argument_type})\n"))
        material = _product(_tagged("$g", viba_ast.ExponentChain([
            viba_ast.Never(), _tagged("$b", argument)])))
        _corner_in(f"exponent argument[{index}] {argument_type}", pool, "m.G",
                   material, expect=expect, resident=False, resident_source=None)


def run_deep_stress():
    """再深一点、再宽一点：容器 12/20 层、1000 个元素、100 个键、64 个成员。"""
    for depth in (12, 20):
        type_text, body = "int", viba_ast.Constant(1)
        for _ in range(depth):
            type_text = f"list[{type_text}]"
            body = viba_ast.TypeApp("ListLiteral", [body])
        pool = _pool(("m.viba", "m", f"Box := Object * $a {type_text}\n"))
        _corner_in(f"depth {depth}", pool, "m.Box",
                   _product(_tagged("$a", body)), expect="ListLiteral[")
    pool = _pool(("m.viba", "m", "Box := Object * $a list[int]\n"))
    thousand = viba_ast.TypeApp("ListLiteral",
                                [viba_ast.Constant(i) for i in range(1000)])
    _corner_in("a list of 1000", pool, "m.Box", _product(_tagged("$a", thousand)),
               expect="ListLiteral[0, 1, 2")
    pool = _pool(("m.viba", "m", "Box := Object * $a dict[str, int]\n"))
    pairs = [viba_ast.Tuple([viba_ast.Constant(f"k{i}"), viba_ast.Constant(i)])
             for i in range(100)]
    _corner_in("a dict of 100", pool, "m.Box",
               _product(_tagged("$a", viba_ast.TypeApp("DictLiteral", pairs))),
               expect='("k99", 99)]')
    pool = _pool(("m.viba", "m",
                  "Box := Object * " + " * ".join(f"$f{i} int" for i in range(64)) + "\n"))
    _corner_in("a product of 64", pool, "m.Box",
               _product(*[_tagged(f"$f{i}", viba_ast.Constant(i))
                          for i in range(64)]), expect="* $f63 63")


def run_head_written_as_unit():
    """产品头写成 nil：写出来的单位还是语言的那个 Object。"""
    _corner_in("the head written as nil",
               _pool(("m.viba", "m", "Box := nil * $a int\n")), "m.Box",
               _product(_tagged("$a", viba_ast.Constant(1))),
               expect="entry :=\n  Object\n  * $a 1\n")
    _corner_in("two unit members and the head",
               _pool(("m.viba", "m", "Box := Object * $u Object * $v Object\n")),
               "m.Box",
               _product(_tagged("$u", viba_ast.Nil()),
                        _tagged("$v", viba_ast.Nil())),
               expect="* $u nil\n  * $v nil")


def run_inline_member_cases():
    """内联成员：第一个不带标签的成员摊进自己的成员，递归；单位元不算成员；
    重标签写不出来；内联环不转圈。"""
    _corner("an inline product member",
            "A := $x int * $y int\nB := A * $z int\n", "B",
            _product(_tagged("$x", viba_ast.Constant(1)),
                     _tagged("$y", viba_ast.Constant(2)),
                     _tagged("$z", viba_ast.Constant(3))),
            expect="entry :=\n  $x 1\n  * $y 2\n  * $z 3\n")
    _corner("an inline chain two deep",
            "A := $x int\nB := A * $y int\nC := B * $z int\n", "C",
            _product(_tagged("$x", viba_ast.Constant(1)),
                     _tagged("$y", viba_ast.Constant(2)),
                     _tagged("$z", viba_ast.Constant(3))),
            expect="entry :=\n  $x 1\n  * $y 2\n  * $z 3\n")
    _corner("a member that is no product keeps its place",
            "Box := int * $a int\n", "Box",
            _product(viba_ast.Constant(7), _tagged("$a", viba_ast.Constant(1))),
            expect="entry :=\n  7\n  * $a 1\n")
    _corner("an inline member after a positional one",
            "A := $x int * $y int\nBox := int * A * $z int\n", "Box",
            _product(viba_ast.Constant(7),
                     _tagged("$x", viba_ast.Constant(1)),
                     _tagged("$y", viba_ast.Constant(2)),
                     _tagged("$z", viba_ast.Constant(3))),
            expect="entry :=\n  7\n  * $x 1\n  * $y 2\n  * $z 3\n")
    _corner("a unit member between positional ones",
            "Box := Object * int * Object * int * $a int\n", "Box",
            _product(viba_ast.Constant(7), viba_ast.Nil(),
                     viba_ast.Constant(8), _tagged("$a", viba_ast.Constant(1))),
            expect="entry :=\n  Object\n  * 7\n  * 8\n  * $a 1\n")
    _corner("the head written as a name over the unit",
            "U := Object\nBox := U * $a int\n", "Box",
            _product(_tagged("$a", viba_ast.Constant(1))),
            expect="entry :=\n  Object\n  * $a 1\n")
    _corner("the member written as a name on the material side",
            "Data := $x 1 * $y 2\nA := $x int * $y int\nB := A * $z int\n", "B",
            _product(viba_ast.TypeRef("Data"), _tagged("$z", viba_ast.Constant(3))),
            expect="entry :=\n  $x 1\n  * $y 2\n  * $z 3\n")
    _corner("an inline member behind a generic application",
            "Inner := $x int * $y int\nBox[T] := T * $z int\nB := Box[Inner]\n", "B",
            _product(_tagged("$x", viba_ast.Constant(1)),
                     _tagged("$y", viba_ast.Constant(2)),
                     _tagged("$z", viba_ast.Constant(3))),
            expect="entry :=\n  $x 1\n  * $y 2\n  * $z 3\n")
    _gap("the same tag twice through an inline",
         "A := $x int\nB := A * $x str\n", "B",
         _product(_tagged("$x", viba_ast.Constant(1))), "written twice")
    _gap("the same tag twice, one of them inlined",
         "A := $x int * $y int\nB := A * $y str\n", "B",
         _product(_tagged("$x", viba_ast.Constant(1)),
                  _tagged("$y", viba_ast.Constant(2))), "written twice")
    _gap("an inline cycle stays one positional member",
         "A := A * $x int\n", "A",
         _product(_tagged("$x", viba_ast.Constant(1))), "no value here")
    _gap("an inline cycle that repeats a tag",
         "A := $x int * A * $y int\n", "A",
         _product(_tagged("$x", viba_ast.Constant(1)),
                  _tagged("$y", viba_ast.Constant(2))), "written twice")


def run():
    for case in (run_fixture_cases, run_empty_container_cases, run_nil_slot_cases,
                 run_set_order_cases, run_exponent_cases, run_code_block_cases,
                 run_shape_corner_cases, run_sum_corner_cases,
                 run_leaf_corner_cases, run_exponent_corner_cases,
                 run_more_gap_cases, run_never_and_key_cases, run_gap_cases,
                 run_leaf_matrix, run_number_and_string_gaps, run_tag_matrix,
                 run_depth_and_width_ladders, run_absent_member_positions,
                 run_alias_ladders, run_sum_ladders, run_shape_matrix,
                 run_container_fills, run_never_positions, run_dict_key_gaps,
                 run_positional_gaps, run_unit_member_shapes,
                 run_subclass_leaf_cases,
                 run_code_block_positions, run_exponent_batteries,
                 run_name_alias_shapes, run_name_gaps, run_more_gap_corners,
                 run_dict_key_aliases, run_cross_module_cases, run_cycle_cases,
                 run_material_root_cases, run_definition_name_cases,
                 run_alias_of_definition_cases, run_sums_in_containers,
                 run_exponent_argument_shapes, run_deep_stress,
                 run_head_written_as_unit, run_inline_member_cases):
        case()
    print(f"serialize: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())

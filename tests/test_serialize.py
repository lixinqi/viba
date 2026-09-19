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
from viba.viba_type_descriptor import empty_pool, parse_viba_file, pool_add_file
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


def run():
    for case in (run_fixture_cases, run_empty_container_cases, run_nil_slot_cases,
                 run_set_order_cases, run_exponent_cases, run_code_block_cases,
                 run_shape_corner_cases, run_sum_corner_cases,
                 run_leaf_corner_cases, run_exponent_corner_cases,
                 run_more_gap_cases, run_never_and_key_cases, run_gap_cases):
        case()
    print(f"serialize: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())

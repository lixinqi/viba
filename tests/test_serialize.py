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
                resident=True):
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
    missing = {address: value for address, value in before.items()
               if after.get(address, "<missing>") != value}
    check(not missing, f"{label}: every address reads the same leaf ({missing})")

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
                 run_never_and_key_cases, run_gap_cases):
        case()
    print(f"serialize: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())

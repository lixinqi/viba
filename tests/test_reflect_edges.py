"""viba.reflect 的边角：Config 与 VibaAccess 的 repr、访问单元的错误、动态取值。

核心反射协议（viba-reflect.md）平时由规则层与序列化那两套语料跑着，这里补的是
它们碰不到的边角：报错里的名字、repr、以及"设计里没有这个地址"和"可序列化数据里没有这
一块"这一对的区别。

语料现搓：一个小池子加一份可序列化数据。

    python3 tests/test_reflect_edges.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import viba_ast
from viba.reflect import (Config, VibaData, VibaReflectError, _key_text, _scalar_name,
                          _type_name, access, at_key, by_field_index, by_tag)
from viba.type import VibaProgramErr, Ok
from viba.viba_type_descriptor import (empty_pool, parse_viba_file, pool_add_file,
                                       pool_find_definition)

PASS = FAIL = 0


def check(ok: bool, label: str):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def same(label: str, got, want):
    check(got == want, f"{label}: {got!r} != {want!r}")


def is_err(label: str, result, needle: str):
    check(isinstance(result, VibaProgramErr) and needle in result.err_msg,
          f"{label}: wanted VibaProgramErr({needle!r}), got {result!r}")


BOX = """Box = Object * $a int * $b str * $xs list[int] * $d dict[str, int]
"""


def _definition(source: str, name: str):
    pool = empty_pool()
    parsed = parse_viba_file(pool, source, "edge.viba", "edge")
    assert isinstance(parsed, Ok), parsed
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    return pool_find_definition(pool, f"edge.{name}").ok_value


def _product(members):
    return viba_ast.ProductChain([viba_ast.TypeRef("Object"), *members])


def _root(definition, body):
    rooted = access.root(definition, VibaData(body))
    assert isinstance(rooted, Ok), rooted
    return rooted.ok_value


def run_reprs():
    """repr 是调试面：说得清自己是什么，且两次运行一样。"""
    same("a bare config", repr(Config()), "Config(never_eqv=[], nil_eqv=[])")
    same("a config says what it was told",
         repr(Config(never_eqv={"Oneof"}, nil_eqv={"Object"})),
         "Config(never_eqv=['Oneof'], nil_eqv=['Object'])")
    same("an accessor says its config", repr(access), f"VibaAccess({access.config!r})")
    same("a step says its kind and value", repr(by_tag("$a")), "by_tag('$a')")
    same("a positional step", repr(by_field_index(3)), "by_field_index(3)")
    same("a key step", repr(at_key("k")), "at_key('k')")

    definition = _definition(BOX, "Box")
    same("a node is named by its address", repr(_root(definition, _product([]))), "VibaNode(root)")
    node = _root(definition, _product([viba_ast.Tagged("$a", viba_ast.Constant(1))]))
    child = access.get(node, by_tag("$a")).ok_value
    same("a child node carries its path", repr(child), "VibaNode(by_tag('$a'))")


def run_name_helpers():
    """只出现在报错里的三个名字助手。"""
    same("a string names itself", _type_name("A"), "A")
    same("a node is named by its name", _type_name(viba_ast.TypeRef("A")), "A")
    definition = _definition(BOX, "Box")
    same("a descriptor by its full name", _type_name(definition), "edge.Box")

    same("true is a bool", _scalar_name(True), "bool")
    same("false too", _scalar_name(False), "bool")
    same("an int is an int", _scalar_name(7), "int")
    same("a float is a float", _scalar_name(1.5), "float")
    same("a str is a str", _scalar_name("x"), "str")
    same("nil is no scalar", _scalar_name(None), None)

    same("no key is no text", _key_text(None), None)
    same("true keys write true", _key_text(True), "true")
    same("false keys write false", _key_text(False), "false")
    same("other keys write themselves", _key_text(7), "7")
    same("str keys write themselves", _key_text("k"), "k")


def run_access_edges():
    """设计里没有这个地址是 VibaProgramErr；可序列化数据里没有那一块是 Ok(nil)/False。"""
    definition = _definition(BOX, "Box")
    full = _product([
        viba_ast.Tagged("$a", viba_ast.Constant(1)),
        viba_ast.Tagged("$b", viba_ast.Constant("two")),
        viba_ast.Tagged("$xs", viba_ast.TypeApp("ListLiteral", [viba_ast.Constant(1)])),
        viba_ast.Tagged("$d", viba_ast.TypeApp("DictLiteral", [
            viba_ast.Tuple([viba_ast.Constant("k"), viba_ast.Constant(2)])])),
    ])
    root = _root(definition, full)

    is_err("an address the design lacks", access.get(root, by_tag("$nope")),
           "the design has no such address")
    same("and has says false", access.has(root, by_tag("$nope")).ok_value, False)
    same("has says true for one that is there",
         access.has(root, by_tag("$a")).ok_value, True)

    thin = _product([viba_ast.Tagged("$a", viba_ast.Constant(1))])
    thin_root = _root(definition, thin)
    same("a piece the viba data lacks reads nil",
         access.get(thin_root, by_tag("$b")).ok_value, None)
    same("and has says false", access.has(thin_root, by_tag("$b")).ok_value, False)

    is_err("a product is no leaf", access.leaf(root), "not a leaf")
    same("a literal is a leaf", access.leaf(root.by_tag("a")).ok_value, 1)
    is_err("a product is no container", access.length(root), "not a container")
    same("a list has a length", access.length(root.by_tag("xs")).ok_value, 1)
    is_err("a list is no dict", access.keys(root.by_tag("xs")), "not a dict")
    same("a dict hands its keys over", access.keys(root.by_tag("d")).ok_value, ["k"])

    bad_key = _product([
        viba_ast.Tagged("$a", viba_ast.Constant(1)),
        viba_ast.Tagged("$b", viba_ast.Constant("two")),
        viba_ast.Tagged("$xs", viba_ast.TypeApp("ListLiteral", [])),
        viba_ast.Tagged("$d", viba_ast.TypeApp("DictLiteral", [
            viba_ast.TypeApp("ListLiteral", [viba_ast.Constant("k")])])),
    ])
    is_err("a key that is not a literal",
           access.keys(_root(definition, bad_key).by_tag("d")), "not a literal")


def run_dynamic_accessors():
    """`node.get_a()` 这类动态取值：取不到就抛，不是给 VibaProgramErr；try_ 那一族给 VibaProgramErr。"""
    definition = _definition(BOX, "Box")
    root = _root(definition, _product([
        viba_ast.Tagged("$a", viba_ast.Constant(1)),
        viba_ast.Tagged("$b", viba_ast.Constant("two")),
        viba_ast.Tagged("$xs", viba_ast.TypeApp("ListLiteral", [viba_ast.Constant(1)])),
        viba_ast.Tagged("$d", viba_ast.TypeApp("DictLiteral", [
            viba_ast.Tuple([viba_ast.Constant("k"), viba_ast.Constant(2)])])),
    ]))
    same("get_a takes the step", access.leaf(root.get_a()).ok_value, 1)
    same("has_b answers a bool", root.has_b(), True)
    same("len() reads a container", len(root.get_xs()), 1)
    same("keys() reads a dict", root.get_d().keys(), ["k"])
    same("values() reads a dict", root.get_d().values()[0].leaf, 2)
    same("items() reads a dict", root.get_d().items()[0][0], "k")
    same("a dict reads by key", access.leaf(root.get_d()["k"]).ok_value, 2)
    same("a list iterates by position",
         [access.leaf(item).ok_value for item in root.get_xs()], [1])
    same("membership works by tag", "a" in root, True)
    check("a tag that is not there is not a member", "nope" not in root)
    try:
        root.get_nope()
        check(False, "a missing tag raises from the dynamic accessor")
    except VibaReflectError as error:
        check("no such address" in str(error),
              f"a missing tag raises from the dynamic accessor: {error}")
    is_err("try_get_ answers VibaProgramErr instead", root.try_get_nope(), "no such address")
    same("dir() lists the accessors it answers to", "get_a" in dir(root), True)

    # 位置成员：按位置的那一族叫 get_field_<i>
    position = _definition("Pos = Object * int * $a int\n", "Pos")
    pos = _root(position, _product([viba_ast.Constant(5),
                                    viba_ast.Tagged("$a", viba_ast.Constant(6))]))
    same("a positional member reads by index", access.leaf(pos.get_field_0()).ok_value, 5)
    same("has_field answers a bool", pos.has_field_0(), True)
    check("dir() names positional accessors too", "get_field_0" in dir(pos))
    is_err("try_get_field_ answers VibaProgramErr for one that is not there",
           pos.try_get_field_9(), "no such address")
    same("and the tag still reads by name", access.leaf(pos.get_a()).ok_value, 6)

    # 一个不是名字的东西当 tag：说清楚要的是字符串，不是 AttributeError
    try:
        by_tag(0)
        check(False, "a tag name that is not a string is refused")
    except TypeError as error:
        check("a tag name is a string" in str(error),
              f"a tag name that is not a string is refused: {error}")


def run():
    run_reprs()
    run_name_helpers()
    run_access_edges()
    run_dynamic_accessors()
    print(f"reflect_edges: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())

"""viba.rule.reflect 的 API 测试。

一个 api 一个 case，函数名就是它测的东西：``test_api_<api>_case_<NNN>``。
``run()`` 按文件里的顺序把这些 case 全跑一遍，最后打一行汇总。
跑语料的那份是 tests/test_rule_reflect_corpus.py（18377 份材料沿图走通）。

材料全是仓库里现成的：

    data/rule_coding_style_check/demo.viba            DemoRule：int / 积 / list / Predicate
    data/rule_coding_style_check/not_rule.viba        not[...] 外壳（分支写在里面）
    data/rule_coding_style_check/sum_rule.viba        和类型
    data/rule_coding_style_check/not_rules/not_rule03.viba  禁止的操作数写成名字（not[Crimes]）
    data/is_sub_type/not/sup061.viba                  禁止直接写成 never <- $not_operand A
    data/type_descriptor/case_000/top.viba            Shape0：位置成员；Node1：list / set / dict / 可选

witness 都在这里手搓（跟 test_rule_coding_style_check.py 的 _bound_witness
一个路子），值写死，所以能对死。语料里没有 (A, B) 这种元组成员，所以
at_index / VibaLength 的元组那一支没有材料可测。

    python tests/test_rule_reflect_api.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import viba_ast
from viba.type import AstNodeType, Err, Ok, custom_module
from viba.viba_type_descriptor import (
    NEVER,
    empty_pool,
    parse_viba_file,
    pool_add_file,
    pool_find_definition,
)
from viba.rule import reflect
from viba.rule.reflect import (
    VibaReflectError,
    Witness,
    at_index,
    at_key,
    by_field_index,
    by_tag,
)

_RULES = Path(__file__).resolve().parent / "data" / "rule_coding_style_check"
_NOT_RULES = _RULES / "not_rules"
_NOT_CASES = Path(__file__).resolve().parent / "data" / "rule_refutation" / "not"
_CASES = Path(__file__).resolve().parent / "data" / "type_descriptor" / "case_000"


# ----------------------------------------------------------------------
# 材料
# ----------------------------------------------------------------------


def _load(path: Path, module_name: str):
    source = path.read_text()
    pool = empty_pool()
    parsed = parse_viba_file(pool, source, path.name, module_name)
    assert isinstance(parsed, Ok), parsed
    return pool_add_file(pool, parsed.ok_value).ok_value


def _definition_of(pool, full_name: str):
    found = pool_find_definition(pool, full_name)
    assert isinstance(found, Ok), found
    return found.ok_value


def _body_of(path: Path, name: str):
    for definition in viba_ast.parse(path.read_text()).body:
        if getattr(definition, "name", None) == name:
            return definition.body
    raise AssertionError(f"{path.name}: no definition named {name}")


def _predicate_node(body):
    for node in viba_ast.walk(body):
        if isinstance(node, viba_ast.TypeApp) and node.constructor == "Predicate":
            return node
    raise AssertionError("no Predicate field")


def _demo_body():
    """照 DemoRule 手搓一份：int、积、list、Predicate。

    Metric[T] := $value T，量出来的值都在自己那一层 $value 底下。
    """
    return viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$code_length", viba_ast.Tagged("$value", viba_ast.Constant(7))),
        viba_ast.Tagged("$coverage", viba_ast.Tagged("$value", viba_ast.ProductChain([
            viba_ast.Tagged("$documented_lines", viba_ast.Constant(5)),
            viba_ast.Tagged("$total_lines", viba_ast.Constant(11)),
        ]))),
        viba_ast.Tagged("$keywords", viba_ast.Tagged("$value", viba_ast.TypeApp(
            "ListLiteral", [viba_ast.Constant("a"), viba_ast.Constant("b")]))),
        viba_ast.Tagged("$assert_code_len_le_24",
                        _predicate_node(_body_of(_RULES / "demo.viba", "DemoRule"))),
    ])


def _node1_body():
    """照 Node1 手搓一份：list / set / dict / 可选。"""
    return viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$items", viba_ast.TypeApp(
            "ListLiteral", [viba_ast.Constant(1), viba_ast.Constant(2)])),
        viba_ast.Tagged("$seen", viba_ast.TypeApp(
            "SetLiteral", [viba_ast.Constant("x")])),
        viba_ast.Tagged("$table", viba_ast.TypeApp("DictLiteral", [
            viba_ast.Tuple([viba_ast.Constant("k"), viba_ast.Constant(1)]),
            viba_ast.Tuple([viba_ast.Constant("m"), viba_ast.Constant(2)]),
        ])),
        viba_ast.Tagged("$maybe", viba_ast.Constant(3)),
    ])


def rule40_body():
    """照 rule40 的 GroupSpec 手搓一份：三层 TypeRef 链上的数据。

    Metric[GroupSpec] → $value → GroupSpec := $bucket Bucket →
    Bucket := list[Slot] → Slot := $name str
    """
    return viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$group_spec", viba_ast.Tagged("$value", viba_ast.ProductChain([
            viba_ast.Tagged("$bucket", viba_ast.TypeApp("ListLiteral", [
                viba_ast.ProductChain([
                    viba_ast.TypeRef("Object"),
                    viba_ast.Tagged("$name", viba_ast.Constant("slot0")),
                ]),
            ])),
        ]))),
    ])


def _named_operand_body():
    """$not_crimes 填成 not[Crimes]：材料也照名字写，两边都展开。"""
    return viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$not_crimes", viba_ast.TypeApp("not", [viba_ast.TypeRef("Crimes")])),
    ])


def _written_operand_body():
    """设计直接写 never <- $not_operand A：材料照写，$a 量出 3。"""
    return viba_ast.Exponent(
        viba_ast.Never(),
        viba_ast.Tagged("$not_operand", viba_ast.Tagged("$a", viba_ast.Constant(3))))


class _Materials:
    """现成材料 + 手搓的几份 witness。"""

    def __init__(self):
        demo = _load(_RULES / "demo.viba", "demo")
        self.demo = _definition_of(demo, "demo.DemoRule")
        self.demo_access = reflect.access
        self.demo_witness = Witness(_demo_body())
        self.demo_node = self.demo_access.root(self.demo, self.demo_witness).ok_value

        top = _load(_CASES / "top.viba", "top")
        self.shape = _definition_of(top, "top.Shape0")
        self.shape_access = reflect.access
        self.shape_node = self.shape_access.root(self.shape, Witness(
            viba_ast.Product(viba_ast.Constant(1), viba_ast.Constant("two")))).ok_value
        self.node1 = _definition_of(top, "top.Node1")
        self.node1_access = reflect.access
        self.node1_node = self.node1_access.root(
            self.node1, Witness(_node1_body())).ok_value

        not_rule = _load(_RULES / "not_rule.viba", "not_rule")
        self.not_rule = _definition_of(not_rule, "not_rule.NoDeathPenaltyRule")
        self.not_access = reflect.access
        self.law_abiding = self.not_access.root(
            self.not_rule, Witness(_body_of(_RULES / "not_rule.viba", "LawAbidingWitness"))).ok_value

        named = _load(_NOT_RULES / "not_rule03.viba", "not_rule03")
        self.named_not = _definition_of(named, "not_rule03.NotRule03")
        self.named_not_access = reflect.access
        self.named_not_node = self.named_not_access.root(
            self.named_not, Witness(_named_operand_body())).ok_value

        written = _load(_NOT_CASES / "sup061.viba", "sup061")
        self.written_not = _definition_of(written, "sup061.F6")
        self.written_not_access = reflect.access
        self.written_not_node = self.written_not_access.root(
            self.written_not, Witness(_written_operand_body())).ok_value

        rule40 = _load(_RULES / "rules" / "rule40.viba", "rule40")
        self.rule40 = _definition_of(rule40, "rule40.Rule40")
        self.rule40_access = reflect.access
        self.rule40_node = self.rule40_access.root(
            self.rule40, Witness(rule40_body())).ok_value

        sum_rule = _load(_RULES / "sum_rule.viba", "sum_rule")
        self.sum_rule = _definition_of(sum_rule, "sum_rule.SumRule")
        self.sum_access = reflect.access
        self.sum_pass = self.sum_access.root(
            self.sum_rule, Witness(_body_of(_RULES / "sum_rule.viba", "SumWitnessPass"))).ok_value

    def _without(self, node, tag):
        """把某个 tag 摘掉的那份材料（造"数据里没有"用）。"""
        kept = [e for e in viba_ast.convert_to_chain_style(node.data).elements
                if not (isinstance(e, viba_ast.Tagged) and e.tag == tag)]
        return reflect.access.root(self.demo,
                                   Witness(viba_ast.ProductChain(kept))).ok_value


_MATERIALS = None


def _materials():
    global _MATERIALS
    if _MATERIALS is None:
        _MATERIALS = _Materials()
    return _MATERIALS


# ----------------------------------------------------------------------
# VibaRoot
# ----------------------------------------------------------------------


def _inline(source: str, name: str):
    """现搓一份定义：inline 模块里的 name。"""
    pool = empty_pool()
    parsed = parse_viba_file(pool, source, "inline.viba", "inline")
    assert isinstance(parsed, Ok), parsed
    return _definition_of(pool_add_file(pool, parsed.ok_value).ok_value,
                          f"inline.{name}")


def _pool_of(*files):
    """(文件名, 模块名, 源码) 编进一个池子。"""
    pool = empty_pool()
    for file_name, module, source in files:
        parsed = parse_viba_file(pool, source, file_name, module)
        assert isinstance(parsed, Ok), parsed
        added = pool_add_file(pool, parsed.ok_value)
        assert isinstance(added, Ok), added
        pool = added.ok_value
    return pool


def test_api_unfold_stops_at_a_cycle():
    """名字绕回自己：展开停在名字上，不转圈。"""
    access = reflect.access
    cyclic = _inline("A := B\nB := A\nBox := Object * $a A\n", "Box")
    node = access.root(cyclic, Witness(viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$a", viba_ast.Constant(1))]))).ok_value
    assert access.unfold(access.member_steps(node)[0][2]).kind == "type_ref"

    growing = _inline("W[T] := W[list[T]]\nBox := Object * $w W[int]\n", "Box")
    node2 = access.root(growing, Witness(viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$w", viba_ast.Constant(1))]))).ok_value
    assert access.unfold(access.member_steps(node2)[0][2]).kind == "type_app"


def test_api_members_across_modules():
    """import 前缀的名字：`d.P` 落到 defs 的 P，`d.U` 落地就是单位。"""
    access = reflect.access
    pool = _pool_of(("defs.viba", "defs", "U := Object\nP := Object * $x int\n"),
                    ("main.viba", "main",
                     "import defs as d\nBox := d.U * $a d.P\n"))
    definition = _definition_of(pool, "main.Box")
    node = access.root(definition, Witness(viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$a", viba_ast.ProductChain([
            viba_ast.Tagged("$x", viba_ast.Constant(1))]))]))).ok_value
    steps = access.member_steps(node)
    assert [tag for tag, _, _ in steps] == ["$a"], steps
    assert access.unfold(steps[0][2]).kind == "product"


def test_api_members_unit_behind_a_name():
    """名字是透明的：`H := Object` 的 H 当产品头就不占一格，`U := Object`
    的成员是单位成员，`N := never` 的成员照旧没有居民。"""
    access = reflect.access
    headed = _inline("H := Object\nBox := H * $a int\n", "Box")
    node = access.root(headed, Witness(viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$a", viba_ast.Constant(1))]))).ok_value
    assert [tag for tag, _, _ in access.member_steps(node)] == ["$a"]

    member = _inline("U := Object\nBox := Object * $u U * $a int\n", "Box")
    node2 = access.root(member, Witness(viba_ast.ProductChain([
        viba_ast.TypeRef("Object"), viba_ast.Tagged("$u", viba_ast.Nil()),
        viba_ast.Tagged("$a", viba_ast.Constant(1))]))).ok_value
    units = [(tag, access._is_unit_descriptor(descriptor))
             for tag, _, descriptor in access.member_steps(node2)]
    assert units == [("$u", True), ("$a", False)], units

    never = _inline("N := never\nBox := Object * $n N\n", "Box")
    node3 = access.root(never, Witness(viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$n", viba_ast.Constant(1))]))).ok_value
    assert access.unfold(access.member_steps(node3)[0][2]).kind == NEVER


def test_api_root_case_001():
    """声明的版本里有这一版：给出起点。"""
    m = _materials()
    rooted = m.demo_access.root(m.demo, m.demo_witness)
    assert isinstance(rooted, Ok)
    assert rooted.ok_value.descriptor is m.demo.body


def test_api_root_case_002():
    """起点只把 (描述符, 数据) 配成对：节点给出来，data 就是那份材料；版本协议不管。"""
    m = _materials()
    rooted = m.demo_access.root(m.demo, Witness(m.demo_witness.node))
    assert isinstance(rooted, Ok) and rooted.ok_value.data is m.demo_witness.node


# ----------------------------------------------------------------------
# VibaLeaf
# ----------------------------------------------------------------------


def test_api_leaf_case_001():
    """int 字面量读得出值：量出来的值是 $value 那一层的字面量。"""
    m = _materials()
    inside = m.demo_node.by_tag("code_length").by_tag("value")
    assert m.demo_access.leaf(inside).ok_value == 7


def test_api_leaf_case_002():
    """float / bool / str 也读得出。"""
    m = _materials()
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$items", viba_ast.Constant(1.5)),
        viba_ast.Tagged("$seen", viba_ast.Constant(True)),
        viba_ast.Tagged("$table", viba_ast.Constant("x")),
        viba_ast.Tagged("$maybe", viba_ast.Constant(3)),
    ])
    node = m.node1_access.root(m.node1, Witness(body)).ok_value
    assert m.node1_access.leaf(node.by_tag("items")).ok_value == 1.5
    assert m.node1_access.leaf(node.by_tag("seen")).ok_value is True
    assert m.node1_access.leaf(node.by_tag("table")).ok_value == "x"


def test_api_leaf_case_003():
    """数据是 nil：读得出 nil 叶子。"""
    m = _materials()
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$items", viba_ast.TypeApp(
            "ListLiteral", [viba_ast.Constant(1)])),
        viba_ast.Tagged("$seen", viba_ast.TypeApp(
            "SetLiteral", [viba_ast.Constant("x")])),
        viba_ast.Tagged("$table", viba_ast.TypeApp("DictLiteral", [])),
        viba_ast.Tagged("$maybe", viba_ast.Nil()),
    ])
    node = m.node1_access.root(m.node1, Witness(body)).ok_value
    leaf = m.node1_access.leaf(node.by_tag("maybe"))
    assert isinstance(leaf, Ok) and leaf.ok_value is None


def test_api_leaf_case_004():
    """Not a leaf: Err."""
    m = _materials()
    assert isinstance(m.demo_access.leaf(m.demo_node.by_tag("coverage")), Err)


# ----------------------------------------------------------------------
# VibaLength
# ----------------------------------------------------------------------


def test_api_length_case_001():
    """list 给元素个数。"""
    m = _materials()
    assert m.node1_access.length(m.node1_node.by_tag("items")).ok_value == 2


def test_api_length_case_002():
    """set 给元素个数。"""
    m = _materials()
    assert m.node1_access.length(m.node1_node.by_tag("seen")).ok_value == 1


def test_api_length_case_003():
    """dict 给键数。"""
    m = _materials()
    assert m.node1_access.length(m.node1_node.by_tag("table")).ok_value == 2


def test_api_length_case_004():
    """不是容器：Err。"""
    m = _materials()
    assert isinstance(m.demo_access.length(m.demo_node.by_tag("code_length")), Err)


# ----------------------------------------------------------------------
# VibaKeys
# ----------------------------------------------------------------------


def test_api_keys_case_001():
    """dict 给键，顺序由实现定。"""
    m = _materials()
    assert m.node1_access.keys(m.node1_node.by_tag("table")).ok_value == ["k", "m"]


def test_api_keys_case_002():
    """不是 dict：Err。"""
    m = _materials()
    assert isinstance(m.node1_access.keys(m.node1_node.by_tag("items")), Err)


# ----------------------------------------------------------------------
# 节点上的 by_tag / by_field_index / at_index / at_key / leaf
# ----------------------------------------------------------------------


def test_api_node_by_tag_case_001():
    """按 tag 取子节点；名字带不带 $ 都认。"""
    m = _materials()
    assert m.demo_node.by_tag("code_length").by_tag("value").value == 7
    assert m.demo_node.by_tag("$code_length").by_tag("$value").value == 7


def test_api_node_by_tag_case_002():
    """取不到就抛。"""
    m = _materials()
    try:
        m.demo_node.by_tag("nope")
        raise AssertionError("本该抛")
    except VibaReflectError:
        pass


def test_api_node_by_field_index_case_001():
    """位置成员按位置取。"""
    m = _materials()
    assert m.shape_node.by_field_index(0).value == 1
    assert m.shape_node.by_field_index(1).value == "two"


def test_api_node_by_field_index_case_002():
    """位置越界就抛。"""
    m = _materials()
    try:
        m.shape_node.by_field_index(9)
        raise AssertionError("本该抛")
    except VibaReflectError:
        pass


def test_api_node_at_index_case_001():
    """list 按下标取。"""
    m = _materials()
    assert m.node1_node.by_tag("items").at_index(1).value == 2


def test_api_node_at_index_case_002():
    """set 也按下标取（顺序由实现定）。"""
    m = _materials()
    assert m.node1_node.by_tag("seen").at_index(0).value == "x"


def test_api_node_at_index_case_003():
    """下标越界就抛。"""
    m = _materials()
    try:
        m.node1_node.by_tag("items").at_index(9)
        raise AssertionError("本该抛")
    except VibaReflectError:
        pass


def test_api_node_at_index_case_004():
    """A dict is not index-addressable: $at_index is for list / set / tuple."""
    m = _materials()
    table = m.node1_node.by_tag("table")
    assert m.node1_access.has(table, at_index(0)).ok_value is False
    assert isinstance(m.node1_access.get(table, at_index(0)), Err)
    assert len(table) == 2  # 数得出来，只是不按下标取


def test_api_node_at_key_case_001():
    """dict 按键取。"""
    m = _materials()
    assert m.node1_node.by_tag("table").at_key("m").value == 2


def test_api_node_at_key_case_002():
    """没有这个键就抛。"""
    m = _materials()
    try:
        m.node1_node.by_tag("table").at_key("nope")
        raise AssertionError("本该抛")
    except VibaReflectError:
        pass


def test_api_node_leaf_case_001():
    """leaf 给常量记录，Python 落点 value 给裸值。"""
    m = _materials()
    node = m.demo_node.by_tag("code_length").by_tag("value")
    assert node.leaf == 7 and node.value == 7


def test_api_node_leaf_case_002():
    """不是叶子：leaf 抛——量出来的值那一层不是叶子，量出来的数才是。"""
    m = _materials()
    for node in (m.demo_node.by_tag("coverage"), m.demo_node.by_tag("coverage").by_tag("value")):
        try:
            node.leaf
            raise AssertionError("本该抛")
        except VibaReflectError:
            pass


# ----------------------------------------------------------------------
# 节点上的形状：is_list / is_set / is_dict
# ----------------------------------------------------------------------


def test_api_node_is_list_case_001():
    """写着 list[...] 的才是 list。"""
    m = _materials()
    assert m.node1_node.by_tag("items").is_list is True
    assert m.node1_node.by_tag("table").is_list is False


def test_api_node_is_list_case_002():
    """写成 Metric[Keywords] 的不算：只看写出来的链头。"""
    m = _materials()
    assert m.demo_node.by_tag("keywords").is_list is False


def test_api_node_is_set_case_001():
    """写着 set[...] 的才是 set。"""
    m = _materials()
    assert m.node1_node.by_tag("seen").is_set is True
    assert m.node1_node.by_tag("items").is_set is False


def test_api_node_is_dict_case_001():
    """写着 dict[...] 的才是 dict。"""
    m = _materials()
    assert m.node1_node.by_tag("table").is_dict is True
    assert m.node1_node.by_tag("seen").is_dict is False


def test_api_node_is_dict_case_002():
    """不是容器：三个都假。"""
    m = _materials()
    node = m.demo_node.by_tag("code_length")
    assert node.is_list is False and node.is_set is False and node.is_dict is False


# ----------------------------------------------------------------------
# 节点上的 Python 落点：get_ / has_ / try_get_ / in
# ----------------------------------------------------------------------


def test_api_node_get_case_001():
    """get_{name}() 取值；能一路点下去（量出来的值先进 $value）。"""
    m = _materials()
    assert m.demo_node.get_code_length().get_value().value == 7
    assert m.demo_node.get_coverage().get_value().get_documented_lines().value == 5


def test_api_node_get_case_002():
    """get_field_{i}() 按位置取值。"""
    m = _materials()
    assert m.shape_node.get_field_0().value == 1
    assert m.shape_node.get_field_1().value == "two"


def test_api_node_get_case_003():
    """设计直接写 never <- $not_operand A 时也一样：get_not_operand() 取到 A，再往下取叶子。"""
    m = _materials()
    node = m.written_not_node
    assert m.written_not_access.has(node, by_tag("$not_operand")).ok_value is True
    operand = node.get_not_operand()
    assert operand.by_tag("a").leaf == 3
    assert operand.get_a().value == 3


def test_api_node_has_case_001():
    """has_{name}() 答真假，永不抛。"""
    m = _materials()
    assert m.demo_node.has_code_length() is True
    assert m.demo_node.has_nope() is False


def test_api_node_has_case_002():
    """has_field_{i}()：位置成员。"""
    m = _materials()
    assert m.shape_node.has_field_0() is True
    assert m.shape_node.has_field_5() is False


def test_api_node_contains_case_001():
    """in 问的是同一件事。"""
    m = _materials()
    assert "code_length" in m.demo_node
    assert "$code_length" in m.demo_node
    assert "nope" not in m.demo_node


def test_api_node_try_get_case_001():
    """try_get_{name}()：取到给 Ok(node)，问不出来给 Err。"""
    m = _materials()
    assert isinstance(m.demo_node.try_get_code_length(), Ok)
    assert isinstance(m.demo_node.try_get_nope(), Err)


def test_api_node_try_get_case_002():
    """try_get_{name}()：数据里没有给 Ok(nil)。"""
    m = _materials()
    sparse = m._without(m.demo_node, "$code_length")
    given = sparse.try_get_code_length()
    assert isinstance(given, Ok) and given.ok_value is None


def test_api_node_try_get_case_003():
    """try_get_field_{i}()：位置越界给 Err。"""
    m = _materials()
    assert isinstance(m.shape_node.try_get_field_5(), Err)


def test_api_node_attribute_case_001():
    """不是我们的接口名：老老实实 AttributeError。"""
    m = _materials()
    try:
        m.demo_node.nope
        raise AssertionError("本该 AttributeError")
    except AttributeError:
        pass


def test_api_node_dir_case_001():
    """dir() 看得见合成出来的名字。"""
    m = _materials()
    names = dir(m.demo_node)
    for wanted in ("get_code_length", "has_code_length", "try_get_code_length", "value"):
        assert wanted in names, wanted


# ----------------------------------------------------------------------
# 节点上的容器写法：[] / len() / 迭代 / keys / values / items
# ----------------------------------------------------------------------


def test_api_node_getitem_case_001():
    """node[i] 是 at_index，node[key] 是 at_key。"""
    m = _materials()
    assert m.node1_node.get_items()[0].value == 1
    assert m.node1_node.get_table()["m"].value == 2


def test_api_node_len_case_001():
    """len(node) 对 list / dict 都给个数。"""
    m = _materials()
    assert len(m.node1_node.get_items()) == 2
    assert len(m.node1_node.get_table()) == 2


def test_api_node_len_case_002():
    """不是容器：len(node) 抛。"""
    m = _materials()
    try:
        len(m.demo_node.get_code_length())
        raise AssertionError("本该抛")
    except VibaReflectError:
        pass


def test_api_node_iter_case_001():
    """迭代就是 len 加 at_index 走一遍。"""
    m = _materials()
    assert [item.value for item in m.node1_node.get_items()] == [1, 2]


def test_api_node_keys_case_001():
    """keys() / values() / items()。"""
    m = _materials()
    table = m.node1_node.get_table()
    assert table.keys() == ["k", "m"]
    assert [v.value for v in table.values()] == [1, 2]
    assert [(k, v.value) for k, v in table.items()] == [("k", 1), ("m", 2)]


# ----------------------------------------------------------------------
# 节点上的两栏与地址
# ----------------------------------------------------------------------


def test_api_node_fields_case_001():
    """节点的两栏：descriptor 是设计侧那一段，data 是材料那一段。

    设计侧写的是 Metric[CodeLength]（定义不动），材料那边是 $value 那一层。
    """
    m = _materials()
    node = m.demo_node.by_tag("code_length")
    assert node.descriptor.kind == "type_app"
    assert isinstance(node.data, viba_ast.Tagged) and node.data.tag == "$value"
    assert isinstance(node.by_tag("value").data, viba_ast.Constant)


def test_api_node_path_case_001():
    """path 是从起点走到这里的 VibaPath。"""
    m = _materials()
    assert m.demo_node.path == ()
    assert m.demo_node.get_coverage().get_value().get_total_lines().path == (
        by_tag("$coverage"), by_tag("$value"), by_tag("$total_lines"))


# ----------------------------------------------------------------------
# 链式调用（第 7.4 节那套写法）
# ----------------------------------------------------------------------


def test_api_chain_case_001():
    """Viba 层：by_tag → at_index / at_key → leaf，一路点下去。"""
    m = _materials()
    assert m.node1_node.by_tag("items").at_index(1).leaf == 2
    assert m.node1_node.by_tag("table").at_key("m").leaf == 2


def test_api_chain_case_002():
    """Python 落点：get_ → get_ → .value；get_ → [] → .value。"""
    m = _materials()
    assert m.demo_node.get_coverage().get_value().get_total_lines().value == 11
    assert m.demo_node.get_keywords().get_value()[1].value == "b"
    assert m.node1_node.get_table()["m"].value == 2


def test_api_chain_case_003():
    """第 7.4 节那段写法：先问有没有，再迭代，再读元素。"""
    m = _materials()
    seen = []
    if "keywords" in m.demo_node:
        for item in m.demo_node.get_keywords().get_value():
            seen.append(item.value)
    assert seen == ["a", "b"]


def test_api_chain_case_005():
    """三层 TypeRef 的链：Metric[GroupSpec] → $value → GroupSpec → Bucket → Slot。"""
    m = _materials()
    node = m.rule40_node
    assert node.by_tag("group_spec").by_tag("value").by_tag("bucket") \
        .at_index(0).by_tag("name").leaf == "slot0"


def test_api_chain_case_006():
    """同一条三层链走 VibaGetByPath；走到容器元素上还不是叶子。"""
    m = _materials()
    path = [by_tag("$group_spec"), by_tag("$value"), by_tag("$bucket"),
            at_index(0), by_tag("$name")]
    assert reflect.access.get_by_path(m.rule40_node, path).ok_value == "slot0"
    assert isinstance(reflect.access.resolve(m.rule40_node, path[:-1]), Ok)
    assert isinstance(reflect.access.get_by_path(m.rule40_node, path[:-1]), Err)


def test_api_chain_case_004():
    """链上断掉就抛，不给半截节点。"""
    m = _materials()
    try:
        m.node1_node.get_table()["nope"]
        raise AssertionError("本该抛")
    except VibaReflectError:
        pass
    try:
        m.demo_node.get_coverage().get_missing()
        raise AssertionError("本该抛")
    except VibaReflectError:
        pass


# ----------------------------------------------------------------------
# VibaResolve / VibaGetByPath
# ----------------------------------------------------------------------


def test_api_resolve_case_001():
    """一条路径走到底，给出那个节点，path 记着走过哪儿。"""
    m = _materials()
    path = [by_tag("$coverage"), by_tag("$value"), by_tag("$documented_lines")]
    resolved = reflect.access.resolve(m.demo_node, path)
    assert isinstance(resolved, Ok)
    assert resolved.ok_value.path == tuple(path)
    assert resolved.ok_value.value == 5


def test_api_resolve_case_002():
    """容器路径：at_index / at_key。"""
    m = _materials()
    assert reflect.access.resolve(m.node1_node, [by_tag("$items"), at_index(1)]).ok_value.value == 2
    assert reflect.access.resolve(m.node1_node, [by_tag("$table"), at_key("k")]).ok_value.value == 1


def test_api_resolve_case_003():
    """路径断了：Err。"""
    m = _materials()
    assert isinstance(reflect.access.resolve(m.demo_node, [by_tag("$nope")]), Err)


def test_api_resolve_case_004():
    """Ending on a missing piece: the path stops there with Ok(nil)."""
    m = _materials()
    sparse = m._without(m.demo_node, "$keywords")
    given = reflect.access.resolve(sparse, [by_tag("$keywords")])
    assert isinstance(given, Ok) and given.ok_value is None


def _bare_sum_material(source: str, outer: str, witness: str):
    """A design built from an inline source: an untagged sum a material skips."""
    pool = empty_pool()
    parsed = parse_viba_file(pool, source, "bare_sum.viba", "bare_sum")
    assert isinstance(parsed, Ok), parsed
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    definition = _definition_of(pool, f"bare_sum.{outer}")
    body = {d.name: d for d in viba_ast.parse(source).body}[witness].body
    return reflect.access, reflect.access.root(definition, Witness(body)).ok_value


def test_api_resolve_case_006():
    """An untagged sum with one inner node may be skipped by the material."""
    source = """Inner := Oneof | $a int | $b str
Outer := Object * $x (Inner | nil)
BareWitness := Object * $x ($a 1)
"""
    accessor, node = _bare_sum_material(source, "Outer", "BareWitness")
    x = node.by_tag("$x")
    assert accessor.has(x, by_field_index(0)).ok_value is True
    assert reflect.access.get_by_path(node, [by_tag("$x"), by_field_index(0), by_tag("$a")]).ok_value == 1
    assert reflect.access.get_by_path(node, [by_tag("$x"), by_tag("$a")]).ok_value == 1
    assert accessor.has(x, by_field_index(1)).ok_value is False  # the nil branch is not taken
    assert isinstance(reflect.access.get_by_path(node, [by_tag("$x"), by_field_index(1)]), Err)


def test_api_resolve_case_007():
    """Leaf branches are told apart by the value; two inner nodes keep the layer."""
    leafy = """Leafy := Object * $x (int | str)
LeafyWitness := Object * $x 7
Optional := Object * $x (int | nil)
NilWitness := Object * $x nil
"""
    accessor, node = _bare_sum_material(leafy, "Leafy", "LeafyWitness")
    x = node.by_tag("$x")
    assert accessor.has(x, by_field_index(0)).ok_value is True
    assert reflect.access.get_by_path(node, [by_tag("$x"), by_field_index(0)]).ok_value == 7
    assert accessor.has(x, by_field_index(1)).ok_value is False

    accessor, node = _bare_sum_material(leafy, "Optional", "NilWitness")
    x = node.by_tag("$x")
    assert accessor.has(x, by_field_index(0)).ok_value is False
    assert reflect.access.get_by_path(node, [by_tag("$x"), by_field_index(1)]).ok_value is None

    two = """A := Object * $a int
B := Object * $b str
Two := Object * $x (A | B)
TwoWitness := Object * $x ($a 1)
"""
    accessor, node = _bare_sum_material(two, "Two", "TwoWitness")
    assert accessor.has(node.by_tag("$x"), by_field_index(0)).ok_value is False


def test_api_resolve_case_005():
    """A missing piece with steps left: Err, never a None walked on."""
    m = _materials()
    sparse = m._without(m.demo_node, "$keywords")
    given = reflect.access.resolve(sparse, [by_tag("$keywords"), at_index(0)])
    assert isinstance(given, Err) and given.err_msg == "this step has no value"
    assert isinstance(reflect.access.get_by_path(sparse, [by_tag("$keywords"), at_index(0)]), Err)


def test_api_get_by_path_case_001():
    """先 resolve 再 leaf：一条路径直接读出值。"""
    m = _materials()
    assert reflect.access.get_by_path(m.demo_node, [by_tag("$code_length"), by_tag("$value")]).ok_value == 7
    assert reflect.access.get_by_path(m.demo_node,
                     [by_tag("$keywords"), by_tag("$value"), at_index(1)]).ok_value == "b"
    assert reflect.access.get_by_path(m.node1_node, [by_tag("$table"), at_key("k")]).ok_value == 1


def test_api_get_by_path_case_002():
    """路径断了：Err。"""
    m = _materials()
    assert isinstance(reflect.access.get_by_path(m.demo_node,
                                [by_tag("$keywords"), by_tag("$value"), at_index(9)]), Err)


def test_api_get_by_path_case_003():
    """中途没有值：Err。"""
    m = _materials()
    sparse = m._without(m.demo_node, "$keywords")
    assert isinstance(reflect.access.get_by_path(sparse, [by_tag("$keywords")]), Err)


def test_api_get_by_path_case_004():
    """尽头不是叶子：Err（resolve 到得了，leaf 读不出）。"""
    m = _materials()
    assert isinstance(reflect.access.resolve(m.demo_node, [by_tag("$coverage")]), Ok)
    assert isinstance(reflect.access.get_by_path(m.demo_node, [by_tag("$coverage")]), Err)


# ----------------------------------------------------------------------
# VibaListFields
# ----------------------------------------------------------------------


def test_api_list_fields_case_001():
    """按 DefinitionMembers 的顺序列出来。"""
    m = _materials()
    listed = reflect.access.list_fields(m.demo_node, m.demo)
    assert isinstance(listed, Ok)
    assert [n.path[-1].value for n in listed.ok_value] == [
        "$code_length", "$coverage", "$keywords", "$assert_code_len_le_24"]


def test_api_list_fields_case_002():
    """数据里缺的字段不进表。"""
    m = _materials()
    sparse = m._without(m.demo_node, "$code_length")
    listed = reflect.access.list_fields(sparse, m.demo)
    assert [n.path[-1].value for n in listed.ok_value] == [
        "$coverage", "$keywords", "$assert_code_len_le_24"]


def test_api_list_fields_case_003():
    """这一段没有成员时是空表，不是 Err。"""
    m = _materials()
    listed = reflect.access.list_fields(m.demo_node.by_tag("code_length"), m.demo)
    assert isinstance(listed, Ok) and listed.ok_value == []


# ----------------------------------------------------------------------
# 本层的遍历（体检表用）
# ----------------------------------------------------------------------


def test_api_walk_case_001():
    """走出来的坐标不重不漏；字面量都读得出。"""
    m = _materials()
    walked = m.demo_access._walk(m.demo_node)
    paths = [n.path for n in walked]
    assert len(paths) == len(set(map(repr, paths)))
    assert () in paths
    assert (by_tag("$coverage"), by_tag("$value"), by_tag("$total_lines")) in paths
    assert (by_tag("$keywords"), by_tag("$value"), at_index(0)) in paths
    leaves = [m.demo_access.leaf(n).ok_value for n in walked
              if isinstance(n.data, (viba_ast.Constant, viba_ast.Nil))]
    assert leaves == [7, 5, 11, "a", "b"]


def test_api_walk_case_002():
    """容器那一支的坐标也在图上。"""
    m = _materials()
    paths = [n.path for n in m.node1_access._walk(m.node1_node)]
    assert (by_tag("$table"), at_key("m")) in paths
    assert (by_tag("$seen"), at_index(0)) in paths


# ----------------------------------------------------------------------
# VibaStep
# ----------------------------------------------------------------------


def test_api_step_case_001():
    """相等与不等（跨支也不等）。"""
    assert by_tag("$count") == by_tag("$count")
    assert by_tag("$count") != by_tag("$ratio")
    assert by_tag("$count") != at_index(0)


def test_api_step_case_002():
    """可哈希。"""
    assert hash(by_tag("$count")) == hash(by_tag("$count"))
    assert len({by_tag("$count"), at_index(1), at_key("k")}) == 3


def test_api_step_case_003():
    """可打印。"""
    assert "$count" in repr(by_tag("$count"))
    assert repr(at_key("k")) == "at_key('k')"


# ----------------------------------------------------------------------
# Data 的绑定
# ----------------------------------------------------------------------


def test_api_witness_case_001():
    """也收 AstNodeType：拆出它的 ast 节点。"""
    m = _materials()
    typed = Witness(AstNodeType(m.demo_witness.node, custom_module("")))
    assert typed.node is m.demo_witness.node


# ----------------------------------------------------------------------
# 写材料要用的两块：成员换算成步子、槽位让不让 nil
# ----------------------------------------------------------------------


def test_api_member_steps_case_001():
    """有 tag 的按 tag、没 tag 的按位置，顺序就是成员的顺序。"""
    m = _materials()
    steps = reflect.access.member_steps(m.demo_node)
    assert [tag for tag, _, _ in steps] == [
        "$code_length", "$coverage", "$keywords", "$assert_code_len_le_24"]
    assert [repr(step) for _, step, _ in steps] == [
        "by_tag('$code_length')", "by_tag('$coverage')", "by_tag('$keywords')",
        "by_tag('$assert_code_len_le_24')"]


def test_api_member_steps_case_002():
    """位置成员自己数自己的：tag 的不占号。"""
    m = _materials()
    steps = reflect.access.member_steps(m.shape_node)
    assert [(tag, repr(step)) for tag, step, _ in steps] == [
        (None, "by_field_index(0)"), (None, "by_field_index(1)")]


def test_api_member_steps_case_003():
    """拿到的步子就是 VibaGet 收的步子，直接走得通。"""
    m = _materials()
    for _, step, _ in reflect.access.member_steps(m.demo_node):
        assert isinstance(reflect.access.get(m.demo_node, step), Ok)


def test_api_member_steps_case_004():
    """不是积/和的那一段没有成员，步子表是空的。"""
    m = _materials()
    assert reflect.access.member_steps(m.node1_node.by_tag("items")) == []


def test_api_member_steps_case_005():
    """不带标签的积成员是内联位：它的 tag 直接列出来，不带那一跳。"""
    pool = empty_pool()
    parsed = parse_viba_file(pool, "A := $x int * $y str\nB := A * $z bool\n",
                             "inline.viba", "inline")
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    definition = pool_find_definition(pool, "inline.B").ok_value
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$x", viba_ast.Constant(1)),
        viba_ast.Tagged("$y", viba_ast.Constant("s")),
        viba_ast.Tagged("$z", viba_ast.Constant(True)),
    ])
    node = reflect.access.root(definition, Witness(body)).ok_value
    steps = reflect.access.member_steps(node)
    assert [tag for tag, _, _ in steps] == ["$x", "$y", "$z"]
    assert [repr(step) for _, step, _ in steps] == [
        "by_tag('$x')", "by_tag('$y')", "by_tag('$z')"]
    assert reflect.access.get_by_path(node, [by_tag("$x")]).ok_value == 1


def test_api_member_steps_case_006():
    """单位元不是成员：写在头里、写在中间、藏在名字后面，都不算。"""
    pool = empty_pool()
    parsed = parse_viba_file(pool, "U := Object\n"
                                   "Box := Object * U * $a int * Object * $b str\n",
                             "units.viba", "units")
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    definition = pool_find_definition(pool, "units.Box").ok_value
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"), viba_ast.Nil(),
        viba_ast.Tagged("$a", viba_ast.Constant(1)), viba_ast.Nil(),
        viba_ast.Tagged("$b", viba_ast.Constant("s")),
    ])
    node = reflect.access.root(definition, Witness(body)).ok_value
    assert [tag for tag, _, _ in reflect.access.member_steps(node)] == ["$a", "$b"]
    assert reflect.access.get_by_path(node, [by_tag("$b")]).ok_value == "s"


def test_api_list_fields_case_004():
    """内联进来的字段也在表里，位置按摊开后的算。"""
    pool = empty_pool()
    parsed = parse_viba_file(pool, "A := $x int * $y str\nB := int * A * $z bool\n",
                             "fields.viba", "fields")
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    definition = pool_find_definition(pool, "fields.B").ok_value
    body = viba_ast.ProductChain([
        viba_ast.Constant(7),
        viba_ast.Tagged("$x", viba_ast.Constant(1)),
        viba_ast.Tagged("$y", viba_ast.Constant("s")),
        viba_ast.Tagged("$z", viba_ast.Constant(True)),
    ])
    node = reflect.access.root(definition, Witness(body)).ok_value
    listed = reflect.access.list_fields(node, definition)
    assert isinstance(listed, Ok)
    assert [repr(n.path[-1]) for n in listed.ok_value] == [
        "by_field_index(0)", "by_tag('$x')", "by_tag('$y')", "by_tag('$z')"]


def test_api_inline_cycle_case_001():
    """内联成环要问得出来：直接、绕别名、绕两个定义都算。"""
    pool = empty_pool()
    parsed = parse_viba_file(pool, "A := A * $x int\n"
                                   "Alias := Tail * $x int\nTail := Alias\n"
                                   "Mutual := Other * $m int\nOther := Mutual * $o int\n"
                                   "Fine := Inner * $x int\nInner := Object * $y int\n",
                             "cycles.viba", "cycles")
    pool = pool_add_file(pool, parsed.ok_value).ok_value

    def cycle_of(name):
        definition = pool_find_definition(pool, f"cycles.{name}").ok_value
        return reflect.access.inline_cycle(definition.body)

    assert cycle_of("A") == "A"
    assert cycle_of("Alias") == "Tail"
    assert cycle_of("Mutual") == "Other"
    assert cycle_of("Fine") is None
    assert reflect.access.inline_cycle(
        pool_find_definition(pool, "cycles.Inner").ok_value.body) is None


def test_api_carries_nil_case_001():
    """和里有一支是 nil：这个槽位让 nil。"""
    m = _materials()
    body = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$maybe", viba_ast.Nil()),
    ])
    node = m.node1_access.root(m.node1, Witness(body)).ok_value
    assert reflect.access.carries_nil(node.get_maybe().descriptor) is True


def test_api_carries_nil_case_002():
    """不是和就不让 nil。"""
    m = _materials()
    assert reflect.access.carries_nil(m.demo_node.descriptor) is False


def test_api_carries_nil_case_003():
    """别名与泛型应用先展开再看。"""
    pool = empty_pool()
    parsed = parse_viba_file(pool, "Maybe[T] := T | nil\nPlain[T] := T\n"
                                   "Holder := Object * $a Maybe[int] * $b Plain[int]\n",
                             "holder.viba", "holder")
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    definition = pool_find_definition(pool, "holder.Holder").ok_value
    node = reflect.access.root(definition, Witness(viba_ast.TypeRef("Object"))).ok_value
    fields = {tag: descriptor
              for tag, _, descriptor in reflect.access.member_steps(node)}
    assert reflect.access.carries_nil(fields["$a"]) is True
    assert reflect.access.carries_nil(fields["$b"]) is False


def test_api_carries_nil_case_004():
    """容器是名字而不是和，不让 nil。"""
    m = _materials()
    assert reflect.access.carries_nil(m.node1_node.get_items().descriptor) is False


# ----------------------------------------------------------------------
# 跑
# ----------------------------------------------------------------------


def run():
    cases = [value for name, value in globals().items()
             if name.startswith("test_api_") and callable(value)]
    for case in cases:
        case()
    print(f"rule_reflect_api: {len(cases)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())

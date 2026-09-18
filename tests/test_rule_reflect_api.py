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
    empty_pool,
    parse_viba_file,
    pool_add_file,
    pool_find_definition,
)
from viba.rule.reflect import (
    VibaReflectError,
    Witness,
    access,
    at_index,
    at_key,
    by_field_index,
    by_tag,
    viba_list_fields,
    viba_get_by_path,
    viba_resolve,
    viba_version_matches,
)

_RULES = Path(__file__).resolve().parent / "data" / "rule_coding_style_check"
_NOT_RULES = _RULES / "not_rules"
_NOT_CASES = Path(__file__).resolve().parent / "data" / "is_sub_type" / "not"
_CASES = Path(__file__).resolve().parent / "data" / "type_descriptor" / "case_000"


# ----------------------------------------------------------------------
# 材料
# ----------------------------------------------------------------------


def _load(path: Path, module_name: str):
    source = path.read_text()
    pool = empty_pool()
    parsed = parse_viba_file(pool, source, path.name, module_name)
    assert isinstance(parsed, Ok), parsed
    return pool_add_file(pool, parsed.value).value


def _definition_of(pool, full_name: str):
    found = pool_find_definition(pool, full_name)
    assert isinstance(found, Ok), found
    return found.value


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
        self.demo_access = access(self.demo)
        self.demo_witness = Witness(_demo_body(), {self.demo.file_hash})
        self.demo_node = self.demo_access.root(self.demo_witness).value

        top = _load(_CASES / "top.viba", "top")
        self.shape = _definition_of(top, "top.Shape0")
        self.shape_access = access(self.shape)
        self.shape_node = self.shape_access.root(Witness(
            viba_ast.Product(viba_ast.Constant(1), viba_ast.Constant("two")),
            {self.shape.file_hash})).value
        self.node1 = _definition_of(top, "top.Node1")
        self.node1_access = access(self.node1)
        self.node1_node = self.node1_access.root(
            Witness(_node1_body(), {self.node1.file_hash})).value

        not_rule = _load(_RULES / "not_rule.viba", "not_rule")
        self.not_rule = _definition_of(not_rule, "not_rule.NoDeathPenaltyRule")
        self.not_access = access(self.not_rule)
        self.law_abiding = self.not_access.root(
            Witness(_body_of(_RULES / "not_rule.viba", "LawAbidingWitness"),
                    {self.not_rule.file_hash})).value

        named = _load(_NOT_RULES / "not_rule03.viba", "not_rule03")
        self.named_not = _definition_of(named, "not_rule03.NotRule03")
        self.named_not_access = access(self.named_not)
        self.named_not_node = self.named_not_access.root(
            Witness(_named_operand_body(), {self.named_not.file_hash})).value

        written = _load(_NOT_CASES / "sup061.viba", "sup061")
        self.written_not = _definition_of(written, "sup061.F6")
        self.written_not_access = access(self.written_not)
        self.written_not_node = self.written_not_access.root(
            Witness(_written_operand_body(), {self.written_not.file_hash})).value

        rule40 = _load(_RULES / "rules" / "rule40.viba", "rule40")
        self.rule40 = _definition_of(rule40, "rule40.Rule40")
        self.rule40_access = access(self.rule40)
        self.rule40_node = self.rule40_access.root(
            Witness(rule40_body(), {self.rule40.file_hash})).value

        sum_rule = _load(_RULES / "sum_rule.viba", "sum_rule")
        self.sum_rule = _definition_of(sum_rule, "sum_rule.SumRule")
        self.sum_access = access(self.sum_rule)
        self.sum_pass = self.sum_access.root(
            Witness(_body_of(_RULES / "sum_rule.viba", "SumWitnessPass"),
                    {self.sum_rule.file_hash})).value

    def _without(self, node, tag):
        """把某个 tag 摘掉的那份材料（造"数据里没有"用）。"""
        kept = [e for e in viba_ast.convert_to_chain_style(node.data).elements
                if not (isinstance(e, viba_ast.Tagged) and e.tag == tag)]
        return node._access.root(Witness(viba_ast.ProductChain(kept),
                                         {node._access.definition.file_hash})).value


_MATERIALS = None


def _materials():
    global _MATERIALS
    if _MATERIALS is None:
        _MATERIALS = _Materials()
    return _MATERIALS


# ----------------------------------------------------------------------
# VibaRoot
# ----------------------------------------------------------------------


def test_api_root_case_001():
    """声明的版本里有这一版：给出起点。"""
    m = _materials()
    rooted = m.demo_access.root(m.demo_witness)
    assert isinstance(rooted, Ok)
    assert rooted.value.descriptor is m.demo.body


def test_api_root_case_002():
    """起点不核版本：数据没声明版本照样给节点，核不核是上层的事。"""
    m = _materials()
    rooted = m.demo_access.root(Witness(m.demo_witness.node, set()))
    assert isinstance(rooted, Ok) and rooted.value.data is m.demo_witness.node


def test_api_root_case_003():
    """版本对不上也给节点；要拦就上层拿 file_hashes / viba_version_matches 自己拦。"""
    m = _materials()
    rooted = m.demo_access.root(Witness(m.demo_witness.node, {"deadbeef"}))
    assert isinstance(rooted, Ok)
    assert m.demo_access.file_hashes(Witness(m.demo_witness.node, {"deadbeef"})).value == {"deadbeef"}
    assert viba_version_matches(m.demo.pool, Witness(m.demo_witness.node, {"deadbeef"})).value is False


def test_api_root_case_004():
    """声明里多带一个哈希不算错。"""
    m = _materials()
    rooted = m.demo_access.root(Witness(m.demo_witness.node, {m.demo.file_hash, "x"}))
    assert isinstance(rooted, Ok)


# ----------------------------------------------------------------------
# VibaFileHashes
# ----------------------------------------------------------------------


def test_api_file_hashes_case_001():
    """原样给出数据声明的集合。"""
    m = _materials()
    data = Witness(m.demo_witness.node, {m.demo.file_hash, "other"})
    assert m.demo_access.file_hashes(data).value == {m.demo.file_hash, "other"}


# ----------------------------------------------------------------------
# VibaVersionMatches
# ----------------------------------------------------------------------


def test_api_version_matches_case_001():
    """数据声明的都在池子里：真。"""
    m = _materials()
    assert viba_version_matches(m.demo.pool, m.demo_witness).value is True


def test_api_version_matches_case_002():
    """有一个找不到：假。"""
    m = _materials()
    data = Witness(m.demo_witness.node, {m.demo.file_hash, "nope"})
    assert viba_version_matches(m.demo.pool, data).value is False


# ----------------------------------------------------------------------
# VibaHas
# ----------------------------------------------------------------------


def test_api_has_case_001():
    """数据里有这一段：真。"""
    m = _materials()
    assert m.demo_access.has(m.demo_node, by_tag("$code_length")).value is True


def test_api_has_case_002():
    """图上没这个坐标：假（不是 Err）。"""
    m = _materials()
    assert m.demo_access.has(m.demo_node, by_tag("$missing")).value is False


def test_api_has_case_003():
    """数据里没这一段：假。"""
    m = _materials()
    sparse = m._without(m.demo_node, "$code_length")
    assert m.demo_access.has(sparse, by_tag("$code_length")).value is False


def test_api_has_case_004():
    """和类型：只有选中的那一支是真的。"""
    m = _materials()
    assert m.sum_access.has(m.sum_pass, by_tag("$small")).value is True
    assert m.sum_access.has(m.sum_pass, by_tag("$big")).value is False


def test_api_has_case_005():
    """not[A]：$not_operand 是一个坐标（和积/和一样），分支在它下面。"""
    m = _materials()
    shell = m.law_abiding.by_tag("not_crimes")
    assert m.not_access.has(shell, by_tag("$not_operand")).value is True
    assert m.not_access.has(shell, by_tag("$homicide")).value is False
    operand = shell.get_not_operand()
    for branch in ("$homicide", "$arson", "$robbery"):
        assert m.not_access.has(operand, by_tag(branch)).value is True


def test_api_has_case_006():
    """三种链一个规矩：$elements 就是成员，链头的单位元不算。"""
    m = _materials()
    metric = m.demo_node.by_tag("code_length")
    assert m.demo_access.has(metric, by_tag("$value")).value is True
    check = m.demo_node.by_tag("assert_code_len_le_24")
    assert m.demo_access.has(check, by_tag("$assert_cond")).value is True
    assert m.demo_access.has(check, by_tag("$assert_python_code")).value is True
    assert m.demo_access.has(check, by_tag("$never_there")).value is False


def test_api_has_case_007():
    """同一份定义在两边都展开：设计和材料的成员对得上。"""
    m = _materials()
    shell = m.law_abiding.by_tag("not_crimes")
    operand = shell.get_not_operand()
    assert m.not_access.has(operand, by_tag("$not_operand")).value is False
    assert operand.descriptor.kind == "sum"


def test_api_has_case_008():
    """积 / 和 / 指数三种链的成员读法一致：$elements 是成员，链头单位元不算。"""
    m = _materials()
    # 积：字段是成员，链头的 RuleObject 不是
    assert m.demo_node.has_code_length() is True
    assert m.demo_access.has(m.demo_node, by_tag("$RuleObject")).value is False
    assert "get_ruleobject" not in dir(m.demo_node)
    # 和：分支是成员
    assert m.sum_access.has(m.sum_pass, by_tag("$small")).value is True
    # 指数：操作数是成员，链头的 never 不是
    shell = m.law_abiding.by_tag("not_crimes")
    assert m.not_access.has(shell, by_tag("$not_operand")).value is True
    assert m.not_access.has(shell, by_tag("$never")).value is False
    assert "get_never" not in dir(shell)


def test_api_has_case_009():
    """禁止的操作数在材料里写成名字（not[Crimes]）时，也展开成定义体。"""
    m = _materials()
    shell = m.named_not_node.by_tag("not_crimes")
    assert m.named_not_access.has(shell, by_tag("$not_operand")).value is True
    operand = shell.get_not_operand()
    for branch in ("$homicide", "$arson", "$robbery"):
        assert m.named_not_access.has(operand, by_tag(branch)).value is True


# ----------------------------------------------------------------------
# VibaGet
# ----------------------------------------------------------------------


def test_api_get_case_001():
    """走一步：回来的是新节点，描述符与数据都换到那一段。

    量出来的值在 $value 那一层，所以坐标是 $code_length → $value。
    """
    m = _materials()
    given = m.demo_access.get(m.demo_node, by_tag("$code_length"))
    assert isinstance(given, Ok)
    assert isinstance(given.value.data, viba_ast.Tagged)
    inside = m.demo_access.get(given.value, by_tag("$value"))
    assert isinstance(inside, Ok) and inside.value.data.value == 7
    assert given.value.descriptor is not m.demo_node.descriptor


def test_api_get_case_002():
    """数据里没有这一段：Ok(nil)。"""
    m = _materials()
    sparse = m._without(m.demo_node, "$keywords")
    given = m.demo_access.get(sparse, by_tag("$keywords"))
    assert isinstance(given, Ok) and given.value is None


def test_api_get_case_003():
    """图上压根没这个坐标：Err。"""
    m = _materials()
    assert isinstance(m.demo_access.get(m.demo_node, by_tag("$missing")), Err)


def test_api_get_case_004():
    """和类型取没选中的那一支：Ok(nil)。"""
    m = _materials()
    given = m.sum_access.get(m.sum_pass, by_tag("$big"))
    assert isinstance(given, Ok) and given.value is None


# ----------------------------------------------------------------------
# VibaLeaf
# ----------------------------------------------------------------------


def test_api_leaf_case_001():
    """int 字面量读得出值：量出来的值是 $value 那一层的字面量。"""
    m = _materials()
    inside = m.demo_node.by_tag("code_length").by_tag("value")
    assert m.demo_access.leaf(inside).value.value == 7


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
    node = m.node1_access.root(Witness(body, {m.node1.file_hash})).value
    assert m.node1_access.leaf(node.by_tag("items")).value.value == 1.5
    assert m.node1_access.leaf(node.by_tag("seen")).value.value is True
    assert m.node1_access.leaf(node.by_tag("table")).value.value == "x"


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
    node = m.node1_access.root(Witness(body, {m.node1.file_hash})).value
    leaf = m.node1_access.leaf(node.by_tag("maybe"))
    assert isinstance(leaf, Ok) and leaf.value.kind == "nil"


def test_api_leaf_case_004():
    """这一段不是叶子：Err。"""
    m = _materials()
    assert isinstance(m.demo_access.leaf(m.demo_node.by_tag("coverage")), Err)


# ----------------------------------------------------------------------
# VibaLength
# ----------------------------------------------------------------------


def test_api_length_case_001():
    """list 给元素个数。"""
    m = _materials()
    assert m.node1_access.length(m.node1_node.by_tag("items")).value == 2


def test_api_length_case_002():
    """set 给元素个数。"""
    m = _materials()
    assert m.node1_access.length(m.node1_node.by_tag("seen")).value == 1


def test_api_length_case_003():
    """dict 给键数。"""
    m = _materials()
    assert m.node1_access.length(m.node1_node.by_tag("table")).value == 2


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
    assert m.node1_access.keys(m.node1_node.by_tag("table")).value == ["k", "m"]


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
    assert m.demo_node.by_tag("code_length").by_tag("value").value.value == 7
    assert m.demo_node.by_tag("$code_length").by_tag("$value").value.value == 7


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
    assert m.shape_node.by_field_index(0).value.value == 1
    assert m.shape_node.by_field_index(1).value.value == "two"


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
    assert m.node1_node.by_tag("items").at_index(1).value.value == 2


def test_api_node_at_index_case_002():
    """set 也按下标取（顺序由实现定）。"""
    m = _materials()
    assert m.node1_node.by_tag("seen").at_index(0).value.value == "x"


def test_api_node_at_index_case_003():
    """下标越界就抛。"""
    m = _materials()
    try:
        m.node1_node.by_tag("items").at_index(9)
        raise AssertionError("本该抛")
    except VibaReflectError:
        pass


def test_api_node_at_key_case_001():
    """dict 按键取。"""
    m = _materials()
    assert m.node1_node.by_tag("table").at_key("m").value.value == 2


def test_api_node_at_key_case_002():
    """没有这个键就抛。"""
    m = _materials()
    try:
        m.node1_node.by_tag("table").at_key("nope")
        raise AssertionError("本该抛")
    except VibaReflectError:
        pass


def test_api_node_leaf_case_001():
    """leaf 与 Python 落点 value 是同一个东西。"""
    m = _materials()
    node = m.demo_node.by_tag("code_length").by_tag("value")
    assert node.leaf.value == 7 and node.value.value == 7


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
    assert m.demo_node.get_code_length().get_value().value.value == 7
    assert m.demo_node.get_coverage().get_value().get_documented_lines().value.value == 5


def test_api_node_get_case_002():
    """get_field_{i}() 按位置取值。"""
    m = _materials()
    assert m.shape_node.get_field_0().value.value == 1
    assert m.shape_node.get_field_1().value.value == "two"


def test_api_node_get_case_003():
    """设计直接写 never <- $not_operand A 时也一样：get_not_operand() 取到 A，再往下取叶子。"""
    m = _materials()
    node = m.written_not_node
    assert m.written_not_access.has(node, by_tag("$not_operand")).value is True
    operand = node.get_not_operand()
    assert operand.by_tag("a").leaf.value == 3
    assert operand.get_a().value.value == 3


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
    assert isinstance(given, Ok) and given.value is None


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
    assert m.node1_node.get_items()[0].value.value == 1
    assert m.node1_node.get_table()["m"].value.value == 2


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
    assert [item.value.value for item in m.node1_node.get_items()] == [1, 2]


def test_api_node_keys_case_001():
    """keys() / values() / items()。"""
    m = _materials()
    table = m.node1_node.get_table()
    assert table.keys() == ["k", "m"]
    assert [v.value.value for v in table.values()] == [1, 2]
    assert [(k, v.value.value) for k, v in table.items()] == [("k", 1), ("m", 2)]


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
    assert m.node1_node.by_tag("items").at_index(1).leaf.value == 2
    assert m.node1_node.by_tag("table").at_key("m").leaf.value == 2


def test_api_chain_case_002():
    """Python 落点：get_ → get_ → .value；get_ → [] → .value。"""
    m = _materials()
    assert m.demo_node.get_coverage().get_value().get_total_lines().value.value == 11
    assert m.demo_node.get_keywords().get_value()[1].value.value == "b"
    assert m.node1_node.get_table()["m"].value.value == 2


def test_api_chain_case_003():
    """第 7.4 节那段写法：先问有没有，再迭代，再读元素。"""
    m = _materials()
    seen = []
    if "keywords" in m.demo_node:
        for item in m.demo_node.get_keywords().get_value():
            seen.append(item.value.value)
    assert seen == ["a", "b"]


def test_api_chain_case_005():
    """三层 TypeRef 的链：Metric[GroupSpec] → $value → GroupSpec → Bucket → Slot。"""
    m = _materials()
    node = m.rule40_node
    assert node.by_tag("group_spec").by_tag("value").by_tag("bucket") \
        .at_index(0).by_tag("name").leaf.value == "slot0"


def test_api_chain_case_006():
    """同一条三层链走 VibaGetByPath；走到容器元素上还不是叶子。"""
    m = _materials()
    path = [by_tag("$group_spec"), by_tag("$value"), by_tag("$bucket"),
            at_index(0), by_tag("$name")]
    assert viba_get_by_path(m.rule40_node, path).value.value == "slot0"
    assert isinstance(viba_resolve(m.rule40_node, path[:-1]), Ok)
    assert isinstance(viba_get_by_path(m.rule40_node, path[:-1]), Err)


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
    resolved = viba_resolve(m.demo_node, path)
    assert isinstance(resolved, Ok)
    assert resolved.value.path == tuple(path)
    assert resolved.value.value.value == 5


def test_api_resolve_case_002():
    """容器路径：at_index / at_key。"""
    m = _materials()
    assert viba_resolve(m.node1_node, [by_tag("$items"), at_index(1)]).value.value.value == 2
    assert viba_resolve(m.node1_node, [by_tag("$table"), at_key("k")]).value.value.value == 1


def test_api_resolve_case_003():
    """路径断了：Err。"""
    m = _materials()
    assert isinstance(viba_resolve(m.demo_node, [by_tag("$nope")]), Err)


def test_api_resolve_case_004():
    """中途没有值：Ok(nil)。"""
    m = _materials()
    sparse = m._without(m.demo_node, "$keywords")
    given = viba_resolve(sparse, [by_tag("$keywords")])
    assert isinstance(given, Ok) and given.value is None


def test_api_get_by_path_case_001():
    """先 resolve 再 leaf：一条路径直接读出值。"""
    m = _materials()
    assert viba_get_by_path(m.demo_node, [by_tag("$code_length"), by_tag("$value")]).value.value == 7
    assert viba_get_by_path(m.demo_node,
                     [by_tag("$keywords"), by_tag("$value"), at_index(1)]).value.value == "b"
    assert viba_get_by_path(m.node1_node, [by_tag("$table"), at_key("k")]).value.value == 1


def test_api_get_by_path_case_002():
    """路径断了：Err。"""
    m = _materials()
    assert isinstance(viba_get_by_path(m.demo_node,
                                [by_tag("$keywords"), by_tag("$value"), at_index(9)]), Err)


def test_api_get_by_path_case_003():
    """中途没有值：Err。"""
    m = _materials()
    sparse = m._without(m.demo_node, "$keywords")
    assert isinstance(viba_get_by_path(sparse, [by_tag("$keywords")]), Err)


def test_api_get_by_path_case_004():
    """尽头不是叶子：Err（resolve 到得了，leaf 读不出）。"""
    m = _materials()
    assert isinstance(viba_resolve(m.demo_node, [by_tag("$coverage")]), Ok)
    assert isinstance(viba_get_by_path(m.demo_node, [by_tag("$coverage")]), Err)


# ----------------------------------------------------------------------
# VibaListFields
# ----------------------------------------------------------------------


def test_api_list_fields_case_001():
    """按 DefinitionMembers 的顺序列出来。"""
    m = _materials()
    listed = viba_list_fields(m.demo_node, m.demo)
    assert isinstance(listed, Ok)
    assert [n.path[-1].value for n in listed.value] == [
        "$code_length", "$coverage", "$keywords", "$assert_code_len_le_24"]


def test_api_list_fields_case_002():
    """数据里缺的字段不进表。"""
    m = _materials()
    sparse = m._without(m.demo_node, "$code_length")
    listed = viba_list_fields(sparse, m.demo)
    assert [n.path[-1].value for n in listed.value] == [
        "$coverage", "$keywords", "$assert_code_len_le_24"]


def test_api_list_fields_case_003():
    """这一段没有成员时是空表，不是 Err。"""
    m = _materials()
    listed = viba_list_fields(m.demo_node.by_tag("code_length"), m.demo)
    assert isinstance(listed, Ok) and listed.value == []


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
    leaves = [m.demo_access.leaf(n).value.value for n in walked
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
    """版本声明存成集合。"""
    m = _materials()
    assert Witness(m.demo_witness.node, [m.demo.file_hash]).hashes == {m.demo.file_hash}


def test_api_witness_case_002():
    """也收 AstNodeType：拆出它的 ast 节点。"""
    m = _materials()
    typed = Witness(AstNodeType(m.demo_witness.node, custom_module("")),
                    {m.demo.file_hash})
    assert typed.node is m.demo_witness.node


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

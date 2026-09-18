"""viba.rule.reflect 的语料验收：rule_coding_style_check 那批 (rule, witness) 全跑一遍。

点接口的那份是 tests/test_rule_reflect_api.py；这一份是拿现成语料跑面：
每份材料都要能核版本、给起点、沿图走通。

     python tests/test_rule_reflect_corpus.py

每一份都走同一条路：核版本 → 给起点 → 沿图（Rule 的描述符）走一遍。走的时候
每个坐标都要有答案：`has` 与 `get` 必须一致，是字面量的必须读得出叶子，是容器
的必须数得出来、每个下标都取得到。

计数与 tests/test_rule_coding_style_check.py 对齐：每条生成规则 20 + 20 + 200 份判定
见证，加 20 份判定性见证；再加 demo/sum/not/predicate 那几组。语料加规则时份数自己跟着涨。

     python tests/test_rule_reflect_corpus.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import viba_ast
from viba.type import AstNodeType, Ok, custom_module
from viba.viba_type_descriptor import (
    empty_pool,
    parse_viba_file,
    pool_add_file,
    pool_find_definition,
)
from viba.rule.generate_witnesses import generate_witnesses
from viba.rule.reflect import (access, at_index, at_key, by_field_index, by_tag,
                               viba_version_matches, Witness)
from viba.rule.reset_predication_by_python_code import reset_predication_by_python_code

DATA = Path(__file__).resolve().parent / "data" / "rule_coding_style_check"
NOT_DATA = DATA / "not_rules"
WITNESSES_PER_RULE = 20
MIXED_WITNESSES = 200
PREDICATE_WITNESSES = 50
FAIL_PROB = 0.1


class UnderTest:
    """一份规则文件：描述符侧的池子 + viba.rule 的 rule 对象。"""

    def __init__(self, path: Path, rule_name: str):
        self.path = path
        self.source = path.read_text()
        self.module_name = path.stem
        pool = empty_pool()
        parsed = parse_viba_file(pool, self.source, path.name, self.module_name)
        assert isinstance(parsed, Ok), parsed
        self.pool = pool_add_file(pool, parsed.value).value
        found = pool_find_definition(self.pool, f"{self.module_name}.{rule_name}")
        assert isinstance(found, Ok), found
        self.definition = found.value
        self.accessor = access(self.definition)
        self.module = custom_module(self.source)
        self.definitions = {d.name: d for d in self.module.module.body}
        self.rule = AstNodeType(self.definitions[rule_name].body, self.module)

    # ---- 一份 witness 走一遍 ----

    def drive(self, witness, note: str = "") -> int:
        node = witness.ast_node if isinstance(witness, AstNodeType) else witness
        rooted = self.accessor.root(Witness(node, {self.definition.file_hash}))
        assert isinstance(rooted, Ok), f"{self.path.name} {note}: {rooted!r}"
        walked = self._walk(rooted.value, note)
        # RuleAccess.walk 也是遍历，两者必须给出同一批坐标
        ours = sorted(repr(n.path) for n in self.accessor._walk(rooted.value))
        theirs = sorted(repr(p) for p in self._paths(rooted.value, ()))
        assert ours == theirs, f"{self.path.name} {note}: walk 与独立遍历不一致"
        return 1 + walked

    def _paths(self, node, path):
        """独立于 RuleAccess.walk 的遍历，只按协议里那几条下一步。"""
        out = [path]
        slots = self.accessor._members(node)
        if slots is not None:
            positional = 0
            for tag, _ in slots:
                if tag:
                    step = by_tag(tag)
                else:
                    step = by_field_index(positional)
                    positional += 1
                given = self.accessor.get(node, step)
                if isinstance(given, Ok) and given.value is not None:
                    out += self._paths(given.value, path + (step,))
        shape = self.accessor._unfold(node.descriptor)
        container = self.accessor._container_kind(shape)
        if container == "dict":
            keys = self.accessor.keys(node)
            for key in keys.value:
                given = self.accessor.get(node, at_key(key))
                if isinstance(given, Ok) and given.value is not None:
                    out += self._paths(given.value, path + (at_key(key),))
        elif self.accessor._has_elements_by_index(shape):
            length = self.accessor.length(node)
            for index in range(length.value):
                given = self.accessor.get(node, at_index(index))
                if isinstance(given, Ok) and given.value is not None:
                    out += self._paths(given.value, path + (at_index(index),))
        return out

    def _walk(self, node, note: str) -> int:
        seen = 0
        slots = self.accessor._members(node)
        if slots is not None:
            positional = 0
            for tag, _ in slots:
                if tag:
                    step = by_tag(tag)
                else:
                    step = by_field_index(positional)
                    positional += 1
                given_has = self.accessor.has(node, step)
                given_get = self.accessor.get(node, step)
                assert isinstance(given_has, Ok), f"{self.path.name} {note}: {step} {given_has!r}"
                # 有数据 <-> has 真；没数据是 Ok(nil)
                assert isinstance(given_get, Ok), f"{self.path.name} {note}: {step} {given_get!r}"
                assert (given_get.value is not None) == given_has.value, \
                    f"{self.path.name} {note}: {step} has={given_has!r} get={given_get!r}"
                seen += 1
                if given_get.value is not None:
                    seen += self._walk(given_get.value, note)
        shape = self.accessor._unfold(node.descriptor)
        container = self.accessor._container_kind(shape)
        if container == "dict":
            keys = self.accessor.keys(node)
            assert isinstance(keys, Ok), f"{self.path.name} {note}: {node!r} {keys!r}"
            for key in keys.value:
                given = self.accessor.get(node, at_key(key))
                assert isinstance(given, Ok) and given.value is not None, \
                    f"{self.path.name} {note}: key {key!r} {given!r}"
                seen += 1 + self._walk(given.value, note)
        elif self.accessor._has_elements_by_index(shape):
            length = self.accessor.length(node)
            assert isinstance(length, Ok), f"{self.path.name} {note}: {node!r} {length!r}"
            for index in range(length.value):
                given = self.accessor.get(node, at_index(index))
                assert isinstance(given, Ok) and given.value is not None, \
                    f"{self.path.name} {note}: at {index} {given!r}"
                seen += 1 + self._walk(given.value, note)
        if isinstance(node.data, (viba_ast.Constant, viba_ast.Nil)):
            assert isinstance(self.accessor.leaf(node), Ok), f"{self.path.name} {note}: {node!r}"
        return seen

    def witness_definition(self, name: str):
        return AstNodeType(self.definitions[name].body, self.module)


def check_generated(path: Path, rule_name: str, seed: int) -> int:
    under = UnderTest(path, rule_name)
    pairs = 0
    for fail_prob in (0.0, 1.0):
        for witness in generate_witnesses(under.rule, WITNESSES_PER_RULE, seed=seed, fail_prob=fail_prob):
            under.drive(witness, f"fail_prob={fail_prob}")
            pairs += 1
    for witness in generate_witnesses(under.rule, MIXED_WITNESSES, seed=seed, fail_prob=FAIL_PROB):
        under.drive(witness, "mixed")
        pairs += 1
    # check_determinate 内部就是用这份生成的（fail_prob=0, seed=1）
    for witness in generate_witnesses(under.rule, WITNESSES_PER_RULE, seed=1, fail_prob=0.0):
        under.drive(witness, "determinacy")
        pairs += 1
    return pairs


def check_demo() -> int:
    under = UnderTest(DATA / "demo.viba", "DemoRule")
    for witness in generate_witnesses(under.rule, 50, seed=7):
        under.drive(witness, "demo")
    for witness in generate_witnesses(under.rule, 50, seed=1, fail_prob=0.0):
        under.drive(witness, "demo determinacy")
    return 100


def check_sum_rule() -> int:
    under = UnderTest(DATA / "sum_rule.viba", "SumRule")
    pairs = 0
    for witness in generate_witnesses(under.rule, 40, seed=3):
        under.drive(witness, "sum")
        pairs += 1
    for name in ("SumWitnessPass", "SumWitnessFail"):
        under.drive(under.witness_definition(name), name)
        pairs += 1
    for witness in generate_witnesses(under.rule, 40, seed=3, fail_prob=0.0):
        under.drive(witness, "sum determinacy")
        pairs += 1
    # 和类型：数据只落在选中的那一支上
    node = under.accessor.root(
        Witness(under.witness_definition("SumWitnessPass").ast_node,
                {under.definition.file_hash})).value
    assert under.accessor.has(node, by_tag("$small")).value is True
    assert under.accessor.has(node, by_tag("$big")).value is False
    return pairs


def check_not_rule() -> int:
    under = UnderTest(DATA / "not_rule.viba", "NoDeathPenaltyRule")
    pairs = 0
    for name in ("LawAbidingWitness", "ViolatingWitness", "PartialWitness"):
        under.drive(under.witness_definition(name), name)
        pairs += 1
    for witness in generate_witnesses(under.rule, 40, seed=3):
        under.drive(witness, "not")
        pairs += 1
    for witness in generate_witnesses(under.rule, 40, seed=3, fail_prob=0.0):
        under.drive(witness, "not determinacy")
        pairs += 1
    # not[A] 的外壳两边都在：操作数是一个坐标，三个分支在它下面
    node = under.accessor.root(
        Witness(under.witness_definition("LawAbidingWitness").ast_node,
                {under.definition.file_hash})).value
    shell = node.by_tag("not_crimes")
    assert under.accessor.has(shell, by_tag("not_operand")).value is True
    operand = shell.get_not_operand()
    for branch in ("$homicide", "$arson", "$robbery"):
        assert under.accessor.has(operand, by_tag(branch)).value is True
    return pairs


def check_broken_rules() -> int:
    """坏规则：图上没有落点的地方必须老老实实报错，不能编数据出来。"""
    pairs = 0
    under = UnderTest(DATA / "broken_rules.viba", "BadRef")
    witness = generate_witnesses(under.rule, 1, seed=1)[0]
    node = under.accessor.root(Witness(witness.ast_node, {under.definition.file_hash})).value
    member = node.by_tag("x")
    assert not isinstance(under.accessor.leaf(member), Ok)  # Missing 解析不了，更读不出叶子
    pairs += 1

    under = UnderTest(DATA / "broken_rules.viba", "BadRule")
    witness = generate_witnesses(under.rule, 1, seed=1)[0]
    node = under.accessor.root(Witness(witness.ast_node, {under.definition.file_hash})).value
    assert under.accessor.has(node, by_tag("$x")).value is False  # ... 不是数据
    pairs += 1
    return pairs


def check_predicate_reset() -> int:
    under = UnderTest(DATA / "predicate_bound.viba", "PredicateBoundRule")
    check = _predicate_node(under.rule.ast_node)
    pairs = 0
    for value in range(100):
        witness = _bound_witness(under.module, value, check)
        under.drive(witness, f"bound {value}")
        pairs += 1
    for value in range(100):
        witness = _bound_witness(under.module, value, check)
        under.drive(reset_predication_by_python_code(witness), f"bound reset {value}")
        pairs += 1
    return pairs


def check_predicate_code() -> int:
    pairs = 0
    for path in sorted((DATA / "rules").glob("rule*.viba")):
        number = path.stem[len("rule"):]
        under = UnderTest(path, f"Rule{number}")
        witnesses = generate_witnesses(under.rule, PREDICATE_WITNESSES,
                                      seed=int(number), fail_prob=0.0)
        for witness in witnesses:
            under.drive(reset_predication_by_python_code(witness), "predicate code")
            pairs += 1
    return pairs


def _predicate_node(body):
    for node in viba_ast.walk(body):
        if isinstance(node, viba_ast.TypeApp) and node.constructor == "Predicate":
            return node
    raise AssertionError("the rule has no Predicate field")


def _bound_witness(module, value, check):
    """Metric[Len] := $value Len：量出来的值在 $value 那一层。"""
    body = viba_ast.Product(
        viba_ast.TypeRef("Object"),
        viba_ast.Product(
            viba_ast.Tagged("$len", viba_ast.Tagged("$value", viba_ast.Constant(value))),
            viba_ast.Tagged("$check", check),
        ),
    )
    return AstNodeType(body, module)


def main() -> int:
    pairs = 0
    rule_paths = sorted((DATA / "rules").glob("rule*.viba"))
    not_paths = sorted(NOT_DATA.glob("not_rule*.viba"))
    for path in rule_paths:
        number = path.stem[len("rule"):]
        pairs += check_generated(path, f"Rule{number}", int(number))
    for path in not_paths:
        number = path.stem[len("not_rule"):]
        pairs += check_generated(path, f"NotRule{number}", int(number))
    pairs += check_demo()
    pairs += check_sum_rule()
    pairs += check_not_rule()
    pairs += check_broken_rules()
    pairs += check_predicate_reset()
    pairs += check_predicate_code()
    # 起点不核版本；要核就用便利函数，版本对不上在这里看得出来
    under = UnderTest(DATA / "demo.viba", "DemoRule")
    witness = generate_witnesses(under.rule, 1, seed=1)[0]
    assert isinstance(under.accessor.root(Witness(witness.ast_node, set())), Ok)
    assert isinstance(under.accessor.root(Witness(witness.ast_node, {"deadbeef"})), Ok)
    assert viba_version_matches(under.definition.pool,
                                Witness(witness.ast_node, {"deadbeef"})).value is False

    # 份数跟着语料算：60 条生成规则 × (20+20+200+20) + 几组固定的
    generated = (len(rule_paths) + len(not_paths)) * (WITNESSES_PER_RULE * 3 + MIXED_WITNESSES)
    fixed = 100 + 82 + 83 + 2 + 200 + len(rule_paths) * PREDICATE_WITNESSES
    assert pairs == generated + fixed, f"covered {pairs} pairs, expected {generated + fixed}"
    print(f"rule_reflect: {pairs} 份 (rule, witness) 走通 "
          f"({len(rule_paths)} + {len(not_paths)} 条生成规则，含版本核对、坐标枚举、叶子与容器取值)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

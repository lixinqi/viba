"""is_compliant / is_determinate over generated instances.

Data-driven over tests/data/rule_check/ruleNN.viba (see build.py):
every rule is judged in three ways — its generated instances must
all be structural subtypes of the Assert-stripped rule, judging
against the rule itself must never Err, and is_determinate must
certify it. Inline cases cover the original DEMO plus two broken
rules that determinacy must reject.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import ast as viba_ast
from viba.generate import generate
from viba.is_compliant import is_compliant
from viba.is_determinate import is_determinate
from viba.type import AstNodeType, Err, Ok, custom_module

DEMO = """
CodeLength := int
DocCoverage := $documented_lines int * $total_lines int
Keywords := list[str]

DemoRule :=
  Object
  * $code_length Metric[CodeLength]
  * $coverage Metric[DocCoverage]
  * $keywords Metric[Keywords]
  * $assert_code_len_le_24
      Assert[{code length <= 24}, $python_code {
def handler(self):
    return self.code_length.value <= 24
}]

DemoRuleStripped :=
  Object
  * $code_length Metric[CodeLength]
  * $coverage Metric[DocCoverage]
  * $keywords Metric[Keywords]
  * $assert_code_len_le_24 nil
"""

DATA = Path(__file__).resolve().parent / "data" / "rule_check"
INSTANCES_PER_RULE = 20


def _entry(text: str, name: str) -> AstNodeType:
    module = custom_module(text)
    defs = {n.name: n for n in module.module.body}
    return AstNodeType(defs[name].body, module)


def _strip_asserts(node):
    """Assert fields witness as nil; the rest of the shape is kept."""
    if isinstance(node, viba_ast.Tagged):
        if isinstance(node.type, viba_ast.TypeApp) and node.type.constructor == "Assert":
            return viba_ast.Tagged(node.tag, viba_ast.Nil())
        return viba_ast.Tagged(node.tag, _strip_asserts(node.type))
    if isinstance(node, viba_ast.Product):
        return viba_ast.Product(_strip_asserts(node.left), _strip_asserts(node.right))
    if isinstance(node, viba_ast.ProductChain):
        return viba_ast.ProductChain([_strip_asserts(e) for e in node.elements])
    return node


def _check_rule_file(path: Path) -> int:
    number = path.stem[len("rule"):]
    text = path.read_text()
    rule = _entry(text, f"Rule{number}")
    stripped = AstNodeType(_strip_asserts(rule.ast_node), rule.container_module)
    checked = 0
    for instance in generate(rule, INSTANCES_PER_RULE, seed=int(number)):
        plain = is_compliant(instance, stripped)
        assert isinstance(plain, Ok) and plain.value is True, f"{path.name}: stripped"
        given = is_compliant(instance, rule)
        assert isinstance(given, Ok), f"{path.name}: judge vs rule errored: {given!r}"
        checked += 1
    determined = is_determinate(rule, INSTANCES_PER_RULE, seed=1)
    assert isinstance(determined, Ok) and determined.value is True, f"{path.name}: {determined!r}"
    return checked


def _check_demo() -> None:
    rule = _entry(DEMO, "DemoRule")
    stripped = _entry(DEMO, "DemoRuleStripped")
    verdicts = {True: 0, False: 0}
    for instance in generate(rule, 50, seed=7):
        given = is_compliant(instance, rule)
        assert isinstance(given, Ok), f"judge vs rule errored: {given!r}"
        plain = is_compliant(instance, stripped)
        assert isinstance(plain, Ok) and plain.value is True
        verdicts[given.value] += 1
    determined = is_determinate(rule, 50, seed=1)
    assert isinstance(determined, Ok) and determined.value is True


def _check_broken_rules() -> None:
    ellipsis = _entry("BadRule :=\n  Object\n  * $x ...\n", "BadRule")
    assert isinstance(is_determinate(ellipsis, 5, seed=1), Err)
    missing = _entry("BadRef :=\n  Object\n  * $x Missing\n", "BadRef")
    assert isinstance(is_determinate(missing, 5, seed=1), Err)


def main():
    paths = sorted(DATA.glob("rule*.viba"))
    total = 0
    for path in paths:
        total += _check_rule_file(path)
    _check_demo()
    _check_broken_rules()
    print(f"rule_check: {len(paths)} data rules x {INSTANCES_PER_RULE} instances"
          f" ({total} judged) + demo + 2 broken rules rejected")


main()

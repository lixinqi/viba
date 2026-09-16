"""is_compliant / is_determinate over generated instances.

Data-driven over tests/data/rule_check/ruleNN.viba (see build.py):
for every rule, each generated instance is judged against the rule
itself and must not Err, and is_determinate must certify the rule.
Inline cases cover the original DEMO plus two broken rules that
determinacy must reject.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
"""

DATA = Path(__file__).resolve().parent / "data" / "rule_check"
INSTANCES_PER_RULE = 20


def _entry(text: str, name: str) -> AstNodeType:
    module = custom_module(text)
    defs = {n.name: n for n in module.module.body}
    return AstNodeType(defs[name].body, module)


def _check_rule_file(path: Path) -> int:
    number = path.stem[len("rule"):]
    rule = _entry(path.read_text(), f"Rule{number}")
    checked = 0
    for instance in generate(rule, INSTANCES_PER_RULE, seed=int(number)):
        given = is_compliant(instance, rule)
        assert isinstance(given, Ok), f"{path.name}: judge errored: {given!r}"
        checked += 1
    determined = is_determinate(rule, INSTANCES_PER_RULE, seed=1)
    assert isinstance(determined, Ok) and determined.value is True, f"{path.name}: {determined!r}"
    return checked


def _check_demo() -> None:
    rule = _entry(DEMO, "DemoRule")
    verdicts = {True: 0, False: 0}
    for instance in generate(rule, 50, seed=7):
        given = is_compliant(instance, rule)
        assert isinstance(given, Ok), f"judge errored: {given!r}"
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

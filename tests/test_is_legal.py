"""is_compliant / is_determinate over generated instances.

The pipeline: instance_generator turns a rule into instances; every
instance must judge true/false without error (is_compliant never
Errs on a legal rule); is_determinate certifies the rule itself by
quantifying over generated instances.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.is_legal import is_compliant, is_determinate
from viba.instance_generator import generate
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


def _entry(text: str, name: str) -> AstNodeType:
    module = custom_module(text)
    defs = {n.name: n for n in module.module.body}
    return AstNodeType(defs[name].body, module)


def main():
    rule = _entry(DEMO, "DemoRule")
    stripped = _entry(DEMO, "DemoRuleStripped")
    instances = generate(rule, 50, seed=7)
    assert len(instances) == 50
    verdicts = {True: 0, False: 0}
    for instance in instances:
        given = is_compliant(instance, rule)
        assert isinstance(given, Ok), f"judge vs rule errored: {given!r}"
        plain = is_compliant(instance, stripped)
        assert isinstance(plain, Ok) and plain.value is True
        verdicts[given.value] += 1
    determined = is_determinate(rule, 50, seed=1)
    assert isinstance(determined, Ok) and determined.value is True
    ellipsis = _entry("BadRule :=\n  Object\n  * $x ...\n", "BadRule")
    assert isinstance(is_determinate(ellipsis, 5, seed=1), Err)
    missing = _entry("BadRef :=\n  Object\n  * $x Missing\n", "BadRef")
    assert isinstance(is_determinate(missing, 5, seed=1), Err)
    print(f"is_legal: 50 instances judged, verdicts {verdicts};"
          " determinate ok, two broken rules rejected")


main()

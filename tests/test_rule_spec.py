"""Executable checks for the Rule spec (rule.md).

The spec is a program: its ```viba block must parse, round-trip, and
its demo judgments must hold — DemoResultPass seats in
DemoRuleStripped, DemoResultFail does not, poisoned sup errs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import ast as viba_ast
from viba.type import AstNodeType, Err, Ok, custom_module
from viba.is_sub_type import is_sub_type

SPEC = Path(__file__).resolve().parent.parent / "rule.md"


def _spec_source():
    block = SPEC.read_text().split("```viba\n", 1)[1]
    return block.split("\n```", 1)[0]


# 剥离与取证的产物不在 Rule 规范内，由测试自带：
# pass 写 nil 对位，fail 同 tag 写 AssertionFailed（永不入围）。
DEMO = """
DemoRuleStripped :=
  Object
  * $code_length Metric[CodeLength]
  * $assert_code_len_le_24 nil

DemoResultFail :=
  Object
  * $code_length 30
  * $assert_code_len_le_24 AssertionFailed[int, str]

DemoResultPass :=
  Object
  * $code_length 20
  * $assert_code_len_le_24 nil

DemoSumRule :=
  OneofRule
  | $small int
  | $big str

DemoSumFive :=
  $small 5
"""


def _round_trip(text: str):
    once = viba_ast.unparse(viba_ast.parse(text))
    twice = viba_ast.unparse(viba_ast.parse(once))
    assert once == twice, f"unstable round-trip:\n{once}\n!=\n{twice}"


def _entry(module, name: str):
    defs = {n.name: n for n in module.module.body}
    return AstNodeType(defs[name].body, module)


def _judgment(module, sub_name: str, sup_name: str):
    result = is_sub_type(_entry(module, sub_name), _entry(module, sup_name))
    assert isinstance(result, Ok), f"{sub_name} <: {sup_name}: {result!r}"
    return result.value


def main():
    _round_trip(_spec_source())
    module = custom_module(_spec_source() + DEMO)
    assert _judgment(module, "DemoResultPass", "DemoRuleStripped") is True
    assert _judgment(module, "DemoResultFail", "DemoRuleStripped") is False
    poison_sup = is_sub_type(_entry(module, "DemoResultPass"), _entry(module, "DemoResultFail"))
    assert isinstance(poison_sup, Err), f"poison on sup side must be Err: {poison_sup!r}"
    names = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert names == ["DemoRule", "DemoSumRule"], f"rule markers: {names}"
    assert _judgment(module, "DemoSumFive", "DemoSumRule") is True
    print("rule spec: round-trip + 4 judgment checks + marker scan passed")


main()

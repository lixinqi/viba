"""Executable checks for the rule coding style (viba-rule.md).

The style document is a program: its DemoRule ```viba block must parse,
round-trip, satisfy check_rule_coding_style, and its demo judgments must
hold — DemoWitnessPass seats in DemoRuleStripped, DemoWitnessFail does
not.
The stripped/pass/fail companions live in
tests/data/rule_coding_style_check/spec_demo.viba; the corpus coverage
lives in test_rule_coding_style_check.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import viba_ast
from viba.type import AstNodeType, Ok, custom_module
from viba.is_sub_type import is_sub_type
from viba.rule import check_rule_coding_style

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data" / "rule_coding_style_check"


def _spec_source():
    """The DemoRule block of viba-rule.md, wherever it sits."""
    text = (ROOT / "viba-rule.md").read_text()
    for chunk in text.split("```viba\n")[1:]:
        source = chunk.split("\n```", 1)[0]
        if "DemoRule" in source:
            return source
    raise AssertionError("viba-rule.md has no DemoRule ```viba block")


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
    demo = (DATA / "spec_demo.viba").read_text()
    module = custom_module(_spec_source() + demo)
    assert _judgment(module, "DemoWitnessPass", "DemoRuleStripped") is True
    assert _judgment(module, "DemoWitnessFail", "DemoRuleStripped") is False
    names = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert names == ["DemoRule"], f"rule markers: {names}"
    spec = check_rule_coding_style(_entry(module, "DemoRule"))
    assert isinstance(spec, Ok) and spec.value is None, f"rule style check: {spec!r}"
    print("rule coding style: round-trip + style check + 2 judgment checks + marker scan passed")


main()

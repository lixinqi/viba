"""is_compliant / is_determinate over generated instances.

Data-driven over tests/data/rule_check/: ruleNN.viba (see build.py)
each contribute a marked rule whose generated instances must judge
against the rule itself without Err and whose is_determinate must
certify it; demo.viba, sum_rule.viba and broken_rules.viba cover the
original DEMO, an OneofRule over Assert-carrying branch rules, and
two broken rules determinacy must reject. All Viba source lives in
data files — this file is pure checking logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import ast as viba_ast
from viba.generate import generate
from viba.is_compliant import is_compliant
from viba.is_determinate import is_determinate
from viba.type import AstNodeType, Err, Ok, custom_module

DATA = Path(__file__).resolve().parent / "data" / "rule_check"
INSTANCES_PER_RULE = 20


def _load(name: str):
    module = custom_module((DATA / name).read_text())
    return module, {n.name: n for n in module.module.body}


def _entry(defs, module, name: str) -> AstNodeType:
    return AstNodeType(defs[name].body, module)


def _check_rule_file(path: Path) -> int:
    number = path.stem[len("rule"):]
    module, defs = _load(path.name)
    rule = _entry(defs, module, f"Rule{number}")
    found = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert found == [f"Rule{number}"], f"{path.name}: marker scan {found}"
    checked = 0
    for instance in generate(rule, INSTANCES_PER_RULE, seed=int(number)):
        given = is_compliant(instance, rule)
        assert isinstance(given, Ok), f"{path.name}: judge errored: {given!r}"
        checked += 1
    determined = is_determinate(rule, INSTANCES_PER_RULE, seed=1)
    assert isinstance(determined, Ok) and determined.value is True, f"{path.name}: {determined!r}"
    return checked


def _check_demo() -> None:
    module, defs = _load("demo.viba")
    rule = _entry(defs, module, "DemoRule")
    verdicts = {True: 0, False: 0}
    for instance in generate(rule, 50, seed=7):
        given = is_compliant(instance, rule)
        assert isinstance(given, Ok), f"judge errored: {given!r}"
        verdicts[given.value] += 1
    determined = is_determinate(rule, 50, seed=1)
    assert isinstance(determined, Ok) and determined.value is True


def _check_sum_rule() -> None:
    module, defs = _load("sum_rule.viba")
    rule = _entry(defs, module, "SumRule")
    names = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert names == ["SmallRule", "BigRule", "SumRule"], f"markers: {names}"
    verdicts = set()
    for instance in generate(rule, 40, seed=3):
        judged = is_compliant(instance, rule)
        assert isinstance(judged, Ok), f"sum judge errored: {judged!r}"
        verdicts.add(judged.value)
    assert verdicts == {True, False}, f"sum verdicts: {verdicts}"
    determined = is_determinate(rule, 40, seed=3)
    assert isinstance(determined, Ok) and determined.value is True
    for witness, want in (("SumWitnessPass", True), ("SumWitnessFail", False)):
        sub = _entry(defs, module, witness)
        got = is_compliant(sub, rule)
        assert isinstance(got, Ok) and got.value is want, f"{witness}: {got!r}"


def _check_broken_rules() -> None:
    module, defs = _load("broken_rules.viba")
    for name in ("BadRule", "BadRef"):
        rule = _entry(defs, module, name)
        assert isinstance(is_determinate(rule, 5, seed=1), Err), name


def main():
    paths = sorted(DATA.glob("rule*.viba"))
    total = 0
    for path in paths:
        total += _check_rule_file(path)
    _check_demo()
    _check_sum_rule()
    _check_broken_rules()
    print(f"rule_check: {len(paths)} data rules x {INSTANCES_PER_RULE} instances"
          f" ({total} judged) + demo + sum-rule + 2 broken rules rejected")


main()

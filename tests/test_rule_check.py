"""is_compliant / is_determinate over generated witnesses.

Data-driven over tests/data/rule_check/rules/ruleNN.viba (see build.py)
each contribute a marked rule whose generated witnesses must judge
against the rule itself: with fail_prob 0 every Predicate passes and
every witness must be compliant (True), with fail_prob 1 every
Predicate fails and every witness must be rejected (False), and with
the default fail_prob the True share must match
(1 - fail_prob) ** predicates. is_determinate must certify each rule. demo.viba, sum_rule.viba, not_rule.viba and
broken_rules.viba cover the original DEMO, an OneofRule over
Predicate-carrying branch rules, a prohibitive not[...] rule with
per-branch refutation witnesses, and two broken rules determinacy must
reject. All Viba source lives in data files — this file is pure
checking logic.
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
WITNESSES_PER_RULE = 20
MIXED_WITNESSES = 200
FAIL_PROB = 0.1


def _load(path: Path):
    module = custom_module(path.read_text())
    return module, {n.name: n for n in module.module.body}


def _entry(defs, module, name: str) -> AstNodeType:
    return AstNodeType(defs[name].body, module)


def _judge(rule, witnesses, path: Path):
    verdicts = set()
    for witness in witnesses:
        given = is_compliant(witness, rule)
        assert isinstance(given, Ok), f"{path.name}: judge errored: {given!r}"
        verdicts.add(given.value)
    return verdicts


def _judge_counts(rule, witnesses, path: Path):
    counts = {True: 0, False: 0}
    for witness in witnesses:
        given = is_compliant(witness, rule)
        assert isinstance(given, Ok), f"{path.name}: judge errored: {given!r}"
        counts[given.value] += 1
    return counts


def _predicate_count(rule) -> int:
    return sum(1 for node in viba_ast.walk(rule.ast_node)
               if isinstance(node, viba_ast.TypeApp) and node.constructor == "Predicate")


def _check_rule_file(path: Path) -> int:
    number = path.stem[len("rule"):]
    module, defs = _load(path)
    rule = _entry(defs, module, f"Rule{number}")
    found = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert found == [f"Rule{number}"], f"{path.name}: marker scan {found}"
    checked = 0
    for fail_prob, want in ((0.0, True), (1.0, False)):
        witnesses = generate(rule, WITNESSES_PER_RULE, seed=int(number), fail_prob=fail_prob)
        verdicts = _judge(rule, witnesses, path)
        assert verdicts == {want}, f"{path.name}: fail_prob={fail_prob} gave {verdicts}"
        checked += len(witnesses)
    # A witness passes only when none of its Predicate fields flipped, so
    # the True share is (1 - fail_prob) ** predicates; the sample must sit
    # inside a 4-sigma band of that.
    passes = (1 - FAIL_PROB) ** _predicate_count(rule)
    witnesses = generate(rule, MIXED_WITNESSES, seed=int(number), fail_prob=FAIL_PROB)
    counts = _judge_counts(rule, witnesses, path)
    assert counts[True] > 0 and counts[False] > 0, f"{path.name}: mixed {counts}"
    expected = MIXED_WITNESSES * passes
    spread = 4 * (MIXED_WITNESSES * passes * (1 - passes)) ** 0.5 + 1
    assert abs(counts[True] - expected) <= spread, (
        f"{path.name}: True {counts[True]}/{MIXED_WITNESSES}, model {expected:.1f}±{spread:.1f}")
    checked += MIXED_WITNESSES
    determined = is_determinate(rule, WITNESSES_PER_RULE, seed=1)
    assert isinstance(determined, Ok) and determined.value is True, f"{path.name}: {determined!r}"
    return checked


def _check_demo() -> None:
    module, defs = _load(DATA / "demo.viba")
    rule = _entry(defs, module, "DemoRule")
    verdicts = _judge(rule, generate(rule, 50, seed=7), DATA / "demo.viba")
    assert verdicts == {True, False}, f"demo verdicts: {verdicts}"
    determined = is_determinate(rule, 50, seed=1)
    assert isinstance(determined, Ok) and determined.value is True


def _check_sum_rule() -> None:
    module, defs = _load(DATA / "sum_rule.viba")
    rule = _entry(defs, module, "SumRule")
    names = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert names == ["SmallRule", "BigRule", "SumRule"], f"markers: {names}"
    verdicts = set()
    for witness in generate(rule, 40, seed=3):
        judged = is_compliant(witness, rule)
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
    module, defs = _load(DATA / "broken_rules.viba")
    for name in ("BadRule", "BadRef"):
        rule = _entry(defs, module, name)
        assert isinstance(is_determinate(rule, 5, seed=1), Err), name


def _check_not_rule() -> None:
    module, defs = _load(DATA / "not_rule.viba")
    rule = _entry(defs, module, "DeathPenaltyRule")
    names = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert names == ["DeathPenaltyRule"], f"markers: {names}"
    for witness, want in (("LawAbidingWitness", True),
                          ("ViolatingWitness", False),
                          ("PartialWitness", False)):
        sub = _entry(defs, module, witness)
        got = is_compliant(sub, rule)
        assert isinstance(got, Ok) and got.value is want, f"{witness}: {got!r}"
    verdicts = set()
    for witness in generate(rule, 40, seed=3):
        judged = is_compliant(witness, rule)
        assert isinstance(judged, Ok), f"not judge errored: {judged!r}"
        verdicts.add(judged.value)
    assert verdicts == {True, False}, f"not verdicts: {verdicts}"
    determined = is_determinate(rule, 40, seed=3)
    assert isinstance(determined, Ok) and determined.value is True


def main():
    paths = sorted((DATA / "rules").glob("rule*.viba"))
    total = 0
    for path in paths:
        total += _check_rule_file(path)
    determinacy = len(paths) * WITNESSES_PER_RULE
    _check_demo()
    _check_sum_rule()
    _check_not_rule()
    _check_broken_rules()
    print(f"rule_check: {len(paths)} data rules x {WITNESSES_PER_RULE} witnesses"
          f" ({total} verdict witnesses + {determinacy} determinacy witnesses)"
          f" + demo + sum-rule + not-rule + 2 broken rules rejected")


main()

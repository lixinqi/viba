"""is_compliant / is_determinate over generated witnesses.

Data-driven over tests/data/rule_coding_style_check/rules/ruleNN.viba and
not_rules/not_ruleNN.viba (see build.py): the 40 Metric/Predicate
corpus rules and 20 prohibition rules covering the classic not[...]
shapes. Each contributes a marked rule whose generated witnesses must
judge against the rule itself: with fail_prob 0 every flip passes and
every witness must be compliant (True), with fail_prob 1 every flip
fails and every witness must be rejected (False), and with the default
fail_prob the True share must match (1 - fail_prob) ** flip_sites,
where a flip site is a positive Predicate field or a tagged not branch.
is_determinate must certify each rule. demo.viba, sum_rule.viba, not_rule.viba and
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
from viba.generate import flip_sites, generate
from viba.is_compliant import is_compliant
from viba.is_determinate import is_determinate
from viba.rule_coding_style_check import rule_coding_style_check
from viba.type import AstNodeType, Err, Ok, custom_module

DATA = Path(__file__).resolve().parent / "data" / "rule_coding_style_check"
NOT_DATA = DATA / "not_rules"
WITNESSES_PER_RULE = 20
MIXED_WITNESSES = 200
FAIL_PROB = 0.1


def _load(path: Path):
    module = custom_module(path.read_text())
    return module, {n.name: n for n in module.module.body}


def _entry(defs, module, name: str) -> AstNodeType:
    return AstNodeType(defs[name].body, module)


def _spec(defs, module, name: str) -> None:
    given = rule_coding_style_check(_entry(defs, module, name))
    assert isinstance(given, Ok) and given.value is True, f"{name}: style {given!r}"


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


def _check_generated_rule(path: Path, name: str, seed: int) -> int:
    module, defs = _load(path)
    rule = _entry(defs, module, name)
    found = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert found == [name], f"{path.name}: marker scan {found}"
    _spec(defs, module, name)
    checked = 0
    for fail_prob, want in ((0.0, True), (1.0, False)):
        witnesses = generate(rule, WITNESSES_PER_RULE, seed=seed, fail_prob=fail_prob)
        verdicts = _judge(rule, witnesses, path)
        assert verdicts == {want}, f"{path.name}: fail_prob={fail_prob} gave {verdicts}"
        checked += len(witnesses)
    # A witness passes only when none of generate's flips lands, so the
    # True share is (1 - fail_prob) ** flip_sites; the sample must sit
    # inside a 4-sigma band of that.
    sites = flip_sites(rule)
    passes = (1 - FAIL_PROB) ** sites
    witnesses = generate(rule, MIXED_WITNESSES, seed=seed, fail_prob=FAIL_PROB)
    counts = _judge_counts(rule, witnesses, path)
    assert counts[True] > 0 and counts[False] > 0, f"{path.name}: mixed {counts}"
    expected = MIXED_WITNESSES * passes
    spread = 4 * (MIXED_WITNESSES * passes * (1 - passes)) ** 0.5 + 1
    assert abs(counts[True] - expected) <= spread, (
        f"{path.name}: True {counts[True]}/{MIXED_WITNESSES}, model {expected:.1f}±{spread:.1f} (sites={sites})")
    checked += MIXED_WITNESSES
    determined = is_determinate(rule, WITNESSES_PER_RULE, seed=1)
    assert isinstance(determined, Ok) and determined.value is True, f"{path.name}: {determined!r}"
    return checked


def _check_rule_file(path: Path) -> int:
    number = path.stem[len("rule"):]
    return _check_generated_rule(path, f"Rule{number}", int(number))


def _check_not_rule_file(path: Path) -> int:
    number = path.stem[len("not_rule"):]
    return _check_generated_rule(path, f"NotRule{number}", int(number))


def _check_demo() -> None:
    path = DATA / "demo.viba"
    module, defs = _load(path)
    _spec(defs, module, "DemoRule")
    rule = _entry(defs, module, "DemoRule")
    verdicts = _judge(rule, generate(rule, 50, seed=7), path)
    assert verdicts == {True, False}, f"demo verdicts: {verdicts}"
    determined = is_determinate(rule, 50, seed=1)
    assert isinstance(determined, Ok) and determined.value is True


def _check_sum_rule() -> None:
    path = DATA / "sum_rule.viba"
    module, defs = _load(path)
    rule = _entry(defs, module, "SumRule")
    names = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert names == ["SmallRule", "BigRule", "SumRule"], f"markers: {names}"
    for name in names:
        _spec(defs, module, name)
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
    path = DATA / "not_rule.viba"
    module, defs = _load(path)
    rule = _entry(defs, module, "NoDeathPenaltyRule")
    names = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert names == ["NoDeathPenaltyRule"], f"markers: {names}"
    _spec(defs, module, "NoDeathPenaltyRule")
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
    not_paths = sorted(NOT_DATA.glob("not_rule*.viba"))
    total = 0
    for path in paths:
        total += _check_rule_file(path)
    for path in not_paths:
        total += _check_not_rule_file(path)
    determinacy = (len(paths) + len(not_paths)) * WITNESSES_PER_RULE
    _check_demo()
    _check_sum_rule()
    _check_not_rule()
    _check_broken_rules()
    print(f"rule_coding_style_check: {len(paths)} rules + {len(not_paths)} not-rules"
          f" ({total} verdict witnesses + {determinacy} determinacy witnesses)"
          f" + demo + sum-rule + not-rule + 2 broken rules rejected")


main()

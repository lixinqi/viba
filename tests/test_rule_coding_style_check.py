"""is_compliant / check_determinate over generated witnesses.

Data-driven over tests/data/rule_coding_style_check/rules/ruleNN.viba and
not_rules/not_ruleNN.viba (see build.py): the 40 Metric/Predicate
corpus rules and 20 prohibition rules covering the classic not[...]
shapes. Each contributes a marked rule whose generated witnesses must
judge against the rule itself: with fail_prob 0 every flip passes and
every witness must be compliant (True), with fail_prob 1 every flip
fails and every witness must be rejected (False), and with the default
fail_prob the True share must match (1 - fail_prob) ** flip_sites,
where a flip site is a positive Predicate field or a tagged not branch.
check_determinate must certify each rule. demo.viba, sum_rule.viba, not_rule.viba and
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
from viba.rule import (
    check_determinate,
    generate_witnesses,
    is_compliant,
    reset_predication_by_python_code,
    check_rule_coding_style,
)
from viba.type import AstNodeType, Err, Ok, custom_module, module_get_type

DATA = Path(__file__).resolve().parent / "data" / "rule_coding_style_check"
NOT_DATA = DATA / "not_rules"
WITNESSES_PER_RULE = 20
MIXED_WITNESSES = 200
PREDICATE_WITNESSES = 50
FAIL_PROB = 0.1


def _load(path: Path):
    module = custom_module(path.read_text())
    return module, {n.name: n for n in module.module.body}


def _entry(defs, module, name: str) -> AstNodeType:
    return AstNodeType(defs[name].body, module)


def _spec(defs, module, name: str) -> None:
    given = check_rule_coding_style(_entry(defs, module, name))
    assert isinstance(given, Ok) and given.value is None, f"{name}: style {given!r}"


def _flip_sites(rule) -> int:
    """Independent flips generate_witnesses makes for a rule: one per positive
    Predicate field, one per tagged not branch. A witness passes only
    when none of them flips, so the True share is (1 - p) ** sites.
    Test-side model of generate_witnesses, kept here because it is not API."""
    return _count_flips(rule.ast_node, rule.container_module)


def _count_flips(node, module) -> int:
    if isinstance(node, viba_ast.Tagged):
        return _count_flips(node.type, module)
    if isinstance(node, viba_ast.Product):
        return _count_flips(node.left, module) + _count_flips(node.right, module)
    if isinstance(node, viba_ast.ProductChain):
        return sum(_count_flips(e, module) for e in node.elements)
    if isinstance(node, viba_ast.Tuple):
        return sum(_count_flips(e, module) for e in node.elements)
    if isinstance(node, viba_ast.TypeApp):
        if node.constructor == "Predicate":
            return 1
        if node.constructor == "not":
            branches = _not_branches(node.args[0], module)
            return 0 if branches is None else len(branches)
        return 0
    if isinstance(node, viba_ast.TypeRef):
        body, home = _unfold_ref(node, module)
        return 0 if body is node else _count_flips(body, home)
    return 0


def _not_branches(node, module):
    """The tagged branches of a not argument; None if one is untagged."""
    node, module = _unfold_ref(node, module)
    if isinstance(node, viba_ast.Sum):
        left = _not_branches(node.left, module)
        right = _not_branches(node.right, module)
        return None if left is None or right is None else left + right
    if isinstance(node, viba_ast.SumChain):
        out = []
        for element in node.elements:
            part = _not_branches(element, module)
            if part is None:
                return None
            out += part
        return out
    return [node] if isinstance(node, viba_ast.Tagged) else None


def _unfold_ref(node, module):
    """A TypeRef to a plain TypeDefinition unfolds to its body."""
    if not isinstance(node, viba_ast.TypeRef):
        return node, module
    resolved = module_get_type(module, node.name)
    if not isinstance(resolved, Ok) or not isinstance(resolved.value, AstNodeType):
        return node, module
    target = resolved.value
    if not isinstance(target.ast_node, viba_ast.TypeDefinition):
        return node, module
    return target.ast_node.body, target.container_module


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
        witnesses = generate_witnesses(rule, WITNESSES_PER_RULE, seed=seed, fail_prob=fail_prob)
        verdicts = _judge(rule, witnesses, path)
        assert verdicts == {want}, f"{path.name}: fail_prob={fail_prob} gave {verdicts}"
        checked += len(witnesses)
    # A witness passes only when none of generate_witnesses' flips lands, so the
    # True share is (1 - fail_prob) ** flip_sites; the sample must sit
    # inside a 4-sigma band of that.
    sites = _flip_sites(rule)
    passes = (1 - FAIL_PROB) ** sites
    witnesses = generate_witnesses(rule, MIXED_WITNESSES, seed=seed, fail_prob=FAIL_PROB)
    counts = _judge_counts(rule, witnesses, path)
    assert counts[True] > 0 and counts[False] > 0, f"{path.name}: mixed {counts}"
    expected = MIXED_WITNESSES * passes
    spread = 4 * (MIXED_WITNESSES * passes * (1 - passes)) ** 0.5 + 1
    assert abs(counts[True] - expected) <= spread, (
        f"{path.name}: True {counts[True]}/{MIXED_WITNESSES}, model {expected:.1f}±{spread:.1f} (sites={sites})")
    checked += MIXED_WITNESSES
    determined = check_determinate(rule, WITNESSES_PER_RULE, seed=1)
    assert isinstance(determined, Ok) and determined.value is None, f"{path.name}: {determined!r}"
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
    verdicts = _judge(rule, generate_witnesses(rule, 50, seed=7), path)
    assert verdicts == {True, False}, f"demo verdicts: {verdicts}"
    determined = check_determinate(rule, 50, seed=1)
    assert isinstance(determined, Ok) and determined.value is None


def _check_sum_rule() -> None:
    path = DATA / "sum_rule.viba"
    module, defs = _load(path)
    rule = _entry(defs, module, "SumRule")
    names = [d.name for d in viba_ast.rule_definitions(module.module)]
    assert names == ["SmallRule", "BigRule", "SumRule"], f"markers: {names}"
    for name in names:
        _spec(defs, module, name)
    verdicts = set()
    for witness in generate_witnesses(rule, 40, seed=3):
        judged = is_compliant(witness, rule)
        assert isinstance(judged, Ok), f"sum judge errored: {judged!r}"
        verdicts.add(judged.value)
    assert verdicts == {True, False}, f"sum verdicts: {verdicts}"
    determined = check_determinate(rule, 40, seed=3)
    assert isinstance(determined, Ok) and determined.value is None
    for witness, want in (("SumWitnessPass", True), ("SumWitnessFail", False)):
        sub = _entry(defs, module, witness)
        got = is_compliant(sub, rule)
        assert isinstance(got, Ok) and got.value is want, f"{witness}: {got!r}"


def _check_broken_rules() -> None:
    module, defs = _load(DATA / "broken_rules.viba")
    for name in ("BadRule", "BadRef"):
        rule = _entry(defs, module, name)
        assert isinstance(check_determinate(rule, 5, seed=1), Err), name


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
    for witness in generate_witnesses(rule, 40, seed=3):
        judged = is_compliant(witness, rule)
        assert isinstance(judged, Ok), f"not judge errored: {judged!r}"
        verdicts.add(judged.value)
    assert verdicts == {True, False}, f"not verdicts: {verdicts}"
    determined = check_determinate(rule, 40, seed=3)
    assert isinstance(determined, Ok) and determined.value is None


def _check_predicate_reset():
    """One int metric, a predicate requiring value < 50, and 100
    witnesses whose metric runs 0..99: before running the code every
    witness is compliant, after reset_predication_by_python_code only
    the 50 below 50 are."""
    module, defs = _load(DATA / "predicate_bound.viba")
    _spec(defs, module, "PredicateBoundRule")
    rule = _entry(defs, module, "PredicateBoundRule")
    check = _predicate_node(rule.ast_node)
    witnesses = [_bound_witness(module, value, check) for value in range(100)]
    before = sum(is_compliant(w, rule).value for w in witnesses)
    after = sum(is_compliant(reset_predication_by_python_code(w), rule).value for w in witnesses)
    assert before == 100, f"before reset: compliant {before}/100"
    assert after == 50, f"after reset: compliant {after}/100"
    print(f"predicate reset: {before}/100 -> {after}/100")


def _predicate_node(body):
    for node in viba_ast.walk(body):
        if isinstance(node, viba_ast.TypeApp) and node.constructor == "Predicate":
            return node
    raise AssertionError("the rule has no Predicate field")


def _bound_witness(module, value, check):
    body = viba_ast.Product(
        viba_ast.TypeRef("Object"),
        viba_ast.Product(
            viba_ast.Tagged("$len", viba_ast.Constant(value)),
            viba_ast.Tagged("$check", check),
        ),
    )
    return AstNodeType(body, module)


def _check_broken_predicates():
    """$python_code that cannot run — syntax error, raise, missing
    attribute, wrong signature, undefined name, no predicate entry —
    must make the rule non-determinate."""
    broken = sorted((DATA / "broken_predicates").glob("*.viba"))
    assert broken, "no broken predicate samples"
    for path in broken:
        module, defs = _load(path)
        rule = _entry(defs, module, "BrokenRule")
        given = check_determinate(rule, 3, seed=1)
        assert isinstance(given, Err), f"{path.name}: {given!r}"
    print(f"broken predicate code: {len(broken)} samples rejected")


def _check_predicate_code():
    """generate_witnesses(fail_prob=0) leaves every Predicate in place; then each
    compiled $python_code runs and a false predication becomes the
    poison. The compliant share is therefore a measured value between 0
    and 1, not 100%."""
    total = compliant = 0
    for path in sorted((DATA / "rules").glob("rule*.viba")):
        number = path.stem[len("rule"):]
        module, defs = _load(path)
        rule = _entry(defs, module, f"Rule{number}")
        witnesses = generate_witnesses(rule, PREDICATE_WITNESSES, seed=int(number), fail_prob=0.0)
        for witness in witnesses:
            given = is_compliant(reset_predication_by_python_code(witness), rule)
            assert isinstance(given, Ok), f"{path.name}: judge errored: {given!r}"
            compliant += given.value
            total += 1
    assert 0 < compliant < total, f"predicate code: compliant {compliant}/{total}"
    print(f"predicate code: compliant {compliant}/{total} = {compliant / total:.1%}")


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
    _check_broken_predicates()
    _check_predicate_reset()
    _check_predicate_code()
    print(f"rule_coding_style_check: {len(paths)} rules + {len(not_paths)} not-rules"
          f" ({total} verdict witnesses + {determinacy} determinacy witnesses)"
          f" + demo + sum-rule + not-rule + 2 broken rules rejected"
          f" + broken predicates rejected + predicate reset + predicate code")


main()

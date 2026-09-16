"""Rule determinacy check: is the rule well-formed, and can every
generated witness be judged?

Result[None]: Ok(None) when the rule is determinate, Err(reason)
otherwise. First check_rule_coding_style(rule) (viba-rule.md): a rule that
violates the writing conventions is not determinate. Then quantify over
generate_witnesses(rule, count, seed, fail_prob=0): no Predicate is flipped by the
generator, every witness runs its compiled $python_code through
reset_predication_by_python_code, and is_compliant judges the result.
An Err — an unresolvable name, an ellipsis, or a predicate that raises —
makes the rule not determinate.
"""

from viba.rule.generate_witnesses import generate_witnesses
from viba.rule.is_compliant import is_compliant
from viba.rule.reset_predication_by_python_code import reset_predication_by_python_code
from viba.rule.check_rule_coding_style import check_rule_coding_style
from viba.type import Err, Ok


def check_determinate(rule, count: int, seed=None):
    """Result[None]: Ok(None) when the rule follows viba-rule.md, its
    predicate code runs, and every generated witness judges without
    error; Err(reason) otherwise."""
    style = check_rule_coding_style(rule)
    if isinstance(style, Err):
        return style
    witnesses = generate_witnesses(rule, count, seed, fail_prob=0.0)
    for index, witness in enumerate(witnesses):
        try:
            witness = reset_predication_by_python_code(witness)
        except Exception as exc:  # the predicate code must actually run
            return Err(f"witness #{index}: predicate raised {exc!r}")
        judged = is_compliant(witness, rule)
        if isinstance(judged, Err):
            return Err(f"witness #{index}: {judged.message}")
    return Ok(None)

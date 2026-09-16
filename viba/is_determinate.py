"""Rule determinacy check: is the rule well-formed, and can every
generated witness be judged?

First rule_coding_style_check(rule) (rule.md): a rule that violates the
writing conventions is not determinate. Then quantify over
generate(rule, count, seed): if any witness makes is_compliant Err, the
rule is not determinate and the Err carries the failing witness's index.
Ok(True) means the rule is well-formed and every sampled witness judged
cleanly true/false.
"""

from viba.generate import generate
from viba.is_compliant import is_compliant
from viba.rule_coding_style_check import rule_coding_style_check
from viba.type import Err, Ok


def is_determinate(rule, count: int, seed=None):
    """Result[bool]: the rule follows rule.md and every generated
    witness judges without error."""
    style = rule_coding_style_check(rule)
    if isinstance(style, Err):
        return style
    for index, witness in enumerate(generate(rule, count, seed)):
        judged = is_compliant(witness, rule)
        if isinstance(judged, Err):
            return Err(f"witness #{index}: {judged.message}")
    return Ok(True)

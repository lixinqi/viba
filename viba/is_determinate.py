"""Rule determinacy check: can every generated witness be judged?

Quantifies over generate(rule, count, seed): if any witness makes
is_compliant Err, the rule is not determinate and the Err carries
the failing witness's index. Ok(True) means every sampled witness
judged cleanly true/false.
"""

from viba.generate import generate
from viba.is_compliant import is_compliant
from viba.type import Err, Ok


def is_determinate(rule, count: int, seed=None):
    """Result[bool]: all generated witnesses judge without error."""
    for index, witness in enumerate(generate(rule, count, seed)):
        judged = is_compliant(witness, rule)
        if isinstance(judged, Err):
            return Err(f"witness #{index}: {judged.message}")
    return Ok(True)

"""Rule determinacy check: can every generated instance be judged?

Quantifies over generate(rule, count, seed): if any instance makes
is_compliant Err, the rule is not determinate and the Err carries
the failing instance's index. Ok(True) means every sampled instance
judged cleanly true/false.
"""

from viba.generate import generate
from viba.is_compliant import is_compliant
from viba.type import Err, Ok


def is_determinate(rule, count: int, seed=None):
    """Result[bool]: all generated instances judge without error."""
    for index, instance in enumerate(generate(rule, count, seed)):
        judged = is_compliant(instance, rule)
        if isinstance(judged, Err):
            return Err(f"instance #{index}: {judged.message}")
    return Ok(True)

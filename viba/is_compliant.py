"""Compliance verdict: does one witness satisfy the rule?

Thin wrapper over is_sub_type — a witness is compliant iff it is
a structural subtype of the rule. On a legal rule the verdict is
always Ok(True) or Ok(False); Err reports judgment errors
(unresolvable references, ellipsis, PredicationFailed on the
rule side).
"""

from viba.is_sub_type import is_sub_type


def is_compliant(witness, rule):
    """Result[bool]: witness <: rule."""
    return is_sub_type(witness, rule)

"""Compliance judgment: does one instance satisfy the rule?

Thin wrapper over is_sub_type — an instance is compliant iff it is
a structural subtype of the rule. On a legal rule the verdict is
always Ok(True) or Ok(False); Err reports judgment errors
(unresolvable references, ellipsis, AssertionViolated on the
rule side).
"""

from viba.is_sub_type import is_sub_type


def is_compliant(instance, rule):
    """Result[bool]: instance <: rule."""
    return is_sub_type(instance, rule)

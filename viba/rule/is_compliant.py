"""Compliance verdict: does one witness satisfy the rule?

Thin wrapper over is_sub_type — a witness is compliant iff it is
a structural subtype of the rule. On a legal rule the verdict is
always Ok(True) or Ok(False); Err reports judgment errors
(unresolvable references, ellipsis).
"""

from viba.is_sub_type import is_sub_type

# What a never-headed chain (a prohibition, once unfolded) accepts at a branch
# where `never` itself would do. Naming it is this layer's business: the core
# is told the word, it does not know it.
TERMINATORS = frozenset({"PredicationFailed"})


def is_compliant(witness, rule):
    """Result[bool]: witness <: rule."""
    return is_sub_type(witness, rule, terminators=TERMINATORS)

"""Compliance verdict: does one witness satisfy the rule?

Thin wrapper over is_sub_type — a witness is compliant iff it is
a structural subtype of the rule. On a legal rule the verdict is
always Ok(True) or Ok(False); Err reports judgment errors
(unresolvable references, ellipsis).
"""

from viba.is_sub_type import is_sub_type
from viba.reflect import Config, language_config

# What a never-headed chain (a prohibition, once unfolded) accepts at a branch
# where `never` itself would do. Naming it is this layer's business: the core
# is told the word, it does not know it.
TERMINATORS = frozenset({"PredicationFailed"})

# The units a rule judgment reads: the language's own (a metric function's
# trailing `Hint[$python_code {...}]`, a metric object's `Appendix[{...}]`)
# plus this layer's rule marker, which a witness does not write
# (viba-rule.md section 8). Predicate and PredicationFailed are deliberately
# not units here — a predication is a type to compare, which is what makes a
# failed one judge False. The address layer's own config says otherwise
# because it reads, not judges: viba/rule/reflect.py.
judgment_config = Config(never_eqv=set(language_config.never_eqv),
                         nil_eqv=set(language_config.nil_eqv) | {"RuleObject"})


def is_compliant(witness, rule):
    """Result[bool]: witness <: rule.

    The judgment reads the units in judgment_config above."""
    return is_sub_type(witness, rule, terminators=TERMINATORS,
                       config=judgment_config)

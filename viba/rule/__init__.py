"""The rule layer: a Viba application for declaring and judging rules.

The core stays in viba — is_sub_type over the Type model, the AST and
the parser. The rule vocabulary (RuleObject, Predicate,
Metric, PredicationFailed, not) lives in viba/builtin.viba, and these
modules carry the accumulated rule-level API:

- generate_witnesses: random witnesses of a rule;
- reset_predication_by_python_code: run each Predicate's $python_code
  and swap a false predication for the poison;
- is_compliant: witness <: rule;
- is_shape_compatible: the witness shape alone, predication leaves erased;
- check_rule_coding_style: the viba-rule.md writing conventions;
- check_determinate: well-formed, predicate code runs, every witness
  judges without error.
"""

from viba.rule.check_determinate import check_determinate
from viba.rule.generate_witnesses import generate_witnesses
from viba.rule.is_compliant import is_compliant
from viba.rule.is_shape_compatible import is_shape_compatible
from viba.rule.reset_predication_by_python_code import reset_predication_by_python_code
from viba.rule.check_rule_coding_style import check_rule_coding_style

__all__ = [
    "generate_witnesses",
    "reset_predication_by_python_code",
    "is_compliant",
    "is_shape_compatible",
    "check_rule_coding_style",
    "check_determinate",
]

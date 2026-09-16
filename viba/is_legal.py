"""The two verdicts over rules and instances.

- is_compliant(instance, rule): does the instance satisfy the rule?
  Ok(True/False) is the compliance verdict; Err is an authoring
  mistake, never a verdict.
- is_determinate(rule): is the rule unambiguous? Instances are
  generated and judged; Ok(True) means every instance judged true
  or false without error, Err reports a counterexample instance.
"""

from viba.instance_generator import generate
from viba.is_sub_type import is_sub_type
from viba.type import AstNodeType, Err, Ok, Result


def is_compliant(instance: AstNodeType, rule: AstNodeType) -> Result:
    """Ok(True/False) is the compliance verdict; Err is authoring."""
    return is_sub_type(instance, rule)


def is_determinate(rule: AstNodeType, count: int = 100, seed=None) -> Result:
    """Ok(True): every generated instance judged true/false cleanly.
    Err: the rule is ambiguous — some instance made the judgment
    error out; the message identifies the counterexample."""
    for index, instance in enumerate(generate(rule, count, seed)):
        verdict = is_sub_type(instance, rule)
        if isinstance(verdict, Err):
            return Err(f"instance #{index}: {verdict.message}")
    return Ok(True)

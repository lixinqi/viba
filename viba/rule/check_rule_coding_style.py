"""Rule coding-style check: does a rule follow viba-rule.md?

check_rule_coding_style(rule) -> Result[None]. Ok(None) when the rule
follows the writing conventions; Err names the first violation.

Checked (viba-rule.md section in parentheses):
- the chain head is RuleObject or OneofRule (2);
- a RuleObject body is a product, a OneofRule body is a sum (2);
- a OneofRule branch names another rule (2);
- untagged product fields come before tagged ones (3);
- Predicate[Cond, $python_code Code] carries exactly those two
  arguments, and Code defines `predicate` (4);
- Metric[Name] carries exactly one argument (5);
- Predicate and not are tagged product fields, never sum branches (6);
- not[A] carries exactly one argument (7).
"""

from viba.ast import nodes as ast_nodes
from viba.ast.rules import PRODUCT_MARKER, SUM_MARKER, body_marker
from viba.type import AstNodeType, Err, Ok, Result, module_get_type


class _Violation(Exception):
    pass


def check_rule_coding_style(rule: AstNodeType) -> Result:
    body = getattr(rule.ast_node, "body", rule.ast_node)
    module = rule.container_module
    marker = body_marker(body)
    if marker is None:
        return Err("missing RuleObject/OneofRule marker")
    try:
        if marker == PRODUCT_MARKER:
            _check_product(body, module, top=True)
        else:
            _check_oneof(body, module)
    except _Violation as exc:
        return Err(str(exc))
    return Ok(None)


def _product_elements(node):
    if isinstance(node, ast_nodes.Product):
        return _product_elements(node.left) + _product_elements(node.right)
    if isinstance(node, ast_nodes.ProductChain):
        return list(node.elements)
    return [node]


def _sum_elements(node):
    if isinstance(node, ast_nodes.Sum):
        return _sum_elements(node.left) + _sum_elements(node.right)
    if isinstance(node, ast_nodes.SumChain):
        return list(node.elements)
    return [node]


def _check_product(node, module, top=False):
    elements = _product_elements(node)
    if top:
        if not elements or not _is_marker(elements[0], PRODUCT_MARKER):
            raise _Violation("a RuleObject body must open with RuleObject")
        elements = elements[1:]
    tagged_seen = False
    for element in elements:
        if isinstance(element, ast_nodes.Tagged):
            tagged_seen = True
            _check_field(element.type, module, tagged=True, in_sum=False)
        else:
            if tagged_seen:
                raise _Violation("untagged product field after a tagged one")
            _check_field(element, module, tagged=False, in_sum=False)


def _check_oneof(node, module):
    elements = _sum_elements(node)
    if not elements or not _is_marker(elements[0], SUM_MARKER):
        raise _Violation("a OneofRule body must open with OneofRule")
    for element in elements[1:]:
        reference = element.type if isinstance(element, ast_nodes.Tagged) else element
        if not isinstance(reference, ast_nodes.TypeRef):
            raise _Violation("a OneofRule branch must name another rule")
        if not _names_a_rule(reference.name, module):
            raise _Violation(f"OneofRule branch {reference.name!r} is not a rule")


def _check_field(node, module, tagged, in_sum):
    if isinstance(node, ast_nodes.Tagged):
        _check_field(node.type, module, tagged=True, in_sum=in_sum)
        return
    if isinstance(node, (ast_nodes.Product, ast_nodes.ProductChain)):
        _check_product(node, module)
        return
    if isinstance(node, (ast_nodes.Sum, ast_nodes.SumChain)):
        _check_sum(node, module)
        return
    if isinstance(node, ast_nodes.TypeApp):
        if node.constructor == "Predicate":
            _check_predicate(node, tagged, in_sum)
        elif node.constructor == "Metric":
            _check_tagged(tagged, "Metric")
            _expect_args(node, 1, "Metric")
        elif node.constructor == "not":
            _check_tagged(tagged, "not")
            _expect_args(node, 1, "not")


def _check_sum(node, module):
    for element in _sum_elements(node):
        if isinstance(element, ast_nodes.Tagged):
            _check_field(element.type, module, tagged=True, in_sum=True)
        else:
            _check_field(element, module, tagged=False, in_sum=True)


def _check_predicate(node, tagged, in_sum):
    if in_sum:
        raise _Violation("Predicate may not sit in a sum branch")
    _check_tagged(tagged, "Predicate")
    _expect_args(node, 2, "Predicate")
    cond, code = node.args
    if not isinstance(cond, ast_nodes.CodeBlock):
        raise _Violation("Predicate's first argument must be a {...} condition")
    if not (isinstance(code, ast_nodes.Tagged) and code.tag == "$python_code"):
        raise _Violation("Predicate's second argument must be $python_code {...}")
    if not isinstance(code.type, ast_nodes.CodeBlock):
        raise _Violation("$python_code must carry a {...} block")
    if "def predicate(" not in code.type.code:
        raise _Violation("$python_code must define predicate")


def _check_tagged(tagged, name):
    if not tagged:
        raise _Violation(f"{name} must be a tagged field")


def _expect_args(node, count, name):
    if len(node.args) != count:
        raise _Violation(f"{name} takes exactly {count} argument(s)")


def _names_a_rule(name, module):
    resolved = module_get_type(module, name)
    if not isinstance(resolved, Ok) or not isinstance(resolved.value, AstNodeType):
        return False
    node = resolved.value.ast_node
    return body_marker(getattr(node, "body", node)) is not None


def _is_marker(node, name):
    return isinstance(node, ast_nodes.TypeRef) and node.name == name

"""Shape comparison: is this witness shaped like that rule?

is_shape_compatible(sub, sup) -> Result[bool]. Only the predication leaves
are erased: every Predicate[...] and PredicationFailed[...] — including one
hidden behind a type name, which unfolds first — becomes the same leaf. So
the two are interchangeable here, while everything else (metric values and
containers, tags, products, sums, not[...] shells) is left alone.

The witness is then required to be a subtype of the rule. That is weaker
than is_compliant only in the predications: the metrics must still have the
right data type, but it does not matter whether an assertion passed. So it
answers "is the witness shape ready".

Two notes on the shape of this:
- the erased leaf is PredicationFailed, not nil: a not[...] branch must be
  PredicationFailed for the shell rule to hold, so not[nil] <: not[nil] is
  False and nil leaves would fail every prohibitive rule;
- the comparison is one-way (sub <: sup), not mutual: a sum-typed metric
  unfolds to a sum while a witness carries one chosen branch, which is a
  legitimate shape but not equal to the sum.
"""

from viba.viba_ast import nodes as ast_nodes
from viba.type import AstNodeType, Err, Ok, Result, module_get_type
from viba.is_sub_type import is_sub_type


def is_shape_compatible(sub: AstNodeType, sup: AstNodeType) -> Result:
    """Ok(True) when the sub shape fits the sup shape."""
    fits = is_sub_type(erase_predications(sub), erase_predications(sup))
    if isinstance(fits, Err):
        return fits
    return Ok(bool(fits.value))


def erase_predications(rule: AstNodeType) -> AstNodeType:
    """The rule with every predication leaf replaced by the poison."""
    return AstNodeType(_erase(rule.ast_node, rule.container_module), rule.container_module)


def _leaf():
    return ast_nodes.TypeApp("PredicationFailed", [ast_nodes.Nil(), ast_nodes.TypeRef("str")])


def _erase(node, module):
    if isinstance(node, ast_nodes.Tagged):
        return ast_nodes.Tagged(node.tag, _erase(node.type, module))
    if isinstance(node, ast_nodes.Product):
        return ast_nodes.Product(_erase(node.left, module), _erase(node.right, module))
    if isinstance(node, ast_nodes.ProductChain):
        return ast_nodes.ProductChain([_erase(e, module) for e in node.elements])
    if isinstance(node, ast_nodes.Sum):
        return ast_nodes.Sum(_erase(node.left, module), _erase(node.right, module))
    if isinstance(node, ast_nodes.SumChain):
        return ast_nodes.SumChain([_erase(e, module) for e in node.elements])
    if isinstance(node, ast_nodes.Tuple):
        return ast_nodes.Tuple([_erase(e, module) for e in node.elements])
    if isinstance(node, ast_nodes.Exponent):
        return ast_nodes.Exponent(_erase(node.result, module), _erase(node.argument, module))
    if isinstance(node, ast_nodes.ExponentChain):
        return ast_nodes.ExponentChain([_erase(e, module) for e in node.elements])
    if isinstance(node, ast_nodes.TypeRef):
        body, home = _unfold(node, module)
        return _erase(body, home) if body is not node else node
    if isinstance(node, ast_nodes.TypeApp):
        if node.constructor in ("Predicate", "PredicationFailed"):
            return _leaf()
        if node.constructor == "not" and len(node.args) == 1:
            return ast_nodes.TypeApp("not", [_erase(node.args[0], module)])
    return node


def _unfold(node, module):
    """A TypeRef to a plain TypeDefinition unfolds to its body."""
    resolved = module_get_type(module, node.name)
    if not isinstance(resolved, Ok) or not isinstance(resolved.value, AstNodeType):
        return node, module
    target = resolved.value
    if not isinstance(target.ast_node, ast_nodes.TypeDefinition):
        return node, module
    return target.ast_node.body, target.container_module

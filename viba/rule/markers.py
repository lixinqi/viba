"""The rule marker: RuleObject.

A rule is a plain type definition whose chain head carries the marker,
RuleObject (like Object, product identity, cardinality 1). The name is a
syntactic synonym of Object in every judgment; it exists so a module can
declare which of its definitions are rules. A definition whose body is a
sum is not a rule: the sum identity is Oneof, and it says nothing about
rules.

The marker is this layer's word, so it lives here and not in the syntax
package: viba.viba_ast knows nodes, not rules.
"""

from typing import List, Optional

from viba.viba_ast import nodes as ast_nodes

PRODUCT_MARKER = "RuleObject"


def rule_marker(defn) -> Optional[str]:
    """PRODUCT_MARKER if defn is a rule, else None."""
    return body_marker(getattr(defn, "body", None))


def body_marker(body) -> Optional[str]:
    """PRODUCT_MARKER if a definition body is a rule body. The marker may be
    written through an import (`rule.RuleObject`): what counts is the name,
    not the prefix it was reached by."""
    head = _chain_head(body)
    if isinstance(head, ast_nodes.TypeRef) and head.name.split(".")[-1] == PRODUCT_MARKER:
        return PRODUCT_MARKER
    return None


def rule_definitions(module) -> List:
    """The definitions of `module` that carry a rule marker."""
    kinds = (ast_nodes.TypeDefinition, ast_nodes.GenericDefinition)
    return [d for d in module.body if isinstance(d, kinds) and rule_marker(d)]


def _chain_head(node):
    while isinstance(node, (ast_nodes.Product, ast_nodes.Sum)):
        node = node.left
    if isinstance(node, (ast_nodes.ProductChain, ast_nodes.SumChain)) and node.elements:
        return node.elements[0]
    return node

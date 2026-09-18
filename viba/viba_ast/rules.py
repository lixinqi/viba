"""Rule markers: RuleObject / OneofRule.

A rule is a plain type definition whose chain head carries a marker:
RuleObject (like Object, product identity, cardinality 1) or
OneofRule (like Oneof, sum identity, cardinality 0). The markers are
syntactic synonyms of Object/Oneof in every judgment; they exist so a
module can declare which of its definitions are rules.
"""

from typing import List, Optional

from viba.viba_ast.nodes import (
    GenericDefinition,
    Module,
    Product,
    ProductChain,
    Sum,
    SumChain,
    TypeDefinition,
    TypeRef,
)

PRODUCT_MARKER = "RuleObject"
SUM_MARKER = "OneofRule"


def rule_marker(defn) -> Optional[str]:
    """PRODUCT_MARKER | SUM_MARKER if defn is a rule, else None."""
    return body_marker(getattr(defn, "body", None))


def body_marker(body) -> Optional[str]:
    """PRODUCT_MARKER | SUM_MARKER if a definition body is a rule body."""
    head = _chain_head(body)
    if isinstance(head, TypeRef) and head.name in (PRODUCT_MARKER, SUM_MARKER):
        return head.name
    return None


def rule_definitions(module: Module) -> List[TypeDefinition]:
    """The definitions of `module` that carry a rule marker."""
    kinds = (TypeDefinition, GenericDefinition)
    return [d for d in module.body if isinstance(d, kinds) and rule_marker(d)]


def _chain_head(node):
    while isinstance(node, (Product, Sum)):
        node = node.left
    if isinstance(node, (ProductChain, SumChain)) and node.elements:
        return node.elements[0]
    return node

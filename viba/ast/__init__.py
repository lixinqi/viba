"""viba.ast — python-ast-style interface over Viba source.

Public API (cf. the standard-library ast module):
    parse(source)   -> Module
    unparse(tree)   -> canonical chain-style Viba source
    canonical(tree) -> chain-converted Module
    dump(node)      -> repr of a node tree
    walk / iter_child_nodes
    NodeVisitor / NodeTransformer

The parser is imported lazily inside parse() to avoid a package cycle
(viba.parser builds viba.ast.nodes objects in its grammar actions).
"""

from collections import deque
from typing import Any, Iterator, List, Optional

from viba.ast.nodes import (
    AST,
    Module,
    TypeDefinition,
    GenericDefinition,
    Import,
    Sum,
    Product,
    Exponent,
    Tagged,
    TypeApp,
    Tuple,
    TypeRef,
    Constant,
    Nil,
    Never,
    Ellipsis,
    CodeBlock,
    SumChain,
    ProductChain,
    ExponentChain,
)
from viba.ast.chain import convert_to_chain_style, convert_from_chain_style
from viba.ast.unparse import unparse_module
from viba.ast.rules import (
    PRODUCT_MARKER,
    SUM_MARKER,
    rule_definitions,
    rule_marker,
)

__all__ = [
    "AST", "Module", "TypeDefinition", "GenericDefinition", "Import",
    "Sum", "Product", "Exponent", "Tagged",
    "TypeApp", "Tuple", "TypeRef", "Constant", "Nil", "Never", "Ellipsis",
    "CodeBlock", "SumChain", "ProductChain", "ExponentChain",
    "parse", "unparse", "canonical", "dump",
    "convert_to_chain_style", "convert_from_chain_style",
    "iter_child_nodes", "walk", "NodeVisitor", "NodeTransformer",
    "PRODUCT_MARKER", "SUM_MARKER", "rule_definitions", "rule_marker",
]


def parse(source: str) -> Module:
    """Parse Viba source into a Module (cf. ast.parse)."""
    from viba.parser import parser as _ply_parser

    program = _ply_parser.parse(source) or []
    return Module(program)


def canonical(tree: Module) -> Module:
    """Return a chain-style canonical copy of the Module."""
    return Module([convert_to_chain_style(defn) for defn in tree.body])


def unparse(tree: Module) -> str:
    """Convert a Module back to canonical Viba source (cf. ast.unparse)."""
    return unparse_module(canonical(tree))


# ----------------------------------------------------------------------
# Traversal (cf. ast.iter_child_nodes / ast.walk)
# ----------------------------------------------------------------------


def iter_child_nodes(node: AST) -> Iterator[AST]:
    """Yield all direct child nodes of `node`."""
    for name in node._fields:
        value = getattr(node, name, None)
        if isinstance(value, AST):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, AST):
                    yield item


def walk(node: AST) -> Iterator[AST]:
    """Breadth-first traversal of `node` and all its descendants."""
    queue = deque([node])
    while queue:
        current = queue.popleft()
        yield current
        queue.extend(iter_child_nodes(current))


# ----------------------------------------------------------------------
# Visitors (cf. ast.NodeVisitor / ast.NodeTransformer)
# ----------------------------------------------------------------------


class NodeVisitor:
    """Walk the AST calling visit_<ClassName> methods."""

    def visit(self, node: AST) -> Any:
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: AST) -> Any:
        for child in iter_child_nodes(node):
            self.visit(child)


class NodeTransformer(NodeVisitor):
    """Walk the AST and allow visit methods to replace or drop nodes."""

    def generic_visit(self, node: AST) -> AST:
        for name in node._fields:
            value = getattr(node, name, None)
            if isinstance(value, AST):
                new_node = self.visit(value)
                if new_node is None:
                    setattr(node, name, None)
                elif not isinstance(new_node, AST):
                    raise TypeError(
                        f"{self.__class__.__name__}: visit method must "
                        f"return an AST node or None, got {type(new_node)}"
                    )
                else:
                    setattr(node, name, new_node)
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if not isinstance(item, AST):
                        new_list.append(item)
                        continue
                    new_item = self.visit(item)
                    if new_item is None:
                        continue
                    if not isinstance(new_item, AST):
                        raise TypeError(
                            f"{self.__class__.__name__}: visit method must "
                            f"return an AST node or None, got {type(new_item)}"
                        )
                    new_list.append(new_item)
                setattr(node, name, new_list)
        return node


# ----------------------------------------------------------------------
# dump (cf. ast.dump)
# ----------------------------------------------------------------------


def _dump_value(value: Any, annotate_fields: bool, indent: Optional[int], level: int) -> str:
    if isinstance(value, AST):
        return _dump_node(value, annotate_fields, indent, level + 1)
    if isinstance(value, list):
        return "[" + ", ".join(
            _dump_value(item, annotate_fields, indent, level)
            for item in value
        ) + "]"
    return repr(value)


def _dump_node(node: AST, annotate_fields: bool, indent: Optional[int], level: int) -> str:
    cls = node.__class__.__name__
    fields = []
    for name in node._fields:
        value = getattr(node, name, None)
        rendered = _dump_value(value, annotate_fields, indent, level + 1)
        fields.append(f"{name}={rendered}" if annotate_fields else rendered)

    if indent is None:
        return f"{cls}({', '.join(fields)})"

    # Indented, one node per line
    pad = " " * (indent * (level + 1))
    closing = " " * (indent * level)
    if not fields:
        return f"{cls}()"
    sep = ",\n" + pad
    return f"{cls}(\n{pad}{sep.join(fields)},\n{closing})"


def dump(node: AST, annotate_fields: bool = True, *, indent: Optional[int] = None) -> str:
    """Return a string representation of the AST (cf. ast.dump)."""
    return _dump_node(node, annotate_fields, indent, 0)

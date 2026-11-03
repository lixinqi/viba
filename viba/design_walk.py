"""viba.design_walk — the small readings of a design's source that its walks share.

Two layers that walk a design and resolve its names — `extract_sub_repo` and
`check_convertable_to_json_schema` — need the same few readings: the elements of a chain,
the head and arguments of a call, a type name taken apart, and what tells one
walk of a definition from another. They are collected here so both walks read a
design the same way.

Nothing here resolves a name or holds a module: these are the readings of the
syntax, and of the bindings a decision hands over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from viba import viba_ast
from viba.type import BUILTIN_DIR

__all__ = [
    "chain_elements", "call_parts", "split_type_name", "is_definition",
    "is_function",
    "with_params", "bindings_key", "longest_import", "is_under",
    "under_builtin", "module_label",
]


def chain_elements(node):
    """Sum / product / exponent as the elements of the main chain, in source order."""
    if isinstance(node, (viba_ast.SumChain, viba_ast.ProductChain,
                         viba_ast.ExponentChain)):
        return list(node.elements)
    if isinstance(node, (viba_ast.Sum, viba_ast.Product)):
        elements, current = [], node
        while isinstance(current, type(node)):
            elements.append(current.right)
            current = current.left
        elements.append(current)
        elements.reverse()
        return elements
    if isinstance(node, viba_ast.Exponent):
        elements, current = [], node
        while isinstance(current, viba_ast.Exponent):
            elements.append(current.argument)
            current = current.result
        elements.append(current)
        elements.reverse()
        return elements
    return None


def call_parts(node):
    """A call taken apart: (the head, the arguments in the order the source has them)."""
    arguments = []
    while isinstance(node, viba_ast.Partial):
        arguments.append(node.argument)
        node = node.function
    return node, list(reversed(arguments))


def split_type_name(name: str):
    """`pkg.mod.Name` -> (`pkg.mod`, `Name`); the name before the last dot is the module."""
    module_name, dot, definition = name.rpartition(".")
    if not dot or not module_name or not definition:
        return "", name
    return module_name, definition


def is_function(definition) -> bool:
    """A definition whose body is a function (`A <- B`) or a call (`f << ...`)."""
    body = definition.body
    return isinstance(body, (viba_ast.Partial, viba_ast.Exponent,
                             viba_ast.ExponentChain))


def is_definition(node) -> bool:
    return isinstance(node, (viba_ast.TypeDefinition, viba_ast.GenericDefinition))


def with_params(definition, bindings: dict) -> dict:
    """The bindings with the definition's own generic parameters standing as placeholders."""
    inner = dict(bindings)
    for param in getattr(definition, "generic_params", None) or ():
        inner.setdefault(param, None)
    return inner


def bindings_key(bindings: dict) -> tuple:
    """What tells one walk of a definition from another: the arguments it stands in."""
    return tuple(sorted(
        (name, "placeholder" if bound is None
         else (bound[1].file_path or "", viba_ast.dump(bound[0])))
        for name, bound in bindings.items()))


def longest_import(imports: dict, name: str) -> Optional[str]:
    """The import local name a dotted name goes through, or None."""
    for local in sorted(imports, key=len, reverse=True):
        if name == local or name.startswith(local + "."):
            return local
    return None


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def under_builtin(file_path) -> bool:
    """Is this file one of the package's own (the interpreter's vocabulary)?"""
    return is_under(Path(file_path).resolve(), BUILTIN_DIR)


def module_label(module) -> str:
    return getattr(module, "file_path", None) or getattr(module, "name", "") or "?"

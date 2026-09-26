"""Class-based pattern dispatch over AST nodes (package-internal).

Handlers are passed by class name, e.g.:

    viba_type_match(node, Sum=lambda s: ..., Product=lambda p: ...,
                    _=lambda t: ...)
"""

from typing import Any, Callable

from viba.viba_ast.nodes import AST


def viba_type_match(
    node: AST,
    Module: Callable[[Any], Any] = None,
    TypeDefinition: Callable[[Any], Any] = None,
    GenericDefinition: Callable[[Any], Any] = None,
    Import: Callable[[Any], Any] = None,
    Sum: Callable[[Any], Any] = None,
    Product: Callable[[Any], Any] = None,
    Exponent: Callable[[Any], Any] = None,
    Partial: Callable[[Any], Any] = None,
    Tagged: Callable[[Any], Any] = None,
    Member: Callable[[Any], Any] = None,
    Let: Callable[[Any], Any] = None,
    Binding: Callable[[Any], Any] = None,
    TypeApp: Callable[[Any], Any] = None,
    Tuple: Callable[[Any], Any] = None,
    TypeRef: Callable[[Any], Any] = None,
    Constant: Callable[[Any], Any] = None,
    Nil: Callable[[Any], Any] = None,
    Never: Callable[[Any], Any] = None,
    Any: Callable[[Any], Any] = None,
    Ellipsis: Callable[[Any], Any] = None,
    CodeBlock: Callable[[Any], Any] = None,
    SumChain: Callable[[Any], Any] = None,
    ProductChain: Callable[[Any], Any] = None,
    ExponentChain: Callable[[Any], Any] = None,
    _: Callable[[AST], Any] = None,  # default/wildcard handler
    strict: bool = True,  # if True, raise when no handler matches
) -> Any:
    """Dispatch on the node's concrete class name.

    Args:
        node: The AST instance to match against.
        <ClassName>: Handler for that node class.
        _: Default handler (if no specific handler matches).
        strict: If True (default), raise TypeError when no handler
            matches. If False, return None.

    Returns:
        The result of the matched handler.
    """
    class_name = node.__class__.__name__
    handler = locals().get(class_name)

    if handler is not None:
        return handler(node)

    if _ is not None:
        return _(node)

    if strict:
        raise TypeError(f"No handler matched for node: {class_name}")
    return None


def match_builder(strict: bool = True, **handlers):
    """Build a partial-match function with predefined handlers.

    Example:
        describe = match_builder(
            Sum=lambda s: "a sum",
            TypeRef=lambda r: f"ref {r.name}",
        )
        describe(node)  # TypeError if no handler matches and strict=True
    """

    def matcher(node: AST) -> Any:
        return viba_type_match(node, strict=strict, **handlers)

    return matcher

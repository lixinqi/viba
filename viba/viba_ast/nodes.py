"""Viba AST node classes (syntax and semantics in one layer).

The parser (viba.parser) builds these nodes directly; unparse, chain
canonicalization, dump and visitors all work on this single hierarchy.

Composite nodes:
    Module, TypeDefinition, GenericDefinition, Sum, Product, Exponent,
    Tagged, TypeApp, Tuple
Atomic nodes:
    TypeRef, Constant, Nil, Never, Ellipsis, CodeBlock
Canonical (chain-style) nodes, produced by convert_to_chain_style:
    SumChain, ProductChain, ExponentChain
"""

from typing import Any


class AST:
    """Base class of all Viba AST nodes (cf. ast.AST)."""

    _fields: tuple = ()

    def __init__(self, *args: Any, **kwargs: Any):
        if len(args) > len(self._fields):
            raise TypeError(
                f"{self.__class__.__name__} expected at most "
                f"{len(self._fields)} positional arguments, got {len(args)}"
            )
        for name, value in zip(self._fields, args):
            setattr(self, name, value)
        for name in self._fields[len(args):]:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
            else:
                setattr(self, name, None)
        if kwargs:
            raise TypeError(
                f"{self.__class__.__name__} got unexpected keyword "
                f"argument(s): {', '.join(kwargs)}"
            )


class Module(AST):
    """A whole .viba file: a sequence of definitions."""

    _fields = ("body",)


class TypeDefinition(AST):
    """name := body — a definition without generic parameters."""

    _fields = ("name", "body")


class GenericDefinition(AST):
    """name[T, U] := body — a definition with generic parameters."""

    _fields = ("name", "generic_params", "body")


class Sum(AST):
    """left | right"""

    _fields = ("left", "right")


class Product(AST):
    """left * right"""

    _fields = ("left", "right")


class Exponent(AST):
    """result <- argument (function type)"""

    _fields = ("result", "argument")


class Apply(AST):
    """function << argument — the written argument is given to the function."""

    _fields = ("function", "argument")


class Tagged(AST):
    """$tag Type"""

    _fields = ("tag", "type")


class TypeApp(AST):
    """Constructor[Arg, ...]"""

    _fields = ("constructor", "args")


class Tuple(AST):
    """(A, B, C) — positional product; distinct from Product (A * B)."""

    _fields = ("elements",)


class TypeRef(AST):
    """A named type reference."""

    _fields = ("name",)


class Constant(AST):
    """Literal int / float / str / bool value (cf. ast.Constant)."""

    _fields = ("value",)


class Nil(AST):
    """nil — the product identity. `void`/`None` parse here (aliases).
    The empty tuple () is NOT this node: it is Tuple([]), its own type
    that happens to share the cardinality 1."""

    _fields = ()


class Never(AST):
    """never — the sum identity."""

    _fields = ()


class Ellipsis(AST):
    """... — open/variadic marker."""

    _fields = ()


class CodeBlock(AST):
    """{ ... } — opaque code content, kept verbatim."""

    _fields = ("code",)


class Import(AST):
    """import a.b.c [as d] — module reference at the top level."""

    _fields = ("module", "alias")


class SumChain(AST):
    """Canonical main chain of a sum: the left-nested run of |.

    An element may itself be a SumChain — that is a branch, kept whole
    (`A | (B | C)` is `SumChain([A, SumChain([B, C])])`).
    """

    _fields = ("elements",)


class ProductChain(AST):
    """Canonical main chain of a product: the left-nested run of *.

    An element may itself be a ProductChain: a branch keeps its grouping
    (`A * (B * C)` is `ProductChain([A, ProductChain([B, C])])`).
    """

    _fields = ("elements",)


class ExponentChain(AST):
    """Canonical main chain of an exponent, in written order.

    Same shape as SumChain / ProductChain: `elements[0]` is the result
    (the leftmost thing written), the rest are the arguments in the order
    they are written. An element may itself be an ExponentChain — that is
    a branch: `A <- (B <- C)` is
    `ExponentChain([A, ExponentChain([B, C])])`.
    """

    _fields = ("elements",)

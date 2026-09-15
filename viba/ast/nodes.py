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
    """nil — the product identity. () parses to this as well.
    `void` and `None` are accepted aliases, canonicalized at parse."""

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
    """Canonical flat form of a sum: elements joined by |."""

    _fields = ("elements",)


class ProductChain(AST):
    """Canonical flat form of a product: elements joined by *."""

    _fields = ("elements",)


class ExponentChain(AST):
    """Canonical flat form of an exponent chain.

    `args` is application order: args[0] is fed first. The Viba chain
    syntax lists arguments right-to-left.
    """

    _fields = ("result", "args")

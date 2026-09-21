"""Unparse AST nodes back to Viba source."""

import re
from typing import List, Union

from viba.viba_ast._match import viba_type_match
from viba.viba_ast.nodes import (
    AST,
    Module,
    TypeDefinition,
    GenericDefinition,
    Import,
    Sum,
    Product,
    Exponent,
    Apply,
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

_BINARY_NAMES = (
    "Sum", "Product", "Exponent", "SumChain", "ProductChain", "ExponentChain",
)


def unparse_module(tree: Module, indent: int = 2) -> str:
    """Convert a Module (or a list of definitions) to Viba source."""
    definitions = tree.body if isinstance(tree, Module) else list(tree)
    return "\n\n".join(_unparse_type(d, indent, 0) for d in definitions)


def unparse_type(node: AST, indent: int = 2) -> str:
    """Convert a single type expression to Viba source."""
    return _unparse_type(node, indent, 0)


def _unparse_type(node: AST, indent: int, depth: int) -> str:
    """Convert a single node to Viba code."""

    def unparse_tag(t):
        return t.tag if t.tag.startswith("$") else f"${t.tag}"

    return viba_type_match(
        node,
        TypeDefinition=lambda d: _unparse_type_definition(d, indent, depth),
        GenericDefinition=lambda d: _unparse_generic_definition(d, indent, depth),
        Import=lambda i: _unparse_import(i),
        Sum=lambda s: _unparse_binary(s, " | ", indent, depth),
        Product=lambda p: _unparse_binary(p, " * ", indent, depth),
        Exponent=lambda e: _unparse_exponent(e, indent, depth),
        Apply=lambda a: _unparse_apply(a, indent, depth),
        Tagged=lambda t: f"{unparse_tag(t)}{_tagged_body_parens(t.type, _unparse_type(t.type, indent, depth), ' ' * (indent * depth))}",
        TypeApp=lambda a: _unparse_typeapp(a, indent, depth),
        Tuple=lambda t: _unparse_tuple(t, indent, depth),
        TypeRef=lambda r: r.name,
        Constant=lambda c: _unparse_constant(c),
        Nil=lambda v: "nil",
        Never=lambda n: "never",
        Ellipsis=lambda e: "...",
        CodeBlock=lambda c: _unparse_codeblock(c),
        SumChain=lambda s: _unparse_sumchain(s, indent, depth),
        ProductChain=lambda p: _unparse_productchain(p, indent, depth),
        ExponentChain=lambda e: _unparse_exponentchain(e, indent, depth),
    )


def _unparse_type_definition(defn: TypeDefinition, indent: int, depth: int) -> str:
    """Unparse a TypeDefinition (no generic parameters)."""
    prefix = " " * (indent * (depth + 1))
    body = _dedent(_unparse_type(defn.body, indent, depth + 1), prefix)
    return f"{defn.name} :=\n{prefix}{body}"


def _unparse_generic_definition(defn: GenericDefinition, indent: int, depth: int) -> str:
    """Unparse a GenericDefinition."""
    params = "[" + ", ".join(defn.generic_params) + "]"
    prefix = " " * (indent * (depth + 1))
    body = _dedent(_unparse_type(defn.body, indent, depth + 1), prefix)
    return f"{defn.name}{params} :=\n{prefix}{body}"


def _unparse_import(import_node: Import) -> str:
    """Unparse an Import: `import a.b [as c]`."""
    if import_node.alias:
        return f"import {import_node.module} as {import_node.alias}"
    return f"import {import_node.module}"


def _unparse_binary(bin_node: Union[Sum, Product], op: str, indent: int, depth: int) -> str:
    """Unparse binary Sum/Product trees."""
    left = _unparse_type(bin_node.left, indent, depth + 1)
    right = _unparse_type(bin_node.right, indent, depth + 1)

    left_needs_paren = _needs_parens(bin_node.left, bin_node.__class__.__name__, "left")
    right_needs_paren = _needs_parens(
        bin_node.right, bin_node.__class__.__name__, "right"
    )

    if left_needs_paren:
        left = f"({left})"
    if right_needs_paren:
        right = f"({right})"

    return f"{left}{op}{right}"


def _unparse_exponent(exp_node: Exponent, indent: int, depth: int) -> str:
    """Unparse an Exponent (function type)."""
    result = _unparse_type(exp_node.result, indent, depth + 1)
    argument = _unparse_type(exp_node.argument, indent, depth + 1)

    if _is_binary(exp_node.result):
        result = f"({result})"
    if _is_binary(exp_node.argument):
        argument = f"({argument})"

    return f"{result} <- {argument}"


def _unparse_apply(node: Apply, indent: int, depth: int) -> str:
    """Unparse `function << argument`."""
    function = _unparse_type(node.function, indent, depth + 1)
    argument = _unparse_type(node.argument, indent, depth + 1)

    if _is_binary(node.function):
        function = f"({function})"
    if _is_binary(node.argument):
        argument = f"({argument})"

    return f"{function} << {argument}"


def _unparse_typeapp(app_node: TypeApp, indent: int, depth: int) -> str:
    """Unparse a TypeApp: Name[a, b], and Name[] for an application of none."""
    args = ", ".join(
        _dedent(_unparse_type(arg, indent, depth + 1), " " * (indent * (depth + 1)))
        for arg in app_node.args)
    return f"{app_node.constructor}[{args}]"


def _unparse_tuple(tuple_node: Tuple, indent: int, depth: int) -> str:
    """Unparse a Tuple: (A, B, C), and (A,) for a tuple of one.

    One element needs the comma: `(A)` is just `A` when it is read back.
    """
    args = ", ".join(_unparse_type(e, indent, depth + 1) for e in tuple_node.elements)
    if len(tuple_node.elements) == 1:
        args += ","
    return f"({args})"


def _unparse_constant(const_node: Constant) -> str:
    """Unparse a Constant literal."""
    return literal_spelling(const_node.value)


# The lexer's own spelling of a number: `\d+` for an int, and a plain decimal
# for a float. No sign, no exponent — a viba literal is a non-negative number.
_INT_SPELLING = re.compile(r"^\d+$")
_FLOAT_SPELLING = re.compile(r"^(\d+\.\d*|\.\d+)$")


def literal_spelling(value) -> str:
    """The literal that reads back as this Python value.

    A string gets a delimiter that does not occur in it — double quotes,
    single quotes, or triple quotes when it spans lines — so the text survives
    the round trip untranslated (viba reads no escapes back). A number must be
    spelled the way the lexer reads numbers: viba has no negative literal and
    Python writes large and small floats with an exponent, so those raise
    ValueError rather than being written as something else. Only the four
    builtin literal types themselves are literals; a subclass is not one.
    """
    # The exact types, not subclasses: writing goes through the value's own
    # spelling, and a subclass can spell itself as anything at all — including
    # source that breaks out of the literal.
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is str:
        return _quote_string(value)
    if type(value) is int:
        spelling = str(value)
        if not _INT_SPELLING.match(spelling):
            raise ValueError(f"viba has no literal for this number: {spelling}")
        return spelling
    if type(value) is float:
        spelling = str(value)
        if not _FLOAT_SPELLING.match(spelling):
            raise ValueError(f"viba has no literal for this number: {spelling}")
        return spelling
    raise ValueError(f"viba has no literal for {type(value).__name__}")


def _quote_string(value: str) -> str:
    """One of the three string literals, chosen so the text reads back as is.

    A delimiter inside the text would end the literal early, and a trailing
    backslash would swallow the closing one, so each form is only used when
    the text is free of its delimiter. A text that no form can hold (all three
    delimiters inside it, or a `'''` on a line-spanning text) has no spelling
    in the language and raises ValueError.
    """
    if "\n" not in value and not value.endswith("\\"):
        if '"' not in value:
            return f'"{value}"'
        if "'" not in value:
            return f"'{value}'"
    if "'''" not in value and not value.endswith("'"):
        return f"'''{value}'''"
    raise ValueError("no viba string literal holds this text")


def _unparse_codeblock(codeblock_node: CodeBlock) -> str:
    """Unparse a CodeBlock: content goes back inside braces as-is."""
    return "{" + codeblock_node.code + "}"


def _unparse_sumchain(chain: SumChain, indent: int, depth: int) -> str:
    """Unparse a SumChain."""
    if not chain.elements:
        return "never"

    line_indent = " " * (indent * depth)
    inner_indent = " " * (indent * (depth + 1))

    lines = []
    for i, elem in enumerate(chain.elements):
        elem_str = _dedent(_unparse_type(elem, indent, depth + 1), inner_indent)
        # A same-kind branch keeps its parentheses: written flat it would be
        # read back as part of the main chain.
        if isinstance(elem, SumChain):
            elem_str = f"({elem_str})"
        if i == 0:
            lines.append(f"{line_indent}{elem_str}")
        else:
            lines.append(f"{line_indent}| {elem_str}")

    return "\n".join(lines)


def _unparse_productchain(chain: ProductChain, indent: int, depth: int) -> str:
    """Unparse a ProductChain."""
    if not chain.elements:
        return "nil"

    line_indent = " " * (indent * depth)
    inner_indent = " " * (indent * (depth + 1))

    lines = []
    for i, elem in enumerate(chain.elements):
        elem_str = _dedent(_unparse_type(elem, indent, depth + 1), inner_indent)
        # `*` binds tighter than `|`, so a bare sum element would split the
        # product on reparse: `(A | B) * C` must keep its parentheses.
        # A same-kind branch chain (`A * (B * C)`) needs them for the same
        # reason: without them the branch would reparse as part of the main
        # chain.
        if isinstance(elem, (SumChain, ProductChain)):
            elem_str = f"({elem_str})"
        if i == 0:
            lines.append(f"{line_indent}{elem_str}")
        else:
            lines.append(f"{line_indent}* {elem_str}")

    return "\n".join(lines)


def _unparse_exponentchain(chain: ExponentChain, indent: int, depth: int) -> str:
    """Unparse an ExponentChain (elements in written order)."""
    if not chain.elements:
        return "never"

    line_indent = " " * (indent * depth)
    inner_indent = " " * (indent * (depth + 1))

    # `<-` binds tighter than `|` and `*`, so any composite result or
    # argument must be parenthesized to preserve the grouping.
    def _parenthesize(node: AST, unparsed: str) -> str:
        if node.__class__.__name__ in _BINARY_NAMES:
            return f"({unparsed})"
        return unparsed

    lines = []
    result_str = _parenthesize(
        chain.elements[0],
        _dedent(_unparse_type(chain.elements[0], indent, depth + 1), inner_indent),
    )
    lines.append(f"{line_indent}{result_str}")

    for arg in chain.elements[1:]:
        arg_str = _parenthesize(
            arg, _dedent(_unparse_type(arg, indent, depth + 1), inner_indent))
        lines.append(f"{line_indent}<- {arg_str}")

    return "\n".join(lines)


def _dedent(text: str, prefix: str) -> str:
    """Drop the first line's own indentation: the caller places that line."""
    first, newline, rest = text.partition("\n")
    if prefix and first.startswith(prefix):
        first = first[len(prefix):]
    return first + newline + rest


def _needs_parens(child: AST, parent_name: str, position: str) -> bool:
    """Determine if a child expression needs parentheses."""
    child_name = child.__class__.__name__

    # Always parenthesize chains when they appear inside binary ops
    if child_name in ("SumChain", "ProductChain", "ExponentChain"):
        return True

    # Sum and Product have same precedence, associate left
    if parent_name in ("Sum", "Product"):
        if child_name == parent_name:
            return False  # same operator, no parens needed (left-assoc)
        if child_name in ("Sum", "Product", "Exponent"):
            return True  # different operator, need parens

    # Exponent is right-associative
    if parent_name == "Exponent":
        if child_name == "Exponent" and position == "argument":
            return False  # right side of exponent can chain without parens
        if child_name in ("Sum", "Product", "Exponent"):
            return True

    return False


def _is_binary(node: AST) -> bool:
    """Check if a node is a binary type or a flattened chain of one."""
    return node.__class__.__name__ in _BINARY_NAMES


def _tagged_body_parens(node: AST, unparsed: str, prefix: str = "") -> str:
    """Parenthesize a tagged body if it is a binary type or another tag."""
    unparsed = _dedent(unparsed, prefix)
    if _is_binary(node) or isinstance(node, Tagged):
        return f"({unparsed})"
    return f" {unparsed}"

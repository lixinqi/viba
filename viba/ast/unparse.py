"""Unparse AST nodes back to Viba source."""

from typing import List, Union

from viba.ast._match import viba_type_match
from viba.ast.nodes import (
    AST,
    Module,
    Definition,
    Import,
    Sum,
    Product,
    Exponent,
    Tagged,
    TypeApp,
    Tuple,
    TypeRef,
    Constant,
    Void,
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


def _unparse_type(node: AST, indent: int, depth: int) -> str:
    """Convert a single node to Viba code."""

    def unparse_tag(t):
        return t.tag if t.tag.startswith("$") else f"${t.tag}"

    return viba_type_match(
        node,
        Definition=lambda d: _unparse_definition(d, indent, depth),
        Import=lambda i: _unparse_import(i),
        Sum=lambda s: _unparse_binary(s, " | ", indent, depth),
        Product=lambda p: _unparse_binary(p, " * ", indent, depth),
        Exponent=lambda e: _unparse_exponent(e, indent, depth),
        Tagged=lambda t: f"{unparse_tag(t)}{_tagged_body_parens(t.type, _unparse_type(t.type, indent, depth))}",
        TypeApp=lambda a: _unparse_typeapp(a, indent, depth),
        Tuple=lambda t: _unparse_tuple(t, indent, depth),
        TypeRef=lambda r: r.name,
        Constant=lambda c: _unparse_constant(c),
        Void=lambda v: "void",
        Never=lambda n: "never",
        Ellipsis=lambda e: "...",
        CodeBlock=lambda c: _unparse_codeblock(c),
        SumChain=lambda s: _unparse_sumchain(s, indent, depth),
        ProductChain=lambda p: _unparse_productchain(p, indent, depth),
        ExponentChain=lambda e: _unparse_exponentchain(e, indent, depth),
    )


def _unparse_definition(defn: Definition, indent: int, depth: int) -> str:
    """Unparse a Definition."""
    if defn.generic_params:
        params = "[" + ", ".join(defn.generic_params) + "]"
    else:
        params = ""

    body = _unparse_type(defn.body, indent, depth + 1)

    return f"{defn.name}{params} :=\n{' ' * indent * (depth + 1)}{body}"


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


def _unparse_typeapp(app_node: TypeApp, indent: int, depth: int) -> str:
    """Unparse a TypeApp."""
    if not app_node.args:
        return app_node.constructor

    args = ", ".join(_unparse_type(arg, indent, depth + 1) for arg in app_node.args)
    return f"{app_node.constructor}[{args}]"


def _unparse_tuple(tuple_node: Tuple, indent: int, depth: int) -> str:
    """Unparse a Tuple: (A, B, C)."""
    args = ", ".join(_unparse_type(e, indent, depth + 1) for e in tuple_node.elements)
    return f"({args})"


def _unparse_constant(const_node: Constant) -> str:
    """Unparse a Constant literal."""
    value = const_node.value
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        # Double-quoted strings cannot contain raw newlines (lexer rule);
        # fall back to triple quotes for multi-line content.
        if "\n" in value:
            return f"'''{value}'''"
        return f'"{value}"'
    return str(value)


def _unparse_codeblock(codeblock_node: CodeBlock) -> str:
    """Unparse a CodeBlock: content goes back inside braces as-is."""
    return "{" + codeblock_node.code + "}"


def _unparse_sumchain(chain: SumChain, indent: int, depth: int) -> str:
    """Unparse a SumChain."""
    if not chain.elements:
        return "never"

    base_indent = " " * (indent * depth)
    elem_indent = " " * (indent * (depth + 1))

    lines = []
    for i, elem in enumerate(chain.elements):
        elem_str = _unparse_type(elem, indent, depth + 1)
        if i == 0:
            lines.append(f"{base_indent}{elem_str}")
        else:
            lines.append(f"{elem_indent}| {elem_str}")

    return "\n".join(lines)


def _unparse_productchain(chain: ProductChain, indent: int, depth: int) -> str:
    """Unparse a ProductChain."""
    if not chain.elements:
        return "void"

    base_indent = " " * (indent * depth)
    elem_indent = " " * (indent * (depth + 1))

    lines = []
    for i, elem in enumerate(chain.elements):
        elem_str = _unparse_type(elem, indent, depth + 1)
        # `*` binds tighter than `|`, so a bare sum element would split the
        # product on reparse: `(A | B) * C` must keep its parentheses.
        if isinstance(elem, SumChain):
            elem_str = f"({elem_str})"
        if i == 0:
            lines.append(f"{base_indent}{elem_str}")
        else:
            lines.append(f"{elem_indent}* {elem_str}")

    return "\n".join(lines)


def _unparse_exponentchain(chain: ExponentChain, indent: int, depth: int) -> str:
    """Unparse an ExponentChain."""
    if not chain.args:
        return _unparse_type(chain.result, indent, depth)

    base_indent = " " * (indent * depth)
    arg_indent = " " * (indent * (depth + 1))

    # `<-` binds tighter than `|` and `*`, so any composite result or
    # argument must be parenthesized to preserve the grouping.
    def _parenthesize(node: AST, unparsed: str) -> str:
        if node.__class__.__name__ in _BINARY_NAMES:
            return f"({unparsed})"
        return unparsed

    lines = []
    result_str = _parenthesize(
        chain.result, _unparse_type(chain.result, indent, depth)
    )
    lines.append(f"{base_indent}{result_str}")

    # chain.args is application order (args[0] is fed first); the Viba
    # chain syntax lists arguments right-to-left, so emit in reverse.
    for arg in reversed(chain.args):
        arg_str = _parenthesize(arg, _unparse_type(arg, indent, depth + 1))
        lines.append(f"{arg_indent}<- {arg_str}")

    return "\n".join(lines)


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


def _tagged_body_parens(node: AST, unparsed: str) -> str:
    """Parenthesize a tagged body if it is a binary type or another tag."""
    if _is_binary(node) or isinstance(node, Tagged):
        return f"({unparsed})"
    return f" {unparsed}"

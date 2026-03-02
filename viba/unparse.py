# unparse.py
# VIBA code generation from AST
# Converts Type AST back to VIBA source code

from typing import List
from viba.type import Type
from viba.match import viba_type_match


def unparse(program: List[Type], indent: int = 2) -> str:
    """
    Convert Type AST back to VIBA source code.

    Args:
        program: List of Type definitions (from parse())
        indent: Base indentation level (spaces), default 2

    Returns:
        VIBA source code as string
    """
    lines = []
    for type_def in program:
        lines.append(_unparse_type(type_def, indent, 0))
    return "\n\n".join(lines)


def _unparse_type(type_obj: Type, indent: int, depth: int) -> str:
    """Convert a single Type node to VIBA code."""

    def unparse_tag(t):
        if t.tag.startswith("$"):
            return t.tag
        else:
            return f"${t.tag}"

    return viba_type_match(
        type_obj,
        DefinitionType=lambda d: _unparse_definition(d, indent, depth),
        SumType=lambda s: _unparse_binary(s, " | ", indent, depth),
        ProductType=lambda p: _unparse_binary(p, " * ", indent, depth),
        ExponentType=lambda e: _unparse_exponent(e, indent, depth),
        TaggedType=lambda t: f"{unparse_tag(t)} {_unparse_type(t.type, indent, depth)}",
        TypeAppType=lambda a: _unparse_typeapp(a, indent, depth),
        TypeRefType=lambda r: r.name,
        LiteralType=lambda l: _unparse_literal(l),
        IdentityType=lambda i: _unparse_identity(i),
        EllipsisType=lambda e: "...",
        PureTagType=lambda p: p.name,
        SumChainType=lambda s: _unparse_sumchain(s, indent, depth),
        ProductChainType=lambda p: _unparse_productchain(p, indent, depth),
        ExponentChainType=lambda e: _unparse_exponentchain(e, indent, depth),
        strict=True,
        _=lambda t: f"<unknown:{type(t).__name__}>",
    )


def _unparse_definition(def_type, indent: int, depth: int) -> str:
    """Unparse a DefinitionType."""
    # Generic parameters
    if def_type.generic_params:
        params = "[" + ", ".join(def_type.generic_params) + "]"
    else:
        params = ""

    # Body
    body = _unparse_type(def_type.body, indent, depth + 1)

    return f"{def_type.name}{params} :=\n{' '*indent*(depth + 1)}{body}"


def _unparse_binary(bin_type, op: str, indent: int, depth: int) -> str:
    """Unparse binary types (Sum/Product)."""
    left = _unparse_type(bin_type.left, indent, depth + 1)
    right = _unparse_type(bin_type.right, indent, depth + 1)

    # Check if we need parentheses
    left_needs_paren = _needs_parens(bin_type.left, bin_type.__class__.__name__, "left")
    right_needs_paren = _needs_parens(
        bin_type.right, bin_type.__class__.__name__, "right"
    )

    if left_needs_paren:
        left = f"({left})"
    if right_needs_paren:
        right = f"({right})"

    return f"{left}{op}{right}"


def _unparse_exponent(exp_type, indent: int, depth: int) -> str:
    """Unparse an ExponentType (function type)."""
    result = _unparse_type(exp_type.result, indent, depth + 1)
    argument = _unparse_type(exp_type.argument, indent, depth + 1)

    # Add parentheses if needed
    if _is_binary(exp_type.argument) and not isinstance(
        exp_type.argument, ExponentType
    ):
        argument = f"({argument})"

    return f"{result} <- {argument}"


def _unparse_typeapp(app_type, indent: int, depth: int) -> str:
    """Unparse a TypeAppType."""
    if not app_type.args:
        return app_type.constructor

    args = ", ".join(_unparse_type(arg, indent, depth + 1) for arg in app_type.args)
    return f"{app_type.constructor}[{args}]"


def _unparse_literal(lit_type) -> str:
    """Unparse a LiteralType."""
    if lit_type.val_type == "str":
        return f'"{lit_type.val}"'
    elif lit_type.val_type == "bool":
        return str(lit_type.val).lower()
    else:
        return str(lit_type.val)


def _unparse_identity(ident_type) -> str:
    """Unparse an IdentityType."""
    if ident_type.alias == "()":
        return "()"
    elif ident_type.type == "ProductIdentity":
        return "void"
    else:  # SumIdentity
        return "never"


def _unparse_sumchain(chain, indent: int, depth: int) -> str:
    """Unparse a SumChainType."""
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


def _unparse_productchain(chain, indent: int, depth: int) -> str:
    """Unparse a ProductChainType."""
    if not chain.elements:
        return "void"

    base_indent = " " * (indent * depth)
    elem_indent = " " * (indent * (depth + 1))

    lines = []
    for i, elem in enumerate(chain.elements):
        elem_str = _unparse_type(elem, indent, depth + 1)
        if i == 0:
            lines.append(f"{base_indent}{elem_str}")
        else:
            lines.append(f"{elem_indent}* {elem_str}")

    return "\n".join(lines)


def _unparse_exponentchain(chain, indent: int, depth: int) -> str:
    """Unparse an ExponentChainType."""
    if not chain.args:
        return _unparse_type(chain.result, indent, depth)

    base_indent = " " * (indent * depth)
    arg_indent = " " * (indent * (depth + 1))

    lines = []
    result_str = _unparse_type(chain.result, indent, depth)
    lines.append(f"{base_indent}{result_str}")

    for arg in chain.args:
        arg_str = _unparse_type(arg, indent, depth + 1)
        lines.append(f"{arg_indent}<- {arg_str}")

    return "\n".join(lines)


def _needs_parens(child: Type, parent_type: str, position: str) -> bool:
    """Determine if a child expression needs parentheses."""
    child_name = child.__class__.__name__

    # Always parenthesize chains when they appear inside binary ops
    if child_name in ("SumChainType", "ProductChainType", "ExponentChainType"):
        return True

    # Sum and Product have same precedence, associate left
    if parent_type in ("SumType", "ProductType"):
        if child_name == parent_type:
            return False  # Same operator, no parens needed (left-assoc)
        if child_name in ("SumType", "ProductType", "ExponentType"):
            return True  # Different operator, need parens

    # Exponent is right-associative
    if parent_type == "ExponentType":
        if child_name == "ExponentType" and position == "argument":
            return False  # Right side of exponent can chain without parens
        if child_name in ("SumType", "ProductType", "ExponentType"):
            return True

    return False


def _is_binary(type_obj: Type) -> bool:
    """Check if a type is a binary type (Sum/Product/Exponent)."""
    return type_obj.__class__.__name__ in ("SumType", "ProductType", "ExponentType")


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from viba.type import (
        DefinitionType,
        SumType,
        ProductType,
        ExponentType,
        TaggedType,
        TypeAppType,
        TypeRefType,
        IdentityType,
        EllipsisType,
        PureTagType,
        LiteralType,
        SumChainType,
        ProductChainType,
        ExponentChainType,
    )

    print("Testing unparse with proper indentation...")

    # Test 1: Basic product chain
    print("\n1. Basic product chain:")
    a = TypeRefType(node="TypeRef", name="String")
    b = TypeRefType(node="TypeRef", name="Int")
    c = TypeRefType(node="TypeRef", name="Bool")

    prod_chain = ProductChainType(a, b, c)
    def_type = DefinitionType(
        node="Definition", name="ExampleType", generic_params=[], body=prod_chain
    )

    result = unparse([def_type])
    print(result)

    # Test 2: Basic sum chain
    print("\n2. Basic sum chain:")
    sum_chain = SumChainType(a, b, c)
    def_type = DefinitionType(
        node="Definition", name="ExampleSum", generic_params=[], body=sum_chain
    )

    result = unparse([def_type])
    print(result)

    # Test 3: Exponent chain
    print("\n3. Exponent chain:")
    exp_chain = ExponentChainType(c, a, b)  # result=c, args=[a,b]
    def_type = DefinitionType(
        node="Definition", name="ExampleFunc", generic_params=[], body=exp_chain
    )

    result = unparse([def_type])
    print(result)

    # Test 4: Nested structures
    print("\n4. Nested structures:")
    inner_prod = ProductChainType(a, b)
    outer_sum = SumChainType(inner_prod, c)
    def_type = DefinitionType(
        node="Definition", name="Nested", generic_params=[], body=outer_sum
    )

    result = unparse([def_type])
    print(result)

    # Test 5: Option and Result examples
    print("\n5. Option and Result:")
    t = TypeRefType(node="TypeRef", name="T")
    e = TypeRefType(node="TypeRef", name="E")

    option = DefinitionType(
        node="Definition",
        name="Option",
        generic_params=["T"],
        body=SumChainType(
            TaggedType(node="TaggedType", tag="some", type=t),
            IdentityType(node="Identity", type="ProductIdentity", alias="()"),
        ),
    )

    result_type = DefinitionType(
        node="Definition",
        name="Result",
        generic_params=["T", "E"],
        body=SumChainType(
            TaggedType(node="TaggedType", tag="ok", type=t),
            TaggedType(node="TaggedType", tag="err", type=e),
        ),
    )

    result = unparse([option, result_type])
    print(result)

    print("\nAll tests completed.")

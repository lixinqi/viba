# match.py
# VIBA pattern matching module
# Provides a flexible pattern matching interface for VIBA Type AST

from typing import Any, Callable, Dict, Optional, Type as PyType
from viba.type import (
    Type,
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
)


# ----------------------------------------------------------------------
# Pattern matching function
# ----------------------------------------------------------------------


def viba_type_match(
    type_obj: Type,
    DefinitionType: Callable[[DefinitionType], Any] = None,
    SumType: Callable[[SumType], Any] = None,
    ProductType: Callable[[ProductType], Any] = None,
    ExponentType: Callable[[ExponentType], Any] = None,
    TaggedType: Callable[[TaggedType], Any] = None,
    TypeAppType: Callable[[TypeAppType], Any] = None,
    TypeRefType: Callable[[TypeRefType], Any] = None,
    IdentityType: Callable[[IdentityType], Any] = None,
    EllipsisType: Callable[[EllipsisType], Any] = None,
    PureTagType: Callable[[PureTagType], Any] = None,
    LiteralType: Callable[[LiteralType], Any] = None,
    _: Callable[[Type], Any] = None,  # default/wildcard handler
    strict: bool = True,  # if True, raise exception when no handler matches
) -> Any:
    """
    Pattern match on VIBA Type AST nodes.

    Each keyword argument corresponds to a concrete Type subclass name.
    The handler function for the matching subclass is called with the type object.

    Args:
        type_obj: The Type instance to match against
        DefinitionType: Handler for DefinitionType
        SumType: Handler for SumType
        ProductType: Handler for ProductType
        ExponentType: Handler for ExponentType
        TaggedType: Handler for TaggedType
        TypeAppType: Handler for TypeAppType
        TypeRefType: Handler for TypeRefType
        IdentityType: Handler for IdentityType
        EllipsisType: Handler for EllipsisType
        PureTagType: Handler for PureTagType
        LiteralType: Handler for LiteralType
        _: Default handler for any type (if no specific handler matches)
        strict: If True (default), raise TypeError when no handler matches.
                If False, return None when no handler matches.

    Returns:
        The result of the matched handler.

    Raises:
        TypeError: When strict=True and no handler matches the type.

    Example:
        result = viba_type_match(
            some_type,
            DefinitionType=lambda d: f"definition: {d.name}",
            TypeRefType=lambda r: f"reference: {r.name}",
            _=lambda t: f"other: {type(t).__name__}"
        )
    """
    # Get the concrete class name
    class_name = type_obj.__class__.__name__

    # Get the handler using the class name as key
    handler = locals().get(class_name)

    # Try specific handler
    if handler is not None:
        return handler(type_obj)

    # Fall back to default handler if provided
    if _ is not None:
        return _(type_obj)

    # No handler matched
    if strict:
        raise TypeError(f"No handler matched for type: {class_name}")
    return None


# ----------------------------------------------------------------------
# Convenience wrapper for creating pattern matching functions
# ----------------------------------------------------------------------


def match_builder(strict: bool = True, **handlers):
    """
    Create a pattern matching function with predefined handlers.

    Args:
        strict: Passed through to viba_type_match
        **handlers: Keyword arguments mapping subclass names to handler functions

    Returns:
        A function that takes a Type and applies the predefined patterns.

    Example:
        my_matcher = match_builder(
            DefinitionType=lambda d: d.name,
            TypeRefType=lambda r: r.name,
            _=lambda t: "unknown"
        )
        result = my_matcher(some_type)
    """

    def matcher(type_obj: Type) -> Any:
        return viba_type_match(type_obj, **handlers, strict=strict)

    return matcher


# ----------------------------------------------------------------------
# Examples and tests
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Create some test types
    test_def = DefinitionType(
        node="Definition",
        name="Test",
        generic_params=[],
        body=TypeRefType(node="TypeRef", name="Int"),
    )

    test_sum = SumType(
        node="SumType",
        left=TypeRefType(node="TypeRef", name="A"),
        right=TypeRefType(node="TypeRef", name="B"),
    )

    test_literal = LiteralType(node="Literal", val=42, val_type="int")

    # Example 1: Direct matching with strict=True (default)
    print("=== Direct matching (strict=True) ===")
    for t in [test_def, test_sum, test_literal]:
        result = viba_type_match(
            t,
            DefinitionType=lambda d: f"Definition: {d.name}",
            SumType=lambda s: f"Sum: {s.left} | {s.right}",
            LiteralType=lambda l: f"Literal: {l.val} ({l.val_type})",
            # No _ handler, but strict=True will raise if no match
        )
        print(f"  {result}")

    # Example 2: Missing handler with strict=True raises exception
    print("\n=== Missing handler (strict=True raises) ===")
    try:
        result = viba_type_match(
            test_literal,
            DefinitionType=lambda d: "def",
            SumType=lambda s: "sum",
            # No LiteralType handler, no _
        )
    except TypeError as e:
        print(f"  Caught expected exception: {e}")

    # Example 3: strict=False returns None when no handler matches
    print("\n=== strict=False returns None ===")
    result = viba_type_match(
        test_literal,
        DefinitionType=lambda d: "def",
        SumType=lambda s: "sum",
        strict=False,
    )
    print(f"  Result: {result}")

    # Example 4: Using match_builder with strict
    print("\n=== match_builder with strict=False ===")
    matcher = match_builder(
        strict=False, DefinitionType=lambda d: d.name, TypeRefType=lambda r: r.name
    )

    for t in [test_def, TypeRefType(node="TypeRef", name="Int"), test_literal]:
        print(f"  {t.__class__.__name__} -> {matcher(t)}")

    # Example 5: Extract nested information (strict=True ensures completeness)
    print("\n=== Nested extraction (complete patterns) ===")

    def extract_info(type_obj: Type) -> Dict[str, Any]:
        return viba_type_match(
            type_obj,
            DefinitionType=lambda d: {
                "kind": "definition",
                "name": d.name,
                "params": d.generic_params,
                "body": extract_info(d.body),
            },
            SumType=lambda s: {
                "kind": "sum",
                "left": extract_info(s.left),
                "right": extract_info(s.right),
            },
            ProductType=lambda p: {
                "kind": "product",
                "left": extract_info(p.left),
                "right": extract_info(p.right),
            },
            TypeRefType=lambda r: {"kind": "ref", "name": r.name},
            LiteralType=lambda l: {
                "kind": "literal",
                "value": l.val,
                "type": l.val_type,
            },
            IdentityType=lambda i: {
                "kind": "identity",
                "identity_type": i.type,
                "alias": i.alias,
            },
            EllipsisType=lambda e: {"kind": "ellipsis"},
            PureTagType=lambda p: {"kind": "puretag", "name": p.name, "path": p.path},
            TypeAppType=lambda a: {
                "kind": "typeapp",
                "constructor": a.constructor,
                "args": [extract_info(arg) for arg in a.args],
            },
            TaggedType=lambda t: {
                "kind": "tagged",
                "tag": t.tag,
                "type": extract_info(t.type),
            },
            ExponentType=lambda e: {
                "kind": "exponent",
                "result": extract_info(e.result),
                "argument": extract_info(e.argument),
            },
        )

    nested_type = SumType(
        node="SumType",
        left=ProductType(
            node="ProductType",
            left=TypeRefType(node="TypeRef", name="X"),
            right=TypeRefType(node="TypeRef", name="Y"),
        ),
        right=TypeRefType(node="TypeRef", name="Z"),
    )

    import json

    print(json.dumps(extract_info(nested_type), indent=2))

    # Example 6: Incomplete patterns with strict=True would raise
    print("\n=== Incomplete patterns with strict=True (would raise) ===")

    def incomplete_extract(type_obj: Type) -> Dict[str, Any]:
        return viba_type_match(
            type_obj,
            SumType=lambda s: {
                "kind": "sum",
                "left": incomplete_extract(s.left),
                "right": incomplete_extract(s.right),
            },
            TypeRefType=lambda r: {"kind": "ref", "name": r.name},
            # Missing ProductType handler!
        )

    try:
        result = incomplete_extract(nested_type)
    except TypeError as e:
        print(f"  Caught expected exception: {e}")

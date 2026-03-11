# coding_style.py
# VIBA standard coding style checker
# Validates that Type AST conforms to VIBA standard coding style

from typing import List, Union, Optional
from dataclasses import dataclass
from viba.type import (
    Type,
    DefinitionType,
    ExponentType,
    ExponentChainType,
    SumChainType,
    ProductChainType,
    TaggedType,
    TypeAppType,
    TypeRefType,
    IdentityType,
)
from viba.match import viba_type_match


# ----------------------------------------------------------------------
# Result type for coding style check
# ----------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of coding style check."""

    success: bool
    message: str
    style: Optional["VibaStdCodingStyle"] = None


# ----------------------------------------------------------------------
# Schema definitions following VIBA spec
# ----------------------------------------------------------------------


class VibaStdCodingStyle:
    """Base class for all coding style variants."""

    pass


@dataclass
class ClassDefineStdCodingStyle(VibaStdCodingStyle):
    """Class definition style: contains no exponent type."""

    contains_no_exponent_type: DefinitionType


@dataclass
class FuncDeclareStdCodingStyle(VibaStdCodingStyle):
    """Function declaration style: exactly one exponent type."""

    only_one_exponent_type: DefinitionType


@dataclass
class FuncImplementStdCodingStyle(VibaStdCodingStyle):
    """Function implementation style: exponent chain as top layer."""

    exponent_chain_type_as_top_layer: DefinitionType


# ----------------------------------------------------------------------
# Main checking function
# ----------------------------------------------------------------------


def check_viba_std_coding_style(t: Type) -> CheckResult:
    """
    Check if a Type definition conforms to VIBA standard coding style.

    Args:
        t: Type definition to check

    Returns:
        CheckResult with success status, message, and style if successful
    """
    return viba_type_match(
        t,
        DefinitionType=lambda d: _check_definition_style(d),
        strict=False,
        _=lambda x: CheckResult(
            success=False, message=f"Expected DefinitionType, got {type(x).__name__}"
        ),
    )


def _check_definition_style(def_type: DefinitionType) -> CheckResult:
    """Check if a definition follows one of the standard coding styles."""
    # Case 1: FuncImplementStdCodingStyle - exponent chain as top layer
    if isinstance(def_type.body, ExponentChainType):
        return CheckResult(
            success=True,
            message=f"Definition {def_type.name} follows FuncImplementStdCodingStyle (exponent chain)",
            style=FuncImplementStdCodingStyle(
                exponent_chain_type_as_top_layer=def_type
            ),
        )

    # Check for exponent types in body
    has_exponent = _contains_exponent_type(def_type.body)

    # Case 2: ClassDefineStdCodingStyle - no exponent types
    if not has_exponent:
        return CheckResult(
            success=True,
            message=f"Definition {def_type.name} follows ClassDefineStdCodingStyle (no exponent)",
            style=ClassDefineStdCodingStyle(contains_no_exponent_type=def_type),
        )

    # Case 3: FuncDeclareStdCodingStyle - exactly one exponent at top level
    if _is_single_top_level_exponent(def_type.body):
        return CheckResult(
            success=True,
            message=f"Definition {def_type.name} follows FuncDeclareStdCodingStyle (single exponent)",
            style=FuncDeclareStdCodingStyle(only_one_exponent_type=def_type),
        )

    # No style matches
    return CheckResult(
        success=False,
        message=f"Definition {def_type.name} does not conform to any VIBA standard coding style",
    )


def _contains_exponent_type(t: Type) -> bool:
    """Check if a type contains any ExponentType."""
    return viba_type_match(
        t,
        ExponentType=lambda e: True,
        DefinitionType=lambda d: _contains_exponent_type(d.body),
        SumChainType=lambda s: any(
            _contains_exponent_type(elem) for elem in s.elements
        ),
        ProductChainType=lambda p: any(
            _contains_exponent_type(elem) for elem in p.elements
        ),
        ExponentChainType=lambda e: True,
        TaggedType=lambda tg: _contains_exponent_type(tg.type),
        TypeAppType=lambda a: any(_contains_exponent_type(arg) for arg in a.args),
        strict=False,
        _=lambda _: False,
    )


def _is_single_top_level_exponent(t: Type) -> bool:
    """Check if a type has exactly one ExponentType at the top level."""
    count = 0

    def count_exponents(typ: Type):
        nonlocal count
        if count > 1:
            return

        viba_type_match(
            typ,
            ExponentType=lambda e: set_count(1),
            SumChainType=lambda s: None,  # Sum chains at top level are not allowed for single exponent
            ProductChainType=lambda p: None,  # Product chains at top level are not allowed
            ExponentChainType=lambda e: set_count(1),
            TaggedType=lambda tg: count_exponents(tg.type),
            TypeAppType=lambda a: None,  # Type applications at top level hide exponents
            strict=False,
            _=lambda _: None,
        )

    def set_count(val: int):
        nonlocal count
        count = val

    count_exponents(t)
    return count == 1


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

    print("Testing VIBA standard coding style checker...")

    # Helper to create common type references
    int_ref = TypeRefType(node="TypeRef", name="Int")
    str_ref = TypeRefType(node="TypeRef", name="String")
    bool_ref = TypeRefType(node="TypeRef", name="Bool")

    # ------------------------------------------------------------------
    # Test 1: ClassDefineStdCodingStyle - contains_no_exponent_type
    # ------------------------------------------------------------------
    print("\n1. ClassDefineStdCodingStyle - contains_no_exponent_type:")

    # Simple product type with no exponents
    person_type = ProductChainType(
        TaggedType(node="TaggedType", tag="name", type=str_ref),
        TaggedType(node="TaggedType", tag="age", type=int_ref),
        TaggedType(node="TaggedType", tag="active", type=bool_ref),
    )

    person_def = DefinitionType(
        node="Definition", name="Person", generic_params=[], body=person_type
    )

    result = check_viba_std_coding_style(person_def)
    if result.success:
        print(f"  ✓ Person: {result.message}")
        if isinstance(result.style, ClassDefineStdCodingStyle):
            print(
                f"    → Class definition: {result.style.contains_no_exponent_type.name}"
            )
    else:
        print(f"  ✗ Person: {result.message}")

    # Simple sum type with no exponents
    color_type = SumChainType(
        TaggedType(node="TaggedType", tag="red", type=int_ref),
        TaggedType(node="TaggedType", tag="green", type=int_ref),
        TaggedType(node="TaggedType", tag="blue", type=int_ref),
    )

    color_def = DefinitionType(
        node="Definition", name="Color", generic_params=[], body=color_type
    )

    result = check_viba_std_coding_style(color_def)
    if result.success:
        print(f"  ✓ Color: {result.message}")
        if isinstance(result.style, ClassDefineStdCodingStyle):
            print(
                f"    → Class definition: {result.style.contains_no_exponent_type.name}"
            )
    else:
        print(f"  ✗ Color: {result.message}")

    # ------------------------------------------------------------------
    # Test 2: FuncDeclareStdCodingStyle - only_one_exponent_type
    # ------------------------------------------------------------------
    print("\n2. FuncDeclareStdCodingStyle - only_one_exponent_type:")

    # Simple function type at top level
    func_type = ExponentType(node="ExponentType", result=int_ref, argument=str_ref)

    func_def = DefinitionType(
        node="Definition", name="ParseInt", generic_params=[], body=func_type
    )

    result = check_viba_std_coding_style(func_def)
    if result.success:
        print(f"  ✓ ParseInt: {result.message}")
        if isinstance(result.style, FuncDeclareStdCodingStyle):
            print(
                f"    → Function declaration: {result.style.only_one_exponent_type.name}"
            )
    else:
        print(f"  ✗ ParseInt: {result.message}")

    # Tagged function type at top level
    tagged_func = TaggedType(node="TaggedType", tag="$parser", type=func_type)

    tagged_func_def = DefinitionType(
        node="Definition", name="StringParser", generic_params=[], body=tagged_func
    )

    result = check_viba_std_coding_style(tagged_func_def)
    if result.success:
        print(f"  ✓ StringParser: {result.message}")
        if isinstance(result.style, FuncDeclareStdCodingStyle):
            print(
                f"    → Function declaration: {result.style.only_one_exponent_type.name}"
            )
    else:
        print(f"  ✗ StringParser: {result.message}")

    # ------------------------------------------------------------------
    # Test 3: FuncImplementStdCodingStyle - exponent_chain_type_as_top_layer
    # ------------------------------------------------------------------
    print("\n3. FuncImplementStdCodingStyle - exponent_chain_type_as_top_layer:")

    # Curried function: Int <- String <- Bool
    exp1 = ExponentType(node="ExponentType", result=int_ref, argument=str_ref)
    exp2 = ExponentType(node="ExponentType", result=exp1, argument=bool_ref)

    chain_func_def = DefinitionType(
        node="Definition", name="CurriedFunc", generic_params=[], body=exp2
    )

    result = check_viba_std_coding_style(chain_func_def)
    if result.success:
        print(f"  ✓ CurriedFunc: {result.message}")
        if isinstance(result.style, FuncImplementStdCodingStyle):
            print(
                f"    → Function implementation: {result.style.exponent_chain_type_as_top_layer.name}"
            )
    else:
        print(f"  ✗ CurriedFunc: {result.message}")

    # ExponentChainType (flattened representation)
    exp_chain = ExponentChainType(
        int_ref, str_ref, bool_ref
    )  # result=int_ref, args=[str_ref, bool_ref]

    chain_def = DefinitionType(
        node="Definition", name="FlattenedChain", generic_params=[], body=exp_chain
    )

    result = check_viba_std_coding_style(chain_def)
    if result.success:
        print(f"  ✓ FlattenedChain: {result.message}")
        if isinstance(result.style, FuncImplementStdCodingStyle):
            print(
                f"    → Function implementation: {result.style.exponent_chain_type_as_top_layer.name}"
            )
    else:
        print(f"  ✗ FlattenedChain: {result.message}")

    # ------------------------------------------------------------------
    # Test 4: Invalid cases - should return failed CheckResult
    # ------------------------------------------------------------------
    print("\n4. Invalid cases (should fail):")

    # Multiple exponents at top level (in sum chain)
    exp_a = ExponentType(node="ExponentType", result=int_ref, argument=str_ref)
    exp_b = ExponentType(node="ExponentType", result=str_ref, argument=bool_ref)

    multi_exp_sum = SumChainType(exp_a, exp_b)

    multi_exp_def = DefinitionType(
        node="Definition", name="MultiExponent", generic_params=[], body=multi_exp_sum
    )

    result = check_viba_std_coding_style(multi_exp_def)
    if not result.success:
        print(f"  ✓ MultiExponent: {result.message}")
    else:
        print(f"  ✗ MultiExponent: Should have failed but passed")

    # Multiple exponents at top level (in product chain)
    multi_exp_prod = ProductChainType(exp_a, exp_b)

    multi_exp_prod_def = DefinitionType(
        node="Definition",
        name="MultiExponentProd",
        generic_params=[],
        body=multi_exp_prod,
    )

    result = check_viba_std_coding_style(multi_exp_prod_def)
    if not result.success:
        print(f"  ✓ MultiExponentProd: {result.message}")
    else:
        print(f"  ✗ MultiExponentProd: Should have failed but passed")

    # Nested exponent (not top level)
    nested_type = ProductChainType(
        str_ref,
        TaggedType(
            node="TaggedType",
            tag="$func",
            type=ExponentType(node="ExponentType", result=int_ref, argument=str_ref),
        ),
    )

    nested_exp_def = DefinitionType(
        node="Definition", name="NestedExponent", generic_params=[], body=nested_type
    )

    result = check_viba_std_coding_style(nested_exp_def)
    if not result.success:
        print(f"  ✓ NestedExponent: {result.message}")
    else:
        print(f"  ✗ NestedExponent: Should have failed but passed")

    # Not a DefinitionType
    not_def = TypeRefType(node="TypeRef", name="Int")

    result = check_viba_std_coding_style(not_def)
    if not result.success:
        print(f"  ✓ NotDefinition: {result.message}")
    else:
        print(f"  ✗ NotDefinition: Should have failed but passed")

    # ------------------------------------------------------------------
    # Test 5: Real-world examples
    # ------------------------------------------------------------------
    print("\n5. Real-world examples:")

    # Option type (ClassDefineStdCodingStyle)
    t_ref = TypeRefType(node="TypeRef", name="T")
    option = DefinitionType(
        node="Definition",
        name="Option",
        generic_params=["T"],
        body=SumChainType(
            TaggedType(node="TaggedType", tag="some", type=t_ref),
            IdentityType(node="Identity", type="ProductIdentity", alias="()"),
        ),
    )

    result = check_viba_std_coding_style(option)
    if result.success:
        print(f"  ✓ Option[T]: {result.message}")
    else:
        print(f"  ✗ Option[T]: {result.message}")

    # Result type (ClassDefineStdCodingStyle)
    e_ref = TypeRefType(node="TypeRef", name="E")
    result_type = DefinitionType(
        node="Definition",
        name="Result",
        generic_params=["T", "E"],
        body=SumChainType(
            TaggedType(node="TaggedType", tag="ok", type=t_ref),
            TaggedType(node="TaggedType", tag="err", type=e_ref),
        ),
    )

    result = check_viba_std_coding_style(result_type)
    if result.success:
        print(f"  ✓ Result[T, E]: {result.message}")
    else:
        print(f"  ✗ Result[T, E]: {result.message}")

    # Map function (FuncDeclareStdCodingStyle)
    list_b = TypeAppType(
        node="TypeApp", constructor="List", args=[TypeRefType(node="TypeRef", name="B")]
    )
    map_func = DefinitionType(
        node="Definition",
        name="Map",
        generic_params=["A", "B"],
        body=ExponentType(
            node="ExponentType",
            result=list_b,
            argument=ExponentType(
                node="ExponentType",
                result=TypeRefType(node="TypeRef", name="B"),
                argument=TypeRefType(node="TypeRef", name="A"),
            ),
        ),
    )

    result = check_viba_std_coding_style(map_func)
    if result.success:
        print(f"  ✓ Map[A, B]: {result.message}")
    else:
        print(f"  ✗ Map[A, B]: {result.message}")

    # Pipeline (FuncImplementStdCodingStyle)
    pipeline = DefinitionType(
        node="Definition",
        name="Pipeline",
        generic_params=["A", "B", "C"],
        body=ExponentChainType(
            TypeRefType(node="TypeRef", name="C"),  # result
            TypeRefType(node="TypeRef", name="A"),  # arg1
            TypeRefType(node="TypeRef", name="B"),  # arg2
        ),
    )

    result = check_viba_std_coding_style(pipeline)
    if result.success:
        print(f"  ✓ Pipeline[A, B, C]: {result.message}")
    else:
        print(f"  ✗ Pipeline[A, B, C]: {result.message}")

    print("\nAll tests completed.")

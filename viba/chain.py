# chain.py
# VIBA chain type conversion module
# Provides flattened representations for associative type constructors

from typing import List, Optional
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
    SumChainType,
    ProductChainType,
    ExponentChainType,
)
from viba.match import viba_type_match


# ----------------------------------------------------------------------
# Convert nested binary trees to flattened chains
# ----------------------------------------------------------------------


def convert_to_chain_style(type_obj: Type) -> Type:
    return viba_type_match(
        type_obj,
        SumType=lambda s: _flatten_sum(s),
        ProductType=lambda p: _flatten_product(p),
        ExponentType=lambda e: _flatten_exponent(e),
        DefinitionType=lambda d: DefinitionType(
            node=d.node,
            name=d.name,
            generic_params=d.generic_params,
            body=convert_to_chain_style(d.body),
        ),
        TaggedType=lambda t: TaggedType(
            node=t.node, tag=t.tag, type=convert_to_chain_style(t.type)
        ),
        TypeAppType=lambda a: TypeAppType(
            node=a.node,
            constructor=a.constructor,
            args=[convert_to_chain_style(arg) for arg in a.args],
        ),
        TypeRefType=lambda r: r,
        IdentityType=lambda i: i,
        EllipsisType=lambda e: e,
        PureTagType=lambda p: p,
        LiteralType=lambda l: l,
        SumChainType=lambda s: s,
        ProductChainType=lambda p: p,
        ExponentChainType=lambda e: e,
        strict=False,
        _=lambda t: t,
    )


def _flatten_sum(sum_type: SumType) -> SumChainType:
    elements = []

    def collect(t: Type):
        t2 = convert_to_chain_style(t)
        if isinstance(t2, SumChainType):
            elements.extend(t2.elements)
        else:
            elements.append(t2)

    collect(sum_type.left)
    collect(sum_type.right)

    return SumChainType(*elements)


def _flatten_product(product_type: ProductType) -> ProductChainType:
    elements = []

    def collect(t: Type):
        t2 = convert_to_chain_style(t)
        if isinstance(t2, ProductChainType):
            elements.extend(t2.elements)
        else:
            elements.append(t2)

    collect(product_type.left)
    collect(product_type.right)

    return ProductChainType(*elements)


def _flatten_exponent(exponent_type: ExponentType) -> ExponentChainType:
    def collect(e: ExponentType, args_so_far: List[Type]) -> tuple[List[Type], Type]:
        arg = convert_to_chain_style(e.argument)
        res = convert_to_chain_style(e.result)

        if isinstance(res, ExponentType):
            return collect(res, args_so_far + [arg])
        elif isinstance(res, ExponentChainType):
            return (args_so_far + [arg] + res.args, res.result)
        else:
            return (args_so_far + [arg], res)

    args, result = collect(exponent_type, [])
    return ExponentChainType(result, *args)


# ----------------------------------------------------------------------
# Reconstruct binary trees from chains
# ----------------------------------------------------------------------


def convert_from_chain_style(type_obj: Type) -> Type:
    return viba_type_match(
        type_obj,
        SumChainType=lambda s: _reconstruct_sum(s),
        ProductChainType=lambda p: _reconstruct_product(p),
        ExponentChainType=lambda e: _reconstruct_exponent(e),
        DefinitionType=lambda d: DefinitionType(
            node=d.node,
            name=d.name,
            generic_params=d.generic_params,
            body=convert_from_chain_style(d.body),
        ),
        TaggedType=lambda t: TaggedType(
            node=t.node, tag=t.tag, type=convert_from_chain_style(t.type)
        ),
        TypeAppType=lambda a: TypeAppType(
            node=a.node,
            constructor=a.constructor,
            args=[convert_from_chain_style(arg) for arg in a.args],
        ),
        strict=False,
        _=lambda t: t,
    )


def _reconstruct_sum(chain: SumChainType) -> Type:
    if not chain.elements:
        return IdentityType(node="Identity", type="SumIdentity")

    result = chain.elements[0]
    for elem in chain.elements[1:]:
        result = SumType(node="SumType", left=result, right=elem)
    return result


def _reconstruct_product(chain: ProductChainType) -> Type:
    if not chain.elements:
        return IdentityType(node="Identity", type="ProductIdentity")

    result = chain.elements[0]
    for elem in chain.elements[1:]:
        result = ProductType(node="ProductType", left=result, right=elem)
    return result


def _reconstruct_exponent(chain: ExponentChainType) -> Type:
    result = chain.result
    for arg in reversed(chain.args):
        result = ExponentType(node="ExponentType", result=result, argument=arg)
    return result


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------


def is_chain_type(type_obj: Type) -> bool:
    return isinstance(type_obj, (SumChainType, ProductChainType, ExponentChainType))


def as_sum_chain(type_obj: Type) -> Optional[SumChainType]:
    if isinstance(type_obj, SumChainType):
        return type_obj
    converted = convert_to_chain_style(type_obj)
    return converted if isinstance(converted, SumChainType) else None


def as_product_chain(type_obj: Type) -> Optional[ProductChainType]:
    if isinstance(type_obj, ProductChainType):
        return type_obj
    converted = convert_to_chain_style(type_obj)
    return converted if isinstance(converted, ProductChainType) else None


def as_exponent_chain(type_obj: Type) -> Optional[ExponentChainType]:
    if isinstance(type_obj, ExponentChainType):
        return type_obj
    converted = convert_to_chain_style(type_obj)
    return converted if isinstance(converted, ExponentChainType) else None


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing chain type conversions...")

    # Test data: create nested types
    a = TypeRefType(node="TypeRef", name="A")
    b = TypeRefType(node="TypeRef", name="B")
    c = TypeRefType(node="TypeRef", name="C")
    d = TypeRefType(node="TypeRef", name="D")
    int_type = TypeRefType(node="TypeRef", name="Int")

    # 1. Test sum chain: ((A | B) | (C | D))
    sum1 = SumType(node="SumType", left=a, right=b)
    sum2 = SumType(node="SumType", left=c, right=d)
    sum_nested = SumType(node="SumType", left=sum1, right=sum2)

    print("\n1. Sum chain test:")
    print(f"   Original: {sum_nested}")
    sum_chain = convert_to_chain_style(sum_nested)
    print(f"   Flattened: {sum_chain}")
    sum_reconstructed = convert_from_chain_style(sum_chain)
    print(f"   Reconstructed: {sum_reconstructed}")
    print(f"   Elements: {[e.__class__.__name__ for e in sum_chain.elements]}")

    # 2. Test product chain: ((A * B) * (C * D))
    prod1 = ProductType(node="ProductType", left=a, right=b)
    prod2 = ProductType(node="ProductType", left=c, right=d)
    prod_nested = ProductType(node="ProductType", left=prod1, right=prod2)

    print("\n2. Product chain test:")
    print(f"   Original: {prod_nested}")
    prod_chain = convert_to_chain_style(prod_nested)
    print(f"   Flattened: {prod_chain}")
    prod_reconstructed = convert_from_chain_style(prod_chain)
    print(f"   Reconstructed: {prod_reconstructed}")
    print(f"   Elements: {[e.__class__.__name__ for e in prod_chain.elements]}")

    # 3. Test exponent chain: (((D <- C) <- B) <- A)
    exp1 = ExponentType(node="ExponentType", result=d, argument=c)
    exp2 = ExponentType(node="ExponentType", result=exp1, argument=b)
    exp_nested = ExponentType(node="ExponentType", result=exp2, argument=a)

    print("\n3. Exponent chain test:")
    print(f"   Original: {exp_nested}")
    exp_chain = convert_to_chain_style(exp_nested)
    print(f"   Flattened: {exp_chain}")
    exp_reconstructed = convert_from_chain_style(exp_chain)
    print(f"   Reconstructed: {exp_reconstructed}")
    print(f"   Args: {[e.__class__.__name__ for e in exp_chain.args]}")
    print(f"   Result: {exp_chain.result.__class__.__name__}")

    # 4. Test mixed with other types
    print("\n4. Mixed type test (definition containing chain):")
    tagged_a = TaggedType(node="TaggedType", tag="$input", type=a)
    tagged_b = TaggedType(node="TaggedType", tag="$output", type=b)

    body_sum = SumType(node="SumType", left=tagged_a, right=tagged_b)
    body_prod = ProductType(node="ProductType", left=body_sum, right=int_type)

    definition = DefinitionType(
        node="Definition", name="TestType", generic_params=["T"], body=body_prod
    )

    print(f"   Original definition body: {definition.body}")
    converted_def = convert_to_chain_style(definition)
    print(f"   After conversion: {converted_def.body}")

    # 5. Test analysis helpers
    print("\n5. Analysis helpers test:")

    sum_chain_from_helper = as_sum_chain(sum_nested)
    prod_chain_from_helper = as_product_chain(prod_nested)
    exp_chain_from_helper = as_exponent_chain(exp_nested)

    print(
        f"   as_sum_chain on sum_nested: {type(sum_chain_from_helper).__name__ if sum_chain_from_helper else None}"
    )
    print(
        f"   as_product_chain on prod_nested: {type(prod_chain_from_helper).__name__ if prod_chain_from_helper else None}"
    )
    print(
        f"   as_exponent_chain on exp_nested: {type(exp_chain_from_helper).__name__ if exp_chain_from_helper else None}"
    )
    print(f"   as_sum_chain on a (TypeRef): {as_sum_chain(a)}")

    # 6. Test identity cases
    print("\n6. Edge cases:")

    single_sum = SumType(
        node="SumType", left=a, right=IdentityType(node="Identity", type="SumIdentity")
    )
    single_sum_chain = convert_to_chain_style(single_sum)
    print(f"   Single element sum: {single_sum_chain}")

    single_prod = ProductType(
        node="ProductType",
        left=a,
        right=IdentityType(node="Identity", type="ProductIdentity"),
    )
    single_prod_chain = convert_to_chain_style(single_prod)
    print(f"   Single element product: {single_prod_chain}")

    empty_sum = IdentityType(node="Identity", type="SumIdentity")
    empty_sum_chain = convert_to_chain_style(empty_sum)
    print(f"   Empty sum (identity): {empty_sum_chain}")

    print("\nAll tests completed.")

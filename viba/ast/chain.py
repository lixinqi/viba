"""Chain-style canonicalization of AST nodes.

convert_to_chain_style flattens left-nested Sum/Product/Exponent trees
into flat SumChain/ProductChain/ExponentChain nodes; tuples, tags and
type applications are traversed element-wise. convert_from_chain_style
rebuilds the left-nested binary form.
"""

from typing import List

from viba.ast._match import viba_type_match
from viba.ast.nodes import (
    AST,
    TypeDefinition,
    GenericDefinition,
    Sum,
    Product,
    Exponent,
    Tagged,
    TypeApp,
    Tuple,
    SumChain,
    ProductChain,
    ExponentChain,
)


def convert_to_chain_style(node: AST) -> AST:
    return viba_type_match(
        node,
        Sum=lambda s: _flatten_sum(s),
        Product=lambda p: _flatten_product(p),
        Exponent=lambda e: _flatten_exponent(e),
        TypeDefinition=lambda d: TypeDefinition(
            d.name, convert_to_chain_style(d.body)
        ),
        GenericDefinition=lambda d: GenericDefinition(
            d.name, d.generic_params, convert_to_chain_style(d.body)
        ),
        Tagged=lambda t: Tagged(t.tag, convert_to_chain_style(t.type)),
        TypeApp=lambda a: TypeApp(a.constructor, [convert_to_chain_style(arg) for arg in a.args]),
        Tuple=lambda t: Tuple([convert_to_chain_style(e) for e in t.elements]),
        SumChain=lambda s: s,
        ProductChain=lambda p: p,
        ExponentChain=lambda e: e,
        strict=False,
        _=lambda t: t,
    )


def _flatten_sum(sum_type: Sum) -> SumChain:
    elements = []

    def collect(t: AST):
        t2 = convert_to_chain_style(t)
        if isinstance(t2, SumChain):
            elements.extend(t2.elements)
        else:
            elements.append(t2)

    collect(sum_type.left)
    collect(sum_type.right)

    return SumChain(elements)


def _flatten_product(product_type: Product) -> ProductChain:
    elements = []

    def collect(t: AST):
        t2 = convert_to_chain_style(t)
        if isinstance(t2, ProductChain):
            elements.extend(t2.elements)
        else:
            elements.append(t2)

    collect(product_type.left)
    collect(product_type.right)

    return ProductChain(elements)


def _flatten_exponent(exponent_type: Exponent) -> ExponentChain:
    def collect(e: Exponent, args_so_far: List[AST]) -> "tuple[List[AST], AST]":
        arg = convert_to_chain_style(e.argument)
        res = convert_to_chain_style(e.result)

        if isinstance(res, Exponent):
            return collect(res, args_so_far + [arg])
        elif isinstance(res, ExponentChain):
            return (args_so_far + [arg] + res.args, res.result)
        else:
            return (args_so_far + [arg], res)

    args, result = collect(exponent_type, [])
    return ExponentChain(result, args)


# ----------------------------------------------------------------------
# Reconstruct binary trees from chains
# ----------------------------------------------------------------------


def convert_from_chain_style(node: AST) -> AST:
    return viba_type_match(
        node,
        SumChain=lambda s: _reconstruct_sum(s),
        ProductChain=lambda p: _reconstruct_product(p),
        ExponentChain=lambda e: _reconstruct_exponent(e),
        TypeDefinition=lambda d: TypeDefinition(
            d.name, convert_from_chain_style(d.body)
        ),
        GenericDefinition=lambda d: GenericDefinition(
            d.name, d.generic_params, convert_from_chain_style(d.body)
        ),
        Tagged=lambda t: Tagged(t.tag, convert_from_chain_style(t.type)),
        TypeApp=lambda a: TypeApp(a.constructor, [convert_from_chain_style(arg) for arg in a.args]),
        Tuple=lambda t: Tuple([convert_from_chain_style(e) for e in t.elements]),
        strict=False,
        _=lambda t: t,
    )


def _reconstruct_sum(chain: SumChain) -> AST:
    if not chain.elements:
        return Never()

    result = chain.elements[0]
    for elem in chain.elements[1:]:
        result = Sum(result, elem)
    return result


def _reconstruct_product(chain: ProductChain) -> AST:
    if not chain.elements:
        return Void()

    result = chain.elements[0]
    for elem in chain.elements[1:]:
        result = Product(result, elem)
    return result


def _reconstruct_exponent(chain: ExponentChain) -> AST:
    result = chain.result
    for arg in reversed(chain.args):
        result = Exponent(result, arg)
    return result


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------


def is_chain_type(node: AST) -> bool:
    return isinstance(node, (SumChain, ProductChain, ExponentChain))

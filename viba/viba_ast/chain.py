"""Chain-style canonicalization of AST nodes.

`|` / `*` / `<-` 都左结合，所以写出来的一串里，主链就是一路 `$left`
（指数是一路 `$result`）的嵌套，压成 SumChain / ProductChain /
ExponentChain；`$right`（指数的 `$argument`）那侧的嵌套是支链，原样
算主链里的一个元素，它自己也照同样的规矩递归处理（支链的主链压成
次级链，支链的支链再往深一层）。分组因此不丢：`A * (B * C)` 是
``ProductChain([A, ProductChain([B, C])])``，与 `A * B * C` 不同形。
元组、标签、类型应用照元素遍历。convert_from_chain_style 反向重建
左嵌套的二元形式。
"""

from typing import List

from viba.viba_ast._match import viba_type_match
from viba.viba_ast.nodes import (
    AST,
    TypeDefinition,
    GenericDefinition,
    Sum,
    Product,
    Exponent,
    Partial,
    Tagged,
    TypeApp,
    Tuple,
    Nil,
    Any,
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
        Partial=lambda a: Partial(convert_to_chain_style(a.function),
                                  convert_to_chain_style(a.argument)),
        TypeDefinition=lambda d: TypeDefinition(
            d.name, convert_to_chain_style(d.body)
        ),
        GenericDefinition=lambda d: GenericDefinition(
            d.name, d.generic_params, convert_to_chain_style(d.body)
        ),
        Tagged=lambda t: Tagged(t.tag, convert_to_chain_style(t.type)),
        TypeApp=lambda a: TypeApp(a.constructor, [convert_to_chain_style(arg) for arg in a.args]),
        Tuple=lambda t: Tuple([convert_to_chain_style(e) for e in t.elements]),
        Any=lambda a: a,
        SumChain=lambda s: s,
        ProductChain=lambda p: p,
        ExponentChain=lambda e: e,
        strict=False,
        _=lambda t: t,
    )


def _flatten_sum(sum_type: Sum) -> SumChain:
    """主链（一路 $left）并进这一条，支链（$right）留成一个元素。"""
    elements = _main_chain_elements(sum_type.left, SumChain)
    elements.append(convert_to_chain_style(sum_type.right))
    return SumChain(elements)


def _flatten_product(product_type: Product) -> ProductChain:
    elements = _main_chain_elements(product_type.left, ProductChain)
    elements.append(convert_to_chain_style(product_type.right))
    return ProductChain(elements)


def _main_chain_elements(node: AST, chain_kind) -> List[AST]:
    """把已经规范化过的 left 那一段摊成主链元素表。"""
    converted = convert_to_chain_style(node)
    if isinstance(converted, chain_kind):
        return list(converted.elements)
    return [converted]


def _flatten_exponent(exponent_type: Exponent) -> ExponentChain:
    """主链（一路 $result）并成一条指数链，元素按书写顺序；支链留成一个元素。

    与和/积同形：``A <- B <- C`` 是 ``ExponentChain([A, B, C])``，
    ``A <- (B <- C)`` 是 ``ExponentChain([A, ExponentChain([B, C])])``。
    """
    def collect(e: Exponent, tail: List[AST]) -> List[AST]:
        arg = convert_to_chain_style(e.argument)
        res = convert_to_chain_style(e.result)
        if isinstance(res, ExponentChain):
            return list(res.elements) + [arg] + tail
        return [res, arg] + tail

    return ExponentChain(collect(exponent_type, []))


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

    result = convert_from_chain_style(chain.elements[0])
    for elem in chain.elements[1:]:
        result = Sum(result, convert_from_chain_style(elem))
    return result


def _reconstruct_product(chain: ProductChain) -> AST:
    if not chain.elements:
        return Nil()

    result = convert_from_chain_style(chain.elements[0])
    for elem in chain.elements[1:]:
        result = Product(result, convert_from_chain_style(elem))
    return result


def _reconstruct_exponent(chain: ExponentChain) -> AST:
    result = convert_from_chain_style(chain.elements[0])
    for arg in chain.elements[1:]:
        result = Exponent(result, convert_from_chain_style(arg))
    return result


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------


def is_chain_type(node: AST) -> bool:
    return isinstance(node, (SumChain, ProductChain, ExponentChain))

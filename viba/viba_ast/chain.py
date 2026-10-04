"""Chain-style canonicalization of AST nodes.

`|`, `*` and `<-` are left-associative, so in a written run the main chain is the
nesting that goes down `$left` (for an exponent, `$result`), folded into
SumChain / ProductChain / ExponentChain; the nesting on the `$right` side (an
exponent's `$argument`) is a branch, kept whole as one element of the main chain
and canonicalized the same way in turn (a branch's main chain becomes a nested
chain, and a branch of that branch goes one level deeper). Grouping is therefore
kept: `A * (B * C)` is ``ProductChain([A, ProductChain([B, C])])``, a different
thing from `A * B * C`. Tuples, tags and type applications are walked element by
element. convert_from_chain_style rebuilds the left-nested binary form.
"""

from typing import List

from viba.viba_ast._match import viba_type_match
from viba.viba_ast.nodes import (
    AST,
    TypeDefinition,
    GenericDefinition,
    Pattern,
    Sum,    Product,
    Exponent,
    Partial,
    Tagged,
    Member,
    MemberRead,
    TypeApp,
    Tuple,
    Nil,
    Never,
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
        Pattern=lambda s: Pattern(convert_to_chain_style(s.pattern)),
        Tagged=lambda t: Tagged(t.tag, convert_to_chain_style(t.type)),
        Member=lambda m: m,
        MemberRead=lambda m: MemberRead(convert_to_chain_style(m.owner), m.name),
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
    """The main chain (a run of $left) folds into this one; a branch ($right) stays one element."""
    elements = _main_chain_elements(sum_type.left, SumChain)
    elements.append(convert_to_chain_style(sum_type.right))
    return SumChain(elements)


def _flatten_product(product_type: Product) -> ProductChain:
    elements = _main_chain_elements(product_type.left, ProductChain)
    elements.append(convert_to_chain_style(product_type.right))
    return ProductChain(elements)


def _main_chain_elements(node: AST, chain_kind) -> List[AST]:
    """Flatten the already canonicalized `left` part into the main chain's elements."""
    converted = convert_to_chain_style(node)
    if isinstance(converted, chain_kind):
        return list(converted.elements)
    return [converted]


def _flatten_exponent(exponent_type: Exponent) -> ExponentChain:
    """The main chain (a run of $result) folds into one exponent chain, elements in
    written order; a branch stays one element.

    Laid out like sums and products: ``A <- B <- C`` is ``ExponentChain([A, B, C])``
    and ``A <- (B <- C)`` is ``ExponentChain([A, ExponentChain([B, C])])``.
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
        Pattern=lambda s: Pattern(convert_from_chain_style(s.pattern)),
        Tagged=lambda t: Tagged(t.tag, convert_from_chain_style(t.type)),
        MemberRead=lambda m: MemberRead(convert_from_chain_style(m.owner), m.name),
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
    if not chain.elements:
        return Never()      # no result to head it: the empty function is bottom
    result = convert_from_chain_style(chain.elements[0])
    for arg in chain.elements[1:]:
        result = Exponent(result, convert_from_chain_style(arg))
    return result


# ----------------------------------------------------------------------
# Analysis helpers
# ----------------------------------------------------------------------


def is_chain_type(node: AST) -> bool:
    return isinstance(node, (SumChain, ProductChain, ExponentChain))

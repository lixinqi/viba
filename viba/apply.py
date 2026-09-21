"""`T << X`: the function T with the written argument X given to it.

Partial computation, written in the design itself:

    (A <- $b B <- $c C) << $b B            is  A <- $c C
    (A <- $b B <- $c C) << $c C            is  A <- $b B
    (A <- $b B <- $c C) << $c C << $b B    is  A

Giving every argument leaves the result: the chain is gone, not empty. The
argument is matched the way the layers address one — by its tag when it is
tagged, by its written form otherwise — and a design that gives an argument
the function does not have is a mistake (`ApplyError`), not a judgment.

Names are unfolded through the caller's own resolution, so the reduction
happens where the syntax is read: the judgment with its resolver, the
descriptor layer with the pool's.
"""

from typing import Callable, Optional, Tuple

from viba import viba_ast
from viba.type import ApplyError

_EXP_NODES = (viba_ast.Exponent, viba_ast.ExponentChain)


def reduce_apply(node, module, resolve: Callable) -> Tuple[object, object]:
    """(node, module) with every `<<` given.

    `resolve(name, module) -> (body, home) | None` is how a written name is
    unfolded; an alias is followed to the end of the chain.
    """
    while isinstance(node, viba_ast.Apply):
        base, base_module = reduce_apply(node.function, module, resolve)
        node, module = _give(base, base_module, node.argument, resolve)
    return node, module


def _give(base, module, argument, resolve):
    base, module = _unfold(base, module, resolve)
    if not isinstance(base, _EXP_NODES):
        raise ApplyError(
            f"only a function has arguments to give, not {viba_ast.unparse_type(base)}")
    elements = _elements(base)
    for index, written in enumerate(elements[1:], start=1):
        if _matches(written, argument):
            rest = elements[:index] + elements[index + 1:]
            if len(rest) == 1:
                return rest[0], module
            return viba_ast.ExponentChain(rest), module
    raise ApplyError(
        f"the function has no such argument: {viba_ast.unparse_type(argument)}")


def _unfold(node, module, resolve):
    """A name runs to its body, name after name."""
    seen = set()
    while isinstance(node, viba_ast.TypeRef):
        if node.name in seen:
            return node, module
        seen.add(node.name)
        target = resolve(node.name, module)
        if target is None:
            return node, module
        node, module = target
    return node, module


def _matches(written, given) -> bool:
    """The written argument and the given one are the same address: a tag when
    both carry one (that is how a field is addressed), the written form else."""
    if isinstance(written, viba_ast.Tagged) and isinstance(given, viba_ast.Tagged):
        return written.tag == given.tag
    return viba_ast.unparse_type(written) == viba_ast.unparse_type(given)


def _elements(node):
    """The exponent's elements in written order: result first."""
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    if isinstance(node, viba_ast.Exponent):
        return _elements(node.result) + [node.argument]
    return [node]


__all__ = ["reduce_apply"]

"""`T << X`: the function T with the written argument X given to it.

Partial computation, written in the design itself:

    (A <- $b B <- $c C) << $b B            is  A <- $c C
    (A <- $b B <- $c C) << $c C            is  A <- $b B
    (A <- $b B <- $c C) << $c C << $b B    is  A

Giving every argument leaves the result: the chain is gone, not empty, and
documentation (`Hint[$python_code {...}]`) is no argument, so a function that
ends in one still lands on its result. The
argument is matched the way the layers address one — by its tag when it is
tagged, by its written form otherwise — and a design that gives an argument
the function does not have is a mistake (`PartialError`), not a judgment.

Names are unfolded through the caller's own resolution, so the reduction
happens where the syntax is read: the judgment with its resolver, the
descriptor layer with the pool's.
"""

from typing import Callable, Optional, Tuple

from viba import viba_ast
from viba.type import PartialError

_EXP_NODES = (viba_ast.Exponent, viba_ast.ExponentChain)


def reduce_partial(node, module, resolve: Callable) -> Tuple[object, object]:
    """(node, module) with every `<<` given.

    `resolve(name, module) -> (body, home) | None` is how a written name is
    unfolded; an alias is followed to the end of the chain.
    """
    while isinstance(node, viba_ast.Partial):
        base, base_module = reduce_partial(node.function, module, resolve)
        node, module = _give(base, base_module, node.argument, resolve)
    return node, module


def _give(base, module, argument, resolve):
    base, module = _unfold(base, module, resolve)
    if not isinstance(base, _EXP_NODES):
        raise PartialError(
            f"only a function has arguments to give, not {_written(base)}")
    elements = _elements(base)
    for index, written in enumerate(elements[1:], start=1):
        if _matches(written, argument):
            rest = elements[:index] + elements[index + 1:]
            if all(_is_documentation(element) for element in rest[1:]):
                # Nothing but the result and documentation is left: documentation
                # is no argument (the judgment drops it too), so this is the
                # result itself - what a finished call declares at that position.
                return rest[0], module
            return viba_ast.ExponentChain(rest), module
    raise PartialError(f"the function has no such argument: {_written(argument)}")


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


def _written(node) -> str:
    """The piece as one line: error messages read better without the layout."""
    return " ".join(viba_ast.unparse_type(node).split())


def _is_documentation(node) -> bool:
    """A piece that carries a code block: documentation, not an argument."""
    return any(isinstance(part, viba_ast.CodeBlock) for part in viba_ast.walk(node))


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


__all__ = ["reduce_partial"]

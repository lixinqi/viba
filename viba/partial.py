"""`T << X`: the function T with the written argument X given to it.

Internal: this is the reduction `<<` goes through, used by `viba.is_sub_type`
and the descriptor layer. What a caller uses is the judgment (`is_sub_type`)
and the reading of a module as a function, not this module.

Partial computation, written in the design itself:

    (A <- $b B <- $c C) << $b B            is  A <- $c C
    (A <- $b B <- $c C) << $c C            is  A <- $b B
    (A <- $b B <- $c C) << $c C << $b B    is  A

Giving every argument leaves the result: the chain is gone, not empty, and
documentation (`Hint[$python_code {...}]`) is no argument, so a function that
ends in one still lands on its result.

The argument is matched the way the layers address one — by its tag when it is
tagged, by its written form otherwise — and what is given has to *fit* the slot
it is given to: `(A <- $b B) << $b C` needs `C <: B`. The judge is the
caller's (the judgment passes its own walk, the descriptor layer passes
`is_sub_type`), and names are unfolded through the caller's resolution too, so
a design is refused where it is built and where it is judged. Giving an
argument the function does not have, or one that does not fit, is a mistake
(`PartialError`), not a judgment.
"""

from typing import Callable, Optional, Tuple

from viba import viba_ast
from viba.type import BUILTIN_MODULE, AstNodeType, NilType, Ok, PartialError

_EXP_NODES = (viba_ast.Exponent, viba_ast.ExponentChain)


RET_NAME = "__ret__"
ARGS_NAME = "__args__"
ENVIRON_TAG = "$env"
ENVIRON_TYPE = "Environment"

# The tag that makes a function's arguments lazy (viba/builtin.viba).
LAZY_MARKER_TAG = "$__param_lazy_evaluated_tag_yanatutt__"


def marked_function(node, module):
    """The function type inside `ParametersLazyEvaluated[F]`, else None.

    A marked function *is* the function it marks: the marker says how the
    arguments are given (one at a time, to the host, viba-interpreter.md), not
    what the function is. Every layer reads through it, so all three agree on
    what is being called.

    The name alone decides nothing — a module may define
    `ParametersLazyEvaluated` itself, and a local definition wins — so the
    definition the constructor names has to carry the reserved tag.
    """
    if not isinstance(node, viba_ast.TypeApp) or len(node.args or []) != 1:
        return None
    body = None
    local = _definition(module, node.constructor)
    if local is not None:
        body = local.body
    else:
        builtin = BUILTIN_MODULE.lookup(node.constructor)
        if isinstance(builtin, Ok) and isinstance(builtin.ok_value, AstNodeType):
            body = builtin.ok_value.ast_node
    if body is None:
        return None
    if not any(isinstance(part, viba_ast.Tagged) and part.tag == LAZY_MARKER_TAG
               for part in viba_ast.walk(body)):
        return None
    return node.args[0]


def module_as_function(module, name):
    """(body, home) for a bare import name read as a function, or None.

    A module is a function too: `__ret__ <- $env Environment <- __args__`, its
    input the environment and its arguments, its output `__ret__`. A module that
    declares no `__args__` still has that argument slot — an empty product, which
    is written `()` — so a call says what it gives and an incomplete chain cannot
    pass for a call. `name` has to be one of this module's imports — the name the
    import binds, not a member of it (a dotted name is that module's own
    definition and resolves as it always did) — and the module it names has to be
    a program, `__ret__` and all.
    """
    imports = getattr(module, "imports", None)
    if not imports or name not in imports:
        return None
    imported = module.module_environment(imports[name])
    if not isinstance(imported, Ok):
        return None
    ret = _definition(imported.ok_value, RET_NAME)
    if ret is None:
        return None
    elements = [ret.body,
                viba_ast.Tagged(ENVIRON_TAG, viba_ast.TypeRef(ENVIRON_TYPE))]
    elements += _argument_slots(imported.ok_value)
    return (viba_ast.ExponentChain(elements), imported.ok_value)


def _argument_slots(module):
    """A module call's argument slots: its `__args__` members, or the empty `()`.

    `__args__` is a product type and its members are the call's arguments, in
    written order; no `__args__` (or an empty product) is a module with no
    arguments, and its slot is written `()`.
    """
    definition = _definition(module, ARGS_NAME)
    if definition is None:
        return [viba_ast.Tuple([])]
    body = definition.body
    if isinstance(body, (viba_ast.Product, viba_ast.ProductChain)):
        slots = [factor for factor in product_elements(body)
                 if not _is_product_unit(factor)]
        return slots or [viba_ast.Tuple([])]
    if _is_product_unit(body):
        return [viba_ast.Tuple([])]
    raise PartialError(f"{ARGS_NAME} is not a product: {_written(body)}")


def product_elements(node):
    """The factors of a written product, flattened in written order.

    One definition for the three layers that read a product apart: the
    interpreter (a product is its factors), the judgment and this module.
    """
    if isinstance(node, viba_ast.Product):
        return product_elements(node.left) + product_elements(node.right)
    if isinstance(node, viba_ast.ProductChain):
        return list(node.elements)
    return [node]


def _is_product_unit(node) -> bool:
    """The multiplicative unit as written: `Object`, `nil`, or a bare nil."""
    if isinstance(node, viba_ast.Nil):
        return True
    if isinstance(node, viba_ast.TypeRef):
        builtin = BUILTIN_MODULE.lookup(node.name)
        return isinstance(builtin, Ok) and isinstance(builtin.ok_value, NilType)
    return False


def _definition(module, name):
    for node in getattr(module, "module", module).body:
        if getattr(node, "name", None) == name:
            return node
    return None


def reduce_partial(node, module, resolve: Callable, judge: Callable) -> Tuple[object, object]:
    """(node, module) with every `<<` given.

    `resolve(name, module) -> (body, home) | None` is how a written name is
    unfolded; an alias is followed to the end of the chain.
    `judge(sub, sub_module, sup, sup_module) -> bool` says whether a given
    argument fits the slot it is written to.
    """
    while isinstance(node, viba_ast.Partial):
        base, base_module = reduce_partial(node.function, module, resolve, judge)
        node, module = _give(base, base_module, node.argument, module, resolve, judge)
    return node, module


def _give(base, module, argument, argument_module, resolve, judge):
    base, module = _unfold(base, module, resolve)
    if not isinstance(base, _EXP_NODES):
        raise PartialError(
            f"only a function has arguments to give, not {_written(base)}")
    elements = _elements(base)
    for index, written in enumerate(elements[1:], start=1):
        if _matches(written, argument, module, argument_module, judge):
            rest = elements[:index] + elements[index + 1:]
            if all(_is_documentation(element) for element in rest[1:]):
                # Nothing but the result and documentation is left: documentation
                # is no argument (the judgment drops it too), so this is the
                # result itself - what a finished call declares at that position.
                return rest[0], module
            return viba_ast.ExponentChain(rest), module
    raise PartialError(f"the function has no such argument: {_written(argument)}")


def _unfold(node, module, resolve):
    """A name runs to its body, name after name — and a marked function is the
    function it marks, so the marker is read through here."""
    seen = set()
    while True:
        marked = marked_function(node, module)
        if marked is not None:
            node = marked
            continue
        if not isinstance(node, viba_ast.TypeRef) or node.name in seen:
            return node, module
        seen.add(node.name)
        target = resolve(node.name, module)
        if target is None:
            return node, module
        node, module = target


def _written(node) -> str:
    """The piece as one line: error messages read better without the layout."""
    return " ".join(viba_ast.unparse_type(node).split())


def _is_documentation(node) -> bool:
    """A piece that carries a code block: documentation, not an argument."""
    return any(isinstance(part, viba_ast.CodeBlock) for part in viba_ast.walk(node))


def _matches(written, given, module, given_module, judge) -> bool:
    """Does the given argument go to this written slot?

    A tag when both carry one (that is how a field is addressed) — and then the
    given type has to fit the declared one, so `(A <- $b B) << $b C` is legal
    only when `C <: B`. An argument written without a tag takes the next free
    slot, the way a call gives one, if it fits there: that is how a module's
    `__args__` are given one by one. Without a tag on either side there is no
    address to name: the two are the same piece, or they are not.
    """
    if isinstance(written, viba_ast.Tagged) and isinstance(given, viba_ast.Tagged):
        if written.tag != given.tag:
            return False
        if judge(given.type, given_module, written.type, module):
            return True
        raise PartialError(
            f"{_written(given)} does not fit {_written(written)}: "
            f"{_written(given.type)} <: {_written(written.type)} does not hold")
    if isinstance(written, viba_ast.Tagged):
        if judge(given, given_module, written.type, module):
            return True
        raise PartialError(
            f"{_written(given)} does not fit {_written(written)}: "
            f"{_written(given)} <: {_written(written.type)} does not hold")
    return viba_ast.unparse_type(written) == viba_ast.unparse_type(given)


def _elements(node):
    """The exponent's elements in written order: result first."""
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    if isinstance(node, viba_ast.Exponent):
        return _elements(node.result) + [node.argument]
    return [node]


__all__ = ["reduce_partial", "module_as_function", "marked_function",
           "product_elements"]

"""Host implementations of the two selectors declared in branch.viba.

The branch value's slot is `CalledByNeed[Any]`, so that one argument is handed
over as something to call rather than a value: `id_or_never(env, condition,
get_v)` calls `get_v` only on the branch it takes, and the argument expression
of the branch it does not take is never evaluated. A switch that is not taken
answers `never` (`never * value = never`), and the surrounding sum drops it.
`id` is the identity: the product selector that keeps the value.

    def id_or_never(env, condition, get_v):
        if condition.value:
            return get_v()
        return _never()

The environment arrives as itself and the condition as its `VibaNode` — the
other slots are ordinary ones — so the leaf of a `bool` is `.value`.
"""

from viba import viba_ast
from viba.reflect import VibaNode, access
from viba.type import AstNodeType, custom_module
from viba.viba_type_descriptor import descriptor_of

_MODULE = custom_module("")


def _unit(node) -> VibaNode:
    return VibaNode(access, descriptor_of(AstNodeType(node, _MODULE)), node)


def _never() -> VibaNode:
    return _unit(viba_ast.Never())


def id_or_never(env, condition, get_v):
    """Answer the value when the condition holds; otherwise never."""
    if condition.value:
        return get_v()
    return _never()


def never_or_id(env, condition, get_v):
    """Answer never when the condition holds; otherwise the value."""
    if condition.value:
        return _never()
    return get_v()


def get_func(module_path, func_name):
    """Route the two selectors; anything else defers to whoever called us."""
    if func_name == "id_or_never":
        return id_or_never
    if func_name == "never_or_id":
        return never_or_id
    return None

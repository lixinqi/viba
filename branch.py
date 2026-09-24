"""Host implementations of the two selectors declared in branch.viba.

A switch takes the branch value. When selected it returns that value
(nil * value = value); otherwise it returns never (never * value = never).
The surrounding sum then drops the eliminated branch.
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


def nil_or_never(env, condition, value):
    """Answer value when condition holds; otherwise never."""
    return value if condition.value else _never()


def never_or_nil(env, condition, value):
    """Answer never when condition holds; otherwise value."""
    return _never() if condition.value else value


def get_func(module_path, func_name):
    """Route the two selectors; anything else defers to whoever called us."""
    if func_name == "nil_or_never":
        return nil_or_never
    if func_name == "never_or_nil":
        return never_or_nil
    return None

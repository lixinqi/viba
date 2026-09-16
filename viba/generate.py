"""Random instance generator: Metric leaves become random literals.

Given a rule (viba.type.AstNodeType), produce instances structurally
parallel to it: every Metric[Name] field is replaced by a random
literal of Name's data shape; everything else — including Assert
fields — is kept as-is. On a legal rule no instance can trigger a
judgment error — running instances through is_compliant yields
Ok(True) or Ok(false) only.
"""

import random

from viba import ast as viba_ast
from viba.type import (
    AstNodeType,
    BoolType,
    BuiltinGenericType,
    FloatType,
    IntType,
    ModuleType,
    NilType,
    NeverType,
    Ok,
    StrType,
    module_get_type,
)

_WORDS = ("alpha", "beta", "gamma", "delta")


def generate(rule: AstNodeType, count: int, seed=None, fail_prob: float = 0.1) -> list:
    """`count` random instances of `rule` (deterministic via seed).

    Each Assert field passes unchanged with probability 1-fail_prob
    and is witnessed by AssertionFailed with probability fail_prob.
    """
    rng = random.Random(seed)
    return [_instance_of(rule, rng, fail_prob) for _ in range(count)]


def _instance_of(rule: AstNodeType, rng, fail_prob: float) -> AstNodeType:
    node = _gen_node(rule.ast_node, rule.container_module, rng, fail_prob)
    return AstNodeType(node, rule.container_module)


def _gen_node(node, module: ModuleType, rng, fail_prob):
    if isinstance(node, viba_ast.Tagged):
        return viba_ast.Tagged(node.tag, _gen_node(node.type, module, rng, fail_prob))
    if isinstance(node, viba_ast.Product):
        left = _gen_node(node.left, module, rng, fail_prob)
        right = _gen_node(node.right, module, rng, fail_prob)
        return viba_ast.Product(left, right)
    if isinstance(node, viba_ast.ProductChain):
        elems = [_gen_node(e, module, rng, fail_prob) for e in node.elements]
        return viba_ast.ProductChain(elems)
    if isinstance(node, viba_ast.Tuple):
        return viba_ast.Tuple([_gen_node(e, module, rng, fail_prob) for e in node.elements])
    if isinstance(node, viba_ast.Sum):
        branches = [node.left, node.right]
        return _gen_node(rng.choice(branches), module, rng, fail_prob)
    if isinstance(node, viba_ast.SumChain):
        return _gen_node(rng.choice(list(node.elements)), module, rng, fail_prob)
    if isinstance(node, viba_ast.TypeApp):
        return _gen_typeapp(node, module, rng, fail_prob)
    if isinstance(node, viba_ast.TypeRef):
        return _gen_typeref(node, module, rng, fail_prob)
    return node


def _gen_typeapp(node, module: ModuleType, rng, fail_prob: float):
    if node.constructor == "Metric":
        return _metric_literal(node.args[0], module, rng, fail_prob)
    if node.constructor == "Assert" and rng.random() < fail_prob:
        return _failed_assert()
    resolved = module_get_type(module, node.constructor)
    if isinstance(resolved, Ok) and isinstance(resolved.value, BuiltinGenericType):
        return _container_literal(node, resolved.value, module, rng, fail_prob)
    return node


def _failed_assert():
    """A failed assertion: poison in place of the Assert field."""
    return viba_ast.TypeApp("AssertionFailed", [viba_ast.Nil(), viba_ast.TypeRef("str")])


def _container_literal(node, builtin, module: ModuleType, rng, fail_prob):
    if builtin.name == "dict":
        pairs = [_dict_pair(node, module, rng, fail_prob) for _ in range(rng.randint(0, 2))]
        return viba_ast.TypeApp("DictLiteral", pairs)
    elems = [_typed_literal(node.args[0], module, rng, fail_prob) for _ in range(rng.randint(0, 2))]
    ctor = "ListLiteral" if builtin.name == "list" else "SetLiteral"
    return viba_ast.TypeApp(ctor, elems)


def _dict_pair(node, module: ModuleType, rng, fail_prob):
    key = _typed_literal(node.args[0], module, rng, fail_prob)
    val = _typed_literal(node.args[1], module, rng, fail_prob)
    return viba_ast.Tuple([key, val])


def _typed_literal(node, module: ModuleType, rng, fail_prob):
    if isinstance(node, viba_ast.TypeRef):
        resolved = module_get_type(module, node.name)
        if isinstance(resolved, Ok):
            return _literal_of_type(resolved.value, rng, fail_prob)
    return _gen_node(node, module, rng, fail_prob)


def _metric_literal(name_node, module: ModuleType, rng, fail_prob):
    if isinstance(name_node, viba_ast.TypeRef):
        resolved = module_get_type(module, name_node.name)
        if isinstance(resolved, Ok):
            return _literal_of_type(resolved.value, rng, fail_prob)
    return viba_ast.Nil()


def _literal_of_type(t, rng, fail_prob):
    if isinstance(t, BoolType):
        return viba_ast.Constant(rng.random() < 0.5)
    if isinstance(t, IntType):
        return viba_ast.Constant(rng.randint(0, 100))
    if isinstance(t, FloatType):
        return viba_ast.Constant(round(rng.uniform(0, 100), 2))
    if isinstance(t, StrType):
        return viba_ast.Constant(rng.choice(_WORDS))
    if isinstance(t, AstNodeType) and isinstance(t.ast_node, viba_ast.TypeDefinition):
        return _gen_node(t.ast_node.body, t.container_module, rng, fail_prob)
    return viba_ast.Nil()


def _gen_typeref(node, module: ModuleType, rng, fail_prob):
    resolved = module_get_type(module, node.name)
    if not isinstance(resolved, Ok):
        return node
    t = resolved.value
    if isinstance(t, (NilType, NeverType)):
        return node
    if isinstance(t, (BoolType, IntType, FloatType, StrType)):
        return _literal_of_type(t, rng, fail_prob)
    if isinstance(t, AstNodeType) and isinstance(t.ast_node, viba_ast.TypeDefinition):
        return _gen_node(t.ast_node.body, t.container_module, rng, fail_prob)
    return node

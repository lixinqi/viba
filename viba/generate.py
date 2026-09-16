"""Random witness generator: Metric leaves become random literals.

Given a rule (viba.type.AstNodeType), produce witnesses structurally
parallel to it: every Metric[Name] field is replaced by a random
literal of Name's data shape; everything else is kept as-is, except
Predicate fields (replaced by PredicationFailed with probability
fail_prob) and not[...] fields (the shell is kept, each branch leaf
swapped for a same-tag PredicationFailed). On a legal rule no witness
can trigger a judgment error — running witnesses through is_compliant
yields Ok(True) or Ok(false) only.
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
    """`count` random witnesses of `rule` (deterministic via seed).

    Each Predicate field passes unchanged with probability 1-fail_prob
    and is witnessed by PredicationFailed with probability fail_prob.
    """
    rng = random.Random(seed)
    return [_witness_of(rule, rng, fail_prob) for _ in range(count)]


def _witness_of(rule: AstNodeType, rng, fail_prob: float) -> AstNodeType:
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
    if node.constructor == "Predicate" and rng.random() < fail_prob:
        return _failed_predication()
    if node.constructor == "not":
        return _not_witness(node, module, rng, fail_prob)
    resolved = module_get_type(module, node.constructor)
    if isinstance(resolved, Ok) and isinstance(resolved.value, BuiltinGenericType):
        return _container_literal(node, resolved.value, module, rng, fail_prob)
    return node


def _not_witness(node, module: ModuleType, rng, fail_prob: float):
    """A not[oneof] field: the witness keeps the shell and swaps every
    branch leaf for the poison PredicationFailed. With probability
    fail_prob a branch stays positive (its own Predicate type), so that
    branch carries no refutation and the witness judges False."""
    branches = _sum_branches(node.args[0], module)
    if branches is None:
        return node
    leaves = []
    for tag, branch in branches:
        if rng.random() < fail_prob:
            leaves.append(viba_ast.Tagged(tag, branch))
        else:
            leaves.append(viba_ast.Tagged(tag, _failed_predication()))
    return viba_ast.TypeApp("not", [_oneof(leaves)])


def _oneof(nodes):
    out = nodes[0]
    for node in nodes[1:]:
        out = viba_ast.Sum(out, node)
    return out


def flip_sites(rule: AstNodeType) -> int:
    """Independent flips generate makes for this rule: one per positive
    Predicate field, one per tagged not branch. A witness passes only
    when none of them flips, so its True share is
    (1 - fail_prob) ** flip_sites."""
    return _flip_sites(rule.ast_node, rule.container_module)


def _flip_sites(node, module: ModuleType) -> int:
    if isinstance(node, viba_ast.Tagged):
        return _flip_sites(node.type, module)
    if isinstance(node, viba_ast.Product):
        return _flip_sites(node.left, module) + _flip_sites(node.right, module)
    if isinstance(node, viba_ast.ProductChain):
        return sum(_flip_sites(e, module) for e in node.elements)
    if isinstance(node, viba_ast.Tuple):
        return sum(_flip_sites(e, module) for e in node.elements)
    if isinstance(node, viba_ast.TypeApp):
        if node.constructor == "Predicate":
            return 1
        if node.constructor == "not":
            branches = _sum_branches(node.args[0], module)
            return 0 if branches is None else len(branches)
        return 0
    if isinstance(node, viba_ast.TypeRef):
        body, home = _unfold(node, module)
        return 0 if body is node else _flip_sites(body, home)
    return 0


def _sum_branches(node, module: ModuleType):
    """Flatten a not argument into [(tag, type), ...], unfolding named
    definitions; None if any branch is not tagged."""
    node, module = _unfold(node, module)
    if isinstance(node, viba_ast.Sum):
        left = _sum_branches(node.left, module)
        right = _sum_branches(node.right, module)
        return None if left is None or right is None else left + right
    if isinstance(node, viba_ast.SumChain):
        out = []
        for element in node.elements:
            part = _sum_branches(element, module)
            if part is None:
                return None
            out += part
        return out
    if isinstance(node, viba_ast.Tagged):
        return [(node.tag, node.type)]
    return None


def _unfold(node, module: ModuleType):
    """A TypeRef to a plain TypeDefinition unfolds to its body."""
    if not isinstance(node, viba_ast.TypeRef):
        return node, module
    resolved = module_get_type(module, node.name)
    if not isinstance(resolved, Ok) or not isinstance(resolved.value, AstNodeType):
        return node, module
    target = resolved.value
    if not isinstance(target.ast_node, viba_ast.TypeDefinition):
        return node, module
    return target.ast_node.body, target.container_module


def _failed_predication():
    """A failed predication: the failure marker for a positive
    Predicate field. It never seats in a Predicate slot, so a witness
    carrying it judges False."""
    return viba_ast.TypeApp("PredicationFailed", [viba_ast.Nil(), viba_ast.TypeRef("str")])


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

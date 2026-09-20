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

from viba import viba_ast
from viba.reflect import VibaNode, access as reflect_access
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
from viba.rule.markers import PRODUCT_MARKER
from viba.viba_type_descriptor import descriptor_of

_WORDS = ("alpha", "beta", "gamma", "delta")


def generate_witness(rule: AstNodeType, prepared: AstNodeType) -> AstNodeType:
    """The witness of a metric-carrying rule for one prepared call.

    Every `Metric[CoreFunc]` field becomes the metric object: the marker,
    `$func` naming CoreFunc, and `$call_instance` carrying
    `<Result>[<value>] <- (<the prepared call>)`, where value is what
    CoreFunc's `Hint[$python_code {...}]` answers for that call. The other
    fields are the rule's own: a positive Predicate stays positive, because
    running the code is reset_predication_by_python_code's step.
    """
    module = rule.container_module
    out = []
    for member in _chain(rule.ast_node):
        if _is_marker(member):
            continue                      # a witness carries no rule marker
        metric = _metric_of(member, module)
        out.append(metric(prepared) if metric is not None else member)
    return AstNodeType(viba_ast.ProductChain(out), module)


def _is_marker(member):
    """The rule's own marker: the rule says it is a rule, the witness does not."""
    return (isinstance(member, viba_ast.TypeRef)
            and member.name.split(".")[-1] == PRODUCT_MARKER)


def _chain(node):
    return list(_chain_node(node).elements)


def _chain_node(node):
    if isinstance(node, (viba_ast.ProductChain, viba_ast.SumChain,
                         viba_ast.ExponentChain)):
        return node
    if isinstance(node, (viba_ast.Product, viba_ast.Sum, viba_ast.Exponent)):
        return viba_ast.convert_to_chain_style(node)
    return viba_ast.ProductChain([node])


def _metric_of(member, module):
    """A builder for this member when it is a metric object, else None."""
    if not isinstance(member, viba_ast.Tagged):
        return None
    written = member.type
    if not (isinstance(written, viba_ast.TypeApp)
            and written.constructor.split(".")[-1] == "Metric"):
        return None
    return _MetricMember(member.tag, written, module)


class _MetricMember:
    """One `$tag Metric[CoreFunc]` member of the rule."""

    def __init__(self, tag, written, module):
        self.tag = tag
        self.written = written
        self.module = module
        resolved = module_get_type(module, written.constructor)
        if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
            raise ValueError(f"{written.constructor} is not a metric definition")
        self.definition = resolved.ok_value.ast_node
        self.home = resolved.ok_value.container_module
        if not isinstance(self.definition, viba_ast.GenericDefinition):
            raise ValueError(f"{written.constructor} is not generic")
        self.core = written.args[0]

    def __call__(self, prepared):
        return viba_ast.Tagged(self.tag, viba_ast.ProductChain([
            viba_ast.TypeRef("Object"),
            viba_ast.Tagged(self._marker_tag(), viba_ast.Nil()),
            viba_ast.Tagged("$func", self.core),
            viba_ast.Tagged("$call_instance", self._call_instance(prepared)),
        ]))

    def _marker_tag(self) -> str:
        """The reserved tag: the tagged member of the object's anchor."""
        for member in _chain(self.definition.body):
            if not isinstance(member, viba_ast.TypeRef):
                continue
            resolved = module_get_type(self.home, member.name)
            if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
                continue
            for inner in _chain(resolved.ok_value.ast_node.body):
                if isinstance(inner, viba_ast.Tagged) and isinstance(inner.type, viba_ast.Nil):
                    return inner.tag
        raise ValueError("the metric definition carries no marker tag")

    def _result_name(self) -> str:
        """How `Result` is written where this field was written: the metric's
        own prefix is the one the rule used (`metric.Metric` -> `metric`)."""
        prefix = (self.written.constructor.rsplit(".", 1)[0]
                  if "." in self.written.constructor else "")
        for member in _chain(self.definition.body):
            if not (isinstance(member, viba_ast.Tagged) and member.tag == "$call_instance"):
                continue
            call = _chain(member.type)
            if call and isinstance(call[0], viba_ast.TypeApp):
                name = call[0].constructor.split(".")[-1]
                return f"{prefix}.{name}" if prefix else name
        raise ValueError("the metric definition has no $call_instance")

    def _call_instance(self, prepared):
        return viba_ast.ExponentChain([
            viba_ast.TypeApp(self._result_name(), [viba_ast.Constant(self._measure(prepared))]),
            _chain_node(prepared.ast_node),
        ])

    def _measure(self, prepared):
        """What CoreFunc's own code answers for this prepared call."""
        function, home = _function_of(self.core, self.module)
        code = _hint_code(function, self.core)
        root = VibaNode(reflect_access, descriptor_of(AstNodeType(function.body, home)),
                        prepared.ast_node, data_module=self.module)
        arguments = []
        positional = 0
        for element in _chain(function.body)[1:]:
            if isinstance(element, viba_ast.Tagged):
                arguments.append(root.by_tag(element.tag))
            elif (isinstance(element, viba_ast.TypeApp)
                  and element.constructor.split(".")[-1] == "Hint"):
                continue                      # documentation, not an argument
            else:
                positional += 1
                arguments.append(root.by_field_index(positional))
        namespace = {}
        exec(compile(code, "<metric_func>", "exec"), namespace)
        return namespace["metric_func"](*arguments)


def _function_of(core, module):
    """(definition node, its home module) of the CoreFunc name."""
    resolved = module_get_type(module, core.name)
    if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
        raise ValueError(f"{core.name} names no metric function")
    return resolved.ok_value.ast_node, resolved.ok_value.container_module


def _hint_code(function, core):
    """The code the function's trailing Hint carries."""
    for element in _chain(function.body):
        if (isinstance(element, viba_ast.TypeApp)
                and element.constructor.split(".")[-1] == "Hint"):
            return element.args[0].type.code
    raise ValueError(f"{core.name} carries no Hint[$python_code {{...}}]")


def generate_witnesses(rule: AstNodeType, count: int, seed=None, fail_prob: float = 0.1) -> list:
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
    if isinstance(resolved, Ok) and isinstance(resolved.ok_value, BuiltinGenericType):
        return _container_literal(node, resolved.ok_value, module, rng, fail_prob)
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
    if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
        return node, module
    target = resolved.ok_value
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
            return _literal_of_type(resolved.ok_value, rng, fail_prob)
    return _gen_node(node, module, rng, fail_prob)


def _metric_literal(name_node, module: ModuleType, rng, fail_prob):
    """A metric's measured value goes wherever the Metric definition puts
    it: with Metric[T] := $value T the witness is that tag carrying the
    literal, not the bare literal."""
    if isinstance(name_node, viba_ast.TypeRef):
        resolved = module_get_type(module, name_node.name)
        if isinstance(resolved, Ok):
            literal = _literal_of_type(resolved.ok_value, rng, fail_prob)
            return _under_metric_slot(module, literal)
    return _under_metric_slot(module, viba_ast.Nil())


def _under_metric_slot(module: ModuleType, literal):
    """The tag the Metric definition body puts the measured value under;
    a body that is not one tagged layer takes the literal as is."""
    resolved = module_get_type(module, "Metric")
    if isinstance(resolved, Ok) and isinstance(resolved.ok_value, AstNodeType):
        body = resolved.ok_value.ast_node
        if isinstance(body, viba_ast.GenericDefinition):
            if isinstance(body.body, viba_ast.Tagged):
                return viba_ast.Tagged(body.body.tag, literal)
    return literal


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
    t = resolved.ok_value
    if isinstance(t, (NilType, NeverType)):
        return node
    if isinstance(t, (BoolType, IntType, FloatType, StrType)):
        return _literal_of_type(t, rng, fail_prob)
    if isinstance(t, AstNodeType) and isinstance(t.ast_node, viba_ast.TypeDefinition):
        return _gen_node(t.ast_node.body, t.container_module, rng, fail_prob)
    return node

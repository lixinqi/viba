"""Run a witness's Predicate code and swap a false predication for the
poison.

reset_predication_by_python_code(witness) -> witness. Every Predicate
node in the witness gets its compiled $python_code executed against a
`self` built from the witness's tagged fields (viba-rule.md section 4: tag
minus $ becomes the attribute, so the metric definition's own $value slot
is what the predicate reads as .value, a nested product keeps expanding as
attributes, a tuple by index). A predicate that returns false is replaced
by PredicationFailed[nil, str].

Reading a field follows the shape rules the judgment does, so the two
agree on where a value sits:

- a tagged field is an attribute (`.distance`, `.call_instance`);
- a product expands into its tagged fields;
- an exponent is the call it writes: the result first, the arguments
  after, addressed as `field_0`, `field_1`, ... — the measurement of a
  metric object is `self.distance.call_instance.field_0.ok`;
- a generic application binds its actuals into the definition's body, so
  `Result[5]` is the `$ok` branch carrying 5 while `$err` carries nothing;
- a sum value sits on the branches that carry a value.
"""

import copy
import types

from viba import viba_ast
from viba.viba_ast import nodes as ast_nodes
from viba.type import AstNodeType, Ok, module_get_type


def reset_predication_by_python_code(witness: AstNodeType) -> AstNodeType:
    self_obj = _self_object(witness.ast_node, witness.container_module)
    node = _reset(witness.ast_node, self_obj)
    return AstNodeType(node, witness.container_module)


def _reset(node, self_obj):
    if isinstance(node, ast_nodes.Product):
        return ast_nodes.Product(_reset(node.left, self_obj), _reset(node.right, self_obj))
    if isinstance(node, ast_nodes.ProductChain):
        return ast_nodes.ProductChain([_reset(e, self_obj) for e in node.elements])
    if isinstance(node, ast_nodes.Tagged):
        return ast_nodes.Tagged(node.tag, _reset(node.type, self_obj))
    if isinstance(node, ast_nodes.Sum):
        return ast_nodes.Sum(_reset(node.left, self_obj), _reset(node.right, self_obj))
    if isinstance(node, ast_nodes.SumChain):
        return ast_nodes.SumChain([_reset(e, self_obj) for e in node.elements])
    if isinstance(node, ast_nodes.Tuple):
        return ast_nodes.Tuple([_reset(e, self_obj) for e in node.elements])
    if _is_predicate(node):
        return _predicate(node, self_obj)
    return node


def _predicate(node, self_obj):
    code = node.args[1].type.code
    namespace = {}
    exec(compile(code, "<predicate>", "exec"), namespace)
    if namespace["predicate"](self_obj):
        return node
    return ast_nodes.TypeApp("PredicationFailed", [ast_nodes.Nil(), ast_nodes.TypeRef("str")])


def _self_object(node, module):
    while isinstance(node, ast_nodes.Tagged):
        node = node.type
    fields = {}
    for tag, body in _product_tags(node):
        if not _is_predicate(body):
            fields[tag[1:]] = _value(body, module)
    return types.SimpleNamespace(**fields)


def _product_tags(node):
    if isinstance(node, ast_nodes.Product):
        return _product_tags(node.left) + _product_tags(node.right)
    if isinstance(node, ast_nodes.ProductChain):
        out = []
        for element in node.elements:
            out += _product_tags(element)
        return out
    if isinstance(node, ast_nodes.Tagged):
        return [(node.tag, node.type)]
    return []


_SUM_NODES = (ast_nodes.Sum, ast_nodes.SumChain)
_EXP_NODES = (ast_nodes.Exponent, ast_nodes.ExponentChain)
_CONTAINERS = ("ListLiteral", "SetLiteral", "DictLiteral")


def _flatten(node, binary, chain):
    if isinstance(node, binary):
        if isinstance(node, ast_nodes.Exponent):
            return _flatten(node.result, binary, chain) + [node.argument]
        return _flatten(node.left, binary, chain) + _flatten(node.right, binary, chain)
    if isinstance(node, chain):
        return list(node.elements)
    return [node]


def _value(node, module):
    """The Python value this piece writes, or None when it writes none."""
    if isinstance(node, ast_nodes.Constant):
        return node.value
    if isinstance(node, ast_nodes.Tagged):
        return types.SimpleNamespace(**{node.tag[1:]: _value(node.type, module)})
    if isinstance(node, (ast_nodes.Product, ast_nodes.ProductChain)):
        fields = {tag[1:]: _value(body, module) for tag, body in _product_tags(node)}
        return types.SimpleNamespace(**fields)
    if isinstance(node, ast_nodes.Tuple):
        return tuple(_value(e, module) for e in node.elements)
    if isinstance(node, _SUM_NODES):
        return _sum_value(node, module)
    if isinstance(node, _EXP_NODES):
        elements = _flatten(node, ast_nodes.Exponent, ast_nodes.ExponentChain)
        return types.SimpleNamespace(**{
            f"field_{index}": _value(element, module)
            for index, element in enumerate(elements)})
    if isinstance(node, ast_nodes.TypeApp):
        if node.constructor in _CONTAINERS:
            return _container(node, module)
        applied = _applied_generic(node, module)
        if applied is not None:
            body, home = applied
            return _value(body, home)
    return None


def _sum_value(node, module):
    """A sum value: the branches that carry a value, merged by tag."""
    fields = {}
    for element in _flatten(node, ast_nodes.Sum, ast_nodes.SumChain):
        if isinstance(element, ast_nodes.Tagged):
            part = _value(element.type, module)
            if part is not None:
                fields[element.tag[1:]] = part
            continue
        part = _value(element, module)
        if isinstance(part, types.SimpleNamespace):
            fields.update(vars(part))
    return types.SimpleNamespace(**fields)


def _applied_generic(node, module):
    """(body, home module) of a generic application with its actuals bound:
    `Result[5]` is Result's body with T := 5, which is what its branches
    then carry. None when the name is not a generic or the arity differs."""
    resolved = module_get_type(module, node.constructor)
    if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
        return None
    definition = resolved.ok_value.ast_node
    if not isinstance(definition, ast_nodes.GenericDefinition):
        return None
    params = list(definition.generic_params or [])
    if len(params) != len(node.args):
        return None
    return (_fill_params(definition.body, dict(zip(params, node.args))),
            resolved.ok_value.container_module)


class _ParamFiller(viba_ast.NodeTransformer):
    def __init__(self, bindings):
        self.bindings = bindings

    def visit_TypeRef(self, node):
        return self.bindings.get(node.name, node)


def _fill_params(body, bindings):
    return _ParamFiller(bindings).visit(copy.deepcopy(body))


def _container(node, module):
    if node.constructor == "ListLiteral":
        return [_value(a, module) for a in node.args]
    if node.constructor == "SetLiteral":
        return set(_value(a, module) for a in node.args)
    pairs = []
    for pair in node.args:
        key, value = pair.elements
        pairs.append((_value(key, module), _value(value, module)))
    return dict(pairs)


def _is_predicate(node):
    return isinstance(node, ast_nodes.TypeApp) and node.constructor == "Predicate"

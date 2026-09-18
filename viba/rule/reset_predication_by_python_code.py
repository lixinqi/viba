"""Run a witness's Predicate code and swap a false predication for the
poison.

reset_predication_by_python_code(witness) -> witness. Every Predicate
node in the witness gets its compiled $python_code executed against a
`self` built from the witness's tagged fields (viba-rule.md section 4: tag
minus $ becomes the attribute, so the metric definition's own $value slot
is what the predicate reads as .value, a nested product keeps expanding as
attributes, a tuple by index). A predicate that returns false is replaced
by PredicationFailed[nil, str].
"""

import types

from viba.viba_ast import nodes as ast_nodes
from viba.type import AstNodeType


def reset_predication_by_python_code(witness: AstNodeType) -> AstNodeType:
    self_obj = _self_object(witness.ast_node)
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


def _self_object(node):
    while isinstance(node, ast_nodes.Tagged):
        node = node.type
    fields = {}
    for tag, body in _product_tags(node):
        if not _is_predicate(body):
            fields[tag[1:]] = _value(body)
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


def _value(node):
    if isinstance(node, ast_nodes.Constant):
        return node.value
    if isinstance(node, ast_nodes.Tagged):
        return types.SimpleNamespace(**{node.tag[1:]: _value(node.type)})
    if isinstance(node, (ast_nodes.Product, ast_nodes.ProductChain)):
        fields = {tag[1:]: _value(body) for tag, body in _product_tags(node)}
        return types.SimpleNamespace(**fields)
    if isinstance(node, ast_nodes.Tuple):
        return tuple(_value(e) for e in node.elements)
    if isinstance(node, ast_nodes.TypeApp) and node.constructor in _CONTAINERS:
        return _container(node)
    return None


_CONTAINERS = ("ListLiteral", "SetLiteral", "DictLiteral")


def _container(node):
    if node.constructor == "ListLiteral":
        return [_value(a) for a in node.args]
    if node.constructor == "SetLiteral":
        return set(_value(a) for a in node.args)
    pairs = []
    for pair in node.args:
        key, value = pair.elements
        pairs.append((_value(key), _value(value)))
    return dict(pairs)


def _is_predicate(node):
    return isinstance(node, ast_nodes.TypeApp) and node.constructor == "Predicate"

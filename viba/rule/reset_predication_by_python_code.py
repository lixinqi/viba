"""Run a witness's Predicate code and swap a false predication for the
poison.

reset_predication_by_python_code(witness, rule) -> witness. Every Predicate
node in the witness gets its compiled $python_code executed, and a
predicate that returns false is replaced by PredicationFailed[nil, str].

`self` is the reflect node: `viba.reflect.VibaNode` over the rule's design
and the witness's material, so the code reads the material the way every
other reader does and no per-rule Python object is built anywhere. The
addresses are the protocol's (viba-reflect.md): a tag is `get_<tag>()`, a
position is `get_field_<i>()`, and the value a piece carries is `.value` -
the measurement of a metric object is
`self.get_distance().get_call_instance().get_field_0().get_ok().value`.

The predicate code therefore knows addresses, not shapes: the same body
works for any witness of that rule, and a rule whose fields move is
followed by changing the addresses in its own text.
"""

from viba import viba_ast
from viba.reflect import VibaAccess, VibaNode, language_config
from viba.type import AstNodeType
from viba.viba_type_descriptor import descriptor_of

access = VibaAccess(language_config)


def reset_predication_by_python_code(witness: AstNodeType, rule: AstNodeType) -> AstNodeType:
    self_node = VibaNode(access, descriptor_of(rule), witness.ast_node,
                         data_module=witness.container_module)
    node = _reset(witness.ast_node, self_node)
    return AstNodeType(node, witness.container_module)


def _reset(node, self_node):
    if isinstance(node, viba_ast.Product):
        return viba_ast.Product(_reset(node.left, self_node), _reset(node.right, self_node))
    if isinstance(node, viba_ast.ProductChain):
        return viba_ast.ProductChain([_reset(e, self_node) for e in node.elements])
    if isinstance(node, viba_ast.Tagged):
        return viba_ast.Tagged(node.tag, _reset(node.type, self_node))
    if isinstance(node, viba_ast.Sum):
        return viba_ast.Sum(_reset(node.left, self_node), _reset(node.right, self_node))
    if isinstance(node, viba_ast.SumChain):
        return viba_ast.SumChain([_reset(e, self_node) for e in node.elements])
    if isinstance(node, viba_ast.Tuple):
        return viba_ast.Tuple([_reset(e, self_node) for e in node.elements])
    if _is_predicate(node):
        return _predicate(node, self_node)
    return node


def _predicate(node, self_node):
    code = node.args[1].type.code
    namespace = {}
    exec(compile(code, "<predicate>", "exec"), namespace)
    if namespace["predicate"](self_node):
        return node
    return viba_ast.TypeApp("PredicationFailed", [viba_ast.Nil(), viba_ast.TypeRef("str")])


def _is_predicate(node):
    return isinstance(node, viba_ast.TypeApp) and node.constructor == "Predicate"

"""viba.serialize — write a material out as viba source (the mirror of reflect).

    from viba.reflect import access, VibaData
    from viba import serialize

    root = access.root(definition, VibaData(material))
    serialize.serialize("entry", access, root.ok_value)
    # -> Ok('entry :=\n  Object\n  * $left 1\n  * $right 2\n')

Every value is asked for through the protocol's cells — the same ones a reader
uses — so this knows nothing about a design beyond its shape, and nothing about
how Data is bound. What comes out is canonical viba source (it goes through
viba.builder), and `builder.check` reads it back before it is handed over.

A piece the design has no spelling for gives Err rather than a wrong spelling:
that is a material the design cannot carry, and the caller decides what to do
about it.
"""

from __future__ import annotations

from viba import builder
from viba.reflect import VibaAccess, VibaNode, at_index, at_key
from viba.type import Err, Ok, Result
from viba.viba_type_descriptor import (
    CODE_BLOCK,
    EXPONENT,
    NEVER,
    NIL,
    PRODUCT,
    SUM,
    TAGGED,
    TUPLE,
    TYPE_REF,
)

__all__ = ["serialize", "SerializeGap"]


class SerializeGap(Exception):
    """No viba spelling for this part of the design."""


# Written names come out of a builder nobody defines anything on: _NAMES.Object
# is the name `Object`, _NAMES.Object[...] an application of it.
_NAMES = builder.Builder()

def serialize(name: str, access: VibaAccess, node: VibaNode) -> Result:
    """A piece of the design in hand (a VibaNode) -> viba source: the definition
    named `name` whose body is that piece's spelling."""
    try:
        expression = _emit(access, node)
        vb = builder.Builder()
        setattr(vb, name, expression)
        builder.check(vb)               # read the written source back
    except SerializeGap as gap:
        return Err(str(gap))
    except (TypeError, ValueError) as error:
        return Err(str(error))
    return Ok(str(vb))


def _emit(access: VibaAccess, node: VibaNode):
    """This part of the material, as a builder expression.

    The value side is asked with the protocol's own cells, never by looking at
    what the binding put in the node: VibaLeaf answers "the value is itself this
    piece" (a literal, a unit) and the rest is walked one step at a time.
    """
    given = access.leaf(node)
    if isinstance(given, Ok):
        return _emit_leaf(given.ok_value)
    shape = access.unfold(node.descriptor)
    if shape.kind == SUM:
        return _emit_sum(access, node)
    if shape.kind == PRODUCT:
        return _emit_product(access, node, shape)
    if shape.kind == TUPLE:
        return _emit_tuple(access, node)
    container = access.container_kind(shape)
    if container is not None:
        return _emit_container(access, node, container, shape)
    if shape.kind == TAGGED:
        return _emit_tagged(access, node)
    if shape.kind == EXPONENT:
        return _emit_exponent(access, node, shape)
    if shape.kind == CODE_BLOCK:
        return _emit_code_block(shape)
    if shape.kind == NIL:
        return None
    if shape.kind == NEVER:
        raise SerializeGap("nothing resides in never")
    raise SerializeGap(f"cannot write this piece out: {shape.kind}")


def _emit_sum(access: VibaAccess, node: VibaNode):
    """Sum: write the selected branch; a branch reached through a name keeps
    that name's own spelling, so nothing here names the branch."""
    for tag, step, _ in access.member_steps(node):
        given = access.get(node, step)
        if not isinstance(given, Ok) or given.ok_value is None:
            continue
        inner = _emit(access, given.ok_value)
        return inner if tag is None else _tag(tag)(inner)
    raise SerializeGap(f"no branch of this sum carries a value: {node!r}")


def _unit_expression(descriptor):
    """The product identity as the language writes it: `Object`.

    A design may head a product with a word of its own layer. Writing the
    *design's* word would carry that layer's vocabulary into every material this
    writes, so the language's own unit goes out instead: `Object` is the product
    identity and a builtin name.
    """
    return _NAMES.Object


def _emit_product(access: VibaAccess, node: VibaNode, shape):
    """Product: the leading unit, then every member in order.

    A member the design writes as a unit is written as the unit — under its
    tag when the design gave it one; a member with a value is written from it
    (tagged members under their tag); a member with no value is written `nil`
    when its written type admits nil, and is a gap otherwise. Positional
    members are addressed by position, tagged ones by tag.
    """
    written = []
    elements = list(shape.payload.elements)
    if elements and access._is_unit_descriptor(elements[0]):
        written.append(_unit_expression(elements[0]))
    for tag, step, descriptor in access.member_steps(node):
        if access._is_unit_descriptor(descriptor) and descriptor.kind != NEVER:
            unit = None                         # the design itself writes a unit
            written.append(unit if tag is None else _tag(tag)(unit))
            continue
        given = access.get(node, step)
        if isinstance(given, Ok) and given.ok_value is not None:
            inner = _emit(access, given.ok_value)
        elif _may_be_nil(access, step, node):
            inner = None
        else:
            raise SerializeGap(f"no value here: {tag or step}")
        written.append(inner if tag is None else _tag(tag)(inner))
    product = written[0]
    for element in written[1:]:
        product = product * element
    return product


def _emit_tuple(access: VibaAccess, node: VibaNode):
    """Tuple: a product by position, written as the tuple it is."""
    length = access.length(node)
    if isinstance(length, Err):
        raise SerializeGap(length.err_msg)
    return tuple(_emit(access, _child(access, node, at_index(index)))
                 for index in range(length.ok_value))


def _emit_tagged(access: VibaAccess, node: VibaNode):
    """One written tag whose body is the design's: the material keeps the tag
    (that is how a reader finds `$value` under a metric)."""
    for tag, step, _ in access.member_steps(node):
        return _tag(tag)(_emit(access, _child(access, node, step)))
    raise SerializeGap(f"this tagged piece has no member: {node!r}")


def _emit_code_block(shape):
    """`{ ... }`: opaque, and the material's own text is not reachable through
    the protocol — no cell hands it over. What cannot be read is not invented:
    the unit goes out in its place."""
    return None


def _emit_exponent(access: VibaAccess, node: VibaNode, shape):
    """Exponent chain: the result, then every argument under its own tag.

    Written the way the design writes it — never <- $not_operand (...) is just
    one such chain — with whatever the material has at each address filled in.
    """
    elements = list(shape.payload.elements)
    head = elements[0]
    steps = access.member_steps(node)
    if access._is_unit_descriptor(head):
        chain = _NAMES.never if head.kind == NEVER else _NAMES.nil
    else:
        if not steps:
            raise SerializeGap(f"no value for the result of {shape.kind}")
        _, step, _ = steps[0]
        chain = _emit(access, _child(access, node, step))
        steps = steps[1:]
    for tag, step, _ in steps:
        given = access.get(node, step)
        if isinstance(given, Ok) and given.ok_value is not None:
            inner = _emit(access, given.ok_value)
        elif _may_be_nil(access, step, node):
            inner = None
        else:
            raise SerializeGap(f"no value here: {tag or step}")
        chain = chain ** (inner if tag is None else _tag(tag)(inner))
    return chain


def _may_be_nil(access: VibaAccess, step, node) -> bool:
    """May this slot's written type be nil (a sum with a unit branch)?"""
    target = access.target_of(node, step, access.members(node))
    return target is not None and access.carries_nil(target)


def _emit_container(access: VibaAccess, node: VibaNode, container: str, shape):
    """Container: a literal (ListLiteral / SetLiteral / DictLiteral).

    The members are written in the order the material has them, not sorted: a
    set has no order of its own, but both sides walk it by position, so the
    written order is the order that was read.
    """
    if container == "dict":
        keys = access.keys(node)
        key_type = shape.payload.args[0] if shape.payload.args else None
        if not (key_type is not None and key_type.kind == TYPE_REF
                and key_type.payload.type_name == "str"):
            raise SerializeGap("the protocol hands dict keys over as strings; "
                               "this one is written with another key type")
        if isinstance(keys, Err):
            raise SerializeGap(keys.err_msg)
        pairs = tuple((key, _emit(access, _child(access, node, at_key(key))))
                      for key in keys.ok_value)
        return _NAMES.DictLiteral[pairs]
    length = access.length(node)
    if isinstance(length, Err):
        raise SerializeGap(length.err_msg)
    payloads = tuple(_emit(access, _child(access, node, at_index(index)))
                     for index in range(length.ok_value))
    if container == "list":
        return _NAMES.ListLiteral[payloads]
    if container == "set":
        return _NAMES.SetLiteral[payloads]
    raise SerializeGap(f"cannot write this container out: {container}")


def _emit_leaf(value):
    """A leaf the language can write: nil, or one of the four literals."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    raise SerializeGap(f"cannot write this leaf out: {type(value).__name__}")


def _tag(name: str):
    """The builder's spelling of one written tag ($agent_name -> tag.agent_name)."""
    return getattr(builder.tag, name.lstrip("$"))


def _child(access: VibaAccess, node: VibaNode, step):
    given = access.get(node, step)
    if not isinstance(given, Ok) or given.ok_value is None:
        raise SerializeGap(f"this step has no value: {step}")
    return given.ok_value

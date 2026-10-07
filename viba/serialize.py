"""viba.serialize — write a viba data out as viba source (the mirror of reflect).

    from viba.reflect import access, VibaData
    from viba import serialize

    root = access.root(definition, VibaData(viba_data))
    serialize.serialize("entry", root.ok_value)
    # -> Ok('entry =\n  Object\n  * $left 1\n  * $right 2\n')

Every value is asked for through the protocol's cells — the same ones a reader
uses — so this knows nothing about a design beyond what is written, and nothing about
how Data is bound. What comes out is canonical viba source (it goes through
viba.builder), and `builder.check` reads it back before it is handed over.

A piece the design has no spelling for gives VibaProgramErr rather than a wrong spelling:
that is a viba data the design cannot carry, and the caller decides what to do
about it.
"""

from __future__ import annotations

from viba import builder
from viba import viba_ast
from viba.reflect import LITERAL_KINDS, VibaAccess, VObject, at_index, at_key
from viba.type import VibaProgramErr, Ok, Result
from viba.viba_type_descriptor import (
    CODE_BLOCK,
    EXPONENT,
    NEVER,
    NIL,
    PRODUCT,
    SUM,
    TAGGED,
    TUPLE,
    TYPE_APP,
    TYPE_REF,
)

__all__ = ["serialize", "SerializeGap"]


class SerializeGap(Exception):
    """No viba spelling for this part of the design."""


# Written names come out of a builder nobody defines anything on: _NAMES.Object
# is the name `Object`, _NAMES.Object[...] an application of it.
_NAMES = builder.Builder()

def serialize(name: str, node: VObject) -> Result:
    """A piece of the design in hand (a VObject) -> viba source: the definition
    named `name` whose body is that piece's spelling.

    The node carries the accessor it was read through, so this reads the piece
    exactly as the caller did, and asks for no accessor of its own.
    """
    access = node._access
    try:
        expression = _emit(access, node)
        vb = builder.Builder()
        setattr(vb, name, expression)
        builder.check(vb)               # read the written source back
    except SerializeGap as gap:
        return VibaProgramErr(str(gap))
    except (TypeError, ValueError) as error:
        return VibaProgramErr(str(error))
    return Ok(str(vb))


def _call_parts(node):
    """A written call as (head, arguments in written order)."""
    written = []
    while isinstance(node, viba_ast.Partial):
        written.append(node.argument)
        node = node.function
    return node, list(reversed(written))


def _bare(piece, access: VibaAccess, written_in=None) -> VObject:
    """One written piece as a node of its own: a closure keeps its arguments as
    the viba data they already are, read in the module the call was written in
    (`written_in`) — a name among them is a name there, not nowhere."""
    from viba.type import AstNodeType
    from viba.viba_type_descriptor import descriptor_of
    return VObject(access, descriptor_of(AstNodeType(piece, written_in)), piece)


def _written_in(node):
    """The module the piece at this node was written in, or None when the node
    does not say — the module its descriptor resolved its type in."""
    written = getattr(node.descriptor, "resolvable_type", None)
    return getattr(written, "container_module", None)


def _emit_closure(access: VibaAccess, node: VObject):
    """A closure: a call whose environment has not been given.

    The design reads such a node as the type it would have when executed, which
    is not what the node holds — the node is the written chain itself. So the
    spelling comes from that chain: the name at the head, then every argument
    that is already computed.
    """
    head, arguments = _call_parts(node.data)
    if not isinstance(head, viba_ast.TypeRef):
        # A chain headed by `$tag` is the same call, but the tag is no name to
        # write back through a builder; the value layer keeps such a chain as a
        # call in progress, not as viba data.
        raise SerializeGap("a chain headed by a member has no written form here")
    expression = builder.name(head.name)
    written_in = _written_in(node)
    for argument in arguments:
        piece = _emit(access, _bare(argument, access, written_in))
        if not isinstance(argument, viba_ast.Tagged):
            # An argument that names no field is written as a group, so that it
            # stays one argument: a call inside it does not run on into the chain.
            piece = builder.tag(piece)
        expression = expression << piece
    return expression


def _emit(access: VibaAccess, node: VObject):
    data = getattr(node, "data", None)
    if isinstance(data, viba_ast.Partial):
        return _emit_closure(access, node)
    """This part of the viba data, as a builder expression.

    The value side is asked with the protocol's own cells, never by looking at
    what the binding put in the node: VibaLeaf answers "the value is itself this
    piece" (a literal, a unit) and the rest is walked one step at a time.

    The written type is asked first, for two things only: a piece the design
    calls `never` — however many names that takes — has no resident, and a
    product whose inline chain comes back to where it started has no reading.
    Either way a viba data that carries something there is not a viba data of
    this design.
    """
    unfolded = access.unfold(node.descriptor)
    if isinstance(data, viba_ast.TypeRef) and unfolded.kind == EXPONENT:
        # A name the design calls a function — a call that wants its arguments
        # still. Written as the name, it reads back as that same call; written as
        # the type it would have when executed, it would not.
        return _emit_closure(access, node)
    if unfolded.kind == NEVER:
        raise SerializeGap("nothing resides in never")
    if unfolded.kind == PRODUCT:
        # An untagged member is an inline slot, so the chain has to bottom out:
        # a product whose chain comes back to where it started has no reading,
        # and what was written for it would mean something else when read back.
        cycle = access.inline_cycle(unfolded)
        if cycle is not None:
            raise SerializeGap(f"the inline chain comes back to {cycle!r}")
    given = access.leaf(node)
    if isinstance(given, Ok):
        return _emit_leaf(given.ok_value)
    if unfolded.kind == SUM:
        return _emit_sum(access, node)
    if unfolded.kind == PRODUCT:
        return _emit_product(access, node, unfolded)
    if unfolded.kind == TUPLE:
        return _emit_tuple(access, node)
    container = access.container_kind(unfolded)
    if container is not None:
        return _emit_container(access, node, container, unfolded)
    if unfolded.kind == TAGGED:
        return _emit_tagged(access, node)
    if unfolded.kind == EXPONENT:
        return _emit_exponent(access, node, unfolded)
    if unfolded.kind == CODE_BLOCK:
        return _emit_code_block(unfolded)
    if unfolded.kind == NIL:
        return None
    raise SerializeGap(f"cannot write this piece out: {unfolded.kind}")


def _emit_sum(access: VibaAccess, node: VObject):
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
    *design's* word would carry that layer's vocabulary into every viba_data this
    writes, so the language's own unit goes out instead: `Object` is the product
    identity and a builtin name.
    """
    return _NAMES.Object


def _emit_product(access: VibaAccess, node: VObject, unfolded):
    """Product: the leading unit (when the design wrote one), then every member
    in order.

    A product the design heads with no unit is written as the design writes it;
    when it does write one, the language's own unit leads (`Object`), so a
    product under a block is a product a reader can read back.

    The members come from the map, so an untagged member that stands for a
    product has already handed its own members over and units are gone. A
    member the design writes as a unit is written as the unit — under its tag
    when the design gave it one; a member with a value is written from it
    (tagged members under their tag); a member with no value is written `nil`
    when its written type admits nil, and is a gap otherwise. Positional members
    are addressed by position, tagged ones by tag; the same tag twice is a gap,
    since the two pieces could not be told apart when read back.
    """
    written = []
    elements = list(unfolded.payload.elements)
    if elements and access._is_product_unit(elements[0]):
        written.append(_unit_expression(elements[0]))
    seen_tags = set()
    for tag, step, descriptor in access.member_steps(node):
        # A tag holds one member: the same tag twice is a design this cannot
        # write, since the piece each occurrence carries could not be told
        # apart when read back.
        if tag is not None:
            if tag in seen_tags:
                raise SerializeGap(f"the tag {tag} is written twice in one product")
            seen_tags.add(tag)
        # The design itself writes a unit here — unless the name it writes
        # lands on `never`, which has no resident for a viba data to hold.
        if (access._is_unit_descriptor(descriptor)
                and access.unfold(descriptor).kind != NEVER):
            unit = None
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
    # `builder.literal` leaves an expression it is given alone and wraps a bare
    # value, so `*` here is always the builder's product: `1 * 2` is a product
    # of two members, not Python's two.
    product = builder.literal(written[0])
    for element in written[1:]:
        product = product * builder.literal(element)
    return product


def _emit_tuple(access: VibaAccess, node: VObject):
    """Tuple: a product by position, written as the tuple it is."""
    length = access.length(node)
    if isinstance(length, VibaProgramErr):
        raise SerializeGap(length.msg)
    return tuple(_emit(access, _child(access, node, at_index(index)))
                 for index in range(length.ok_value))


def _emit_tagged(access: VibaAccess, node: VObject):
    """One written tag whose body is the design's: the viba data keeps the tag
    (that is how a reader finds `$value` under a metric)."""
    for tag, step, _ in access.member_steps(node):
        return _tag(tag)(_emit(access, _child(access, node, step)))
    raise SerializeGap(f"this tagged piece has no member: {node!r}")


def _emit_code_block(unfolded):
    """`{ ... }`: opaque, and the viba data's own text is not reachable through
    the protocol — no cell hands it over. What cannot be read is not invented:
    the unit goes out in its place."""
    return None


def _emit_exponent(access: VibaAccess, node: VObject, unfolded):
    """Exponent chain: the result, then every argument under its own tag.

    Written the way the design writes it — never <- $not_operand (...) is just
    one such chain — with whatever the viba data has at each address filled in.
    """
    elements = list(unfolded.payload.elements)
    head = elements[0]
    steps = access.member_steps(node)
    if access._is_unit_descriptor(head):
        chain = (_NAMES.never if access.unfold(head).kind == NEVER
                 else _NAMES.nil)
    else:
        if not steps:
            raise SerializeGap(f"no value for the result of {unfolded.kind}")
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
        chain = chain ** _argument(inner, tag)
    return chain


def _argument(inner, tag):
    """One argument of a chain: under its tag when the design gave it one.

    `**` takes a tagged field or a group, so an argument the design left
    untagged goes out as the group keeping it whole: `<result> <- body`, which
    is what the group writes — it is not a node, only the operand as it is.
    """
    return builder.tag(inner) if tag is None else _tag(tag)(inner)


def _may_be_nil(access: VibaAccess, step, node) -> bool:
    """May this slot's written type be nil (a sum with a unit branch)?"""
    target = access.target_of(node, step, access.members(node))
    return target is not None and access.carries_nil(target)


def _emit_container(access: VibaAccess, node: VObject, container: str, unfolded):
    """Container: a literal (ListLiteral / SetLiteral / DictLiteral).

    The members are written in the order the viba data has them, not sorted: a
    set has no order of its own, but both sides walk it by position, so the
    written order is the order that was read.
    """
    if container == "dict":
        keys = access.keys(node)
        if isinstance(keys, VibaProgramErr):
            raise SerializeGap(keys.msg)
        if _is_literal(unfolded):
            # The design is the literal itself, so there is no key type to ask:
            # the keys are in the piece, and every one of them has to be the
            # string the protocol hands keys over as.
            if not _literal_keys_are_strings(node):
                raise SerializeGap("the protocol hands dict keys over as strings; "
                                   "this one is written with another key type")
        else:
            # The written key type, through however many names: `S = str` and
            # `G[V] = str` are the str the protocol hands keys over as.
            key_type = (access.unfold(unfolded.payload.args[0])
                        if unfolded.payload.args else None)
            if not (key_type is not None and key_type.kind == TYPE_REF
                    and key_type.payload.type_name == "str"):
                raise SerializeGap("the protocol hands dict keys over as strings; "
                                   "this one is written with another key type")
        pairs = tuple((key, _emit(access, _child(access, node, at_key(key))))
                      for key in keys.ok_value)
        return _NAMES.DictLiteral[pairs]
    length = access.length(node)
    if isinstance(length, VibaProgramErr):
        raise SerializeGap(length.msg)
    payloads = tuple(_emit(access, _child(access, node, at_index(index)))
                     for index in range(length.ok_value))
    if container == "list":
        return _NAMES.ListLiteral[payloads]
    if container == "set":
        return _NAMES.SetLiteral[payloads]
    raise SerializeGap(f"cannot write this container out: {container}")


def _is_literal(unfolded) -> bool:
    """Whether this design piece is one of the three literal constructors."""
    return (unfolded.kind == TYPE_APP
            and unfolded.payload.constructor_name in LITERAL_KINDS)


def _literal_keys_are_strings(node) -> bool:
    """Whether every key a dict literal writes is a string.

    `True` only when the piece is written as the literal and each pair's key is a
    string constant: a key of another type has no reading the protocol can hand
    back, and writing it as the string it was handed over as would be a wrong
    spelling rather than a missing one.
    """
    data = node.data
    if not (isinstance(data, viba_ast.TypeApp) and data.constructor == "DictLiteral"):
        return False
    for pair in data.args:
        elements = pair.elements if isinstance(pair, viba_ast.Tuple) else ()
        if not elements or not isinstance(elements[0], viba_ast.Constant):
            return False
        if not isinstance(elements[0].value, str):
            return False
    return True


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


def _child(access: VibaAccess, node: VObject, step):
    given = access.get(node, step)
    if not isinstance(given, Ok) or given.ok_value is None:
        raise SerializeGap(f"this step has no value: {step}")
    return given.ok_value

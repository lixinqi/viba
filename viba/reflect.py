"""viba.reflect — treat a design as a map and read a material through it.

This is the access side of ``viba-reflect.md``: the map (the descriptors) comes
from the descriptor side reading the design, the data (Data) is one material's
type expression, and access walks the map's addresses one step at a time. Names
follow the rule ``viba_type_descriptor.py`` keeps for
``viba_type_descriptor.viba``: the protocol's class names stay as they are, its
functions land as methods in snake_case, this layer's own helpers take an
underscore.

    protocol (viba-reflect.md)         Python
    ------------------------------    ------------------------------------
    VibaStep                          VibaStep
    the four VibaStep branches         by_tag / by_field_index / at_index / at_key
    VibaPath                          VibaPath (= list[VibaStep])
    VibaNode[Data]                    VibaNode
    VibaReflectConfig                 Config
    Result[T]                         Ok / Err (viba.type)
    VibaAccess[Data]                  VibaAccess, built with a Config
    VibaRoot / VibaHas / VibaGet      VibaAccess.root / has / get / leaf /
    VibaLeaf / VibaLength / VibaKeys  length / keys
    VibaResolve / VibaGetByPath       VibaAccess.resolve / get_by_path /
    VibaListFields                    list_fields

What the protocol does not name is this layer's binding or helping, not a protocol concept:

    the Data parameter    -> VibaData: one material's type expression
    section 5.4 "throw"   -> VibaReflectError
    reading the map       -> VibaAccess's underscore methods and this module's

Two rules from the protocol:

* ``root`` only pairs (descriptor, data). Versions are none of its business:
  which revision a material was built against is the caller's bookkeeping.
* ``has`` answers true or false only: no such address in the design and no
  such piece in the material are both ``False``. In ``get``, "the material has
  no such piece" is ``Ok(nil)`` and only "the map has no such address" is
  ``Err``; ``leaf`` / ``length`` / ``keys`` give ``Err`` when they cannot.

The config is the one thing an implementation is told and this module is not:
which written names stand for the units ``nil`` and ``never``. Every accessor
carries one, and this module's own ``access`` carries the language layer's
default.

The section 5.4 Python landing lives on VibaNode: ``get_{name}()`` / ``has_{name}()`` /
``in`` / ``try_get_{name}()`` / ``node[i]`` / ``node.value`` / ``len(node)`` /
iteration / ``keys()`` / ``values()`` / ``items()``. The taking ones throw; the asking
ones never throw.
"""

from __future__ import annotations

import copy
from typing import Iterable, List, Optional, Sequence

from viba import viba_ast
from viba.type import AstNodeType, Err, Ok, Result, module_get_type
from viba.viba_type_descriptor import (
    CODE_BLOCK,
    ELLIPSIS,
    EXPONENT,
    LITERAL,
    NEVER,
    NIL,
    PRODUCT,
    SUM,
    TAGGED,
    TUPLE,
    TYPE_APP,
    TYPE_REF,
    VibaChainDescriptor,
    VibaConstantValue,
    VibaDefinitionDescriptor,
    VibaMemberDescriptor,
    VibaPool,
    VibaTaggedDescriptor,
    VibaTupleDescriptor,
    VibaTypeAppDescriptor,
    VibaTypeDescriptor,
    definition_members,
    member_type_name,
)

# The three builtin containers, and the literal shapes implementations use.
SCALAR_NAMES = ("bool", "int", "float", "str")  # builtin leaves a sum can name
CONTAINERS = ("list", "set", "dict")
LITERAL_CTORS = ("ListLiteral", "SetLiteral", "DictLiteral")


class VibaReflectError(Exception):
    """Thrown when the taking path fails; it carries the protocol's ``Err`` text.

    Section 5.4 says "throw" without naming the exception; the name is this
    layer's.
    """


# ----------------------------------------------------------------------
# Binding the config (VibaReflectConfig)
# ----------------------------------------------------------------------


def _type_name(value) -> str:
    """A written name, however the caller wrote it: a string, or a type node."""
    if isinstance(value, str):
        return value
    for attribute in ("type_name", "name", "full_name"):
        name = getattr(value, attribute, None)
        if isinstance(name, str):
            return name
    return str(value)


class Config:
    """Which written names stand for the units ``never`` and ``nil``.

    ``never_eqv`` holds the names equivalent to ``never``, the sum's unit, and
    ``nil_eqv`` the names equivalent to ``nil``, the product's unit. The units
    themselves are units by kind, so naming them again changes nothing.
    Whoever builds an accessor fills these in.

    Either a name or a descriptor/type node may be given; only its written
    name is kept.
    """

    def __init__(self, never_eqv=(), nil_eqv=()):
        self.never_eqv = frozenset(_type_name(name) for name in never_eqv)
        self.nil_eqv = frozenset(_type_name(name) for name in nil_eqv)

    def __repr__(self):
        return f"Config(never_eqv={sorted(self.never_eqv)!r}, nil_eqv={sorted(self.nil_eqv)!r})"

    def is_unit_name(self, name: str) -> bool:
        return name in self.never_eqv or name in self.nil_eqv


# ----------------------------------------------------------------------
# VibaStep and VibaPath
# ----------------------------------------------------------------------


class VibaStep:
    """One step of an address: one of the four branches."""

    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"{self.kind}({self.value!r})"

    def __eq__(self, other):
        return isinstance(other, VibaStep) and (self.kind, self.value) == (other.kind, other.value)

    def __hash__(self):
        return hash((self.kind, self.value))


def by_tag(name: str) -> VibaStep:
    return VibaStep("by_tag", _tag_of(name))


def by_field_index(index: int) -> VibaStep:
    return VibaStep("by_field_index", index)


def at_index(index: int) -> VibaStep:
    return VibaStep("at_index", index)


def at_key(key: str) -> VibaStep:
    return VibaStep("at_key", key)


VibaPath = List[VibaStep]


def _tag_of(name: str) -> str:
    """Chained calls write names without the $; with it counts too."""
    return name if name.startswith("$") else "$" + name


# ----------------------------------------------------------------------
# Binding Data
# ----------------------------------------------------------------------


class VibaData:
    """This layer's binding of the protocol's Data parameter: one material's
    type expression.

    The protocol leaves Data to the implementation; on this side a material is
    written in the design's own language, so it lands as an AST node. Versions
    and the like are carried by the caller, not by the protocol.
    """

    __slots__ = ("node",)

    def __init__(self, node):
        if isinstance(node, AstNodeType):
            node = node.ast_node
        self.node = node


# ----------------------------------------------------------------------
# VibaNode[Data]
# ----------------------------------------------------------------------


class VibaNode:
    """One node: a piece of the design's descriptor plus a piece of the material.

    ``path`` is the VibaPath walked from the root (the protocol's node has no
    such field; this layer keeps it so an address can be stored, printed and
    compared).
    """

    __slots__ = ("_access", "descriptor", "data", "path")

    def __init__(self, access: "VibaAccess", descriptor: VibaTypeDescriptor, data,
                 path: Sequence[VibaStep] = ()):
        self._access = access
        self.descriptor = descriptor
        self.data = data
        self.path = tuple(path)

    def __repr__(self):
        where = ".".join(repr(step) for step in self.path) or "root"
        return f"VibaNode({where})"

    # ---- section 5.4 node accessors: the taking ones throw ----

    def by_tag(self, name: str) -> "VibaNode":
        return self._access._unwrap(self._access.get(self, by_tag(name)))

    def by_field_index(self, index: int) -> "VibaNode":
        return self._access._unwrap(self._access.get(self, by_field_index(index)))

    def at_index(self, index: int) -> "VibaNode":
        return self._access._unwrap(self._access.get(self, at_index(index)))

    def at_key(self, key: str) -> "VibaNode":
        return self._access._unwrap(self._access.get(self, at_key(key)))

    @property
    def leaf(self) -> VibaConstantValue:
        return self._access._unwrap(self._access.leaf(self))

    @property
    def value(self):
        """The section 5.4 Python landing: ``leaf`` lands on ``node.value``, the
        value the literal record carries (bool / int / float / str / None)."""
        return self.leaf.value

    # ---- shapes: the written chain head only, no unfolding ----

    @property
    def is_list(self) -> bool:
        return self._access._container_kind(self.descriptor) == "list"

    @property
    def is_set(self) -> bool:
        return self._access._container_kind(self.descriptor) == "set"

    @property
    def is_dict(self) -> bool:
        return self._access._container_kind(self.descriptor) == "dict"

    def __len__(self) -> int:
        return self._access._unwrap(self._access.length(self))

    def keys(self) -> List[str]:
        return list(self._access._unwrap(self._access.keys(self)))

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, key):
        if isinstance(key, int) and not isinstance(key, bool):
            return self._access._unwrap(self._access.get(self, at_index(key)))
        return self._access._unwrap(self._access.get(self, at_key(str(key))))

    # ---- magic methods of the Python landing ----

    def _has_tag(self, name: str) -> bool:
        given = self._access.has(self, by_tag(name))
        return bool(given.value) if isinstance(given, Ok) else False

    def _has_field(self, index: int) -> bool:
        given = self._access.has(self, by_field_index(index))
        return bool(given.value) if isinstance(given, Ok) else False

    def __contains__(self, name: str) -> bool:
        return self._has_tag(name)

    def __getattr__(self, name: str):
        if name.startswith("get_field_"):
            return lambda: self.by_field_index(int(name[len("get_field_"):]))
        if name.startswith("has_field_"):
            return lambda: self._has_field(int(name[len("has_field_"):]))
        if name.startswith("try_get_field_"):
            return lambda: self._access.get(
                self, by_field_index(int(name[len("try_get_field_"):])))
        if name.startswith("get_"):
            return lambda: self.by_tag(name[len("get_"):])
        if name.startswith("has_"):
            return lambda: self._has_tag(name[len("has_"):])
        if name.startswith("try_get_"):
            return lambda: self._access.get(self, by_tag(name[len("try_get_"):]))
        raise AttributeError(name)

    def __dir__(self):
        names = ["value", "leaf", "is_list", "is_set", "is_dict",
                 "keys", "values", "items", "path", "descriptor", "data"]
        positional = 0
        for tag, _ in self._access._members(self) or []:
            if tag:
                short = tag[1:]
                names += [f"get_{short}", f"has_{short}", f"try_get_{short}"]
            else:
                names += [f"get_field_{positional}", f"has_field_{positional}",
                          f"try_get_field_{positional}"]
                positional += 1
        return sorted(names)


# ----------------------------------------------------------------------
# VibaAccess[Data]
# ----------------------------------------------------------------------


class VibaAccess:
    """An accessor: section 5.3's six cells plus this layer's map-reading
    helpers, all sharing one VibaReflectConfig.

    The definition is handed to VibaRoot, not to the constructor: the same
    accessor reads any design, and the config is what an implementation is
    told once.
    """

    def __init__(self, config: Config):
        self.config = config

    def __repr__(self):
        return f"VibaAccess({self.config!r})"

    # ---- the six protocol cells ----

    def root(self, definition: VibaDefinitionDescriptor, data: VibaData) -> Result:
        """VibaRoot: hand out the (descriptor, data) pair."""
        return Ok(VibaNode(self, definition.body, data.node))

    def has(self, node: VibaNode, step: VibaStep) -> Result:
        """VibaHas: is this step there. No such address in the design and no
        such piece in the material are both false."""
        if not self._knows(node, step, self._members(node)):
            return Ok(False)
        return Ok(self._value_at(node.data, step, node.descriptor) is not None)

    def get(self, node: VibaNode, step: VibaStep) -> Result:
        """VibaGet: take one step. A piece the material lacks is Ok(nil); an
        address the map lacks is Err."""
        slots = self._members(node)
        if not self._knows(node, step, slots):
            return Err(f"the design has no such address: {step} ({node!r})")
        descriptor = self._target_of(node, step, slots)
        if descriptor is None:
            return Err(f"the design has no such address: {step} ({node!r})")
        piece = self._value_at(node.data, step, node.descriptor)
        if piece is None:
            return Ok(None)
        return Ok(VibaNode(self, descriptor, piece, tuple(node.path) + (step,)))

    def leaf(self, node: VibaNode) -> Result:
        data = node.data
        if isinstance(data, viba_ast.Constant):
            return Ok(_constant_value(data.value))
        if isinstance(data, viba_ast.Nil):
            return Ok(VibaConstantValue("nil", None))
        return Err(f"this piece is not a leaf: {node!r}")

    def length(self, node: VibaNode) -> Result:
        elements = self._elements_of(node.data)
        if elements is None:
            return Err(f"this piece is not a container: {node!r}")
        return Ok(len(elements))

    def keys(self, node: VibaNode) -> Result:
        if not isinstance(node.data, viba_ast.TypeApp) or node.data.constructor != "DictLiteral":
            return Err(f"this piece is not a dict: {node!r}")
        out = []
        for pair in node.data.args:
            key = self._key_of(pair)
            if key is None:
                return Err(f"this dict key is not a literal: {node!r}")
            out.append(_key_text(key))
        return Ok(out)

    # ---- the section 5.2 convenience functions ----

    def resolve(self, node: VibaNode, path: Sequence[VibaStep]) -> Result:
        """VibaResolve: walk the path with VibaGet, one step at a time.

        Walking onto "the material has no such piece" (``Ok(nil)``) is as far
        as it goes: with steps left that is an Err; with no steps left, the
        ``Ok(nil)`` is handed out.
        """
        current = node
        for step in path:
            if current is None:
                return Err("this step has no value")
            given = self.get(current, step)
            if isinstance(given, Err):
                return given
            current = given.value
        return Ok(current)

    def get_by_path(self, node: VibaNode, path: Sequence[VibaStep]) -> Result:
        """VibaGetByPath: VibaResolve first, then VibaLeaf."""
        resolved = self.resolve(node, path)
        if isinstance(resolved, Err):
            return resolved
        if resolved.value is None:
            return Err("this step has no value")
        return self.leaf(resolved.value)

    def list_fields(self, node: VibaNode, definition: VibaDefinitionDescriptor) -> Result:
        """VibaListFields: VibaGet each DefinitionMember; take what comes back."""
        members = definition_members(definition)
        if isinstance(members, Err):
            return members
        out = []
        positional = 0
        for member in members.value:
            if member.tag:
                step = by_tag(member.tag)
            else:
                # A unit the config names may be written as a positional member;
                # it is not a field, takes no positional number, and is skipped.
                if self._is_unit_member(member):
                    continue
                step = by_field_index(positional)
                positional += 1
            given = self.get(node, step)
            # Take what comes back; what is missing (including pieces that are
            # neither product nor sum) does not enter the table.
            if isinstance(given, Ok) and given.value is not None:
                out.append(given.value)
        return Ok(out)

    # ---- private: the taking path (Err and Ok(nil) both throw) ----

    def _unwrap(self, given: Result):
        if isinstance(given, Err):
            raise VibaReflectError(given.message)
        if given.value is None:
            raise VibaReflectError("this piece has no value")
        return given.value

    # ---- private: the map (descriptors read as addressable shapes) ----

    def _unfold(self, descriptor: VibaTypeDescriptor,
               seen: Optional[set] = None) -> VibaTypeDescriptor:
        """Unfold a piece written as a name into an addressable shape.

        Two things, both by the definitions in the pool: a name unfolds to its
        definition body, and a generic application folds its arguments in and
        lands on the body too. Sum, product and exponent follow one rule; there
        are no other exceptions.
        """
        seen = seen or set()
        if id(descriptor) in seen:
            return descriptor
        seen = seen | {id(descriptor)}
        if descriptor.kind == TYPE_REF:
            resolved = self._resolve_ref(descriptor)
            if resolved is None:
                return descriptor
            return self._unfold(resolved, seen)
        if descriptor.kind == TYPE_APP:
            applied = self._apply_generic(descriptor)
            if applied is not None:
                return self._unfold(applied, seen)
        return descriptor

    def _apply_generic(self, descriptor: VibaTypeDescriptor) -> Optional[VibaTypeDescriptor]:
        """When the constructor names a generic definition, hand back the body
        descriptor with the arguments filled in."""
        payload = descriptor.payload
        if payload.resolvable_type is None:
            return None
        resolved = module_get_type(payload.resolvable_type.container_module,
                                   payload.constructor_name)
        if isinstance(resolved, Err) or not isinstance(resolved.value, AstNodeType):
            return None
        target = resolved.value
        if not isinstance(target.ast_node, viba_ast.GenericDefinition):
            return None
        params = list(target.ast_node.generic_params or [])
        if len(params) != len(payload.args):
            return None
        # The descriptor side has no public "descriptor for a type expression"
        # entry point yet, so use its builder.
        # The builtin library is loaded as written (never canonicalized), so
        # canonicalize here first.
        from viba.viba_type_descriptor import _build_type

        body_node = viba_ast.convert_to_chain_style(target.ast_node.body)
        body = _build_type(payload.pool, target.container_module, body_node)
        return self._substitute(body, dict(zip(params, payload.args)))


    def _substitute(self, descriptor: VibaTypeDescriptor,
                    bindings: dict) -> VibaTypeDescriptor:
        """Replace parameters by name with their argument descriptors; only the
        name branch is touched."""
        payload = descriptor.payload
        if descriptor.kind == TYPE_REF:
            return bindings.get(payload.type_name, descriptor)
        if descriptor.kind in (PRODUCT, SUM, EXPONENT):
            return VibaTypeDescriptor(descriptor.kind, VibaChainDescriptor(
                payload.pool, payload.resolvable_type,
                [self._substitute(e, bindings) for e in payload.elements]))
        if descriptor.kind == TYPE_APP:
            return VibaTypeDescriptor(TYPE_APP, VibaTypeAppDescriptor(
                payload.pool, payload.resolvable_type, payload.constructor_name,
                [self._substitute(a, bindings) for a in payload.args]))
        if descriptor.kind == TUPLE:
            return VibaTypeDescriptor(TUPLE, VibaTupleDescriptor(
                payload.pool, payload.resolvable_type,
                [self._substitute(e, bindings) for e in payload.elements]))
        if descriptor.kind == TAGGED:
            return VibaTypeDescriptor(TAGGED, VibaTaggedDescriptor(
                payload.pool, payload.resolvable_type, payload.tag,
                self._substitute(payload.tagged_type, bindings)))
        return descriptor

    def _resolve_ref(self, descriptor: VibaTypeDescriptor) -> Optional[VibaTypeDescriptor]:
        """When a name points at a transparent definition, hand back that body's
        descriptor."""
        resolvable = descriptor.payload.resolvable_type
        if resolvable is None:
            return None
        resolved = module_get_type(resolvable.container_module, descriptor.payload.type_name)
        if isinstance(resolved, Err) or not isinstance(resolved.value, AstNodeType):
            return None  # a builtin leaf, a generic parameter: stop here
        target = resolved.value
        if not isinstance(target.ast_node, viba_ast.TypeDefinition):
            return None  # a generic needs arguments to have a body: bare name stops
        # The descriptor side has no public "descriptor for a type expression"
        # entry point yet, so use its builder.
        from viba.viba_type_descriptor import _build_type

        return _build_type(descriptor.payload.pool, target.container_module,
                           target.ast_node.body)

    def _container_kind(self, descriptor: VibaTypeDescriptor) -> Optional[str]:
        """The three builtin containers list / set / dict; a tuple is not one, it
        is a product matched by position."""
        if descriptor.kind == TYPE_APP and descriptor.payload.constructor_name in CONTAINERS:
            return descriptor.payload.constructor_name
        return None

    def _has_elements_by_index(self, descriptor: VibaTypeDescriptor) -> bool:
        """Shapes that can be counted and taken by index: list / set / tuple.

        A dict is not one of them: its elements are read through keys and
        at_key, so asking a dict by index is an address the design lacks.
        """
        return self._container_kind(descriptor) in ("list", "set") or descriptor.kind == TUPLE

    def _members(self, node: VibaNode) -> Optional[List[tuple]]:
        """The members of this piece: [(tag or None, descriptor), ...], None when
        it has none."""
        return self._design_members(node.descriptor)

    def _design_members(self, descriptor: VibaTypeDescriptor) -> Optional[List[tuple]]:
        """Same, straight from a design descriptor.

        Sum, product and exponent chains follow one rule: the members are
        $elements, and a unit chain head does not count. A branch (a chain
        nested in $elements) counts as one member like any other element.
        """
        descriptor = self._unfold(descriptor)
        elements = None
        if descriptor.kind in (PRODUCT, SUM, EXPONENT):
            elements = list(descriptor.payload.elements)
        elif descriptor.kind == TAGGED:
            return [(descriptor.payload.tag, descriptor.payload.tagged_type)]
        if elements is None:
            return None
        if elements and self._is_unit_descriptor(elements[0]):
            elements = elements[1:]
        slots = []
        for element in elements:
            if element.kind == TAGGED:
                slots.append((element.payload.tag, element.payload.tagged_type))
            else:
                slots.append((None, element))
        return slots

    def _bare_sum(self, descriptor: VibaTypeDescriptor) -> Optional[List[tuple]]:
        """The positional branches of an untagged sum whose layer a material may
        skip: [(index, shape, descriptor), ...], else None.

        A sum with at most one inner node can be written without the sum layer:
        the material carries that branch's content directly, and leaf branches
        are told apart by the value itself. Two inner nodes cannot be told
        apart, and a tagged sum is addressed by tag, so neither qualifies.
        """
        descriptor = self._unfold(descriptor)
        if descriptor.kind != SUM:
            return None
        elements = list(descriptor.payload.elements)
        if elements and self._is_unit_descriptor(elements[0]):
            elements = elements[1:]
        members, inner = [], 0
        for index, element in enumerate(elements):
            if element.kind == TAGGED:
                return None
            shape = self._branch_shape(element)
            inner += 1 if shape == "inner" else 0
            members.append((index, shape, element))
        return members if inner <= 1 else None

    def _branch_shape(self, descriptor: VibaTypeDescriptor) -> str:
        """How one branch of a sum reads: "unit", "leaf" or "inner"."""
        descriptor = self._unfold(descriptor)
        if descriptor.kind in (NIL, NEVER):
            return "unit"
        if descriptor.kind == LITERAL:
            return "leaf"
        if descriptor.kind == TYPE_REF and descriptor.payload.type_name in SCALAR_NAMES:
            return "leaf"
        return "inner"

    def _bare_sum_tag_target(self, descriptor: VibaTypeDescriptor, tag: str):
        """The descriptor a tag names when it lives in a skipped sum's branch."""
        members = self._bare_sum(descriptor)
        if members is None:
            return None
        for _, shape, branch in members:
            if shape != "inner":
                continue
            for branch_tag, branch_type in self._design_members(branch) or []:
                if branch_tag == tag:
                    return branch_type
        return None

    def _knows(self, node: VibaNode, step: VibaStep, slots: Optional[List[tuple]]) -> bool:
        if step.kind == "by_tag":
            if any(tag == step.value for tag, _ in slots or []):
                return True
            return self._bare_sum_tag_target(node.descriptor, step.value) is not None
        if step.kind == "by_field_index":
            positional = [tag for tag, _ in slots or [] if tag is None]
            return 0 <= step.value < len(positional)
        if step.kind == "at_index":
            return self._has_elements_by_index(self._unfold(node.descriptor))
        if step.kind == "at_key":
            return self._container_kind(self._unfold(node.descriptor)) == "dict"
        return False

    def _target_of(self, node: VibaNode, step: VibaStep, slots: Optional[List[tuple]]):
        if step.kind == "by_tag":
            for tag, descriptor in slots or []:
                if tag == step.value:
                    return descriptor
            return self._bare_sum_tag_target(node.descriptor, step.value)
        if step.kind == "by_field_index":
            positional = [descriptor for tag, descriptor in slots or [] if tag is None]
            if 0 <= step.value < len(positional):
                return _as_value(positional[step.value])
            return None
        if step.kind == "at_index":
            shape = self._unfold(node.descriptor)
            if not self._has_elements_by_index(shape):
                return None
            if shape.kind == TUPLE:
                if 0 <= step.value < len(shape.payload.elements):
                    return shape.payload.elements[step.value]
                return None
            return shape.payload.args[0]
        if step.kind == "at_key":
            shape = self._unfold(node.descriptor)
            if self._container_kind(shape) != "dict":
                return None
            return shape.payload.args[1]
        return None

    # ---- private: the data side ----

    def _value_at(self, data, step: VibaStep, design=None):
        """The matching piece in the material; None when there is none (``...``
        is not data)."""
        return _as_value(self._match(data, step, design))

    def _expand_data(self, data, design):
        """Unfold a material written as a name or a generic application: a name
        gives its definition body, an application fills its arguments in.

        One rule on both sides (the design side works on descriptors, this one
        on syntax trees); a name resolves in the module the design piece comes
        from, hence the module carried by ``design``.
        """
        module = _design_module(design)
        if module is None:
            return data
        if isinstance(data, viba_ast.TypeRef):
            body = _definition_body(module, data.name)
            return data if body is None else body
        if isinstance(data, viba_ast.TypeApp):
            definition = _generic_definition(module, data.constructor)
            if definition is None:
                return data
            params = list(definition.generic_params or [])
            if len(params) != len(data.args):
                return data
            bindings = dict(zip(params, data.args))
            return _fill_params(definition.body, bindings)
        return data

    def _match(self, data, step: VibaStep, design=None):
        if step.kind in ("by_tag", "by_field_index"):
            slots = self._data_members(self._expand_data(data, design))
            if slots is not None:
                if step.kind == "by_tag":
                    for tag, piece in slots:
                        if tag == step.value:
                            return piece
                    return None
                positional = [piece for tag, piece in slots if tag is None]
                if 0 <= step.value < len(positional):
                    return positional[step.value]
            if step.kind == "by_field_index":
                return self._bare_sum_piece(data, design, step.value)
            return None
        if step.kind == "at_index":
            elements = self._elements_of(data)
            if elements is None or not (0 <= step.value < len(elements)):
                return None
            return elements[step.value]
        if step.kind == "at_key":
            if not isinstance(data, viba_ast.TypeApp) or data.constructor != "DictLiteral":
                return None
            for pair in data.args:
                key = self._key_of(pair)
                if key is not None and _key_text(key) == step.value:
                    value = pair.elements[1] if len(pair.elements) > 1 else None
                    return value
            return None
        return None

    def _bare_sum_piece(self, data, design, index: int):
        """The piece an untagged-sum design means when the material skips it.

        The single inner branch takes anything that is not a leaf; a leaf branch
        takes the value that matches it (a nil material takes nil).
        """
        members = self._bare_sum(design)
        if members is None:
            return None
        for position, shape, branch in members:
            if position != index:
                continue
            if shape == "inner":
                return None if _is_leaf_data(data) else data
            return data if self._data_fits_leaf(data, branch) else None
        return None

    def _data_fits_leaf(self, data, branch: VibaTypeDescriptor) -> bool:
        """Does this material piece belong to that leaf branch of a sum?"""
        if isinstance(data, viba_ast.Nil):
            return branch.kind == NIL
        if isinstance(data, viba_ast.Never):
            return branch.kind == NEVER
        if not isinstance(data, viba_ast.Constant):
            return False
        literal = _constant_value(data.value)
        if branch.kind == LITERAL:
            return (branch.payload.value.kind == literal.kind
                    and branch.payload.value.value == literal.value)
        return branch.kind == TYPE_REF and branch.payload.type_name == literal.kind

    def _key_of(self, pair):
        """The key of one dict entry; None when it is not a literal."""
        if isinstance(pair, viba_ast.Tuple) and pair.elements:
            key = pair.elements[0]
            if isinstance(key, viba_ast.Constant):
                return key.value
        return None

    def _elements_of(self, data) -> Optional[List]:
        if isinstance(data, viba_ast.Tuple):
            return list(data.elements)
        if isinstance(data, viba_ast.TypeApp) and data.constructor in LITERAL_CTORS:
            return list(data.args)
        return None

    # ---- private: member traversal (VibaListFields and walk) ----

    def _is_unit_descriptor(self, descriptor) -> bool:
        """A unit chain head: nil / never by kind, or a name the config calls
        one."""
        if descriptor.kind in (NIL, NEVER):
            return True
        return (descriptor.kind == TYPE_REF
                and self.config.is_unit_name(descriptor.payload.type_name))

    def _is_unit_member(self, member: VibaMemberDescriptor) -> bool:
        written = member_type_name(member)
        return isinstance(written, Ok) and self.config.is_unit_name(written.value)

    def _is_unit_data(self, node) -> bool:
        """The same on the material side: a unit written as a form or a name."""
        if isinstance(node, (viba_ast.Nil, viba_ast.Never)):
            return True
        return (isinstance(node, viba_ast.TypeRef)
                and self.config.is_unit_name(node.name))

    def _data_members(self, data) -> Optional[List[tuple]]:
        """The members of this data piece: [(tag or None, piece), ...]; the
        three chains follow one rule."""
        elements = _flatten(data)
        if elements is None:
            if isinstance(data, viba_ast.Tagged):
                return [(data.tag, data.type)]
            return None
        if elements and self._is_unit_data(elements[0]):
            elements = elements[1:]
        slots = []
        for element in elements:
            if isinstance(element, viba_ast.Tagged):
                slots.append((element.tag, element.type))
            else:
                slots.append((None, element))
        return slots

    def _walk(self, node: VibaNode) -> List[VibaNode]:
        """Walk every address reachable on this map."""
        out = [node]
        positional = 0
        for tag, _ in self._members(node) or []:
            if tag:
                step = by_tag(tag)
            else:
                step = by_field_index(positional)
                positional += 1
            given = self.get(node, step)
            if isinstance(given, Ok) and given.value is not None:
                out += self._walk(given.value)
        shape = self._unfold(node.descriptor)
        container = self._container_kind(shape)
        if container == "dict":
            given_keys = self.keys(node)
            if isinstance(given_keys, Ok):
                for key in given_keys.value:
                    given = self.get(node, at_key(key))
                    if isinstance(given, Ok) and given.value is not None:
                        out += self._walk(given.value)
        elif self._has_elements_by_index(shape):
            length = self.length(node)
            if isinstance(length, Ok):
                for index in range(length.value):
                    given = self.get(node, at_index(index))
                    if isinstance(given, Ok) and given.value is not None:
                        out += self._walk(given.value)
        return out


# ----------------------------------------------------------------------
# This layer's default accessor
# ----------------------------------------------------------------------

# The language layer's units: what viba/type.viba and the descriptor's own
# docs write for the sum unit and the product unit.
access = VibaAccess(Config(never_eqv={"Oneof"},
                           nil_eqv={"Object", "Assert", "Appendix", "Hint"}))


# ----------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------


def _as_value(piece):
    """``...`` is not data: this piece has nothing to take."""
    if piece is None or isinstance(piece, viba_ast.Ellipsis):
        return None
    return piece


def _flatten(node) -> Optional[List]:
    """Sum / product / exponent read as an element list: the main chain flat,
    a branch as one element.

    The reading matches canonicalization (the chain is the main chain and its
    elements may be branches); a binary tree that was never canonicalized reads
    the same way, so both sides give the same element list.
    """
    if isinstance(node, (viba_ast.ProductChain, viba_ast.SumChain, viba_ast.ExponentChain)):
        return list(node.elements)
    if isinstance(node, (viba_ast.Product, viba_ast.Sum, viba_ast.Exponent)):
        head = node.left if isinstance(node, (viba_ast.Product, viba_ast.Sum)) else node.result
        tail = node.right if isinstance(node, (viba_ast.Product, viba_ast.Sum)) else node.argument
        elements = _flatten(head)
        if elements is None:
            elements = [head]
        return elements + [tail]
    return None


def _is_leaf_data(node) -> bool:
    """A material piece written as a leaf: a literal, nil or never."""
    return isinstance(node, (viba_ast.Constant, viba_ast.Nil, viba_ast.Never))


def _design_module(descriptor):
    """Which module the design piece belongs to (material names resolve in it)."""
    resolvable = getattr(descriptor, "resolvable_type", None)
    return getattr(resolvable, "container_module", None)


def _definition_body(module, name):
    """The plain definition body that name points at; None when it is not one."""
    resolved = module_get_type(module, name)
    if isinstance(resolved, Err) or not isinstance(resolved.value, AstNodeType):
        return None
    node = resolved.value.ast_node
    return node.body if isinstance(node, viba_ast.TypeDefinition) else None


def _generic_definition(module, name):
    """The generic definition that name points at; None when it is not one."""
    resolved = module_get_type(module, name)
    if isinstance(resolved, Err) or not isinstance(resolved.value, AstNodeType):
        return None
    node = resolved.value.ast_node
    return node if isinstance(node, viba_ast.GenericDefinition) else None


class _ParamFiller(viba_ast.NodeTransformer):
    """Replace the parameter names written in a definition body by arguments."""

    def __init__(self, bindings):
        self.bindings = bindings

    def visit_TypeRef(self, node):
        return self.bindings.get(node.name, node)


def _fill_params(body, bindings):
    return _ParamFiller(bindings).visit(copy.deepcopy(body))


def _key_text(key) -> Optional[str]:
    if key is None:
        return None
    if isinstance(key, bool):
        return "true" if key else "false"
    return str(key)


def _constant_value(value) -> VibaConstantValue:
    if isinstance(value, bool):
        return VibaConstantValue("bool", value)
    if isinstance(value, int):
        return VibaConstantValue("int", value)
    if isinstance(value, float):
        return VibaConstantValue("float", value)
    if isinstance(value, str):
        return VibaConstantValue("str", value)
    if value is None:
        return VibaConstantValue("nil", None)
    raise TypeError(f"no constant kind for {value!r}")


# The protocol names (viba-reflect.md sections 4 and 5), and only these.
__all__ = [
    "access", "Config",
    "VibaAccess", "VibaNode", "VibaStep", "VibaPath",
    "by_tag", "by_field_index", "at_index", "at_key",
]

# Bound by this layer, imported by name, not protocol concepts:
#   VibaData         what the Data parameter binds to here
#   VibaReflectError the section 5.4 "taking path throws" exception

"""VibaTypeDescriptor — the descriptor side of the reflection protocol.

The spec is ``viba/viba_type_descriptor.viba``; this module implements it.
Spec names are capitalized (``ParseVibaFile``), the Python here is
snake_case (``parse_viba_file``), as in ``viba/type.py``.

Two things the spec says, and how they land here:

* Every descriptor holds its pool, so any descriptor can answer its own
  questions; type expressions additionally hold the ``viba.type`` value
  they stand for (``resolvable_type``).
* Fields and branches are the same thing (a tag plus a type), so there is
  one ``VibaMemberDescriptor`` and the pool has one member table.

What the spec leaves open, and how it lands here:

* ``VibaPool`` has no mutable state, so adding a file rewrites every
  descriptor in the pool to point at the new pool.  Settled: pools are
  built offline in one go, never added to while in use, so the cost is
  accepted.
"""

from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Optional

from viba import viba_ast
from viba.partial import (environment_result_problem, get_args_product,
                          module_as_function, reduce_partial)
from viba.pattern import (GENERIC_FILE, file_pattern_problem,
                          generic_of_entries, order_of)
from viba.viba_ast.tagged import TAGGED_NAME, symbol_of, tag_of

# The builtin name that reads a call's arguments back as a product.
GET_ARGS_NAME = "__get_args__"
from viba.viba_ast import nodes as ast_nodes
from viba.type import (
    PartialError,
    AstNodeType,
    BUILTIN_CONCEPT,
    BUILTIN_MODULE,
    CustomModuleType,
    VibaProgramErr,
    ModuleType,
    Ok,
    Result,
    module_get_type,
)

# Branch names of the spec's VibaTypeDescriptor sum.
TYPE_REF = "type_ref"
MEMBER_READ = "member_read"
TYPE_APP = "type_app"
TUPLE = "tuple"
TAGGED = "tagged"
SUM = "sum"
PRODUCT = "product"
EXPONENT = "exponent"
LITERAL = "literal"
NIL = "nil"
NEVER = "never"
ANY = "any"
ELLIPSIS = "ellipsis"
CODE_BLOCK = "code_block"

PRODUCT_UNIT = ("Object", "nil")  # the unit of a product chain head
SUM_UNIT = ("Oneof", "never")  # the unit of a sum chain head
EXPONENT_UNIT = ("never",)  # the unit of an exponent chain head (the result element)


# ----------------------------------------------------------------------
# Type expressions: the twelve branches of the spec's sum
# ----------------------------------------------------------------------


class VibaTypeDescriptor:
    """A written type expression: one branch of the spec's sum."""

    __slots__ = ("kind", "payload")

    def __init__(self, kind: str, payload=None):
        self.kind = kind
        self.payload = payload

    @property
    def pool(self) -> Optional["VibaPool"]:
        return getattr(self.payload, "pool", None)

    @property
    def resolvable_type(self) -> Optional[AstNodeType]:
        return getattr(self.payload, "resolvable_type", None)

    def __repr__(self):
        return f"VibaTypeDescriptor({self.kind}, {self.payload!r})"


class VibaMemberReadDescriptor:
    """`g[T].name` — a member of what an application answers.

    Which member that is a definition of is known once the application is
    decided, so the descriptor keeps the owner's own descriptor and the name;
    the layer that reads designs decides, and reads the member there
    (viba-pattern.md).
    """

    __slots__ = ("pool", "resolvable_type", "owner", "name")

    def __init__(self, pool, resolvable_type, owner, name: str):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.owner = owner
        self.name = name

    def __repr__(self):
        return f"VibaMemberReadDescriptor({self.owner!r}, {self.name!r})"


class VibaTypeRefDescriptor:
    """name — a reference, written as is."""

    __slots__ = ("pool", "resolvable_type", "type_name")

    def __init__(self, pool, resolvable_type, type_name: str):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.type_name = type_name

    def __repr__(self):
        return f"VibaTypeRefDescriptor({self.type_name!r})"


class VibaTypeAppDescriptor:
    """Constructor[Arg, ...]"""

    __slots__ = ("pool", "resolvable_type", "constructor_name", "args")

    def __init__(self, pool, resolvable_type, constructor_name: str, args: List[VibaTypeDescriptor]):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.constructor_name = constructor_name
        self.args = args

    def __repr__(self):
        return f"VibaTypeAppDescriptor({self.constructor_name!r}, {len(self.args)} args)"


class VibaTupleDescriptor:
    """(A, B, C) — positional product."""

    __slots__ = ("pool", "resolvable_type", "elements")

    def __init__(self, pool, resolvable_type, elements: List[VibaTypeDescriptor]):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.elements = elements

    def __repr__(self):
        return f"VibaTupleDescriptor({len(self.elements)} elements)"


class VibaTaggedDescriptor:
    """$tag Type"""

    __slots__ = ("pool", "resolvable_type", "tag", "tagged_type")

    def __init__(self, pool, resolvable_type, tag: str, tagged_type: VibaTypeDescriptor):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.tag = tag
        self.tagged_type = tagged_type

    def __repr__(self):
        return f"VibaTaggedDescriptor({self.tag!r})"


class VibaChainDescriptor:
    """Canonical sum chain or product chain."""

    __slots__ = ("pool", "resolvable_type", "elements")

    def __init__(self, pool, resolvable_type, elements: List[VibaTypeDescriptor]):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.elements = elements

    def __repr__(self):
        return f"VibaChainDescriptor({len(self.elements)} elements)"


class VibaLiteralDescriptor:
    """A literal: the value itself (bool / int / float / str / nil)."""

    __slots__ = ("pool", "resolvable_type", "value")

    def __init__(self, pool, resolvable_type, value):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.value = value

    def __repr__(self):
        return f"VibaLiteralDescriptor({self.value!r})"


class VibaCodeBlockDescriptor:
    """{...} — kept verbatim."""

    __slots__ = ("pool", "resolvable_type", "code")

    def __init__(self, pool, resolvable_type, code: str):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.code = code

    def __repr__(self):
        return "VibaCodeBlockDescriptor(...)"


# ----------------------------------------------------------------------
# Members / definitions / imports / files
# ----------------------------------------------------------------------


class VibaMemberDescriptor:
    """A field of a product or a branch of a sum; the same thing."""

    __slots__ = ("pool", "member_index", "tag", "member_type", "containing_full_name")

    def __init__(self, pool, member_index: int, tag, member_type, containing_full_name: str):
        self.pool = pool
        self.member_index = member_index
        self.tag = tag
        self.member_type = member_type
        self.containing_full_name = containing_full_name

    def __repr__(self):
        return f"VibaMemberDescriptor({self.member_index}, {self.tag!r})"


class VibaDefinitionDescriptor:
    __slots__ = (
        "pool", "def_name", "module_name", "file_name", "file_hash",
        "full_name", "generic_params", "body", "members",
    )

    def __init__(self, pool, def_name, module_name, file_name, file_hash,
                 full_name, generic_params, body, members):
        self.pool = pool
        self.def_name = def_name
        self.module_name = module_name
        self.file_name = file_name
        self.file_hash = file_hash
        self.full_name = full_name
        self.generic_params = generic_params
        self.body = body
        self.members = members

    def __repr__(self):
        return f"VibaDefinitionDescriptor({self.full_name!r})"


class VibaImportDescriptor:
    __slots__ = ("pool", "module_name", "local_name")

    def __init__(self, pool, module_name: str, local_name: str):
        self.pool = pool
        self.module_name = module_name
        self.local_name = local_name

    def __repr__(self):
        return f"VibaImportDescriptor({self.module_name!r} as {self.local_name!r})"


class VibaFileDescriptor:
    __slots__ = ("pool", "file_name", "file_hash", "module_name",
                 "imports", "definitions", "_tree")

    def __init__(self, pool, file_name, file_hash, module_name, imports, definitions, tree):
        self.pool = pool
        self.file_name = file_name
        self.file_hash = file_hash
        self.module_name = module_name
        self.imports = imports
        self.definitions = definitions
        self._tree = tree  # private: lets the pool rebind descriptors to itself

    def __repr__(self):
        return f"VibaFileDescriptor({self.module_name!r}, {len(self.definitions)} defs)"


class VibaPool:
    __slots__ = ("files", "file_name2file", "full_name2definition",
                 "full_name2member", "module_environment")

    def __init__(self, files, file_name2file, full_name2definition,
                 full_name2member, module_environment):
        self.files = files
        self.file_name2file = file_name2file
        self.full_name2definition = full_name2definition
        self.full_name2member = full_name2member
        self.module_environment = module_environment

    def __repr__(self):
        return f"VibaPool({len(self.files)} files)"


# ----------------------------------------------------------------------
# Building the pool, building descriptors
# ----------------------------------------------------------------------


def empty_pool() -> VibaPool:
    """A pool with nothing in it; its environment answers module names."""
    pool = VibaPool([], {}, {}, {}, None)
    pool.module_environment = _environment(pool)
    return pool


def _environment(pool: VibaPool) -> Callable[[str], Result]:
    """A module name for a module: looked up in the pool by $module_name, else a VibaProgramErr.

    A generic is not a file but a directory: the `name.number` files in the pool plus its
    `__generic__.viba` marker make one GenericModuleType (viba-pattern.md). The
    bare name of a `builtin.` generic names the same one (`is_closure` is
    `builtin.is_closure`), the way it does wherever else a name is read.
    """
    def environment(module_name: str) -> Result:
        matches = [f for f in pool.files if f.module_name == module_name]
        if not matches:
            generic = _generic_in_pool(pool, module_name)
            if generic is not None:
                return generic
            if "." not in module_name:
                generic = _generic_in_pool(pool, f"{BUILTIN_CONCEPT}.{module_name}")
                if generic is not None:
                    return generic
            return VibaProgramErr(f"no module named {module_name!r} in pool")
        if len(matches) > 1:
            return VibaProgramErr(f"module {module_name!r} is served by {len(matches)} files")
        return Ok(CustomModuleType(matches[0]._tree, pool.module_environment,
                                   _import_locals(matches[0])))
    return environment


def _generic_in_pool(pool: VibaPool, module_name: str):
    """The generic the pool holds under this module name, or None.

    Its files are the ones named under it: the marker `name.__generic__`, and
    one `name.number` per pattern. `name` itself is no file of the pool —
    that is what makes it a generic rather than a module of definitions.
    """
    prefix = module_name + "."
    under = [f for f in pool.files if f.module_name.startswith(prefix)]
    marker_name = prefix + GENERIC_FILE[: -len(".viba")]
    marker = next((f for f in under if f.module_name == marker_name), None)
    if marker is None:
        return None
    parts = []
    for file in under:
        if file is marker:
            continue
        order = order_of(file.module_name[len(prefix):])
        if order is None:
            return VibaProgramErr(
                f"{file.file_name}: a file of the generic {module_name!r} is "
                f"named by its order, a number")
        parts.append((order, file.file_name, file.module_name,
                      CustomModuleType(file._tree, pool.module_environment,
                                       _import_locals(file))))
    return generic_of_entries(module_name, marker.file_name, marker._tree, parts)


def _import_locals_for(tree) -> dict:
    """A syntax tree's import table: binding name -> module name. With `as` that is the
    alias; without `as`, the module's full name is bound (a reference to `import a.b` is
    written a.b.Name)."""
    return {stmt.alias or stmt.module: stmt.module
            for stmt in tree.body if isinstance(stmt, ast_nodes.Import)}


def _import_locals(file) -> dict:
    return {import_.local_name: import_.module_name for import_ in file.imports}


def _normalize_source(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n")


def parse_viba_file(pool: VibaPool, source: str, file_name: str, module_name: str) -> Result:
    """Source compiled into a file descriptor (the descriptor is bound to this pool).

    Which file it is and which module it counts as are both given by the caller; the two
    need not share a name.
    """
    text = _normalize_source(source)
    try:
        tree = viba_ast.canonical(viba_ast.parse(text))
    except Exception as exc:  # syntax error: the parser raises, turn it into VibaProgramErr
        return VibaProgramErr(f"cannot parse: {exc!r}")
    problem = file_pattern_problem(tree, module_name.split(".")[-1])
    if problem is not None:
        return VibaProgramErr(problem)
    file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    try:
        return Ok(_build_file(pool, tree, file_name, module_name, file_hash))
    except PartialError as exc:  # a design mistake, like a duplicate tag
        return VibaProgramErr(str(exc))


def _build_file(pool: VibaPool, tree, file_name: str, module_name: str, file_hash: str) -> VibaFileDescriptor:
    module = CustomModuleType(tree, pool.module_environment,
                              _import_locals_for(tree))
    imports = []
    written = []                    # the definitions in one file, in written order
    where = {}                      # name -> its position in written
    for stmt in tree.body:
        if isinstance(stmt, ast_nodes.Import):
            local = stmt.alias or stmt.module
            imports.append(VibaImportDescriptor(pool, stmt.module, local))
        elif isinstance(stmt, (ast_nodes.TypeDefinition, ast_nodes.GenericDefinition)):
            if stmt.name in where:  # a module may not define the same name twice
                raise PartialError(
                    f"{stmt.name!r} is defined twice: a module defines each "
                    f"name once")
            where[stmt.name] = len(written)
            written.append(stmt)
    definitions = [_build_definition(pool, module, stmt, module_name, file_name, file_hash)
                   for stmt in written]
    return VibaFileDescriptor(pool, file_name, file_hash, module_name, imports, definitions, tree)


def _build_definition(pool, module, stmt, module_name, file_name, file_hash) -> VibaDefinitionDescriptor:
    full_name = f"{module_name}.{stmt.name}"
    params = list(getattr(stmt, "generic_params", None) or [])
    return VibaDefinitionDescriptor(
        pool=pool,
        def_name=stmt.name,
        module_name=module_name,
        file_name=file_name,
        file_hash=file_hash,
        full_name=full_name,
        generic_params=params,
        body=_build_type(pool, module, stmt.body),
        members=_build_members(pool, module, stmt.body, full_name),
    )


def _left_elements(node, kind, chain_kind) -> List:
    """A product/sum written as a chain, read as an element list (written order): collect
    $right along the $left spine."""
    if isinstance(node, chain_kind):
        return list(node.elements)
    elements = []
    while isinstance(node, kind):
        elements.append(node.right)
        node = node.left
    elements.append(node)
    elements.reverse()
    return elements


def _exponent_elements(node) -> List:
    """An exponent written as a chain, read as an element list (written order): collect
    $argument along the $result spine."""
    if isinstance(node, ast_nodes.ExponentChain):
        return list(node.elements)
    elements = []
    while isinstance(node, ast_nodes.Exponent):
        elements.append(node.argument)
        node = node.result
    elements.append(node)
    elements.reverse()
    return elements


def _body_elements(body):
    """A definition body written as a product/sum/exponent chain: (elements, chain-head unit).

    All three chains follow one rule: the members are that chain's elements, and the
    chain-head unit does not count.
    """
    if isinstance(body, (ast_nodes.Product, ast_nodes.ProductChain)):
        return _left_elements(body, ast_nodes.Product, ast_nodes.ProductChain), PRODUCT_UNIT
    if isinstance(body, (ast_nodes.Sum, ast_nodes.SumChain)):
        return _left_elements(body, ast_nodes.Sum, ast_nodes.SumChain), SUM_UNIT
    if isinstance(body, (ast_nodes.Exponent, ast_nodes.ExponentChain)):
        return _exponent_elements(body), EXPONENT_UNIT
    return None, None


def _build_members(pool, module, body, full_name: str) -> List[VibaMemberDescriptor]:
    """The elements of a written product/sum/exponent chain are its members; the chain-head
    unit does not count."""
    elements, unit = _body_elements(body)
    if elements is not None:
        if elements and _is_unit(elements[0], unit):
            elements = elements[1:]
    elif isinstance(body, ast_nodes.Tagged):
        elements = [body]  # the body is one whole tagged type: a product of one member
    else:
        return []
    members = []
    for index, element in enumerate(elements):
        if isinstance(element, ast_nodes.Tagged):
            tag, node = element.tag, element.type
        else:
            tag, node = None, element
        members.append(VibaMemberDescriptor(
            pool=pool,
            member_index=index,
            tag=tag,
            member_type=_build_type(pool, module, node),
            containing_full_name=full_name,
        ))
    return members


def _is_unit(node, names) -> bool:
    return isinstance(node, ast_nodes.TypeRef) and node.name in names


def descriptor_of_tagged(tag, inner: VibaTypeDescriptor, node, written_in) -> VibaTypeDescriptor:
    """A tagged member's descriptor: the tag, over the member's own reading.

    A product's members are addressed by their tags, so the tag has to be part of
    the reading — and the member's own descriptor is what is underneath it.
    """
    return VibaTypeDescriptor(TAGGED, VibaTaggedDescriptor(
        empty_pool(), AstNodeType(node, written_in), tag, inner))


def descriptor_of_values(node, written_in, members) -> VibaTypeDescriptor:
    """A descriptor for a piece built **from values**, not from text.

    Each member already has a descriptor of its own, and with it the module it
    was written in. Reading the whole piece again as one node in one module would
    be wrong the moment a member came from somewhere else — its names would be
    looked up in the wrong place — so the product keeps the members' own
    readings, each under the tag that addresses it.
    """
    resolvable = AstNodeType(node, written_in)
    pieces = _left_elements(node, ast_nodes.Product, ast_nodes.ProductChain)
    elements = []
    for index, member in enumerate(members):
        descriptor = member.node.descriptor
        piece = pieces[index] if index < len(pieces) else None
        if isinstance(piece, ast_nodes.Tagged):
            descriptor = descriptor_of_tagged(piece.tag, descriptor, piece, written_in)
        elements.append(descriptor)
    return VibaTypeDescriptor(PRODUCT, VibaChainDescriptor(
        empty_pool(), resolvable, elements))


def descriptor_of(node) -> VibaTypeDescriptor:
    """A shallow descriptor of a syntax node (AstNodeType).

    Whoever has only syntax and wants to read it through reflection takes the descriptor
    here. The pool is empty: there is no place to look a pool up by name here, so a name is
    resolved in the module this design was written in (the descriptor carries it along)."""
    return _build_type(empty_pool(), node.container_module, node.ast_node)


def _fits(given, given_module, written, written_module):
    """Does the given type fit the slot it is written to? The judgment answers,
    with the language's units, exactly as it would anywhere else."""
    from viba.is_sub_type import is_sub_type
    from viba.reflect import language_config
    judged = is_sub_type(AstNodeType(given, given_module),
                         AstNodeType(written, written_module),
                         config=language_config)
    return isinstance(judged, Ok) and judged.ok_value is True


def _partial_target(pool, name, module):
    """(body, written_in) for the name a `<<` gives to, or None.

    A function written in a module never answers the environment itself: only a
    builtin function does, and the builtin library is the one module the check
    steps aside for. A dotted name may also be a member of the product a name
    stands for — `args.fib` — which is what a chain gives its arguments to
    (`_product_member`).
    """
    resolved = module_get_type(module, name)
    if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
        member = _product_member(pool, name, module)
        if member is not None:
            return member
        return module_as_function(module, name)
    node = resolved.ok_value.ast_node
    if isinstance(node, ast_nodes.GenericDefinition):
        return None
    if isinstance(node, ast_nodes.TypeDefinition):
        node = node.body
    if (resolved.ok_value.container_module is not BUILTIN_MODULE
            and isinstance(node, (ast_nodes.Exponent, ast_nodes.ExponentChain))):
        problem = environment_result_problem(node, name)
        if problem is not None:
            raise PartialError(problem)
    return node, resolved.ok_value.container_module


def _product_member(pool, name, module):
    """(body, written_in) for `args.a`: the member of the product a name stands for.

    `args = __get_args__ << __decl__` is the product of the call's parameters
    (`_get_args_call`), so `args.a` is the `$a` member's declared type: what a
    chain gives its arguments to (`args.fib << …`) and what the judgment layer
    reads through the same dotted name (`is_sub_type._product_member_type`). None
    when the head is no such product, or that member is not there — the name is
    then read the way it always was.
    """
    head, dot, tag = name.rpartition(".")
    if not dot or not head:
        return None
    resolved = module_get_type(module, head)
    if not (isinstance(resolved, Ok) and isinstance(resolved.ok_value, AstNodeType)):
        return None
    body, written_in = resolved.ok_value.ast_node, resolved.ok_value.container_module
    if isinstance(body, ast_nodes.TypeDefinition):
        body = body.body
    if isinstance(body, ast_nodes.Partial):
        product = _get_args_call(pool, body, written_in)
        if product is None:
            return None
        body, written_in = product
    for factor in _left_elements(body, ast_nodes.Product, ast_nodes.ProductChain):
        if isinstance(factor, ast_nodes.Tagged) and factor.tag == "$" + tag:
            return factor.type, written_in
    return None


def _partial_parts(node):
    """(chain head, argument list) for a written `<<` chain, arguments in written order."""
    arguments = []
    while isinstance(node, ast_nodes.Partial):
        arguments.append(node.argument)
        node = node.function
    return node, list(reversed(arguments))


def _get_args_call(pool, node, module):
    """(product, written_in) for `__get_args__ << <chain>`, or None.

    What a call was handed, read as a type: the product of the parameters the
    chain declares, the environment among them — which is how a module names the
    environment (`args.env`) and its arguments (`args.a`), and what makes
    `args = __get_args__ << __decl__` a product here (viba-interpreter.md).
    """
    head, arguments = _partial_parts(node)
    if not (isinstance(head, ast_nodes.TypeRef) and head.name == GET_ARGS_NAME):
        return None
    if len(arguments) != 1:
        return None
    argument, written_in = arguments[0], module
    if isinstance(argument, ast_nodes.TypeRef):
        resolved = module_get_type(module, argument.name)
        if not (isinstance(resolved, Ok) and isinstance(resolved.ok_value, AstNodeType)):
            return None
        argument = resolved.ok_value.ast_node
        written_in = resolved.ok_value.container_module
    if isinstance(argument, ast_nodes.TypeDefinition):
        argument = argument.body
    if not isinstance(argument, (ast_nodes.Exponent, ast_nodes.ExponentChain)):
        return None
    return get_args_product(argument), written_in


def tagged_descriptor(pool, constructor_name: str, args, resolvable):
    """The tag a `tagged[...]` descriptor stands for, or None.

    The symbol is known here when it is a written string — the source folded
    those into the tag already — and when a decision handed it over: a `pattern`
    line extracted the symbol from a tag and `__decl__` builds the tag back with
    it (viba-pattern.md). The answer is the tagged type `$S T`, written as that
    tag, so the descriptor says where the type is addressed and whose module its
    own names resolve in.
    """
    if constructor_name != TAGGED_NAME or len(args) != 2:
        return None
    if args[0].kind != LITERAL:
        return None
    value = args[0].payload.value
    symbol = symbol_of(value) if isinstance(value, str) else None
    if symbol is None:
        return None
    inner = args[1]
    written_in = inner.resolvable_type
    if written_in is None:
        return None
    written = viba_ast.Tagged(tag_of(symbol), written_in.ast_node)
    return VibaTypeDescriptor(TAGGED, VibaTaggedDescriptor(
        pool, AstNodeType(written, written_in.container_module), tag_of(symbol),
        inner))


def _build_type(pool, module, node) -> VibaTypeDescriptor:
    if isinstance(node, ast_nodes.Any):
        return VibaTypeDescriptor(ANY)
    if isinstance(node, ast_nodes.Partial):
        product = _get_args_call(pool, node, module)
        if product is not None:
            return _build_type(pool, product[1], product[0])
        reduced, written_in = reduce_partial(node, module,
                                       lambda name, written_in: _partial_target(pool, name, written_in),
                                       _fits)
        # The reduction may step into another module (a module call: `__impl__` is written there):
        # the read then continues in the module that came back, not the one it came in with.
        return _build_type(pool, written_in, reduced)
    resolvable = AstNodeType(node, module)
    if isinstance(node, (ast_nodes.Product, ast_nodes.ProductChain)):
        return VibaTypeDescriptor(PRODUCT, VibaChainDescriptor(
            pool, resolvable, [_build_type(pool, module, e) for e in
                               _left_elements(node, ast_nodes.Product, ast_nodes.ProductChain)]))
    if isinstance(node, (ast_nodes.Sum, ast_nodes.SumChain)):
        return VibaTypeDescriptor(SUM, VibaChainDescriptor(
            pool, resolvable, [_build_type(pool, module, e) for e in
                               _left_elements(node, ast_nodes.Sum, ast_nodes.SumChain)]))
    if isinstance(node, (ast_nodes.Exponent, ast_nodes.ExponentChain)):
        return VibaTypeDescriptor(EXPONENT, VibaChainDescriptor(
            pool, resolvable,
            [_build_type(pool, module, e) for e in _exponent_elements(node)]))
    if isinstance(node, ast_nodes.TypeApp):
        args = [_build_type(pool, module, a) for a in node.args]
        folded = tagged_descriptor(pool, node.constructor, args, resolvable)
        if folded is not None:
            return folded
        return VibaTypeDescriptor(TYPE_APP, VibaTypeAppDescriptor(
            pool, resolvable, node.constructor, args))
    if isinstance(node, ast_nodes.Tuple):
        return VibaTypeDescriptor(TUPLE, VibaTupleDescriptor(
            pool, resolvable, [_build_type(pool, module, e) for e in node.elements]))
    if isinstance(node, ast_nodes.Tagged):
        return VibaTypeDescriptor(TAGGED, VibaTaggedDescriptor(
            pool, resolvable, node.tag, _build_type(pool, module, node.type)))
    if isinstance(node, ast_nodes.TypeRef):
        return VibaTypeDescriptor(TYPE_REF, VibaTypeRefDescriptor(pool, resolvable, node.name))
    if isinstance(node, ast_nodes.MemberRead):
        # A member read written segment by segment reads as the dotted name it was; the descriptor
        # is built from that name. With an application on the left (`g[T].value`) the member is a
        # definition of the file the decision chose, and is left to the decision layer.
        path = viba_ast.written_path(node)
        if path is None:
            # The left side is an application: which member it is comes out of the decision over
            # that application, so the descriptor keeps the left descriptor and the member name, and
            # the layer that reads the design settles it (reflect._decided_member).
            return VibaTypeDescriptor(MEMBER_READ, VibaMemberReadDescriptor(
                pool, resolvable, _build_type(pool, module, node.owner), node.name))
        return VibaTypeDescriptor(TYPE_REF, VibaTypeRefDescriptor(pool, resolvable, path))
    if isinstance(node, ast_nodes.Constant):
        return VibaTypeDescriptor(LITERAL, VibaLiteralDescriptor(
            pool, resolvable, node.value))
    if isinstance(node, ast_nodes.Nil):
        return VibaTypeDescriptor(NIL)
    if isinstance(node, ast_nodes.Never):
        return VibaTypeDescriptor(NEVER)
    if isinstance(node, ast_nodes.Ellipsis):
        return VibaTypeDescriptor(ELLIPSIS)
    if isinstance(node, ast_nodes.CodeBlock):
        return VibaTypeDescriptor(CODE_BLOCK, VibaCodeBlockDescriptor(pool, resolvable, node.code))
    raise TypeError(f"no descriptor for {type(node).__name__}")


# ----------------------------------------------------------------------
# Queries on the pool
# ----------------------------------------------------------------------


def pool_add_file(pool: VibaPool, file: VibaFileDescriptor) -> Result:
    """A pool with one more file; the descriptors in it are rebound to the new pool."""
    if file.pool is not pool:
        return VibaProgramErr("the file was not built into this pool")
    if file.file_name in pool.file_name2file:
        return VibaProgramErr(f"duplicate file {file.file_name!r}")
    new_pool = VibaPool(
        files=[],
        file_name2file={},
        full_name2definition={},
        full_name2member={},
        module_environment=None,
    )
    new_pool.module_environment = _environment(new_pool)
    stored = [f for f in pool.files] + [file]
    rebuilt = []
    for old in stored:
        fresh = _build_file(new_pool, old._tree, old.file_name, old.module_name, old.file_hash)
        if fresh.file_name in new_pool.file_name2file:
            return VibaProgramErr(f"duplicate file {fresh.file_name!r}")
        new_pool.files.append(fresh)
        new_pool.file_name2file[fresh.file_name] = fresh
        rebuilt.append(fresh)
    for fresh in rebuilt:
        for definition in fresh.definitions:
            if definition.full_name in new_pool.full_name2definition:
                return VibaProgramErr(f"duplicate definition {definition.full_name!r}")
            new_pool.full_name2definition[definition.full_name] = definition
            for member in definition.members:
                if member.tag is None:
                    continue
                full = f"{definition.full_name}.{member.tag}"
                if full in new_pool.full_name2member:
                    return VibaProgramErr(f"duplicate member {full!r}")
                new_pool.full_name2member[full] = member
    return Ok(new_pool)


def pool_find_file(pool: VibaPool, file_name: str) -> Result:
    found = pool.file_name2file.get(file_name)
    return Ok(found) if found is not None else VibaProgramErr(f"no file named {file_name!r}")


def pool_find_definition(pool: VibaPool, full_name: str) -> Result:
    found = pool.full_name2definition.get(full_name)
    return Ok(found) if found is not None else VibaProgramErr(f"no definition named {full_name!r}")


def pool_find_member(pool: VibaPool, full_name: str) -> Result:
    found = pool.full_name2member.get(full_name)
    return Ok(found) if found is not None else VibaProgramErr(f"no member named {full_name!r}")


# ----------------------------------------------------------------------
# Queries on a file
# ----------------------------------------------------------------------


def file_find_import_by_local_name(file: VibaFileDescriptor, local_name: str) -> Result:
    for import_ in file.imports:
        if import_.local_name == local_name:
            return Ok(import_)
    return VibaProgramErr(f"file {file.file_name!r} has no import named {local_name!r}")


# ----------------------------------------------------------------------
# Queries on a definition
# ----------------------------------------------------------------------


def definition_members(definition: VibaDefinitionDescriptor) -> Result:
    return Ok(list(definition.members))


def definition_find_member_by_tag(definition: VibaDefinitionDescriptor, tag: str) -> Result:
    for member in definition.members:
        if member.tag == tag:
            return Ok(member)
    return VibaProgramErr(f"definition {definition.full_name!r} has no member tagged {tag!r}")


def definition_find_member_by_index(definition: VibaDefinitionDescriptor, member_index: int) -> Result:
    if 0 <= member_index < len(definition.members):
        return Ok(definition.members[member_index])
    return VibaProgramErr(f"definition {definition.full_name!r} has no member at {member_index}")


def definition_file(definition: VibaDefinitionDescriptor) -> Result:
    return pool_find_file(definition.pool, definition.file_name)


# ----------------------------------------------------------------------
# Queries on a member
# ----------------------------------------------------------------------


def member_type_name(member: VibaMemberDescriptor) -> Result:
    if member.member_type.kind == TYPE_REF:
        return Ok(member.member_type.payload.type_name)
    return VibaProgramErr(f"member {member.tag!r} is not written as a name")


def member_resolved_definition(member: VibaMemberDescriptor) -> Result:
    """Settle a name in a member type onto a definition: first cut the import prefix (with
    `as` it is an alias, without `as` the module's full name, longest match first), then hand
    it to ModuleGetType."""
    name = member_type_name(member)
    if isinstance(name, VibaProgramErr):
        return name
    pool = member.pool
    containing = pool_find_definition(pool, member.containing_full_name)
    if isinstance(containing, VibaProgramErr):
        return containing
    file = pool_find_file(pool, containing.ok_value.file_name)
    if isinstance(file, VibaProgramErr):
        return file
    written = name.ok_value
    parts = written.split(".")
    module_name = target_name = None
    for cut in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:cut])
        import_ = file_find_import_by_local_name(file.ok_value, prefix)
        if isinstance(import_, Ok):
            module_name = import_.ok_value.module_name
            target_name = ".".join(parts[cut:])
            break
    if module_name is None:
        if len(parts) > 1:
            return VibaProgramErr(f"{parts[0]!r} is neither an import nor a module of this file")
        module_name = file.ok_value.module_name
        target_name = written
    module = pool.module_environment(module_name)
    if isinstance(module, VibaProgramErr):
        return module
    resolved = module_get_type(module.ok_value, target_name)
    if isinstance(resolved, VibaProgramErr):
        return resolved
    node = getattr(resolved.ok_value, "ast_node", None)
    if not isinstance(node, (ast_nodes.TypeDefinition, ast_nodes.GenericDefinition)):
        return VibaProgramErr(f"{written!r} is not a definition")
    if node.name != target_name:
        return VibaProgramErr(f"{written!r} does not name a definition of {module_name!r}")
    return pool_find_definition(pool, f"{module_name}.{node.name}")


def member_containing_definition(member: VibaMemberDescriptor) -> Result:
    return pool_find_definition(member.pool, member.containing_full_name)


__all__ = [
    "VibaTypeDescriptor",
    "VibaTypeRefDescriptor", "VibaTypeAppDescriptor", "VibaTupleDescriptor",
    "VibaTaggedDescriptor", "VibaChainDescriptor",
    "VibaLiteralDescriptor", "VibaCodeBlockDescriptor",
    "VibaMemberDescriptor", "VibaDefinitionDescriptor",
    "VibaImportDescriptor", "VibaFileDescriptor", "VibaPool",
    "empty_pool", "descriptor_of",
    "parse_viba_file", "pool_add_file", "pool_find_file",
    "pool_find_definition", "pool_find_member",
    "file_find_import_by_local_name",
    "definition_members", "definition_find_member_by_tag",
    "definition_find_member_by_index", "definition_file",
    "member_type_name", "member_resolved_definition",
    "member_containing_definition",
]

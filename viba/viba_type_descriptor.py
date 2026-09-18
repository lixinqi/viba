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
from viba.viba_ast import nodes as ast_nodes
from viba.type import (
    AstNodeType,
    CustomModuleType,
    Err,
    ModuleType,
    Ok,
    Result,
    module_get_type,
)

# Branch names of the spec's VibaTypeDescriptor sum.
TYPE_REF = "type_ref"
TYPE_APP = "type_app"
TUPLE = "tuple"
TAGGED = "tagged"
SUM = "sum"
PRODUCT = "product"
EXPONENT = "exponent"
LITERAL = "literal"
NIL = "nil"
NEVER = "never"
ELLIPSIS = "ellipsis"
CODE_BLOCK = "code_block"

PRODUCT_UNIT = ("Object", "nil")  # 积链链头的单位元
SUM_UNIT = ("Oneof", "never")  # 和链链头的单位元
EXPONENT_UNIT = ("never",)  # 指数链链头（结果那一元）的单位元


# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------


class VibaConstantValue:
    """One of the five leaf kinds: bool / int / float / str / nil."""

    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value=None):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"VibaConstantValue({self.kind}, {self.value!r})"


# ----------------------------------------------------------------------
# 类型表达式：和类型的十二支
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


class VibaTaggedDescriptor:
    """$tag Type"""

    __slots__ = ("pool", "resolvable_type", "tag", "tagged_type")

    def __init__(self, pool, resolvable_type, tag: str, tagged_type: VibaTypeDescriptor):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.tag = tag
        self.tagged_type = tagged_type


class VibaChainDescriptor:
    """Canonical sum chain or product chain."""

    __slots__ = ("pool", "resolvable_type", "elements")

    def __init__(self, pool, resolvable_type, elements: List[VibaTypeDescriptor]):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.elements = elements


class VibaLiteralDescriptor:
    """A literal: one of the five leaf kinds."""

    __slots__ = ("pool", "resolvable_type", "value")

    def __init__(self, pool, resolvable_type, value: VibaConstantValue):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.value = value


class VibaCodeBlockDescriptor:
    """{...} — kept verbatim."""

    __slots__ = ("pool", "resolvable_type", "code")

    def __init__(self, pool, resolvable_type, code: str):
        self.pool = pool
        self.resolvable_type = resolvable_type
        self.code = code


# ----------------------------------------------------------------------
# 成员 / 定义 / import / 文件
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
# 建池子、建描述符
# ----------------------------------------------------------------------


def empty_pool() -> VibaPool:
    """A pool with nothing in it; its environment answers module names."""
    pool = VibaPool([], {}, {}, {}, None)
    pool.module_environment = _environment(pool)
    return pool


def _environment(pool: VibaPool) -> Callable[[str], Result]:
    """模块名换模块：在池子里按 $module_name 找，找不到就是 Err。"""
    def environment(module_name: str) -> Result:
        matches = [f for f in pool.files if f.module_name == module_name]
        if not matches:
            return Err(f"no module named {module_name!r} in pool")
        if len(matches) > 1:
            return Err(f"module {module_name!r} is served by {len(matches)} files")
        return Ok(CustomModuleType(matches[0]._tree, pool.module_environment))
    return environment


def _normalize_source(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n")


def parse_viba_file(pool: VibaPool, source: str, file_name: str, module_name: str) -> Result:
    """源码编成文件描述符（描述符绑在这个池子上）。

    文件是哪一个、它算哪个模块，都由调用方给，两者不必同名。
    """
    text = _normalize_source(source)
    try:
        tree = viba_ast.canonical(viba_ast.parse(text))
    except Exception as exc:  # syntax error: the parser raises, turn it into Err
        return Err(f"cannot parse: {exc!r}")
    file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Ok(_build_file(pool, tree, file_name, module_name, file_hash))


def _build_file(pool: VibaPool, tree, file_name: str, module_name: str, file_hash: str) -> VibaFileDescriptor:
    module = CustomModuleType(tree, pool.module_environment)
    imports = []
    definitions = []
    for stmt in tree.body:
        if isinstance(stmt, ast_nodes.Import):
            local = stmt.alias or stmt.module.split(".")[-1]
            imports.append(VibaImportDescriptor(pool, stmt.module, local))
        elif isinstance(stmt, (ast_nodes.TypeDefinition, ast_nodes.GenericDefinition)):
            definitions.append(_build_definition(pool, module, stmt, module_name, file_name, file_hash))
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
    """写成一串的积/和读成元素表（书写顺序）：沿 $left 那条脊柱收 $right。"""
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
    """写成一串的指数读成元素表（书写顺序）：沿 $result 那条脊柱收 $argument。"""
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
    """定义体是写成一串的积/和/指数时，给出 (元素表, 链头单位元)。

    三种链一个规矩：成员就是那一串的元素，链头的单位元不算。
    """
    if isinstance(body, (ast_nodes.Product, ast_nodes.ProductChain)):
        return _left_elements(body, ast_nodes.Product, ast_nodes.ProductChain), PRODUCT_UNIT
    if isinstance(body, (ast_nodes.Sum, ast_nodes.SumChain)):
        return _left_elements(body, ast_nodes.Sum, ast_nodes.SumChain), SUM_UNIT
    if isinstance(body, (ast_nodes.Exponent, ast_nodes.ExponentChain)):
        return _exponent_elements(body), EXPONENT_UNIT
    return None, None


def _build_members(pool, module, body, full_name: str) -> List[VibaMemberDescriptor]:
    """写成一串的积/和/指数的元素就是成员；链头的单位元不算。"""
    elements, unit = _body_elements(body)
    if elements is not None:
        if elements and _is_unit(elements[0], unit):
            elements = elements[1:]
    elif isinstance(body, ast_nodes.Tagged):
        elements = [body]  # 体就是一整个带标签的类型：一个成员的积
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


def _build_type(pool, module, node) -> VibaTypeDescriptor:
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
        return VibaTypeDescriptor(TYPE_APP, VibaTypeAppDescriptor(
            pool, resolvable, node.constructor,
            [_build_type(pool, module, a) for a in node.args]))
    if isinstance(node, ast_nodes.Tuple):
        return VibaTypeDescriptor(TUPLE, VibaTupleDescriptor(
            pool, resolvable, [_build_type(pool, module, e) for e in node.elements]))
    if isinstance(node, ast_nodes.Tagged):
        return VibaTypeDescriptor(TAGGED, VibaTaggedDescriptor(
            pool, resolvable, node.tag, _build_type(pool, module, node.type)))
    if isinstance(node, ast_nodes.TypeRef):
        return VibaTypeDescriptor(TYPE_REF, VibaTypeRefDescriptor(pool, resolvable, node.name))
    if isinstance(node, ast_nodes.Constant):
        return VibaTypeDescriptor(LITERAL, VibaLiteralDescriptor(
            pool, resolvable, _constant_value(node.value)))
    if isinstance(node, ast_nodes.Nil):
        return VibaTypeDescriptor(NIL)
    if isinstance(node, ast_nodes.Never):
        return VibaTypeDescriptor(NEVER)
    if isinstance(node, ast_nodes.Ellipsis):
        return VibaTypeDescriptor(ELLIPSIS)
    if isinstance(node, ast_nodes.CodeBlock):
        return VibaTypeDescriptor(CODE_BLOCK, VibaCodeBlockDescriptor(pool, resolvable, node.code))
    raise TypeError(f"no descriptor for {type(node).__name__}")


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
        # 语法里走不到这里：写 nil 走的是 $nil 那一支；留给访问侧取叶子用。
        return VibaConstantValue("nil", None)
    raise TypeError(f"no constant kind for {value!r}")


# ----------------------------------------------------------------------
# 池子上的查询
# ----------------------------------------------------------------------


def pool_add_file(pool: VibaPool, file: VibaFileDescriptor) -> Result:
    """得到一份多了一个文件的池子；池子里的描述符重新绑到新池子上。"""
    if file.pool is not pool:
        return Err("the file was not built into this pool")
    if file.file_name in pool.file_name2file:
        return Err(f"duplicate file {file.file_name!r}")
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
            return Err(f"duplicate file {fresh.file_name!r}")
        new_pool.files.append(fresh)
        new_pool.file_name2file[fresh.file_name] = fresh
        rebuilt.append(fresh)
    for fresh in rebuilt:
        for definition in fresh.definitions:
            if definition.full_name in new_pool.full_name2definition:
                return Err(f"duplicate definition {definition.full_name!r}")
            new_pool.full_name2definition[definition.full_name] = definition
            for member in definition.members:
                if member.tag is None:
                    continue
                full = f"{definition.full_name}.{member.tag}"
                if full in new_pool.full_name2member:
                    return Err(f"duplicate member {full!r}")
                new_pool.full_name2member[full] = member
    return Ok(new_pool)


def pool_find_file(pool: VibaPool, file_name: str) -> Result:
    found = pool.file_name2file.get(file_name)
    return Ok(found) if found is not None else Err(f"no file named {file_name!r}")


def pool_find_definition(pool: VibaPool, full_name: str) -> Result:
    found = pool.full_name2definition.get(full_name)
    return Ok(found) if found is not None else Err(f"no definition named {full_name!r}")


def pool_find_member(pool: VibaPool, full_name: str) -> Result:
    found = pool.full_name2member.get(full_name)
    return Ok(found) if found is not None else Err(f"no member named {full_name!r}")


# ----------------------------------------------------------------------
# 文件上的查询
# ----------------------------------------------------------------------


def file_find_import_by_local_name(file: VibaFileDescriptor, local_name: str) -> Result:
    for import_ in file.imports:
        if import_.local_name == local_name:
            return Ok(import_)
    return Err(f"file {file.file_name!r} has no import named {local_name!r}")


# ----------------------------------------------------------------------
# 定义上的查询
# ----------------------------------------------------------------------


def definition_members(definition: VibaDefinitionDescriptor) -> Result:
    return Ok(list(definition.members))


def definition_find_member_by_tag(definition: VibaDefinitionDescriptor, tag: str) -> Result:
    for member in definition.members:
        if member.tag == tag:
            return Ok(member)
    return Err(f"definition {definition.full_name!r} has no member tagged {tag!r}")


def definition_find_member_by_index(definition: VibaDefinitionDescriptor, member_index: int) -> Result:
    if 0 <= member_index < len(definition.members):
        return Ok(definition.members[member_index])
    return Err(f"definition {definition.full_name!r} has no member at {member_index}")


def definition_file(definition: VibaDefinitionDescriptor) -> Result:
    return pool_find_file(definition.pool, definition.file_name)


# ----------------------------------------------------------------------
# 成员上的查询
# ----------------------------------------------------------------------


def member_type_name(member: VibaMemberDescriptor) -> Result:
    if member.member_type.kind == TYPE_REF:
        return Ok(member.member_type.payload.type_name)
    return Err(f"member {member.tag!r} is not written as a name")


def member_resolved_definition(member: VibaMemberDescriptor) -> Result:
    """把成员类型里的名字落到定义：先切 import 前缀，再交给 ModuleGetType。"""
    name = member_type_name(member)
    if isinstance(name, Err):
        return name
    pool = member.pool
    containing = pool_find_definition(pool, member.containing_full_name)
    if isinstance(containing, Err):
        return containing
    file = pool_find_file(pool, containing.value.file_name)
    if isinstance(file, Err):
        return file
    written = name.value
    prefix, dot, rest = written.partition(".")
    if dot:
        import_ = file_find_import_by_local_name(file.value, prefix)
        if isinstance(import_, Err):
            return Err(f"{prefix!r} is neither an import nor a module of this file")
        module_name = import_.value.module_name
        target_name = rest
    else:
        module_name = file.value.module_name
        target_name = written
    module = pool.module_environment(module_name)
    if isinstance(module, Err):
        return module
    resolved = module_get_type(module.value, target_name)
    if isinstance(resolved, Err):
        return resolved
    node = getattr(resolved.value, "ast_node", None)
    if not isinstance(node, (ast_nodes.TypeDefinition, ast_nodes.GenericDefinition)):
        return Err(f"{written!r} is not a definition")
    if node.name != target_name:
        return Err(f"{written!r} does not name a definition of {module_name!r}")
    return pool_find_definition(pool, f"{module_name}.{node.name}")


def member_containing_definition(member: VibaMemberDescriptor) -> Result:
    return pool_find_definition(member.pool, member.containing_full_name)


__all__ = [
    "VibaConstantValue",
    "VibaTypeDescriptor",
    "VibaTypeRefDescriptor", "VibaTypeAppDescriptor", "VibaTupleDescriptor",
    "VibaTaggedDescriptor", "VibaChainDescriptor",
    "VibaLiteralDescriptor", "VibaCodeBlockDescriptor",
    "VibaMemberDescriptor", "VibaDefinitionDescriptor",
    "VibaImportDescriptor", "VibaFileDescriptor", "VibaPool",
    "empty_pool",
    "parse_viba_file", "pool_add_file", "pool_find_file",
    "pool_find_definition", "pool_find_member",
    "file_find_import_by_local_name",
    "definition_members", "definition_find_member_by_tag",
    "definition_find_member_by_index", "definition_file",
    "member_type_name", "member_resolved_definition",
    "member_containing_definition",
]

"""is_complete(entry_file_content, library, viba_paths, terminators) -> bool

Treat `entry_file_content` as a design (a `.viba` source): is it complete —
does every address the design can list walk down to a leaf? That is the same
question as "can this design be reflected through" (see viba-reflect.md).

Walking:

- Literals, `nil`, `never`, and the builtin leaves (bool / int / float / str)
  read out, so they end a walk.
- Sum, product and exponent chains behave the same way: the members are the
  elements of the main chain (a unit head does not count), each element is
  walked; tuples are walked element-wise; a tagged piece walks what it tags.
- Names and generic applications resolve through imports and the descriptor
  pool to a definition, whose body is walked next; definitions from the
  builtin library (builtin.viba) count too. A name that does not resolve makes
  the design incomplete. A generic parameter is walked through the actual
  argument an application binds it to; unbound, it is a placeholder and does
  not affect completeness.
- Code blocks, ellipsis, and anything else that has no members but is not in
  terminators make the design incomplete. To end a walk at a code block, name
  the type that wraps it in terminators: the walk then stops at that
  application instead of descending into the code.
- Recursive definitions are coinductive: the same (module, definition,
  arguments) met again while it is being walked counts as fine.

Parameters:

- `entry_file_content`: the entry file's content, taken as the design.
- `library`: (file_path, file_content) pairs that resolve references. The
  module name comes from the path — drop `.viba`, turn `/` into `.`
  (`pkg/mod.viba` is module `pkg.mod`) — so library paths must match the names
  the sources import.
- `viba_paths`: directories to scan recursively for `*.viba`, added to the
  library; a module name is relative to its directory.
- `terminators`: a set of type names (the name of a TypeRef, the constructor
  of a generic application), empty by default. Empty is the strictest: only
  leaves and units end a walk.

Relative module names match the corpora in this repo (`util.viba` is module
`util`); the entry has no path, so its module name is `entry`.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from viba import viba_ast
from viba.type import BUILTIN_MODULE, AstNodeType, Err, Ok
from viba.viba_type_descriptor import (
    empty_pool,
    file_find_import_by_local_name,
    parse_viba_file,
    pool_add_file,
)

ENTRY_FILE_NAME = "entry.viba"
ENTRY_MODULE_NAME = "entry"
BUILTIN_MODULE_NAME = "viba.builtin"

LEAF_NAMES = ("bool", "int", "float", "str")  # scalars a leaf can read
CONTAINER_NAMES = ("list", "set", "dict")  # builtin containers: walk elements
UNIT_NAMES = ("Object", "nil", "Oneof", "never")  # unit chain heads

__all__ = ["is_complete"]


def is_complete(
    entry_file_content: str,
    library: Sequence[Tuple[str, str]] = (),
    viba_paths: Sequence[str] = (),
    terminators: Iterable[str] = frozenset(),
) -> bool:
    """Is this design (entry_file_content) complete?"""
    pool = _add_file(empty_pool(), ENTRY_FILE_NAME, ENTRY_MODULE_NAME, entry_file_content)
    if pool is None:
        return False  # the entry does not even compile
    for file_name, module_name, source in _dependencies(library, viba_paths):
        added = _add_file(pool, file_name, module_name, source)
        if added is not None:  # an unreadable dependency is skipped; needing it
            pool = added      # shows up later as a name that does not resolve
    return _Checker(pool, set(terminators)).complete()


# ----------------------------------------------------------------------
# Compile files, gather dependencies
# ----------------------------------------------------------------------


def _add_file(pool, file_name: str, module_name: str, source: str):
    """Compile one file into the pool; None when it does not compile or the
    module name is taken."""
    parsed = parse_viba_file(pool, source, file_name, module_name)
    if isinstance(parsed, Err):
        return None
    added = pool_add_file(pool, parsed.ok_value)
    if isinstance(added, Err):
        return None
    return added.ok_value


def _dependencies(library, viba_paths) -> List[Tuple[str, str, str]]:
    """(file_name, module_name, source): library first, scanned files after."""
    found: Dict[str, Tuple[str, str]] = {}
    for file_path, file_content in library or ():
        module_name = _module_name(file_path)
        if module_name and module_name not in found:
            found[module_name] = (str(file_path), file_content)
    for directory in viba_paths or ():
        root = Path(directory)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.viba")):
            module_name = _module_name(path.relative_to(root))
            if not module_name or module_name in found:
                continue
            try:
                found[module_name] = (str(path), path.read_text())
            except OSError:
                continue
    return [(file_name, module_name, source)
            for module_name, (file_name, source) in found.items()]


def _module_name(file_path) -> str:
    """pkg/mod.viba -> pkg.mod: drop .viba, turn the separators into dots."""
    text = str(file_path).replace("\\", "/").strip("/")
    if text.endswith(".viba"):
        text = text[: -len(".viba")]
    return text.replace("/", ".")


# ----------------------------------------------------------------------
# The walk
# ----------------------------------------------------------------------


class _Checker:
    """Walks one design with a descriptor pool and a set of terminators."""

    def __init__(self, pool, terminators):
        self.pool = pool
        self.stops = set(terminators)
        self.files = {file.module_name: file for file in pool.files}
        self.definitions: Dict[str, Dict[str, object]] = {}
        self.visiting = set()

    def complete(self) -> bool:
        """Every definition in the entry file has to walk through."""
        for name, definition in self._definitions_of(ENTRY_MODULE_NAME).items():
            if not self._definition(definition, ENTRY_MODULE_NAME):
                return False
        return True

    # ---- resolution ----

    def _definitions_of(self, module_name: str) -> Dict[str, object]:
        if module_name not in self.definitions:
            self.definitions[module_name] = self._read(module_name)
        return self.definitions[module_name]

    def _read(self, module_name: str) -> Dict[str, object]:
        if module_name == BUILTIN_MODULE_NAME:
            return {}  # builtin definitions go through BUILTIN_MODULE.lookup
        module = self.pool.module_environment(module_name)
        if isinstance(module, Err):
            return {}
        return {node.name: node for node in module.ok_value.module.body
                if _is_definition(node)}

    def _resolve(self, name: str, module_name: str):
        """(definition node, module it lives in); None when it does not resolve."""
        local, dot, rest = name.partition(".")
        if dot:
            file = self.files.get(module_name)
            if file is None:
                return None
            imported = file_find_import_by_local_name(file, local)
            if isinstance(imported, Err):
                return None
            module_name, name = imported.ok_value.module_name, rest
        node = self._definitions_of(module_name).get(name)
        if node is not None:
            return node, module_name
        builtin = BUILTIN_MODULE.lookup(name)
        if isinstance(builtin, Ok) and isinstance(builtin.ok_value, AstNodeType):
            return builtin.ok_value.ast_node, BUILTIN_MODULE_NAME
        return None  # not a definition (a builtin scalar and the like)

    # ---- the walk itself ----

    def _walk(self, node, module_name: str, bindings: dict) -> bool:
        if isinstance(node, (viba_ast.Constant, viba_ast.Nil, viba_ast.Never)):
            return True  # shapes a leaf reads out
        if isinstance(node, viba_ast.Ellipsis):
            return "..." in self.stops
        if isinstance(node, viba_ast.CodeBlock):
            return False  # no leaf to read; only the wrapping name can end here
        if isinstance(node, viba_ast.TypeRef):
            return self._name(node.name, module_name, bindings)
        if isinstance(node, viba_ast.TypeApp):
            return self._application(node, module_name, bindings)
        if isinstance(node, viba_ast.Tagged):
            return self._walk(node.type, module_name, bindings)
        if isinstance(node, viba_ast.Tuple):
            return all(self._walk(e, module_name, bindings) for e in node.elements)
        elements = _chain_elements(node)
        if elements is None:
            return False
        if elements and _is_unit_head(elements[0]):
            elements = elements[1:]  # a unit chain head is not a member
        return all(self._walk(e, module_name, bindings) for e in elements)

    def _name(self, name: str, module_name: str, bindings: dict) -> bool:
        if name in bindings:
            bound = bindings[name]
            if bound is None:
                return True  # a generic parameter placeholder: the argument decides
            return self._walk(*bound)
        if name in self.stops:
            return True
        if name in LEAF_NAMES or name in CONTAINER_NAMES or name in UNIT_NAMES:
            return True
        found = self._resolve(name, module_name)
        if found is None:
            return False
        node, module_name = found
        return self._definition(node, module_name)

    def _application(self, node, module_name: str, bindings: dict) -> bool:
        constructor = node.constructor
        if constructor in self.stops:
            return True
        if constructor in CONTAINER_NAMES:
            return all(self._walk(arg, module_name, bindings) for arg in node.args)
        if constructor in LEAF_NAMES or constructor in UNIT_NAMES:
            return True
        caller_module = module_name  # arguments are written at the call site
        found = self._resolve(constructor, caller_module)
        if found is None:
            return False
        definition, target_module = found
        params = list(getattr(definition, "generic_params", None) or [])
        if not isinstance(definition, viba_ast.GenericDefinition):
            return False
        if len(params) != len(node.args):
            return False  # the arity does not match this definition
        inner = dict(bindings)
        for param, arg in zip(params, node.args):
            inner[param] = (arg, caller_module, bindings)
        return self._definition(definition, target_module, inner)

    def _definition(self, node, module_name: str, bindings: dict = None) -> bool:
        bindings = dict(bindings or {})
        for param in getattr(node, "generic_params", None) or ():
            bindings.setdefault(param, None)
        key = (module_name, getattr(node, "name", None), _bindings_key(bindings))
        if key in self.visiting:
            return True  # coinduction: the same walk in progress counts as fine
        self.visiting.add(key)
        try:
            return self._walk(node.body, module_name, bindings)
        finally:
            self.visiting.discard(key)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _is_definition(node) -> bool:
    return isinstance(node, (viba_ast.TypeDefinition, viba_ast.GenericDefinition))


def _is_unit_head(node) -> bool:
    if isinstance(node, (viba_ast.Nil, viba_ast.Never)):
        return True
    return isinstance(node, viba_ast.TypeRef) and node.name in UNIT_NAMES


def _bindings_key(bindings: dict) -> tuple:
    return tuple(sorted(
        (name, "placeholder" if bound is None else (bound[1], viba_ast.dump(bound[0])))
        for name, bound in bindings.items()))


def _chain_elements(node) -> Optional[List]:
    """Sum / product / exponent as the elements of the main chain, in written
    order; None for anything else."""
    if isinstance(node, (viba_ast.SumChain, viba_ast.ProductChain, viba_ast.ExponentChain)):
        return list(node.elements)
    if isinstance(node, (viba_ast.Sum, viba_ast.Product)):
        elements, current = [], node
        while isinstance(current, type(node)):
            elements.append(current.right)
            current = current.left
        elements.append(current)
        elements.reverse()
        return elements
    if isinstance(node, viba_ast.Exponent):
        elements, current = [], node
        while isinstance(current, viba_ast.Exponent):
            elements.append(current.argument)
            current = current.result
        elements.append(current)
        elements.reverse()
        return elements
    return None

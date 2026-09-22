"""The Type model consumed by is_sub_type.

This is the metalanguage layer: Viba types being compared are reified
as values of the `Type` classes below. Structure stays at the viba.viba_ast
layer (wrapped in AstNodeType); this module models leaves, references
and the module machinery they need for lexical resolution.

Decoupling contract: this layer knows how to build Type values from
viba.viba_ast nodes, but nothing about rules, results or compliance.
"""

from pathlib import Path
from typing import Callable, Union

from viba import viba_ast


# ----------------------------------------------------------------------
# Result (cf. Result[T] := Oneof | $ok T | $err str)
# ----------------------------------------------------------------------


class Ok:
    """The $ok branch: its payload is ok_value."""

    def __init__(self, ok_value):
        self.ok_value = ok_value

    def __repr__(self):
        return f"Ok({self.ok_value!r})"


class Err:
    """The $err branch: its payload is err_msg."""

    def __init__(self, err_msg: str):
        self.err_msg = err_msg

    def __repr__(self):
        return f"Err({self.err_msg!r})"


Result = Union[Ok, Err]


# ----------------------------------------------------------------------
# Type leaves
# ----------------------------------------------------------------------


class Type:
    """Base class of the Type universe."""


class NeverType(Type):
    """never — the sum identity (bottom)."""


class AnyType(Type):
    """Any — the top type: every type is a subtype of it."""


class NilType(Type):
    """nil — the product identity (unit). void/None are aliases."""


class BoolType(Type):
    """bool."""


class IntType(Type):
    """int."""


class FloatType(Type):
    """float."""


class StrType(Type):
    """str."""


class BoolLiteralType(Type):
    def __init__(self, value: bool):
        self.value = value


class IntLiteralType(Type):
    def __init__(self, value: int):
        self.value = value


class FloatLiteralType(Type):
    def __init__(self, value: float):
        self.value = value


class StrLiteralType(Type):
    def __init__(self, value: str):
        self.value = value


class BuiltinGenericType(Type):
    """A built-in generic constructor by name (e.g. list).

    Nominal: only same-named constructors are comparable; arguments
    are compared covariantly at the use site (TypeApp layer).
    """

    def __init__(self, name: str):
        self.name = name


# ----------------------------------------------------------------------
# ModuleType
# ----------------------------------------------------------------------


class ModuleType(Type):
    """Base class of module descriptors."""


class BuiltinModuleType(ModuleType):
    """The module that holds built-in types.

    Every lookup is answered by the registry or the builtin
    library (viba/builtin.viba); anything else is an error.
    """

    _BASIC_NAMES = {
        "bool": BoolType,
        "int": IntType,
        "float": FloatType,
        "str": StrType,
        "nil": NilType,
        "void": NilType,  # accepted alias
        "None": NilType,  # accepted alias
        "Object": NilType,  # accepted alias (product identity)
        "never": NeverType,
        "Oneof": NeverType,  # accepted alias (sum identity)
    }
    _GENERIC_NAMES = {
        "List", "list", "set", "dict",
        "ListLiteral", "SetLiteral", "DictLiteral",
    }

    def __init__(self):
        self._library = _load_builtin_library()

    def lookup(self, type_name: str) -> Result:
        if type_name in self._BASIC_NAMES:
            return Ok(self._BASIC_NAMES[type_name]())
        if type_name in self._GENERIC_NAMES:
            return Ok(BuiltinGenericType(type_name))
        if type_name in self._library:
            node = self._library[type_name]
            return Ok(AstNodeType(node, BUILTIN_MODULE))
        return Err(f"no built-in type named {type_name!r}")


def _load_builtin_library() -> dict:
    """Parse viba/builtin.viba into {name: definition node}."""
    path = Path(__file__).with_name("builtin.viba")
    tree = viba_ast.parse(path.read_text())
    kinds = (viba_ast.TypeDefinition, viba_ast.GenericDefinition)
    return {n.name: n for n in tree.body if isinstance(n, kinds)}


# The single shared built-in module instance.
BUILTIN_MODULE = BuiltinModuleType()


class CustomModuleType(ModuleType):
    """A user module: parsed definitions + a lazy environment.

    $module_environment answers `module_name -> Result[ModuleType]`
    for cross-module references; returning Err ends resolution.

    $imports maps a file's import local name to the module it names, so a
    written name that carries an import prefix (`d.Report` under
    `import a.b as d`) lands on the definition it really points at.
    """

    def __init__(
        self,
        module: viba_ast.Module,
        module_environment: Callable[[str], Result] = None,
        imports: dict = None,
    ):
        self.module = module
        if module_environment is None:
            module_environment = lambda name: Err(f"module {name!r} not found")
        self.module_environment = module_environment
        self.imports = dict(imports or {})

    def lookup_local(self, type_name: str) -> Result:
        """Find a top-level definition by name; failing that, through this
        file's imports. The environment is not consulted for either."""
        named = (d for d in self.module.body
            if _is_definition(d) and d.name == type_name)
        found = next(named, None)
        if found is not None:
            return Ok(AstNodeType(found, self))
        return self._lookup_imported(type_name)

    def _lookup_imported(self, type_name: str) -> Result:
        """`d.Name` or `a.b.Name`: the prefix is one of this file's imports, so
        the name is that module's own. What an import binds is its alias when it
        has one and its whole module name when it has none: `import a.b as c`
        answers `c.Name`, `import a.b` answers `a.b.Name`. The longest prefix
        wins, so a dotted module is not read as a shorter one plus a member."""
        parts = type_name.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:cut])
            if prefix not in self.imports:
                continue
            imported = self.module_environment(self.imports[prefix])
            if isinstance(imported, Err):
                return Err(f"{prefix!r} names {self.imports[prefix]!r}: {imported.err_msg}")
            return imported.ok_value.lookup_local(".".join(parts[cut:]))
        return Err(f"no type named {type_name!r} in module")


def _is_definition(node) -> bool:
    return isinstance(node, (viba_ast.TypeDefinition, viba_ast.GenericDefinition))


# ----------------------------------------------------------------------
# AstNodeType — the escape hatch for the structure layer
# ----------------------------------------------------------------------


class AstNodeType(Type):
    """Any structural viba.viba_ast node + the module its TypeRefs resolve in.

    `env_get` models the optional third field of the spec's AstNodeType:
    a fallback resolver for free type names (e.g. generic parameters
    bound by a surrounding TypeApp). Module-local definitions always
    win; env_get only answers names the module cannot resolve.
    """

    def __init__(
        self,
        ast_node: viba_ast.AST,
        container_module: ModuleType,
        env_get: Callable[[str], Result] = None,
    ):
        self.ast_node = ast_node
        self.container_module = container_module
        self.env_get = env_get


# ----------------------------------------------------------------------
# ModuleGetType — Result[Type] <- $module ModuleType <- $type_name str
# ----------------------------------------------------------------------


class UnresolvedTypeError(Exception):
    """A TypeRef could not be resolved to a Type."""


class DuplicateTagError(Exception):
    """One product writes the same tag twice, inlined members counted."""


class PartialError(Exception):
    """A `<<` that cannot be given: the left side is no function, or it has no
    such argument."""


class InlineCycleError(Exception):
    """An inline chain comes back to a definition it is already expanding."""


def module_get_type(module: ModuleType, type_name: str, _seen=None) -> Result:
    """Resolve a type name against a module (ModuleGetType).

    Built-ins are visible from every module: a custom module falls
    back to the built-in registry after its own definitions and
    environment are exhausted.

    The environment is asked for a module too, which is how a nested module's
    own definitions are reached. A module that hands itself (or a module that
    handed us) back would loop — a name that is both a type and a module — so
    the modules met on the way are remembered.
    """
    if isinstance(module, BuiltinModuleType):
        return module.lookup(type_name)
    if isinstance(module, CustomModuleType):
        return _lookup_custom(module, type_name, set() if _seen is None else _seen)
    return Err(f"unknown module kind: {module!r}")


def _lookup_custom(module: CustomModuleType, type_name: str, seen) -> Result:
    local = module.lookup_local(type_name)
    if isinstance(local, Ok):
        return local
    via_env = module.module_environment(type_name)
    # The environment may hand back a fresh module object for the same file, so
    # what is remembered is the syntax tree behind it: one per file.
    if isinstance(via_env, Ok) and id(via_env.ok_value.module) not in seen:
        seen = seen | {id(module.module)}
        return module_get_type(via_env.ok_value, type_name, seen)
    builtin = BUILTIN_MODULE.lookup(type_name)
    if isinstance(builtin, Ok):
        return builtin
    note = via_env.err_msg if isinstance(via_env, Err) else "it names a module already met"
    return Err(f"type {type_name!r} unresolved: {local.err_msg}; {note}")


# ----------------------------------------------------------------------
# Convenience constructors (test-oriented; still compliance-agnostic)
# ----------------------------------------------------------------------


def custom_module(source: str, environment: Callable[[str], Result] = None) -> CustomModuleType:
    """Parse Viba source into a CustomModuleType."""
    return CustomModuleType(viba_ast.parse(source), environment)


def entry_type(source: str, module: ModuleType = None) -> AstNodeType:
    """Wrap an inline type expression as an AstNodeType.

    `source` is a bare expression like `$x int | never`; it is parsed
    as the body of a throwaway definition.
    """
    module = module or CustomModuleType(viba_ast.Module([]))
    tree = viba_ast.parse(f"__entry__ := {source}")
    return AstNodeType(tree.body[0].body, module)

"""The Type model consumed by is_sub_type.

This is the metalanguage layer: Viba types being compared are reified
as values of the `Type` classes below. Structure stays at the viba.ast
layer (wrapped in AstNodeType); this module models leaves, references
and the module machinery they need for nominal resolution.

Decoupling contract: this layer knows how to build Type values from
viba.ast nodes, but nothing about rules, results or compliance.
"""

from typing import Callable, Union

from viba import ast as viba_ast


# ----------------------------------------------------------------------
# Result (cf. Result[T] := $ok T | $err str)
# ----------------------------------------------------------------------


class Ok:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Ok({self.value!r})"


class Err:
    def __init__(self, message: str):
        self.message = message

    def __repr__(self):
        return f"Err({self.message!r})"


Result = Union[Ok, Err]


# ----------------------------------------------------------------------
# Type leaves
# ----------------------------------------------------------------------


class Type:
    """Base class of the Type universe."""


class NeverType(Type):
    """never — the sum identity (bottom)."""


class UnitType(Type):
    """void — the product identity (unit)."""


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
    """A built-in generic constructor by name (e.g. List).

    Nominal: only same-named constructors are comparable; arguments
    are compared covariantly at the use site (TypeApp layer).
    """

    def __init__(self, name: str):
        self.name = name


class PoisonType(Type):
    """AssertionViolated — evidence claims absence; always fails.

    Must never appear on the sup side (lint error if it does).
    """


class OpaqueType(Type):
    """An unresolvable TypeRef, treated as a nominal atom keyed by
    (container module identity, name).

    This is exactly the right semantics for free generic parameters
    (comparing Option[T] against Option[U] compares T against U by
    name, with no substitution). Genuine typos in closed contexts
    simply never match anything.
    """

    def __init__(self, container_module: "ModuleType", name: str):
        self.container_module = container_module
        self.name = name


# ----------------------------------------------------------------------
# ModuleType
# ----------------------------------------------------------------------


class ModuleType(Type):
    """Base class of module descriptors."""


class BuiltinModuleType(ModuleType):
    """The module that holds built-in types.

    Its environment is empty: every lookup is answered by the
    registry, anything else is an error.
    """

    _BASIC_NAMES = {
        "bool": BoolType,
        "int": IntType,
        "float": FloatType,
        "str": StrType,
        "void": UnitType,
        "never": NeverType,
    }
    _GENERIC_NAMES = {"List"}

    def lookup(self, type_name: str) -> Result:
        if type_name in self._BASIC_NAMES:
            return Ok(self._BASIC_NAMES[type_name]())
        if type_name in self._GENERIC_NAMES:
            return Ok(BuiltinGenericType(type_name))
        return Err(f"no built-in type named {type_name!r}")


# The single shared built-in module instance.
BUILTIN_MODULE = BuiltinModuleType()


class CustomModuleType(ModuleType):
    """A user module: parsed definitions + a lazy environment.

    $module_environment answers `module_name -> Result[ModuleType]`
    for cross-module references; returning Err ends resolution.
    """

    def __init__(
        self,
        module: viba_ast.Module,
        module_environment: Callable[[str], Result] = None,
    ):
        self.module = module
        if module_environment is None:
            module_environment = lambda name: Err(f"module {name!r} not found")
        self.module_environment = module_environment

    def lookup_local(self, type_name: str) -> Result:
        """Find a top-level definition by name (no environment fallback)."""
        named = (d for d in self.module.body
            if _is_definition(d) and d.name == type_name)
        found = next(named, None)
        if found is None:
            return Err(f"no type named {type_name!r} in module")
        return Ok(AstNodeType(found, self))


def _is_definition(node) -> bool:
    return isinstance(node, (viba_ast.TypeDefinition, viba_ast.GenericDefinition))


# ----------------------------------------------------------------------
# AstNodeType — the escape hatch for the structure layer
# ----------------------------------------------------------------------


class AstNodeType(Type):
    """Any structural viba.ast node + the module its TypeRefs resolve in."""

    def __init__(self, ast_node: viba_ast.AST, container_module: ModuleType):
        self.ast_node = ast_node
        self.container_module = container_module


# ----------------------------------------------------------------------
# ModuleGetType — Result[Type] <- $module ModuleType <- $type_name str
# ----------------------------------------------------------------------


class UnresolvedTypeError(Exception):
    """A TypeRef could not be resolved to a Type."""


class RuleContainsPoisonError(Exception):
    """AssertionViolated appeared on the sup (Rule) side — lint error."""


def module_get_type(module: ModuleType, type_name: str) -> Result:
    """Resolve a type name against a module (ModuleGetType).

    Built-ins are visible from every module: a custom module falls
    back to the built-in registry after its own definitions and
    environment are exhausted.
    """
    if isinstance(module, BuiltinModuleType):
        return module.lookup(type_name)
    if isinstance(module, CustomModuleType):
        return _lookup_custom(module, type_name)
    return Err(f"unknown module kind: {module!r}")


def _lookup_custom(module: CustomModuleType, type_name: str) -> Result:
    local = module.lookup_local(type_name)
    if isinstance(local, Ok):
        return local
    via_env = module.module_environment(type_name)
    if isinstance(via_env, Ok):
        return module_get_type(via_env.value, type_name)
    builtin = BUILTIN_MODULE.lookup(type_name)
    if isinstance(builtin, Ok):
        return builtin
    return Err(f"type {type_name!r} unresolved: {local.message}; {via_env.message}")


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

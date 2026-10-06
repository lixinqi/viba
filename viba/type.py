"""The Type model consumed by is_sub_type.

This is the metalanguage layer: Viba types being compared are reified
as values of the `Type` classes below. Structure stays at the viba.viba_ast
layer (wrapped in AstNodeType); this module models leaves, references
and the module machinery they need for lexical resolution.

Decoupling contract: this layer knows how to build Type values from
viba.viba_ast nodes, and nothing about how a module runs.
"""

from pathlib import Path
from typing import Callable, List, Optional, Union

from viba import viba_ast


# ----------------------------------------------------------------------
# Result (cf. Result[T] = Oneof | $ok T | $viba_program_err str)
# ----------------------------------------------------------------------

class Ok:
    """The $ok branch: its payload is ok_value."""

    def __init__(self, ok_value):
        self.ok_value = ok_value

    def __repr__(self):
        return f"Ok({self.ok_value!r})"


class Frame:
    """`Frame`: one call site of the chain an error happened in.

        Frame =
            Object
          * $file_path str
          * $lineno int

    `file_path` is the `.viba` file the call is written in, `lineno` the line
    the call is on. The main file is the outermost frame of every chain, and it
    is no call site: its `lineno` is 0.
    """

    def __init__(self, file_path: str, lineno: int):
        self.file_path = file_path
        self.lineno = lineno

    def __eq__(self, other):
        return (isinstance(other, Frame) and self.file_path == other.file_path
                and self.lineno == other.lineno)

    def __hash__(self):
        return hash((self.file_path, self.lineno))

    def __repr__(self):
        return f"Frame({self.file_path!r}, {self.lineno!r})"


Stack = List[Frame]


class InterpretError:
    """`$err InterpretError`: the run stopped, and this says why.

        InterpretError =
            Oneof
          | $viba_program_err ProgramErr           # the program or environment
          | $underlying_viba_op_err UnderlyingOpErr     # a step broke
          | $not_implemented_err UnderlyingOpErr               # nothing implements it
          | $environment_api_invalid_argument_err EnvironmentApiInvalidArgumentErr

    A `Failure` is one payload under two tags: `tag` says which. A `ProgramErr`
    is about the program or the environment itself, so it names no step. An
    `EnvironmentApiInvalidArgumentErr` is about one of the environment's own
    members — the apis viba runs itself, which no `get_func` implements — being
    given something it cannot take.
    """


class VibaProgramErr(InterpretError):
    """`$viba_program_err ProgramErr`: the program or the environment is at fault.

        ProgramErr =
            Object
          * $msg str
          * $stack Stack

    The source does not compile, a name resolves to nothing, a module has no
    `__impl__`, the environment cannot run the file: `msg` says which. `stack` is
    the chain of calls it happened in, outermost first, each frame a call site
    (`Frame`); it names no step — that is what a `Failure` is for.
    """

    def __init__(self, msg: str, stack: Optional[Stack] = None):
        self.msg = msg
        self.stack: Stack = list(stack or [])

    def __repr__(self):
        return f"VibaProgramErr({self.msg!r})"


class EnvironmentApiInvalidArgumentErr(InterpretError, Exception):
    """`EnvironmentApiInvalidArgumentErr`: an environment api was given what it cannot take.

        EnvironmentApiInvalidArgumentErr =
            Object
          * $msg str                               # one sentence, the reason first
          * $api_name str                          # which api: `Environment.sub_env`
          * $args Any                              # what it was given, environment left out

    The environment's members are viba's own side of a run: `get_func` is never
    asked for them, the environment carries them (`args.env.sub_env`,
    `args.env.tmp_env`, the members a host hangs on an environment of its own).
    So one of them refusing what a program handed it — a root that is no ancestor,
    a path with a `.` or `..` segment, an environment with no storage under it —
    is neither a step that failed nor a module that cannot run; it is this: an
    api of the environment, and the arguments it could not take. `api_name` says
    which api, written `Environment.<member>`; `args` says what it was given, as
    viba data — the environment itself is the api's own value, so it is not among
    them. `msg` is one sentence that says what happened and why; it opens with
    the reason (`raised`).

    It is an exception as well, the way every error of this layer is: the same
    news has to cross a callable the run handed to a host.
    """

    def __init__(self, msg: str = "", api_name: str = "", arguments=None):
        super().__init__(msg)
        self.msg = msg
        self.api_name = api_name
        # `$args` — what the api was given. Not `args`: an exception's own `args`
        # is the one `BaseException` keeps, and this is no tuple.
        self.arguments = arguments

    def __repr__(self):
        return f"EnvironmentApiInvalidArgumentErr({self.msg!r}, {self.api_name!r})"


class Err:
    """`$err InterpretError`: the run stopped; `error` says why."""

    def __init__(self, error: InterpretError):
        self.error = error

    def __repr__(self):
        return f"Err({self.error!r})"


Result = Union[Ok, VibaProgramErr]


class UnderlyingOpErr(InterpretError, Exception):
    """`Failure`: this step did not answer. Its two tags are below.

        InterpretResult =
            Oneof
          | $ok VibaNode
          | $viba_program_err str
          | $underlying_viba_op_err UnderlyingOpErr     # it broke
          | $not_implemented_err UnderlyingOpErr               # nothing implements it
          | $environment_api_invalid_argument_err EnvironmentApiInvalidArgumentErr

        UnderlyingOpErr =
            Object
          * $msg str
          * $module_path str
          * $func_name str
          * $call (Any <- $env Env)

    One payload, two tags: a step stopped without answering, and the tag says
    which way — `$not_implemented_err` when `get_func` had nothing for it (the
    ordinary case: this layer ships no library), `$underlying_viba_op_err`
    when it broke. `tag` carries that tag, so the two are told apart without
    reading a message.

    `msg` is one sentence that says what happened and why — it opens with the
    reason (`no implementation`, `refused`, `get_func raised`, `raised`,
    `no leaf`) and goes on with the detail.

    `module_path` and `func_name` are the two `get_func(module_path, func_name)`
    was handed: which definition, at which data path — the same definition at
    another data path is another step. `call` is the call itself in the form that
    can be run again, with the environment left out: `__dyn_call__` and the name
    the host was asked for, or `__dyn_method__` and the member a value carries,
    then the arguments written on it, each with the tag it was written with
    (`__dyn_call__ << "add" << $a 1 << $b 2`). So its type is `Any <- $env Env` —
    a call that still wants its environment, and giving it one runs it, from any
    module, because the name is data. It is viba data, functions and closures
    among the arguments included, so the same call can be made again from this
    result alone, without running anything over; the environment is not part of
    it, another run makes its own (`viba-interpreter.md`).

    It is an exception as well, because that is how the same news crosses a
    callable the run handed to a host; a `get_func` may raise it too, to say it
    has no implementation for a call — raising it with `NOT_IMPLEMENTED_TAG` says
    that, and the run fills in the step and the call it knows, keeping whatever
    the host said.
    """

    def __init__(self, msg: str = "", module_path: str = "", func_name: str = "",
                 call=None, tag: str = None):
        super().__init__(msg)
        self.msg = msg
        self.module_path = module_path
        self.func_name = func_name
        self.call = call
        self.tag = tag or FAILURE_TAG

    def __repr__(self):
        return (f"UnderlyingOpErr({self.msg!r}, {self.module_path!r}, "
                f"{self.func_name!r}, tag={self.tag!r})")


# The two tags a `Failure` answers under: one payload, told apart by the tag.
FAILURE_TAG = "$underlying_viba_op_err"   # the implementation, or `get_func`, broke
NOT_IMPLEMENTED_TAG = "$not_implemented_err"     # nothing implements that step


# The words a failure's `msg` opens with: short, stable, the same ones a reader
# switches on.
REASON_NO_IMPLEMENTATION = "no implementation"   # get_func answered None
REASON_REFUSED = "refused"                       # get_func said so itself
REASON_GET_FUNC_RAISED = "get_func raised"       # get_func broke
REASON_RAISED = "raised"                         # the implementation broke
REASON_NO_LEAF = "no leaf"                       # it answered something with no leaf


# What `interpret` answers, its branches written out:
#
#     InterpretResult =
#         Oneof
#       | $ok VibaNode
#       | $err InterpretError
#
# The error side is `InterpretError` (`$viba_program_err`, an
# `EnvironmentApiInvalidArgumentErr` under `$environment_api_invalid_argument_err`,
# or a `Failure` under one of its two tags). The other APIs keep the two-branch
# `Result`: their work is all here.
InterpretResult = Union[Ok, Err]


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


# The concept in the builtin library that holds the builtin operators: its
# members are written `builtin.add`, and read on their own as `add`.
BUILTIN_CONCEPT = "builtin"


def builtin_directory_name(name: str) -> Optional[str]:
    """The bare name a written name stands for in the builtin directory, or None.

    The built-in vocabulary is the package's own: beside `builtin.viba` sit the
    modules (`Y.viba`, `apply.viba`, `sub_env_run.viba`), and under `builtin/`
    the generics (`builtin/is_closure/`, `builtin/unclosure/`). Those names are
    read from every module, after its own definitions and its imports.
    `sub_env_run` and `builtin.sub_env_run` are the same one, and so are
    `is_closure` and `builtin.is_closure`; a dotted name that carries no
    `builtin.` prefix (`demo.print`) is no builtin name, and neither is `builtin`
    itself. Whether the directory really holds that file is the loader's to say.
    """
    rest = name[len(BUILTIN_CONCEPT) + 1:] if name.startswith(BUILTIN_CONCEPT + ".") else name
    if not rest or "." in rest or rest == BUILTIN_CONCEPT:
        return None
    return rest


class BuiltinModuleType(ModuleType):
    """The module that holds built-in types.

    Every lookup is answered by the registry or the builtin
    library (viba/builtin.viba): the definitions it writes, and the builtin
    operators the concept `builtin` holds — read on their own, `add` is
    `builtin.add`. Anything else is an error.
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
        "tagged",
    }

    def __init__(self):
        self._library = _load_builtin_library()
        self._operators = _load_builtin_operators(self._library)

    def lookup(self, type_name: str) -> Result:
        if type_name in self._BASIC_NAMES:
            return Ok(self._BASIC_NAMES[type_name]())
        if type_name in self._GENERIC_NAMES:
            return Ok(BuiltinGenericType(type_name))
        if type_name in self._library:
            node = self._library[type_name]
            return Ok(AstNodeType(node, BUILTIN_MODULE))
        piece = self.operator(type_name)
        if piece is not None:
            return Ok(AstNodeType(piece, BUILTIN_MODULE))
        return VibaProgramErr(f"no built-in type named {type_name!r}")

    def operator(self, name: str):
        """The written piece of the builtin operator `name`, or None.

        `builtin.add` and `add` are the same member of the concept `builtin`
        (viba/builtin.viba), and this is that member's signature: the chain a
        written call unfolds to, whichever spelling it uses.
        """
        return self._operators.get(name)


def _load_builtin_library() -> dict:
    """Parse viba/builtin.viba into {name: definition node}."""
    path = Path(__file__).with_name("builtin.viba")
    tree = viba_ast.parse(path.read_text())
    kinds = (viba_ast.TypeDefinition, viba_ast.GenericDefinition)
    return {n.name: n for n in tree.body if isinstance(n, kinds)}


def _load_builtin_operators(library: dict) -> dict:
    """{name: written piece} for the tagged members of the builtin concept.

    The members are the builtin operators (`builtin.add`, read on its own as
    `add`), and the concept is one definition in the library: this reads its
    product apart once, so every lookup of an operator is a dictionary read.
    """
    definition = library.get(BUILTIN_CONCEPT)
    if definition is None:
        return {}
    operators = {}
    for factor in _product_factors(definition.body):
        if isinstance(factor, viba_ast.Tagged):
            operators[factor.tag[1:]] = factor.type
    return operators


def _product_factors(node) -> list:
    """The factors of a written product, flattened in written order."""
    if isinstance(node, viba_ast.Product):
        return _product_factors(node.left) + _product_factors(node.right)
    if isinstance(node, viba_ast.ProductChain):
        return list(node.elements)
    return [node]


# The single shared built-in module instance.
BUILTIN_MODULE = BuiltinModuleType()

# The directory the built-in vocabulary lives in: `builtin.viba` and what sits
# next to it (`Y.viba`, `y_helper.viba`, `type.viba`). It is the last stop of the search
# path, so a module that writes `import Y` finds it without naming this
# directory anywhere — the package's own vocabulary is part of the language.
BUILTIN_DIR = Path(__file__).resolve().parent

# The directory the `builtin.` names live in on disk: `builtin/is_closure/` is
# the generic `builtin.is_closure`, whose bare name `is_closure` is read from
# every module too. It is a stop of the search path like `BUILTIN_DIR`, so the
# bare name finds it.
BUILTIN_CONCEPT_DIR = BUILTIN_DIR / BUILTIN_CONCEPT


class CustomModuleType(ModuleType):
    """A user module: parsed definitions + a lazy environment.

    $module_environment answers `module_name -> Result[ModuleType]`
    for cross-module references; returning VibaProgramErr ends resolution.

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
            module_environment = lambda name: VibaProgramErr(f"module {name!r} not found")
        self.module_environment = module_environment
        self.imports = dict(imports or {})

    def lookup_local(self, type_name: str) -> Result:
        """Find a top-level definition by name; failing that, through this
        file's imports. The environment is not consulted for either."""
        found = None
        for node in self.module.body:
            if _is_definition(node) and node.name == type_name:
                found = node          # the parser blocks a name written twice, so take that one
        if found is not None:
            return Ok(AstNodeType(found, self))
        return self._lookup_imported(type_name)

    def _lookup_imported(self, type_name: str) -> Result:
        """`d.Name` or `a.b.Name`: the prefix is one of this file's imports, so
        the name is that module's own. What an import binds is its alias when it
        has one and its whole module name when it has none: `import a.b as c`
        answers `c.Name`, `import a.b` answers `a.b.Name`. The longest prefix
        wins, so a dotted module is not read as a shorter one plus a member.

        A binding that names a generic answers no name at all: a generic is a
        directory of patterns, not a module of definitions, and what it
        has is an application (`gen[T]`, viba-pattern.md)."""
        parts = type_name.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:cut])
            if prefix not in self.imports:
                continue
            imported = self.module_environment(self.imports[prefix])
            if isinstance(imported, VibaProgramErr):
                return VibaProgramErr(f"{prefix!r} names {self.imports[prefix]!r}: {imported.msg}")
            rest = ".".join(parts[cut:])
            if not isinstance(imported.ok_value, CustomModuleType):
                return VibaProgramErr(
                    f"{self.imports[prefix]!r} is a generic: it answers an "
                    f"application ({rest}[T, ...]), not the bare name {type_name!r}")
            return imported.ok_value.lookup_local(rest)
        # The whole name is the binding name (the `c` of `import a.b as c`): not a type but that
        # module — unless it is a generic, and then it stands for an application.
        if len(parts) == 1 and parts[0] in self.imports:
            module_name = self.imports[parts[0]]
            imported = self.module_environment(module_name)
            if isinstance(imported, VibaProgramErr):
                return VibaProgramErr(f"{parts[0]!r} names {module_name!r}: {imported.msg}")
            if not isinstance(imported.ok_value, CustomModuleType):
                return VibaProgramErr(
                    f"{module_name!r} is a generic: it answers an application "
                    f"({parts[0]}[T, ...]), not the bare name {type_name!r}")
            return VibaProgramErr(
                f"{type_name!r} names the module {module_name!r}: a module is no type")
        return VibaProgramErr(f"no type named {type_name!r} in module")


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
        # On a name chain a member read is that name (`a.b.c` reads as `a.b.c`), so the layers that
        # resolve by name (judgment, descriptors) see it; a member read whose left side is an
        # application (`g[T].value`) is no name, and stays as it is for the layer that knows it
        # (viba-style.md).
        if isinstance(ast_node, viba_ast.MemberRead):
            path = viba_ast.written_path(ast_node)
            if path is not None:
                ast_node = viba_ast.TypeRef(path)
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
    return VibaProgramErr(
        f"no type named {type_name!r}: {module!r} is a generic, and a generic "
        f"is applied with [T, ...]")


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
    note = via_env.msg if isinstance(via_env, VibaProgramErr) else "it names a module already met"
    return VibaProgramErr(f"type {type_name!r} unresolved: {local.msg}; {note}")


# ----------------------------------------------------------------------
# Convenience constructors (test-oriented; not tied to how a module runs)
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
    tree = viba_ast.parse(f"__entry__ = {source}")
    return AstNodeType(tree.body[0].body, module)

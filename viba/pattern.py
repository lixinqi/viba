"""Pattern: a generic is a directory, and one of its files answers.

A generic has no body of its own. What it has is a *method*: a type argument as
the source has it, and the answer that argument asks for. Each answer is one file:

    demo/is_base_type/__generic__.viba   the marker: this directory is a generic
    demo/is_base_type/100.viba           pattern bool | int | float | str
    demo/is_base_type/200.viba           pattern A

The file's name is a number — its *decision order*, taken smallest first, and it
need not be contiguous. In front of that number a file may spell how many parts
it takes (`2_200.viba` takes two): the decision counts the parts the arguments in
the source offer and leaves a file its count rules out without taking it.
Inside, one `pattern` line per parameter, in source order. A name the file never
defines is a parameter: what stands in the argument there is extracted. A known
type restricts that argument: the argument must fit it (the judgment layer says
whether it does). Everything else about the file is ordinary — its own
definitions resolve in it, its imports are its own.

Applying the generic takes the files in order and keeps the first whose
patterns fit the arguments as the source has them:

    import demo.is_base_type as is_base_type
    is_bool = is_base_type[bool]        # true, from 100.viba

Nothing fits: that is a program error, not a `never` answer. What the chosen
file answers is its `__decl__` — a type for the design layer, and, when that
type is a function chain and the file gives `__impl__`, the call the file runs
itself where a program runs: a decision is static either way, and a chosen file
that carries its own `__impl__` answers in its own sub-environment, named by its
decision order (viba-pattern.md). The parameter names are bound to the argument
parts as the source has them — each in the module that part comes from in the
source, so a name an enclosing decision bound still stands for what stood at the
call site. `__decl__ = A` answers the extracted type and
`__decl__ = (int <- $env Env <- int)` is a function type like any other.

A tag may also be spelled as a symbol string, and that is where a name that only
exists as a string comes from: `tagged["a", T]` is `$a T`, a `pattern` line
may claim the symbol (`pattern tagged[name, T]` takes `"a"` for `$a int`),
and a `__decl__` builds the tag back out of it (viba-pattern.md). What a string
in the source spells is taken in `viba/viba_ast/tagged.py`; a symbol that is a
name is taken in `tagged_type_of` below, where the decision's bindings are known.

Internal: the layers that parse a design (the interpreter, the judgment, the
descriptor) ask this module; what a caller gives is a generic application in
its own source.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from viba import viba_ast
from viba.partial import (DEF_NAME, is_the_environment, module_as_function,
                          product_elements, reduce_partial)
from viba.viba_ast.tagged import (TAGGED_NAME, literal_symbol, symbol_of,
                                  symbol_problem, tag_of, tagged_node,
                                  tagged_problem)
from viba.type import (AstNodeType, BUILTIN_CONCEPT_DIR, BuiltinGenericType,
                       CustomModuleType, ModuleType, Ok, PartialError, Result,
                       VibaProgramErr, builtin_directory_name, module_get_type)

GENERIC_FILE = "__generic__.viba"

_SUM_NODES = (viba_ast.Sum, viba_ast.SumChain)
_PROD_NODES = (viba_ast.Product, viba_ast.ProductChain)
_EXP_NODES = (viba_ast.Exponent, viba_ast.ExponentChain)

__all__ = [
    "GENERIC_FILE",
    "Choice",
    "GenericModuleType",
    "PatternFile",
    "declared_of",
    "decide",
    "exponent_elements",
    "file_pattern_problem",
    "generic_named",
    "generic_of_entries",
    "load_generic",
    "order_of",
    "patterns_of",
    "reduce_application",
    "structural_pattern_match",
    "sum_elements",
]


# ----------------------------------------------------------------------
# A generic, its files, and what the decision picked
# ----------------------------------------------------------------------


class PatternFile:
    """One file of a generic: the count its name spells, its order, and its text.

    `order` is the number its file name spells; `declared` is the count spelled
    in front of that number (`2_200.viba` declares 2), or None when the name
    spells none; `name` is the module name the file is taken under, so its own
    names resolve in it; `patterns` are the patterns its `pattern` lines give, in
    source order, and `module` is that file as a module.

    The file is taken when the decision reaches it (`take`): a file whose
    declared count the arguments in the source cannot offer is never parsed
    (viba-pattern.md).
    """

    __slots__ = ("order", "path", "name", "declared", "module", "patterns",
                 "_load", "_problem", "_taken", "_checked")

    def __init__(self, order: int, path: str, name: str,
                 declared: Optional[int], load: Optional[Callable[[], Result]]):
        self.order = order
        self.path = path
        self.name = name
        self.declared = declared
        self.module = None                 # until `take`
        self.patterns = None               # until `take`
        self._load = load
        self._problem = None
        self._taken = False
        self._checked = False

    @classmethod
    def ready(cls, order: int, path: str, name: str, declared: Optional[int],
              module: ModuleType, patterns: List[viba_ast.AST]) -> "PatternFile":
        """A file the layer already compiled: the descriptor pool parses its files itself."""
        entry = cls(order, path, name, declared, None)
        entry.module = module
        entry.patterns = list(patterns)
        entry._taken = True
        return entry

    def take(self) -> Result:
        """Ok((module, patterns)) — the file taken once, or why it cannot be taken.

        What the name declares is checked here, where the patterns are at hand: a
        count that its `pattern` lines do not take is a mistake in the name.
        """
        if self._problem is not None:
            return self._problem
        if not self._taken:
            got = self._load()
            if isinstance(got, VibaProgramErr):
                self._problem = got
                return got
            self.module, patterns = got.ok_value
            self.patterns = list(patterns)
            self._taken = True
        if not self._checked:
            self._checked = True
            problem = _declared_problem(self)
            if problem is not None:
                self._problem = VibaProgramErr(problem)
        if self._problem is not None:
            return self._problem
        return Ok((self.module, self.patterns))

    def __repr__(self):
        state = ("not taken" if self.patterns is None
                 else f"{len(self.patterns)} params")
        return f"PatternFile({self.order}, {self.path!r}, {state})"


class GenericModuleType(ModuleType):
    """A generic: the directory whose files are its patterns.

    `name` is the generic's name (the directory's basename), `directory` where
    its files are, and `entries` the pattern files in decision order.
    A generic is no source of type names — a bare generic name is not a type —
    so every layer reaches it through an application, never through a lookup.
    `module` is the marker file taken as a module; it defines nothing, and it is
    here so that a generic answered to code that treats modules alike answers
    "no `__impl__`" instead of breaking.
    """

    def __init__(self, name: str, directory: str, entries: List[PatternFile],
                 module: Optional[viba_ast.Module] = None):
        self.name = name
        self.directory = directory
        self.entries = list(entries)
        self.module = module if module is not None else viba_ast.Module([])

    def __repr__(self):
        return f"GenericModuleType({self.name!r}, {len(self.entries)} patterns)"


class Choice:
    """What the decision picked: the body to take, and the names it binds.

    `body` is the chosen file's `__decl__`, taken in `module` — the file that
    gave it — or None when the file gives none: it is then a module, and a
    member of it is taken by name (`g[T].value`). `bindings` holds every parameter the patterns extracted, each as
    the `AstNodeType` the argument part was. `env_get` is that binding as the
    free-name channel the judgment layer already takes generic parameters
    through (`AstNodeType.env_get`).
    """

    __slots__ = ("entry", "body", "module", "bindings")

    def __init__(self, entry: PatternFile, body: viba_ast.AST,
                 module: ModuleType, bindings: Dict[str, AstNodeType]):
        self.entry = entry
        self.body = body
        self.module = module
        self.bindings = dict(bindings)

    def env_get(self, name: str) -> Result:
        bound = self.bindings.get(name)
        if bound is None:
            return VibaProgramErr(f"unbound parameter {name!r}")
        return Ok(bound)

    def __repr__(self):
        return f"Choice({self.entry.path!r}, {sorted(self.bindings)})"


# ----------------------------------------------------------------------
# Loading a generic's directory
# ----------------------------------------------------------------------


def patterns_of(body) -> List[viba_ast.AST]:
    """The patterns a file's body gives, one per `pattern` line."""
    return [stmt.pattern for stmt in body
            if isinstance(stmt, viba_ast.Pattern)]


def file_pattern_problem(tree, file_name: str) -> Optional[str]:
    """Why this file may not spell `pattern`, or None when it may.

    A pattern is one file of a generic, and its name is its decision order —
    a number, which may have the count it takes spelled in front of it
    (viba-pattern.md). So the name is what tells a pattern file from a module,
    and a file that spells `pattern` under any other name is refused where it is
    compiled: the marker `__generic__.viba` among them, which is no pattern file
    either.
    """
    if not any(isinstance(stmt, viba_ast.Pattern)
               for stmt in getattr(tree, "body", ())):
        return None
    if order_of(file_name) is not None:
        return None
    return (f"{file_name} spells pattern, so it is a pattern file: one of "
            f"the numbered files of its generic's directory")


def load_generic(directory: str, name: str,
                 source_of: Callable[[str], Optional[str]],
                 listing: Callable[[str], Result],
                 module_of: Callable[[str, str, str], Result]) -> Result:
    """Load a generic's whole directory: (GenericModuleType, or why not).

    `directory` holds the files and `name` is what the generic is called.
    `source_of(path)` answers a file's text, or None when that path has no file;
    `listing(directory)` answers the names directly inside it, or a
    VibaProgramErr when the directory cannot be listed; `module_of(path, name,
    source)` compiles one file the way its layer compiles files.

    The marker `__generic__.viba` must be there and must compile; its content is
    otherwise nobody's business. Every other `.viba` file directly inside is a
    pattern: its name is its order and, in front of it, the count it takes, and
    it gives `__decl__`, the type it answers. A `.viba` file named anything else
    is refused — the order is how the decision takes the directory.

    Everything the decision needs to leave a file alone is in its name, so a
    file's text is taken, and parsed, when the decision reaches that file: the
    listing is taken here, the files are not.
    """
    marker = os.path.join(directory, GENERIC_FILE)
    marker_source = source_of(marker)
    if marker_source is None:
        return VibaProgramErr(f"{directory} is no generic: it has no {GENERIC_FILE}")
    try:
        marker_tree = viba_ast.parse(marker_source)
    except SyntaxError as exc:
        return VibaProgramErr(f"cannot parse {marker}: {exc}")
    problem = file_pattern_problem(marker_tree, GENERIC_FILE)
    if problem is not None:
        return VibaProgramErr(f"{marker}: {problem}")

    listed = listing(directory)
    if isinstance(listed, VibaProgramErr):
        return listed
    names = listed.ok_value

    parts = []
    for file_name in sorted(names):
        if file_name == GENERIC_FILE or not file_name.endswith(".viba"):
            continue
        order = order_of(file_name)
        path = os.path.join(directory, file_name)
        if order is None:
            return VibaProgramErr(
                f"{path}: a file of the generic {name!r} is named by its order, "
                f"a number — {file_name!r} is none")
        parts.append((order, path, f"{name}.{order}", declared_of(file_name)))

    return generic_of_files(name, directory, marker_tree, parts, source_of,
                            module_of)


def generic_of_files(name: str, directory: str, marker: Optional[viba_ast.Module],
                     parts, source_of, module_of) -> Result:
    """A generic built from names alone: one `(order, path, module name, declared)`
    per pattern file, each given its text and its module by `source_of` and
    `module_of` when the decision reaches it."""
    entries: List[PatternFile] = []
    for order, path, module_name, declared in parts:
        entries.append(PatternFile(order, path, module_name, declared,
                                   _file_loader(path, module_name, source_of,
                                                module_of)))
    entries.sort(key=lambda entry: entry.order)
    return Ok(GenericModuleType(name, directory, entries, marker))


def _file_loader(path: str, module_name: str, source_of,
                 module_of) -> Callable[[], Result]:
    """How one pattern file is loaded, for the file the decision reaches."""
    def load() -> Result:
        source = source_of(path)
        if source is None:
            return VibaProgramErr(f"cannot load {path}")
        got = module_of(path, module_name, source)
        if isinstance(got, VibaProgramErr):
            return got
        module = got.ok_value
        body = getattr(getattr(module, "module", None), "body", None)
        if body is None:
            return VibaProgramErr(f"{path} is not a module of definitions")
        return Ok((module, patterns_of(body)))
    return load


def generic_of_entries(name: str, directory: str,
                       marker: Optional[viba_ast.Module],
                       parts) -> Result:
    """A generic built from parts the caller already has.

    `parts` is one `(order, path, module name, declared, module)` per pattern
    file, each already compiled by the layer that owns its files. What the file
    declares is taken here — its `pattern` lines in source order, and the
    `__decl__` it answers — so a file means the same thing wherever it was
    compiled. A generic directory a layer serves out of a table rather than out
    of a filesystem goes through here (the descriptor pool does).
    """
    entries: List[PatternFile] = []
    for order, path, module_name, declared, module in parts:
        body = getattr(getattr(module, "module", None), "body", None)
        if body is None:
            return VibaProgramErr(f"{path} is not a module of definitions")
        entries.append(PatternFile.ready(order, path, module_name, declared,
                                         module, patterns_of(body)))
    entries.sort(key=lambda entry: entry.order)
    return Ok(GenericModuleType(name, directory, entries, marker))


def _name_parts(file_name: str):
    """(the count a file name declares, the order it gives), or None when it gives none.

    A pattern file is named by its order, a number, and may spell the count it
    takes in front of it: `2_200.viba` takes two parts and is order 200.
    """
    stem = file_name[: -len(".viba")] if file_name.endswith(".viba") else file_name
    if stem.isdigit():
        return None, int(stem)
    count, underscore, order = stem.partition("_")
    if underscore and count.isdigit() and order.isdigit():
        return int(count), int(order)
    return None


def order_of(file_name: str) -> Optional[int]:
    """The decision order a file name spells, or None when it spells none."""
    parts = _name_parts(file_name)
    return None if parts is None else parts[1]


def declared_of(file_name: str) -> Optional[int]:
    """The count a file name declares, or None when it declares none."""
    parts = _name_parts(file_name)
    return None if parts is None else parts[0]


def _definition(body, name: str):
    """The definition a body gives under this name: the last one."""
    found = None
    for stmt in body:
        if getattr(stmt, "name", None) == name:
            found = stmt
    return found


# ----------------------------------------------------------------------
# Reaching a generic from a name in the source
# ----------------------------------------------------------------------


def generic_named(module: ModuleType, name: str) -> Result:
    """The generic a name in the source names: an import, or the builtin library.

    Ok(the generic) when it is one, Ok(None) when the name is no import of this
    module, is a definition of its own, or names something that is no generic,
    and VibaProgramErr when the binding is there and the module behind it cannot
    be loaded — the caller reports that where it happened rather than taking the
    name as something else. The generics under the builtin library's `builtin/`
    directory (`is_closure`, `unclosure`) are reached from every module, the way
    the modules beside them are: no import is needed, and `builtin.<name>` names
    the same one.

    A generic is reached the way any module is: by what an import binds, the
    longest binding first. `import demo.is_base_type as is_base_type` answers
    `is_base_type`; a name that carries more than the binding names modules
    (`import pkg` then `pkg.mod.G`), so the rest is looked for in the module the
    binding hands over.
    """
    if not isinstance(module, CustomModuleType):
        return Ok(None)
    imports = module.imports
    for prefix in sorted(imports, key=len, reverse=True):
        if name == prefix:
            return _generic_of(module, imports[prefix])
        if name.startswith(prefix + "."):
            rest = name[len(prefix) + 1:]
            handed = module.module_environment(imports[prefix])
            if isinstance(handed, VibaProgramErr):
                return handed
            if isinstance(handed.ok_value, GenericModuleType):
                return Ok(None)                 # a member of a generic is not one
            return generic_named(handed.ok_value, rest)
    if isinstance(module.lookup_local(name), Ok):
        return Ok(None)                 # this module's own definition of that name
    # The generics under the builtin library's `builtin/` directory
    # (`is_closure`, `unclosure`) are reached from every module, the way the
    # modules beside them are: no import is needed, and `builtin.is_closure`
    # names the same one.
    builtin_name = _builtin_generic_name(name)
    if builtin_name is not None:
        return _generic_of(module, builtin_name)
    return Ok(None)


_BUILTIN_GENERIC_NAMES = None


def _builtin_generic_name(name: str):
    """The builtin generic a name in the source stands for, or None.

    Both spellings name it: the bare `is_closure` and `builtin.is_closure`, the
    way `sub_env_run` and `builtin.sub_env_run` name one module. The directory
    `builtin/` is taken once and remembered; a name it does not hold is no
    builtin name, and the caller takes that name the way it always did
    (`builtin.add` is a member of the concept rather than a generic of the
    directory).
    """
    rest = builtin_directory_name(name)
    if rest is None:
        return None
    global _BUILTIN_GENERIC_NAMES
    if _BUILTIN_GENERIC_NAMES is None:
        _BUILTIN_GENERIC_NAMES = {
            one.parent.name
            for one in BUILTIN_CONCEPT_DIR.glob(f"*/{GENERIC_FILE}")}
    return rest if rest in _BUILTIN_GENERIC_NAMES else None


def _generic_of(module: ModuleType, module_name: str) -> Result:
    """Ok(the module) when it is a generic, Ok(None) when it is something else."""
    handed = module.module_environment(module_name)
    if isinstance(handed, VibaProgramErr):
        return handed
    if isinstance(handed.ok_value, GenericModuleType):
        return Ok(handed.ok_value)
    return Ok(None)


def tagged_type_of(node, module: ModuleType, resolve=None) -> Result:
    """The tag a `tagged[...]` in the source stands for, or Ok(None) when it is none.

    `tagged[S, T]` is the tagged type `$S T` and `tagged[S]` is the
    member `$S`, so a symbol can be spelled where only a tag would otherwise
    fit (`viba/viba_ast/tagged.py`). A string literal in the source is already
    folded where the source was parsed; what is taken here is the symbol that is
    a *name* — a `pattern` line's parameter, or one a decision bound.

    `resolve(name, module)` answers the string a name in the source stands for,
    or None when this layer cannot take it there (the judgment resolves through
    the bindings a decision left on the node, the other layers through their
    own). The default resolves the name in the module, a definition's body
    included.

    Ok(None) when this is no tagged application at all, or when its symbol is a
    name this layer cannot take: the caller then takes the application its own
    way, and reports it where it lands. VibaProgramErr when the application is
    no tag: the wrong number of arguments, or a string that spells no symbol.
    """
    if not (isinstance(node, viba_ast.TypeApp) and node.constructor == TAGGED_NAME):
        return Ok(None)
    problem = tagged_problem(node.args)
    if problem is not None:
        return VibaProgramErr(problem)
    symbol = literal_symbol(node.args[0])
    if symbol is None:
        name = node.args[0]
        if not isinstance(name, viba_ast.TypeRef):
            return VibaProgramErr(
                f"{TAGGED_NAME} asks for a symbol spelled as a string, not "
                f"{viba_ast.unparse_type(name)}")
        text = (resolve(name.name, module) if resolve is not None
                else _named_symbol(name.name, module))
        if text is None:
            return Ok(None)
        symbol = symbol_of(text)
        if symbol is None:
            return VibaProgramErr(symbol_problem(text))
    return Ok(tagged_node(symbol, node.args))


def _named_symbol(name: str, module: ModuleType):
    """The string a name in the source stands for here, or None."""
    node, _source_module = _unfold(viba_ast.TypeRef(name), module)
    if isinstance(node, viba_ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def reduce_application(node, module: ModuleType, argument_modules=None,
                       argument_parts=None) -> Result:
    """The chosen body for a generic application, or Ok(None) when it is none.

    `node` is the `G[A, B]` in the source and `module` the module that gave it,
    so the constructor resolves where it was spelled. Ok(None) says the
    constructor is no generic of this module (an ordinary type application, a
    builtin container among them), which the caller takes the way it always
    did; a VibaProgramErr is a decision that failed, or a binding that cannot
    be loaded, and says so.

    `argument_modules` names, for each argument in the source, the module it was
    spelled in. A name a decision bound stands for the argument part that stood
    at the call site, and that part was spelled there, so an argument that is
    such a name is taken in its own module (`decide`). Without it every argument
    is taken in `module`.

    `argument_parts` names, for each argument, the module each of its members was
    spelled in, where the argument is a product built out of values: a member then
    means what it meant where it was made, not what it means in the module that
    piled the members up. Without it every member of an argument is taken in the
    argument's own module.
    """
    if not isinstance(node, viba_ast.TypeApp):
        return Ok(None)
    generic = generic_named(module, node.constructor)
    if isinstance(generic, VibaProgramErr):
        return generic
    if generic.ok_value is None:
        return Ok(None)
    return decide(generic.ok_value, node.args, module,
                  argument_modules=argument_modules,
                  argument_parts=argument_parts)


def decide(generic: GenericModuleType, arguments: List[viba_ast.AST],
           argument_module: ModuleType, argument_modules=None,
           argument_parts=None) -> Result:
    """The first file whose patterns fit, or why none does.

    A generic application is a call made at design time: the chosen file's
    `pattern` parameters are replaced by the arguments the call site gave, so a
    parameter's own name never matters. The files are taken in decision order —
    the numbers, smallest first. A file whose `pattern` line count is not the
    argument count cannot be the one, so it is passed over; the first file every
    pattern fits is the answer. Nothing fitting is a program error: the decision
    failed, and a generic with no answer is no design.

    A file whose name spells how many parts it takes is passed over without
    being taken when the arguments cannot offer that many (`_offered_parts`):
    the count is the bucket the decision jumps to, and the files of one generic
    are then taken only where they can matter. A file the count cannot rule out
    is taken, and its patterns are what decides — the count in the name only
    spares the taking. A sum among the arguments offers no count at all — its
    parts are the branches it gave, and a sum pattern takes exactly those — so
    the application takes every file, and no file of a sum may spell a count in
    its name either.

    Each argument is taken in the module it was spelled in (`argument_modules`;
    `argument_module` where that is not said), because an argument spelled as a
    name a decision bound stands for the part that stood at the call site, and
    that part's own names resolve there.
    """
    where = (list(argument_modules) if argument_modules
             else [argument_module] * len(arguments))
    counted = any(entry.declared is not None for entry in generic.entries)
    offered = _offered_parts(arguments, where) if counted else None
    for entry in generic.entries:
        if (offered is not None and entry.declared is not None
                and entry.declared not in offered):
            continue
        got = entry.take()
        if isinstance(got, VibaProgramErr):
            return got
        module, patterns = got.ok_value
        if len(patterns) != len(arguments):
            continue
        bindings: Dict[str, AstNodeType] = {}
        fits = True
        members = (list(argument_parts) if argument_parts
                   else [None] * len(arguments))
        for pattern, argument, source_module, member_modules in zip(
                patterns, arguments, where, members):
            got = structural_pattern_match(pattern, module, argument,
                                           source_module, bindings,
                                           parts=member_modules)
            if isinstance(got, VibaProgramErr):
                return got
            if got.ok_value is None:
                fits = False
                break
            bindings = got.ok_value
        if not fits:
            continue
        # A file with no `__decl__` puts its "answer" in a named definition (`value`, `impl`): it is
        # its own module, and the caller spells which definition it takes (module semantics).
        declared = _definition(module.module.body, DEF_NAME)
        body = declared.body if declared is not None else None
        return Ok(Choice(entry, body, module, bindings))

    # Nothing fit: every file is taken, whatever its count said, so the answer names what
    # the directory holds and a file that cannot be taken is reported here.
    arities = set()
    for entry in generic.entries:
        got = entry.take()
        if isinstance(got, VibaProgramErr):
            return got
        arities.add(len(entry.patterns))
    arities = sorted(arities)

    spelled = ", ".join(viba_ast.unparse_type(argument) for argument in arguments)
    if arities and len(arguments) not in arities:
        return VibaProgramErr(
            f"generic {generic.name!r} takes {_counted(arities)} parameters, "
            f"not {len(arguments)}: [{spelled}]")
    return VibaProgramErr(
        f"no pattern of {generic.name!r} matches [{spelled}]: "
        f"the decision failed")


# The most totals an application's parts are counted in; past that the walk is not worth
# it, and no file is passed over (viba-pattern.md).
_MOST_TOTALS = 8


def _offered_parts(arguments: List[viba_ast.AST], where) -> Optional[set]:
    """Every number of parts these arguments offer, or None when they offer no count.

    A call in the source offers the arguments it was given, a product its
    members, a chain its positions, a tuple its elements, an application its
    arguments, a tag one part; an argument that unfolds to a structure offers
    that structure's count as well. Totals are summed, since a file takes its
    patterns one per argument in the source. Anything that offers no count — a
    leaf, a name, and above all a sum, whose parts are the branches the argument
    itself gave — leaves the whole application uncounted, and then no file is
    passed over: counting a sum would skip a file that answers sums of other
    branch counts.
    """
    totals = {0}
    for argument, source_module in zip(arguments, where):
        counts = _parts_of_argument(argument, source_module)
        if not counts:
            return None
        totals = {total + count for total in totals for count in counts}
        if len(totals) > _MOST_TOTALS:
            return None
    return totals


def _parts_of_argument(argument, module: ModuleType) -> set:
    """The numbers of parts this argument offers, in every way it may be taken.

    A call counts as the call (`F << A` counts the links it was given) and,
    where it can be given at all, as the chain it stands for; every other
    argument counts as the structure it unfolds to.
    """
    counts = set()
    if isinstance(argument, viba_ast.Partial):
        counts.add(len(_application_parts(argument)[1]))
    try:
        node, _where = _argument_as_type(argument, module)
    except (_BadPattern, PartialError):
        node = None                 # the matching reports it where it lands, if it does
    if node is not None:
        count = _parts_of_node(node)
        if count is not None:
            counts.add(count)
    return counts


def _parts_of_node(node) -> Optional[int]:
    """How many parts this type is taken apart into, or None when it is not.

    A sum is None on purpose: only a sum `pattern` takes a sum apart, and it
    takes one part per branch the *argument* gave, so the number is the
    argument's, not the type's (`_parts_of_pattern`). Counting a sum would put a
    number in the name that the same file contradicts on the next application.
    """
    if isinstance(node, _SUM_NODES):
        return None
    if isinstance(node, _PROD_NODES):
        return len(product_elements(node))
    if isinstance(node, _EXP_NODES):
        return len(exponent_elements(node))
    if isinstance(node, viba_ast.Tuple):
        return len(node.elements)
    if isinstance(node, viba_ast.Tagged):
        return 1
    if isinstance(node, viba_ast.TypeApp):
        return len(node.args)
    return None


def _parts_of_pattern(pattern, module: ModuleType) -> Optional[int]:
    """How many parts this `pattern` line takes, or None when it takes no count.

    The number is what the matching below takes apart: a call in the source
    takes its links, a product its members, a chain its positions, a tuple its
    elements, an application its arguments, a tag one part. A pattern that takes
    no fixed count — a bare name, a type in the source with no parameter, a sum
    — takes whatever it is given, and a file that spells one may not declare a
    count.

    A sum is the one that is not merely unfixed but *harmful* to spell: the
    matching takes a sum argument apart into its branches, so the count belongs
    to the argument, and one file answers sums of different branch counts —
    `pattern A | B | C` fits both `int | str` and `int | str | bool`. Any single
    number in the name would then be wrong for one of them.
    """
    if isinstance(pattern, viba_ast.Ellipsis):
        return None
    if isinstance(pattern, viba_ast.TypeApp) and pattern.constructor == TAGGED_NAME:
        return 1
    if isinstance(pattern, viba_ast.Partial):
        return len(_application_parts(pattern)[1])
    if _is_parameter(pattern, module):
        return None
    if not _has_parameter(pattern, module):
        return None
    if isinstance(pattern, _SUM_NODES):
        return None                 # one part per branch the argument gave
    if isinstance(pattern, _PROD_NODES):
        return len(product_elements(pattern))
    if isinstance(pattern, _EXP_NODES):
        return len(exponent_elements(pattern))
    if isinstance(pattern, viba_ast.Tuple):
        return len(pattern.elements)
    if isinstance(pattern, viba_ast.Tagged):
        return 1
    if isinstance(pattern, viba_ast.TypeApp):
        return len(pattern.args)
    return None


def _declared_problem(entry: PatternFile) -> Optional[str]:
    """Why the count in this file's name disagrees with its patterns, or None."""
    if entry.declared is None:
        return None
    taken = 0
    for pattern in entry.patterns:
        count = _parts_of_pattern(pattern, entry.module)
        if count is None:
            return (f"{entry.path}: the name says the file takes "
                    f"{entry.declared} parts, and a `pattern` line in it takes "
                    f"no count")
        taken += count
    if taken != entry.declared:
        return (f"{entry.path}: the name says the file takes {entry.declared} "
                f"parts, and its `pattern` lines take {taken}")
    return None


def _counted(arities: List[int]) -> str:
    """`1`, `1 or 2`, `1, 2 or 3` — how many parameters the files declare."""
    spelled = [str(number) for number in arities]
    if len(spelled) == 1:
        return spelled[0]
    return " or ".join([", ".join(spelled[:-1]), spelled[-1]])


# ----------------------------------------------------------------------
# Matching one pattern against one type in the source
# ----------------------------------------------------------------------


class _BadPattern(Exception):
    """The pattern itself cannot be taken: this is a mistake, not a mismatch."""


def structural_pattern_match(pattern, pattern_module: ModuleType, argument,
                             argument_module: ModuleType,
                             bindings: Optional[Dict[str, AstNodeType]] = None,
                             parts=None) -> Result:
    """Bind this pattern's parameters to what the argument has in their place.

    Ok(bindings) when the argument fits the pattern: the dict holds every name
    the pattern extracted, each as the `AstNodeType` that stood in the argument
    (spelled where the argument was spelled). Ok(None) when it does not fit —
    the next file is then tried. VibaProgramErr when the pattern cannot be
    taken at all.

    A name the pattern's module never defines is a parameter. Everywhere else
    the pattern is a type in the source, and the argument is judged against it
    with `is_sub_type` — the judgment layer's own judgment, so `true` fits
    `bool` and `int` fits `bool | int`. Where a parameter sits inside structure
    (`list[A]`, `A <- (() | nil)`), the argument is taken apart by the same
    structure, part by part, and each parameter takes the part it stands for.

    A name that appears twice is one parameter, so the second part has to be the
    same type as the first (`pattern A` twice is an equality). `bindings`
    carries what an earlier pattern already extracted, so two `pattern`
    lines of one file share their parameters.

    An argument that spells a call (`add << $a 2`) counts as the type that call
    stands for — the chain with that argument already given — before it is taken
    apart, so a claim about the argument's positions reaches a function a
    caller spelled by giving an argument (viba-pattern.md). A call that cannot
    be given at all is a mistake in the argument, reported here rather than
    counted as a mismatch.
    """
    bindings = {} if bindings is None else bindings
    try:
        matched = _match(pattern, pattern_module, argument, argument_module,
                         bindings, parts)
    except _BadPattern as exc:
        return VibaProgramErr(str(exc))
    except PartialError as exc:
        return VibaProgramErr(str(exc))
    return Ok(matched)


def _match(pattern, pattern_module: ModuleType, argument,
           argument_module: ModuleType, bindings: Dict[str, AstNodeType],
           parts=None):
    """The bindings this part fits with, or None when it does not fit.

    `parts` names, for an argument that is a product, the module each of its
    members was spelled in: the first split of that product hands each member its
    own module, and deeper levels take the module they inherit.
    """
    if isinstance(pattern, viba_ast.Ellipsis):
        raise _BadPattern(
            "ellipsis is not allowed in a `pattern` line: a pattern is one type")
    if isinstance(pattern, viba_ast.TypeApp) and pattern.constructor == TAGGED_NAME:
        return _match_tagged(pattern, pattern_module, argument, argument_module,
                             bindings, parts)
    if isinstance(pattern, viba_ast.Partial):
        return _match_call(pattern, pattern_module, argument, argument_module,
                           bindings)
    if _is_parameter(pattern, pattern_module):
        name = pattern.name
        if name in bindings:
            if _same_type(bindings[name], AstNodeType(argument, argument_module)):
                return bindings
            return None
        bindings[name] = AstNodeType(argument, argument_module)
        return bindings

    if not _has_parameter(pattern, pattern_module):
        # Nothing to extract here: the type in the source is the whole question,
        # and the judgment layer answers it.
        return bindings if _judge(AstNodeType(argument, argument_module),
                                  AstNodeType(pattern, pattern_module)) else None

    node, module = _argument_as_type(argument, argument_module)
    if isinstance(pattern, _SUM_NODES):
        return _match_sum(pattern, pattern_module, node, module, bindings)
    if isinstance(pattern, _PROD_NODES):
        return _match_parts(product_elements(pattern), pattern_module,
                            product_elements(node) if isinstance(node, _PROD_NODES) else [node],
                            module, bindings, parts)
    if isinstance(pattern, _EXP_NODES):
        return _match_parts(exponent_elements(pattern), pattern_module,
                            exponent_elements(node) if isinstance(node, _EXP_NODES) else [node],
                            module, bindings)
    if isinstance(pattern, viba_ast.Tuple):
        if not isinstance(node, viba_ast.Tuple):
            return None
        return _match_parts(pattern.elements, pattern_module, node.elements,
                            module, bindings)
    if isinstance(pattern, viba_ast.Tagged):
        if not isinstance(node, viba_ast.Tagged) or node.tag != pattern.tag:
            return None
        return _match(pattern.type, pattern_module, node.type, module, bindings)
    if isinstance(pattern, viba_ast.TypeApp):
        if not isinstance(node, viba_ast.TypeApp):
            return None
        if _constructor_key(pattern.constructor, pattern_module) != \
                _constructor_key(node.constructor, module):
            return None
        return _match_parts(pattern.args, pattern_module, node.args, module, bindings)
    return None


def _match_call(pattern, pattern_module: ModuleType, argument,
                argument_module: ModuleType, bindings: Dict[str, AstNodeType]):
    """`F << A`: the closure a call in the source is.

    A pattern spelled as an application matches an argument spelled as a call
    that has not been given its environment — a closure. `F` takes the head (the
    api the call is of), and the pattern's remaining links take the arguments
    the caller gave, one for one, in source order: a pattern spells exactly as
    many links as the call has arguments, so `F << A` takes a one-argument call
    and `F << A << B` a two-argument one — one file per length, the way
    `apply_impl` takes a product of a given size. A call that was given an
    environment is no closure (giving it is what runs it), and a bare name or a
    leaf is no call at all.
    """
    if not isinstance(argument, viba_ast.Partial):
        return None
    head, given = _application_parts(argument)
    if any(is_the_environment(one, argument_module, _partial_judge)
           for one in given):
        return None
    wanted_head, wanted = _application_parts(pattern)
    if len(given) != len(wanted):
        return None                     # the pattern spells exactly this many
    matched = _match(wanted_head, pattern_module, head, argument_module, bindings)
    if matched is None:
        return None
    for link, one in zip(wanted, given):
        matched = _match(link, pattern_module, one, argument_module, matched)
        if matched is None:
            return None
    return matched


def _application_parts(node):
    """A call in the source taken apart: (the head, the arguments in source order)."""
    given = []
    while isinstance(node, viba_ast.Partial):
        given.append(node.argument)
        node = node.function
    return node, list(reversed(given))


def _match_tagged(pattern, pattern_module: ModuleType, argument, argument_module,
                  bindings: Dict[str, AstNodeType], parts=None):
    """`tagged[S, T]`: the argument has to be the tag S spells.

    `S` is either a symbol in the source — the tag it has to be — or a
    parameter, which takes the symbol as a string. Taking it as a string is what
    lets the same design hand it back to `tagged` and build the tag again:

        pattern A <- tagged[arg_name, T]           # arg_name is "a" for `$a int`
        __decl__ = A <- int <- tagged[arg_name, T]  # and this is `$a int` again

    One argument in the source claims the member `$S`, two the tagged type `$S T`
    (viba-pattern.md).

    `parts` names, for an argument that is a product, the module each of its
    members was spelled in. A product of one member is that member, so the value
    under the tag means what it meant where that member was made.
    """
    problem = tagged_problem(pattern.args)
    if problem is not None:
        raise _BadPattern(problem)
    node, module = _argument_as_type(argument, argument_module)
    wanted = literal_symbol(pattern.args[0])
    if wanted is not None:
        if not isinstance(node, viba_ast.Tagged) and not isinstance(node, viba_ast.Member):
            return None
        if node.tag != tag_of(wanted):
            return None
    elif isinstance(node, viba_ast.Tagged) or isinstance(node, viba_ast.Member):
        taken = _match(pattern.args[0], pattern_module,
                       viba_ast.Constant(node.tag[1:]), pattern_module, bindings)
        if taken is None:
            return None
        bindings = taken
    else:
        return None
    if len(pattern.args) == 1:
        return bindings
    if not isinstance(node, viba_ast.Tagged):
        return None
    if parts and len(parts) == 1 and parts[0]:
        module = parts[0]
    return _match(pattern.args[1], pattern_module, node.type, module, bindings)


def _match_parts(patterns, pattern_module: ModuleType, arguments, argument_module,
                 bindings: Dict[str, AstNodeType], parts=None):
    """Parts against parts, in order: every pair has to fit, sharing bindings.

    `parts` names, for each part, the module it was spelled in, where that is
    known — a member of a product built out of values keeps the module it was made
    in. A part without one is taken in the module the whole argument was spelled in.
    """
    if len(patterns) != len(arguments):
        return None
    for index, (pattern, argument) in enumerate(zip(patterns, arguments)):
        own = (parts[index] if parts and index < len(parts) and parts[index]
               else argument_module)
        found = _match(pattern, pattern_module, argument, own, bindings)
        if found is None:
            return None
        bindings = found
    return bindings


def _match_sum(pattern, pattern_module: ModuleType, node, module: ModuleType,
               bindings: Dict[str, AstNodeType]):
    """A sum pattern: every branch of the argument fits some branch of it.

    A branch that fits is one branch; an argument that is no sum is one branch
    of its own. Branches are tried in source order, and the first fit is the
    one whose extractions count.
    """
    branches = sum_elements(node) if isinstance(node, _SUM_NODES) else [node]
    for branch in branches:
        found = None
        for alternative in sum_elements(pattern):
            attempt = _match(alternative, pattern_module, branch, module,
                             dict(bindings))
            if attempt is not None:
                found = attempt
                break
        if found is None:
            return None
        bindings = found
    return bindings


# ----------------------------------------------------------------------
# Names, structure, and what fits
# ----------------------------------------------------------------------


def sum_elements(node) -> List[viba_ast.AST]:
    """The branches of a sum in the source, flattened in source order."""
    if isinstance(node, viba_ast.Sum):
        return sum_elements(node.left) + sum_elements(node.right)
    if isinstance(node, viba_ast.SumChain):
        return list(node.elements)
    return [node]


def exponent_elements(node) -> List[viba_ast.AST]:
    """A function in the source taken apart: the result first, the arguments after.

    `A <- B <- C` is `[A, B, C]` — one element per position, in source order,
    which is how the judgment layer takes a chain too. A chain that ends in a
    body — documentation, or the call the chain ends on — counts as the function
    it is: that last element is no position (`parameters_of` and `_slots_of`
    drop it the same way).
    """
    elements = _exponent_elements(node)
    if len(elements) > 1 and isinstance(elements[-1],
                                        (viba_ast.CodeBlock, viba_ast.Partial)):
        elements = elements[:-1]
    return elements


def _exponent_elements(node) -> List[viba_ast.AST]:
    """The elements of an exponent chain in the source, body and all."""
    if isinstance(node, viba_ast.Exponent):
        return _exponent_elements(node.result) + [node.argument]
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    return [node]


def _is_parameter(node, module: ModuleType) -> bool:
    """A name the pattern's module never defines: a parameter to extract.

    Anything else is a type in the source: a definition of that module, a
    builtin name, a name reached through an import. The rule is the design's own
    — a name is what it resolves to, and a name that resolves to nothing is
    standing for whatever the argument has there (viba-pattern.md).
    """
    if not isinstance(node, viba_ast.TypeRef):
        return False
    return isinstance(module_get_type(module, node.name), VibaProgramErr)


def _has_parameter(node, module: ModuleType) -> bool:
    """Whether a parameter sits anywhere in this pattern."""
    return any(_is_parameter(part, module) for part in viba_ast.walk(node))


def _argument_as_type(node, module: ModuleType):
    """The argument as a type: its names unfolded, its `<<` in the source given.

    `add << $a 2` is `add` with that argument already given, so as a type it is
    the chain that is left — the same way the judgment layer takes a call in the
    source (`viba.partial.reduce_partial`). A pattern that takes the argument
    apart needs that chain rather than the `<<` that produces it, while every
    argument that spells no `<<` stays exactly as it stands in the source:
    a chain ending in `()` is a chain with an empty product in it, not the
    result alone.
    """
    node, module = _unfold(node, module)
    tagged = tagged_type_of(node, module)
    if isinstance(tagged, VibaProgramErr):
        raise _BadPattern(tagged.msg)
    if tagged.ok_value is not None:
        node = tagged.ok_value
    if not isinstance(node, viba_ast.Partial):
        return node, module
    return reduce_partial(node, module, _partial_target, _partial_judge)


def _partial_target(name, module: ModuleType):
    """(body, source module) for the name a `<<` in the source gives to, or None.

    A definition is itself, and a bare import name is the module taken as a
    function (`module_as_function`) — the same two answers the judgment layer
    gives a call's head, kept in `viba.partial` so a call in the source means
    one thing wherever a design is taken.
    """
    resolved = module_get_type(module, name)
    if isinstance(resolved, Ok) and isinstance(resolved.ok_value, AstNodeType):
        target = resolved.ok_value
        body = target.ast_node
        if isinstance(body, viba_ast.GenericDefinition):
            return None                     # a bare generic name has no body
        if isinstance(body, viba_ast.TypeDefinition):
            body = body.body
        return body, target.container_module
    return module_as_function(module, name)


def _partial_judge(given, given_module, slot, slot_module) -> bool:
    """Does the argument a `<<` gives fit the slot it is spelled at?"""
    return _judge(AstNodeType(given, given_module),
                  AstNodeType(slot, slot_module))


def _unfold(node, module: ModuleType):
    """A name is transparent: unfold it to the structure it stands for.

    The argument is taken the way every layer takes a name — an alias of an
    alias is the body at the end of the chain — because what a pattern matches
    is the type, not the spelling. A generic's own name is left standing: a
    bare generic has no body to unfold. The module that comes back is the one
    the unfolded body was spelled in, which is where its own names mean
    something.
    """
    seen = set()
    while isinstance(node, viba_ast.TypeRef):
        if node.name in seen:
            return node, module
        seen.add(node.name)
        resolved = module_get_type(module, node.name)
        if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
            # A module name: what it is as a type is the `__decl__` it gave, taken
            # as that chain (the environment position included — this takes the
            # type as the source has it, not the "module as a function" one with
            # the environment dropped).
            from_module = _module_def_body(module, node.name)
            if from_module is None:
                return node, module
            return from_module
        target = resolved.ok_value
        body = target.ast_node
        if isinstance(body, viba_ast.GenericDefinition):
            return node, module
        if isinstance(body, viba_ast.TypeDefinition):
            body = body.body
        node, module = body, target.container_module
    return node, module


def _module_def_body(module: ModuleType, name: str):
    """(the `__decl__` chain of the module this name binds, that module) or None.

    A name in the source may be a module — `import fib_module as F` then `F` —
    and what that module *is* as a type is the function it gave in `__decl__`,
    the environment position included. None when the name is no import here, or
    when what it binds is not a module (a generic has no `__decl__` of its own).
    """
    imports = getattr(module, "imports", None) or {}
    if name not in imports:
        return None
    handed = module.module_environment(imports[name])
    if not isinstance(handed, Ok) or isinstance(handed.ok_value, GenericModuleType):
        return None
    body = getattr(getattr(handed.ok_value, "module", None), "body", None)
    if body is None:
        return None
    definition = _definition(body, DEF_NAME)
    if definition is None:
        return None
    return definition.body, handed.ok_value


def _constructor_key(constructor: str, module: ModuleType) -> str:
    """What a constructor names, as something two sides compare by.

    An alias of a constructor is that constructor (`Alias = list`), and an
    alias of an application is the application's own head (`Mine = list[int]`),
    so both are unfolded before they are compared. The name that comes back is
    the name in the source at the end of the chain.
    """
    node, _module = _unfold(viba_ast.TypeRef(constructor), module)
    if isinstance(node, viba_ast.TypeRef):
        return node.name
    if isinstance(node, viba_ast.TypeApp):
        return node.constructor
    return constructor


def _same_type(left: AstNodeType, right: AstNodeType) -> bool:
    """Whether one type in the source is the other, as the judgment layer judges it.

    A parameter that appears twice is an equality: each side fits the other.
    """
    return _judge(left, right) and _judge(right, left)


def _judge(sub: AstNodeType, sup: AstNodeType) -> bool:
    """`sub <: sup`, as the judgment layer judges it.

    A judgment that cannot be made at all — a name that resolves to nothing, a
    malformed chain — is no fit, not the end of the decision: the next file
    is still worth trying, and a design that fits none of them is reported as
    the decision that failed.
    """
    from viba.is_sub_type import is_sub_type
    judged = is_sub_type(sub, sup)
    if isinstance(judged, VibaProgramErr):
        return False
    return judged.ok_value is True

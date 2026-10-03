"""Specialization: a generic is a directory, and one of its files answers.

A generic has no body of its own. What it has is a *method*: a written type
argument, and the answer that argument asks for. Each answer is one file:

    demo/is_base_type/__generic__.viba   the marker: this directory is a generic
    demo/is_base_type/100.viba           specialize bool | int | float | str
    demo/is_base_type/200.viba           specialize A

The file's name is a number and nothing else — its *decision order*, read
smallest first, and it need not be contiguous. Inside, one `specialize` line
per parameter, in written order. A name the file never defines is a parameter:
what stands in the argument there is extracted. A known type restricts that
argument: the argument must fit it (the judgment layer says whether it does).
Everything else about the file is ordinary — its own definitions resolve in it,
its imports are its own.

Applying the generic reads the files in order and takes the first whose
patterns fit the written arguments:

    import demo.is_base_type as is_base_type
    is_bool = is_base_type[bool]        # true, from 100.viba

Nothing fits: that is a program error, not a `never` answer. What the chosen
file answers is its `__def__`, which is a type and nothing else — a generic is
decided where a design is read, not where a program runs
(viba-specialize.md). The parameter names are bound to the written argument
parts, so `__def__ = A` answers the extracted type and
`__def__ = (int <- $env Env <- int)` is a function type like any other.

Internal: the layers that read a design (the interpreter, the judgment, the
descriptor) ask this module; what a caller writes is a generic application in
its own source.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

from viba import viba_ast
from viba.partial import DEF_NAME, product_elements
from viba.type import (AstNodeType, BuiltinGenericType, CustomModuleType,
                       ModuleType, Ok, Result, VibaProgramErr, module_get_type)

GENERIC_FILE = "__generic__.viba"

_SUM_NODES = (viba_ast.Sum, viba_ast.SumChain)
_PROD_NODES = (viba_ast.Product, viba_ast.ProductChain)
_EXP_NODES = (viba_ast.Exponent, viba_ast.ExponentChain)

__all__ = [
    "GENERIC_FILE",
    "Choice",
    "GenericModuleType",
    "Specialization",
    "decide",
    "exponent_elements",
    "file_specialize_problem",
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


class Specialization:
    """One file of a generic: where it is, and the patterns it writes.

    `order` is the number its file name spells; `name` is the module name the
    file is read under; `module` is that file as a module, so its own names
    resolve in it; `patterns` are its `specialize` patterns in written order.
    """

    __slots__ = ("order", "path", "name", "module", "patterns")

    def __init__(self, order: int, path: str, name: str, module: ModuleType,
                 patterns: List[viba_ast.AST]):
        self.order = order
        self.path = path
        self.name = name
        self.module = module
        self.patterns = list(patterns)

    def __repr__(self):
        return f"Specialization({self.order}, {self.path!r}, {len(self.patterns)} params)"


class GenericModuleType(ModuleType):
    """A generic: the directory whose files are its specializations.

    `name` is the generic's name (the directory's basename), `directory` where
    its files are, and `entries` the specialization files in decision order.
    A generic is no source of type names — a bare generic name is not a type —
    so every layer reaches it through an application, never through a lookup.
    `module` is the marker file read as a module; it defines nothing, and it is
    here so that a generic answered to code that treats modules alike answers
    "no `__ret__`" instead of breaking.
    """

    def __init__(self, name: str, directory: str, entries: List[Specialization],
                 module: Optional[viba_ast.Module] = None):
        self.name = name
        self.directory = directory
        self.entries = list(entries)
        self.module = module if module is not None else viba_ast.Module([])

    def __repr__(self):
        return f"GenericModuleType({self.name!r}, {len(self.entries)} specializations)"


class Choice:
    """What the decision picked: the body to read, and the names it binds.

    `body` is the chosen file's `__def__`, read in `module` — the file that
    wrote it. `bindings` holds every parameter the patterns extracted, each as
    the `AstNodeType` the argument part was. `env_get` is that binding as the
    free-name channel the judgment layer already reads generic parameters
    through (`AstNodeType.env_get`).
    """

    __slots__ = ("entry", "body", "module", "bindings")

    def __init__(self, entry: Specialization, body: viba_ast.AST,
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
# Reading a generic's directory
# ----------------------------------------------------------------------


def patterns_of(body) -> List[viba_ast.AST]:
    """The `specialize` patterns a file's body writes, in written order."""
    return [stmt.pattern for stmt in body
            if isinstance(stmt, viba_ast.Specialize)]


def file_specialize_problem(tree, file_name: str) -> Optional[str]:
    """Why this file may not write `specialize`, or None when it may.

    A specialization is one file of a generic, and its name is its decision
    order — a number (viba-specialize.md). So the name is what tells a
    specialization from a module, and a file that writes `specialize` under any
    other name is refused where it is compiled: the marker `__generic__.viba`
    among them, which is no specialization either.
    """
    if not any(isinstance(stmt, viba_ast.Specialize)
               for stmt in getattr(tree, "body", ())):
        return None
    if order_of(file_name) is not None:
        return None
    return (f"{file_name} writes specialize, so it is a specialization: one of "
            f"the numbered files of its generic's directory")


def load_generic(directory: str, name: str,
                 read: Callable[[str], Optional[str]],
                 listing: Callable[[str], Result],
                 module_of: Callable[[str, str, str], Result]) -> Result:
    """Read a generic's whole directory: (GenericModuleType, or why not).

    `directory` holds the files and `name` is what the generic is called.
    `read(path)` answers a file's text, or None when that path has no file;
    `listing(directory)` answers the names directly inside it, or a
    VibaProgramErr when the directory cannot be read; `module_of(path, name,
    source)` compiles one file the way its layer compiles files.

    The marker `__generic__.viba` must be there and must compile; its content is
    otherwise nobody's business. Every other `.viba` file directly inside is a
    specialization: its name is its order, and it writes `__def__`, the type it
    answers. A `.viba` file named anything else is refused — the order is how
    the decision reads the directory.
    """
    marker = os.path.join(directory, GENERIC_FILE)
    marker_source = read(marker)
    if marker_source is None:
        return VibaProgramErr(f"{directory} is no generic: it has no {GENERIC_FILE}")
    try:
        marker_tree = viba_ast.parse(marker_source)
    except SyntaxError as exc:
        return VibaProgramErr(f"cannot parse {marker}: {exc}")
    problem = file_specialize_problem(marker_tree, GENERIC_FILE)
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
        source = read(path)
        if source is None:
            return VibaProgramErr(f"cannot read {path}")
        module = module_of(path, f"{name}.{order}", source)
        if isinstance(module, VibaProgramErr):
            return module
        parts.append((order, path, f"{name}.{order}", module.ok_value))

    return generic_of_entries(name, directory, marker_tree, parts)


def generic_of_entries(name: str, directory: str,
                       marker: Optional[viba_ast.Module],
                       parts) -> Result:
    """A generic built from parts the caller already has.

    `parts` is one `(order, path, module name, module)` per specialization
    file, each already compiled by the layer that owns its files. What the file
    declares is read here — its `specialize` patterns in written order, and the
    `__def__` it answers — so a file means the same thing wherever it was
    compiled. A generic directory a layer serves out of a table rather than out
    of a filesystem goes through here (the descriptor pool does).
    """
    entries: List[Specialization] = []
    for order, path, module_name, module in parts:
        body = getattr(getattr(module, "module", None), "body", None)
        if body is None:
            return VibaProgramErr(f"{path} is not a module of definitions")
        if _definition(body, DEF_NAME) is None:
            return VibaProgramErr(
                f"{path}: a specialization writes {DEF_NAME}, the type it answers")
        entries.append(Specialization(order, path, module_name, module,
                                      patterns_of(body)))
    entries.sort(key=lambda entry: entry.order)
    return Ok(GenericModuleType(name, directory, entries, marker))


def order_of(file_name: str) -> Optional[int]:
    """The decision order a file name spells, or None when it spells none."""
    stem = file_name[: -len(".viba")] if file_name.endswith(".viba") else file_name
    return int(stem) if stem.isdigit() else None


def _definition(body, name: str):
    """The definition a body writes under this name: the last one."""
    found = None
    for stmt in body:
        if getattr(stmt, "name", None) == name:
            found = stmt
    return found


# ----------------------------------------------------------------------
# Reaching a generic from a written name
# ----------------------------------------------------------------------


def generic_named(module: ModuleType, name: str) -> Result:
    """The generic the written name `name` names through this module's imports.

    Ok(the generic) when it is one, Ok(None) when the name is no import of this
    module or names something that is no generic, and VibaProgramErr when the
    binding is there and the module behind it cannot be loaded — the caller
    reports that where it happened rather than reading the name as something
    else.

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
    return Ok(None)


def _generic_of(module: ModuleType, module_name: str) -> Result:
    """Ok(the module) when it is a generic, Ok(None) when it is something else."""
    handed = module.module_environment(module_name)
    if isinstance(handed, VibaProgramErr):
        return handed
    if isinstance(handed.ok_value, GenericModuleType):
        return Ok(handed.ok_value)
    return Ok(None)


def reduce_application(node, module: ModuleType) -> Result:
    """The chosen body for a generic application, or Ok(None) when it is none.

    `node` is the written `G[A, B]` and `module` the module that wrote it, so
    the constructor resolves where it was written. Ok(None) says the
    constructor is no generic of this module (an ordinary type application, a
    builtin container among them), which the caller reads the way it always
    did; a VibaProgramErr is a decision that failed, or a binding that cannot
    be loaded, and says so.
    """
    if not isinstance(node, viba_ast.TypeApp):
        return Ok(None)
    generic = generic_named(module, node.constructor)
    if isinstance(generic, VibaProgramErr):
        return generic
    if generic.ok_value is None:
        return Ok(None)
    return decide(generic.ok_value, node.args, module)


def decide(generic: GenericModuleType, arguments: List[viba_ast.AST],
           argument_module: ModuleType) -> Result:
    """The first specialization whose patterns fit, or why none does.

    The files are read in decision order — the numbers, smallest first. A file
    whose `specialize` count is not the argument count cannot be the one, so it
    is passed over; the first file every pattern fits is the answer. Nothing
    fitting is a program error: the decision failed, and a generic with no
    answer is no design.
    """
    arities = sorted({len(entry.patterns) for entry in generic.entries})
    for entry in generic.entries:
        if len(entry.patterns) != len(arguments):
            continue
        bindings: Dict[str, AstNodeType] = {}
        fits = True
        for pattern, argument in zip(entry.patterns, arguments):
            got = structural_pattern_match(pattern, entry.module, argument,
                                           argument_module, bindings)
            if isinstance(got, VibaProgramErr):
                return got
            if got.ok_value is None:
                fits = False
                break
            bindings = got.ok_value
        if not fits:
            continue
        body = _definition(entry.module.module.body, DEF_NAME)
        if body is None:
            return VibaProgramErr(
                f"{entry.path}: a specialization writes {DEF_NAME}, the type it answers")
        return Ok(Choice(entry, body.body, entry.module, bindings))

    written = ", ".join(viba_ast.unparse_type(argument) for argument in arguments)
    if arities and len(arguments) not in arities:
        return VibaProgramErr(
            f"generic {generic.name!r} takes {_counted(arities)} parameters, "
            f"not {len(arguments)}: [{written}]")
    return VibaProgramErr(
        f"no specialization of {generic.name!r} matches [{written}]: "
        f"the decision failed")


def _counted(arities: List[int]) -> str:
    """`1`, `1 or 2`, `1, 2 or 3` — how many parameters the files declare."""
    written = [str(number) for number in arities]
    if len(written) == 1:
        return written[0]
    return " or ".join([", ".join(written[:-1]), written[-1]])


# ----------------------------------------------------------------------
# Matching one pattern against one written type
# ----------------------------------------------------------------------


class _BadPattern(Exception):
    """The pattern itself cannot be read: this is a mistake, not a mismatch."""


def structural_pattern_match(pattern, pattern_module: ModuleType, argument,
                             argument_module: ModuleType,
                             bindings: Optional[Dict[str, AstNodeType]] = None) -> Result:
    """Bind this pattern's parameters to what the argument has in their place.

    Ok(bindings) when the argument fits the pattern: the dict holds every name
    the pattern extracted, each as the `AstNodeType` that stood in the argument
    (written where the argument was written). Ok(None) when it does not fit —
    the next specialization is then tried. VibaProgramErr when the pattern
    cannot be read at all.

    A name the pattern's module never defines is a parameter. Everywhere else
    the pattern is a written type, and the argument is judged against it with
    `is_sub_type` — the judgment layer's own reading, so `true` fits `bool` and
    `int` fits `bool | int`. Where a parameter sits inside structure
    (`list[A]`, `A <- (() | nil)`), the argument is read apart by the same
    structure, part by part, and each parameter takes the part it stands for.

    A name written twice is one parameter, so the second part has to be the
    same type as the first (`specialize A` twice is an equality). `bindings`
    carries what an earlier pattern already extracted, so two `specialize`
    lines of one file share their parameters.
    """
    bindings = {} if bindings is None else bindings
    try:
        matched = _match(pattern, pattern_module, argument, argument_module, bindings)
    except _BadPattern as exc:
        return VibaProgramErr(str(exc))
    return Ok(matched)


def _match(pattern, pattern_module: ModuleType, argument,
           argument_module: ModuleType, bindings: Dict[str, AstNodeType]):
    """The bindings this part fits with, or None when it does not fit."""
    if isinstance(pattern, viba_ast.Ellipsis):
        raise _BadPattern(
            "ellipsis is not allowed in a specialize pattern: a pattern is one type")
    if _is_parameter(pattern, pattern_module):
        name = pattern.name
        if name in bindings:
            if _same_type(bindings[name], AstNodeType(argument, argument_module)):
                return bindings
            return None
        bindings[name] = AstNodeType(argument, argument_module)
        return bindings

    if not _has_parameter(pattern, pattern_module):
        # Nothing to extract here: the written type is the whole question, and
        # the judgment layer answers it.
        return bindings if _judge(AstNodeType(argument, argument_module),
                                  AstNodeType(pattern, pattern_module)) else None

    node, module = _unfold(argument, argument_module)
    if isinstance(pattern, _SUM_NODES):
        return _match_sum(pattern, pattern_module, node, module, bindings)
    if isinstance(pattern, _PROD_NODES):
        return _match_parts(product_elements(pattern), pattern_module,
                            product_elements(node) if isinstance(node, _PROD_NODES) else [node],
                            module, bindings)
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


def _match_parts(patterns, pattern_module: ModuleType, arguments, argument_module,
                 bindings: Dict[str, AstNodeType]):
    """Parts against parts, in order: every pair has to fit, sharing bindings."""
    if len(patterns) != len(arguments):
        return None
    for pattern, argument in zip(patterns, arguments):
        found = _match(pattern, pattern_module, argument, argument_module, bindings)
        if found is None:
            return None
        bindings = found
    return bindings


def _match_sum(pattern, pattern_module: ModuleType, node, module: ModuleType,
               bindings: Dict[str, AstNodeType]):
    """A sum pattern: every branch of the argument fits some branch of it.

    A branch that fits is one branch; an argument that is no sum is one branch
    of its own. Branches are tried in written order, and the first fit is the
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
    """The branches of a written sum, flattened in written order."""
    if isinstance(node, viba_ast.Sum):
        return sum_elements(node.left) + sum_elements(node.right)
    if isinstance(node, viba_ast.SumChain):
        return list(node.elements)
    return [node]


def exponent_elements(node) -> List[viba_ast.AST]:
    """A written function read apart: the result first, the arguments after.

    `A <- B <- C` is `[A, B, C]` — one element per position, in written order,
    which is how the judgment layer reads a chain too.
    """
    if isinstance(node, viba_ast.Exponent):
        return exponent_elements(node.result) + [node.argument]
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    return [node]


def _is_parameter(node, module: ModuleType) -> bool:
    """A name the pattern's module never defines: a parameter to extract.

    Anything else is a written type: a definition of that module, a builtin
    name, a name reached through an import. The rule is the design's own — a
    name is what it resolves to, and a name that resolves to nothing is
    standing for whatever the argument has there (viba-specialize.md).
    """
    if not isinstance(node, viba_ast.TypeRef):
        return False
    return isinstance(module_get_type(module, node.name), VibaProgramErr)


def _has_parameter(node, module: ModuleType) -> bool:
    """Whether a parameter sits anywhere in this pattern."""
    return any(_is_parameter(part, module) for part in viba_ast.walk(node))


def _unfold(node, module: ModuleType):
    """A name is transparent: unfold it to the structure it stands for.

    The argument is read the way every layer reads a name — an alias of an
    alias is the body at the end of the chain — because what a pattern matches
    is the type, not the spelling. A generic's own name is left standing: a
    bare generic has no body to unfold. The module that comes back is the one
    the unfolded body was written in, which is where its own names mean
    something.
    """
    seen = set()
    while isinstance(node, viba_ast.TypeRef):
        if node.name in seen:
            return node, module
        seen.add(node.name)
        resolved = module_get_type(module, node.name)
        if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
            return node, module
        target = resolved.ok_value
        body = target.ast_node
        if isinstance(body, viba_ast.GenericDefinition):
            return node, module
        if isinstance(body, viba_ast.TypeDefinition):
            body = body.body
        node, module = body, target.container_module
    return node, module


def _constructor_key(constructor: str, module: ModuleType) -> str:
    """What a constructor names, as something two sides compare by.

    An alias of a constructor is that constructor (`Alias = list`), and an
    alias of an application is the application's own head (`Mine = list[int]`),
    so both are unfolded before they are compared. The name that comes back is
    the written name at the end of the chain.
    """
    node, _module = _unfold(viba_ast.TypeRef(constructor), module)
    if isinstance(node, viba_ast.TypeRef):
        return node.name
    if isinstance(node, viba_ast.TypeApp):
        return node.constructor
    return constructor


def _same_type(left: AstNodeType, right: AstNodeType) -> bool:
    """Whether one written type is the other, as the judgment layer reads it.

    A parameter written twice is an equality: each side fits the other.
    """
    return _judge(left, right) and _judge(right, left)


def _judge(sub: AstNodeType, sup: AstNodeType) -> bool:
    """`sub <: sup`, as the judgment layer reads it.

    A judgment that cannot be made at all — a name that resolves to nothing, a
    malformed chain — is no fit, not the end of the decision: the next
    specialization is still worth trying, and a design that fits none of them
    is reported as the decision that failed.
    """
    from viba.is_sub_type import is_sub_type
    verdict = is_sub_type(sub, sup)
    if isinstance(verdict, VibaProgramErr):
        return False
    return verdict.ok_value is True

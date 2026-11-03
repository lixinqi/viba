"""viba.extract_sub_repo — pull one sub-repo out of a viba repo.

    from viba.extract_sub_repo import extract_sub_repo

    answer = extract_sub_repo("kb", ["kb", "base"], ["kb"], ["report.Report"])
    # Ok(SubRepo(repo="kb-sub", viba_path=["kb-sub", "base"]))

The design is viba/extract_sub_repo.viba and the document is
viba-extract-sub-repo.md; this module is the one layer that takes files apart.

What it does, in the order the document gives it:

- The repo is a directory of `.viba` files. A file's module name is its path
  relative to the repo, without the suffix and with the separators turned into
  dots; a generic is the directory its files sit in (`__generic__.viba` and the
  numbered files beside it), which is the same rule.
- A type name is `xxx.yyy.zzz`: the module `xxx.yyy` is found the way an import
  would find it — on `viba_path`, in order, the builtin directory last — and
  `zzz` is a definition in it.
- The definitions reached from those names are walked. A definition whose file
  is in the repo and under `extract_source_viba_path` is extracted, together
  with the definitions it reaches; everything else is a primitive: not copied,
  its import kept, and the search root it was found under added to the answer's
  `viba_path`.
- A generic application keeps the directory: the marker and the pattern file
  the decision chooses (every file that can take as many arguments as the
  application gives, when the decision cannot be made).
- The sub-repo is written beside the repo (`kb` -> `kb-sub`) and the answer is
  its path with the `viba_path` it needs.

The walk reads the source the way the interpreter's loader does — the same
`_Runner` that serves a run — so a name resolves where it would resolve when
the design runs. Nothing is run: no definition is called, and a code block is
documentation.

A pattern file is kept whole: its `pattern` lines, `__decl__`, `__impl__` and
whatever else it defines go together, because a file missing one of them is no
decision. Every other file keeps the definitions that were reached and the
imports those definitions use; it is spelled again from the syntax tree, so the
`#` lines and the blank lines are not carried over (the members that carry a
code block are part of a definition and do go with it).

The units: `never_eqv` and `nil_eqv` are the names that stand for `never` and
`nil` while the source is read (VibaReflectConfig's parameters,
viba-reflect.md). None means the language's own set, reflect.language_config;
a list given is that list, and an empty one adds no name.

This layer says what to copy; it does not judge the design. A decision that
cannot be made in the source keeps every file that could take the application,
so a generic whose body applies another generic to a parameter still extracts
to a runnable sub-repo.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Set

from viba import viba_ast
from viba.design_walk import (
    bindings_key,
    call_parts,
    chain_elements,
    is_definition,
    is_under,
    longest_import,
    module_label,
    split_type_name,
    under_builtin,
    with_params,
)
from viba.interpret import _Runner
from viba.pattern import GENERIC_FILE, GenericModuleType, PatternFile, decide, generic_named
from viba.type import (
    BUILTIN_CONCEPT_DIR,
    BUILTIN_DIR,
    AstNodeType,
    CustomModuleType,
    ModuleType,
    Ok,
    Result,
    Err,
    module_get_type,
)
from viba.viba_ast.tagged import TAGGED_NAME

__all__ = ["extract_sub_repo", "SubRepo"]

# The language's own units (reflect.language_config): a name that stands for the
# sum's unit or the product's unit. `Object`/`nil`/`never`/`Oneof` resolve as
# builtin names by themselves; the config adds the names that only a document
# block gives (`Assert`, `Appendix`, `Hint`).
DEFAULT_NEVER_EQV = ("Oneof",)
DEFAULT_NIL_EQV = ("Object", "Assert", "Appendix", "Hint")

CONTAINER_NAMES = ("list", "set", "dict")
UNIT_NAMES = ("Object", "nil", "void", "None", "never", "Oneof")


class SubRepo:
    """What one extraction answered: the sub-repo's path and the import path it needs."""

    __slots__ = ("repo", "viba_path")

    def __init__(self, repo: str, viba_path: Sequence[str]):
        self.repo = repo
        self.viba_path = list(viba_path)

    def __eq__(self, other):
        return (isinstance(other, SubRepo)
                and (self.repo, self.viba_path) == (other.repo, other.viba_path))

    def __hash__(self):
        return hash((self.repo, tuple(self.viba_path)))

    def __repr__(self):
        return f"SubRepo({self.repo!r}, {self.viba_path!r})"


def extract_sub_repo(repo, viba_path=(), extract_source_viba_path=(),
                     type_names=(), never_eqv=None, nil_eqv=None) -> Result:
    """Result[SubRepo]: the sub-repo of `repo` that the type names reach.

    `repo` is a directory; `viba_path` the directories an import is looked up
    on, in order; `extract_source_viba_path` the directories of `repo` whose
    definitions are the ones to copy; `type_names` the full names to start from
    (`xxx.yyy.zzz`). `never_eqv` / `nil_eqv` are the unit names, or None for the
    language's own (`reflect.language_config`).

    Err says why no sub-repo was written, one sentence, the first
    thing that does not hold.
    """
    problem = _argument_problem(repo, viba_path, extract_source_viba_path,
                                type_names, never_eqv, nil_eqv)
    if problem is not None:
        return Err(problem)
    slicer = _Slicer(
        str(repo),
        [str(one) for one in viba_path or ()],
        [str(one) for one in extract_source_viba_path or ()],
        [str(one) for one in type_names or ()],
        None if never_eqv is None else [str(one) for one in never_eqv],
        None if nil_eqv is None else [str(one) for one in nil_eqv],
    )
    try:
        return slicer.slice()
    except _SliceErr as exc:
        return Err(str(exc))


def _argument_problem(repo, viba_path, extract_source_viba_path, type_names,
                      never_eqv, nil_eqv) -> Optional[str]:
    if not isinstance(repo, (str, os.PathLike)):
        return f"repo is a path, not {type(repo).__name__}"
    for label, given in (("viba_path", viba_path),
                         ("extract_source_viba_path", extract_source_viba_path),
                         ("type_names", type_names),
                         ("never_eqv", never_eqv),
                         ("nil_eqv", nil_eqv)):
        if given is None:
            if label in ("never_eqv", "nil_eqv"):
                continue
            return f"{label} is a list of strings"
        if isinstance(given, str):
            return f"{label} is a list of strings, not one string"
        try:
            parts = list(given)
        except TypeError:
            return f"{label} is a list of strings, not {type(given).__name__}"
        for one in parts:
            if not isinstance(one, str):
                return f"{label} holds a string, not {type(one).__name__}"
    return None


class _SliceErr(Exception):
    """The first thing that does not hold; the walk stops here."""


class _KeptFile:
    """One file of the sub-repo that keeps some of its definitions."""

    __slots__ = ("module", "names", "imports")

    def __init__(self, module: CustomModuleType):
        self.module = module
        self.names: Set[str] = set()
        self.imports: Set[str] = set()


class _Slicer:
    """One extraction: the loader, what was reached, and where the sub-repo goes."""

    def __init__(self, repo: str, viba_path: Sequence[str],
                 extract_source: Sequence[str], type_names: Sequence[str],
                 never_eqv: Optional[Sequence[str]],
                 nil_eqv: Optional[Sequence[str]]):
        self.repo = Path(os.path.normpath(repo))
        self.repo_root = self.repo.resolve()
        self.extract_roots = [Path(one).resolve() for one in extract_source]
        self.type_names = list(type_names)
        self.never_eqv = set(DEFAULT_NEVER_EQV if never_eqv is None else never_eqv)
        self.nil_eqv = set(DEFAULT_NIL_EQV if nil_eqv is None else nil_eqv)
        self.loader = _Runner(":".join(viba_path) if viba_path else None)
        # What the sub-repo holds: a file with some of its definitions kept, or a
        # file kept whole (a generic's marker and pattern files).
        self.kept = {}
        self.whole = {}
        self.whole_walked = set()
        # The search roots the primitives were found under, in the order they are met.
        self.roots: List[str] = []
        # (file, definition, bindings) already walked.
        self.visited = set()

    # ---- the entry ----

    def slice(self) -> Result:
        if not self.repo.is_dir():
            raise _SliceErr(f"no such repo directory: {self.repo}")
        for name in self.type_names:
            self._start(name)
        return self._write()

    def _start(self, name: str) -> None:
        module_name, definition_name = split_type_name(name)
        if not module_name:
            raise _SliceErr(
                f"{name!r} is no type name: it is a module and a definition, "
                f"like 'pkg.mod.Name'")
        loaded = self.loader.imported(module_name, None)
        if isinstance(loaded, Err):
            raise _SliceErr(
                f"cannot load the module {module_name!r} of {name!r}: {loaded.msg}")
        module = loaded.ok_value
        if not isinstance(module, CustomModuleType):
            raise _SliceErr(
                f"{name!r} names a generic: a generic answers an application, "
                f"not a bare name")
        if not self._is_source_file(module.file_path):
            raise _SliceErr(
                f"{name!r} is not in the extraction source: {module_label(module)}")
        found = module_get_type(module, definition_name)
        if isinstance(found, Err):
            raise _SliceErr(f"no type {name!r}: {found.msg}")
        target = found.ok_value
        if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
            raise _SliceErr(f"{name!r} is no definition")
        self._keep(target.ast_node, target.container_module, {})

    # ---- keeping ----

    def _keep(self, definition, module: ModuleType, bindings: dict) -> None:
        """Keep this definition, and walk what it reaches."""
        if not isinstance(module, CustomModuleType):
            return                              # the builtin vocabulary: the interpreter's
        file_path = module.file_path
        if not file_path or under_builtin(file_path):
            return
        if not self._is_source_file(file_path):
            self._add_root(module.root)
            return
        key = (file_path, definition.name, bindings_key(bindings))
        if key in self.visited:
            return
        self.visited.add(key)
        if self._is_generic_file(file_path):
            # The file is kept whole, so every definition in it is walked; this one
            # also stands in the bindings the decision gave it.
            self._keep_whole(module)
            self._walk(definition.body, module, with_params(definition, bindings))
            return
        self._kept_file(module).names.add(definition.name)
        self._walk(definition.body, module, with_params(definition, bindings))

    def _keep_whole(self, module: CustomModuleType) -> None:
        """Keep the whole file, and walk every definition it has.

        A generic's pattern file is kept whole: which definitions the file needs
        is the file's own business (its `pattern` lines and `__decl__` decide
        what the file answers). What the file's own definitions mention is
        walked with no bindings — a name no decision bound there is a pattern
        parameter, and a pattern parameter is a placeholder.
        """
        file_path = module.file_path
        if not file_path or file_path in self.whole_walked:
            return
        self.whole_walked.add(file_path)
        self.whole[file_path] = module
        for statement in module.module.body:
            if not is_definition(statement):
                continue
            key = (file_path, statement.name, bindings_key({}))
            if key in self.visited:
                continue
            self.visited.add(key)
            self._walk(statement.body, module, with_params(statement, {}), loose=True)

    def _keep_generic(self, generic: GenericModuleType, arguments: int,
                      choice) -> None:
        """Keep the marker and the pattern files the application can take."""
        directory = str(generic.directory)
        if not self._is_source_file(os.path.join(directory, GENERIC_FILE)):
            self._add_root(_generic_root(self.loader, generic))
            return
        marker = os.path.join(directory, GENERIC_FILE)
        self.whole[marker] = generic.module
        if choice is not None:
            self._keep_pattern(choice.entry)
            return
        for entry in generic.entries:
            got = entry.take()
            if isinstance(got, Err):
                raise _SliceErr(f"cannot take {entry.path}: {got.msg}")
            _module, patterns = got.ok_value
            if len(patterns) == arguments:
                self._keep_pattern(entry)

    def _keep_pattern(self, entry: PatternFile) -> None:
        got = entry.take()
        if isinstance(got, Err):
            raise _SliceErr(f"cannot take {entry.path}: {got.msg}")
        module, _patterns = got.ok_value
        if not isinstance(module, CustomModuleType):
            return
        if not self._is_source_file(module.file_path):
            self._add_root(module.root)
            return
        self._keep_whole(module)

    def _kept_file(self, module: CustomModuleType) -> _KeptFile:
        entry = self.kept.get(module.file_path)
        if entry is None:
            entry = _KeptFile(module)
            self.kept[module.file_path] = entry
        return entry

    def _add_root(self, root) -> None:
        if not root:
            return
        path = str(root)
        if Path(path).resolve() in (BUILTIN_DIR, BUILTIN_CONCEPT_DIR):
            return
        if path not in self.roots:
            self.roots.append(path)

    # ---- the walk ----

    def _walk(self, node, module: ModuleType, bindings: dict,
              loose: bool = False) -> None:
        if node is None:
            return
        if isinstance(node, (viba_ast.Constant, viba_ast.Nil, viba_ast.Never,
                             viba_ast.Any, viba_ast.Ellipsis, viba_ast.CodeBlock,
                             viba_ast.Member)):
            return
        if isinstance(node, viba_ast.TypeRef):
            self._name(node.name, module, bindings, loose)
            return
        if isinstance(node, viba_ast.MemberTaken):
            self._member(node, module, bindings, loose)
            return
        if isinstance(node, viba_ast.Tagged):
            self._walk(node.type, module, bindings, loose)
            return
        if isinstance(node, viba_ast.TypeApp):
            self._application(node, module, bindings, loose)
            return
        if isinstance(node, viba_ast.Tuple):
            for element in node.elements:
                self._walk(element, module, bindings, loose)
            return
        if isinstance(node, viba_ast.Partial):
            self._call(node, module, bindings, loose)
            return
        elements = chain_elements(node)
        if elements is not None:
            for element in elements:
                self._walk(element, module, bindings, loose)

    def _name(self, name: str, module: ModuleType, bindings: dict,
              loose: bool) -> None:
        if name in bindings:
            bound = bindings[name]
            if bound is None:
                return                          # a generic parameter, no argument yet
            self._walk(bound[0], bound[1], bound[2])
            return
        if name in self.never_eqv or name in self.nil_eqv:
            return                              # a unit: no definition behind it
        if name in UNIT_NAMES:
            return
        found = module_get_type(module, name)
        if isinstance(found, Err):
            if loose:
                return                          # a pattern parameter of this file
            raise _SliceErr(
                f"cannot resolve {name!r} in {module_label(module)}: {found.msg}")
        target = found.ok_value
        if not isinstance(target, AstNodeType):
            return                              # a builtin scalar, container or form
        if not is_definition(target.ast_node):
            return
        self._note_import(module, name)
        self._keep(target.ast_node, target.container_module, {})

    def _note_import(self, module: ModuleType, name: str) -> None:
        """A kept definition of this file names something through an import."""
        if not isinstance(module, CustomModuleType):
            return
        file_path = module.file_path
        if not file_path or not self._is_source_file(file_path):
            return
        if self._is_generic_file(file_path):
            return                              # whole file: every import stays
        local = longest_import(module.imports, name)
        if local is not None:
            self._kept_file(module).imports.add(local)

    def _application(self, node, module: ModuleType, bindings: dict,
                     loose: bool) -> None:
        constructor = node.constructor
        if constructor == TAGGED_NAME:
            for argument in node.args[1:]:
                self._walk(argument, module, bindings, loose)
            return
        for argument in node.args:
            self._walk(argument, module, bindings, loose)
        if constructor in CONTAINER_NAMES or constructor in UNIT_NAMES:
            return
        if constructor in self.never_eqv or constructor in self.nil_eqv:
            return
        self._note_import(module, constructor)
        generic = generic_named(module, constructor)
        if isinstance(generic, Err):
            raise _SliceErr(
                f"cannot take {constructor!r} in {module_label(module)}: {generic.msg}")
        if generic.ok_value is None:
            self._name(constructor, module, bindings, loose)
            return
        generic = generic.ok_value
        got = decide(generic, node.args, module,
                     argument_modules=self._argument_modules(node, bindings))
        choice = None if isinstance(got, Err) else got.ok_value
        self._keep_generic(generic, len(node.args), choice)
        if choice is None or choice.body is None:
            return
        inner = dict(bindings)
        for pattern_name, bound in choice.bindings.items():
            inner[pattern_name] = (bound.ast_node, bound.container_module, bindings)
        self._walk(choice.body, choice.module, inner)

    def _argument_modules(self, node, bindings: dict):
        """The module each argument was spelled in, where a decision bound it.

        A name a decision bound stands for the argument part that stood at the
        call site, so an argument that is such a name is taken in its own module
        (viba/pattern.py, `decide`).
        """
        kept = []
        for argument in node.args:
            where = None
            if isinstance(argument, viba_ast.TypeRef):
                bound = bindings.get(argument.name)
                if isinstance(bound, tuple):
                    where = bound[1]
            kept.append(where)
        return kept if any(where is not None for where in kept) else None

    def _member(self, node, module: ModuleType, bindings: dict, loose: bool) -> None:
        source_path = viba_ast.source_path(node)
        if source_path is not None:
            self._name(source_path, module, bindings, loose)
            return
        owner = node.owner
        if not isinstance(owner, viba_ast.TypeApp):
            return
        self._note_import(module, owner.constructor)
        generic = generic_named(module, owner.constructor)
        if isinstance(generic, Err):
            raise _SliceErr(
                f"cannot take {owner.constructor!r} in {module_label(module)}: {generic.msg}")
        if generic.ok_value is None:
            # Not a generic directory: the application is taken as it always is,
            # and the member is a member of what it answers.
            self._application(owner, module, bindings, loose)
            return
        generic = generic.ok_value
        for argument in owner.args:
            self._walk(argument, module, bindings, loose)
        got = decide(generic, owner.args, module,
                     argument_modules=self._argument_modules(owner, bindings))
        choice = None if isinstance(got, Err) else got.ok_value
        self._keep_generic(generic, len(owner.args), choice)
        if choice is None:
            return
        inner = dict(bindings)
        for pattern_name, bound in choice.bindings.items():
            inner[pattern_name] = (bound.ast_node, bound.container_module, bindings)
        if choice.body is None:
            # The file is a module: the member names one of its definitions.
            self._keep_named(choice.module, node.name, inner)
            return
        self._walk(choice.body, choice.module, inner)

    def _keep_named(self, module: ModuleType, name: str, bindings: dict) -> None:
        if not isinstance(module, CustomModuleType):
            return
        for statement in module.module.body:
            if is_definition(statement) and statement.name == name:
                self._keep(statement, module, bindings)
                return
        raise _SliceErr(f"no definition {name!r} in {module_label(module)}")

    def _call(self, node, module: ModuleType, bindings: dict, loose: bool) -> None:
        head, arguments = call_parts(node)
        self._walk(head, module, bindings, loose)
        for argument in arguments:
            if isinstance(argument, viba_ast.CodeBlock):
                continue                        # documentation is no argument
            if isinstance(argument, viba_ast.Tagged):
                self._walk(argument.type, module, bindings, loose)
            else:
                self._walk(argument, module, bindings, loose)

    # ---- the files ----

    def _is_source_file(self, file_path) -> bool:
        if not file_path:
            return False
        path = Path(file_path).resolve()
        if not is_under(path, self.repo_root):
            return False
        return any(is_under(path, root) for root in self.extract_roots)

    @staticmethod
    def _is_generic_file(file_path) -> bool:
        """Is this file one of a generic's: the marker, or a numbered file beside it?"""
        path = Path(file_path)
        if path.name == GENERIC_FILE:
            return True
        return (path.parent / GENERIC_FILE).is_file()

    # ---- the answer ----

    def _write(self) -> Result:
        files = {}
        for file_path, entry in self.kept.items():
            files[self._relative(file_path)] = _spell_definitions(entry)
        for file_path in self.whole:
            # A file kept whole is copied as it is: its `#` lines and its layout are
            # part of how a generic reads.
            files[self._relative(file_path)] = Path(file_path).read_text(encoding="utf-8")
        target = self._target()
        self._write_files(target, files)
        return Ok(SubRepo(str(target), [str(target)] + self._ordered_roots()))

    def _ordered_roots(self) -> List[str]:
        """The primitive roots in the order the `viba_path` gave, the others after.

        A root found beside the file that imported it was not on the `viba_path`,
        so it has no place there: it goes after them, in the order it was met.
        """
        places = {os.path.normpath(str(path)): index
                  for index, path in enumerate(self.loader.paths)}

        def place(entry):
            index, root = entry
            found = places.get(os.path.normpath(root))
            return (0, found) if found is not None else (1, index)

        return [root for _index, root in sorted(enumerate(self.roots), key=place)]

    def _relative(self, file_path) -> str:
        return os.path.relpath(Path(file_path).resolve(), self.repo_root)

    def _target(self) -> Path:
        parent = self.repo.parent
        name = self.repo.name + "-sub"
        return Path(os.path.join(str(parent), name)) if str(parent) != "." else Path(name)

    @staticmethod
    def _write_files(target: Path, files: dict) -> None:
        temporary = Path(str(target) + ".tmp")
        if temporary.is_dir():
            shutil.rmtree(temporary)
        elif temporary.exists():
            temporary.unlink()
        temporary.mkdir(parents=True, exist_ok=True)
        for relative, text in sorted(files.items()):
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        os.replace(temporary, target)


def _generic_root(loader, generic: GenericModuleType) -> "Optional[str]":
    """The search root the generic was found under: its module name taken as a path."""
    directory = os.path.normpath(str(generic.directory))
    name = generic.name or ""
    for base in loader.paths:
        place = os.path.normpath(os.path.join(str(base), *name.split(".")))
        if place == directory:
            return str(base)
    return os.path.dirname(directory) or None


# ----------------------------------------------------------------------
# Spelling the files again
# ----------------------------------------------------------------------


def _spell_definitions(entry: _KeptFile) -> str:
    """Imports used, then the kept definitions, in the order the source has them."""
    body = entry.module.module.body
    nodes = [statement for statement in body
             if isinstance(statement, viba_ast.Import)
             and (statement.alias or statement.module) in entry.imports]
    nodes += [statement for statement in body
              if is_definition(statement) and statement.name in entry.names]
    return _spell_nodes(nodes)


def _spell_nodes(nodes) -> str:
    """The statements spelled again: the imports as a block, then the rest.

    Imports stay one per line and the definitions get a blank line between them,
    which is how the source in this repo reads; a file with nothing in it is
    empty (a generic's marker file carries only its `#` comments, which are not
    part of a definition).
    """
    imports = [node for node in nodes if isinstance(node, viba_ast.Import)]
    others = [node for node in nodes if not isinstance(node, viba_ast.Import)]
    head = "\n".join(viba_ast.unparse_type(node) for node in imports)
    rest = "\n\n".join(viba_ast.unparse_type(node) for node in others)
    text = "\n\n".join(part for part in (head, rest) if part)
    return text + "\n" if text else ""

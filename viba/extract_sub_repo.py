"""viba.extract_sub_repo — a sub-repo as one Markdown document.

    from viba.extract_sub_repo import (extract_sub_repo_into_viba_markdown,
                                       extract_sub_viba_markdown)

    document, source_map = extract_sub_repo_into_viba_markdown(
        "kb", ["kb", "base"], ["kb"], ["report.Report"]).ok_value
    # source_map[4] is (module, line): where the document's line 4 came from

    again, to_source = extract_sub_viba_markdown(document, ["report.Report"]).ok_value

The design is viba/extract_sub_repo.viba and the document is
viba/wiki/viba-extract-sub-repo.md; this module is the one layer that takes files
apart.

The answer is a document, not a directory: one section per file, its path the
section's name, its body the lines the source has for the imports it uses and
the definitions kept — comments and blank lines included, so the map can be
exact. Both entries answer that document and a map: the repo entry maps each
code line to the module and the line in it, the document entry to the line of
the source document.

What it does, in the order the document gives it:

- The repo holds `.viba` files and `.viba.md` documents alike. A file's module
  name is its path relative to the repo, without the suffix and with the
  separators turned into dots; a generic is the directory its files sit in
  (`__generic__.viba` and the numbered files beside it), which is the same rule.
  A document's sections are its files, so a definition reached in one is
  extracted like any other and comes out at the path its module has.
- A type name is `xxx.yyy.zzz`: the module `xxx.yyy` is found the way an import
  would find it — on `viba_path`, in order, the builtin directory last — and
  `zzz` is a definition in it.
- The definitions reached from those names are walked. A definition whose file
  is in the repo and under `extract_source_viba_path` is extracted, together
  with the definitions it reaches; everything else is a leaf primitive: its
  definition is not in the document, the import that names it is kept.
- A generic application keeps the directory: the marker and the pattern file
  the decision chooses (every file that can take as many arguments as the
  application gives, when the decision cannot be made). A pattern file's own
  definitions are no names to start from: the file is kept whole by an
  application of its generic, so a type name that lands in one is refused.
- A file kept whole (a generic's marker and pattern file) is its own text; any
  other keeps the definitions that were reached and the imports those
  definitions use, as the source has them.

The walk reads the source the way the interpreter's loader does — the same
`_Runner` that serves a run — so a name resolves where it would resolve when
the design runs. Nothing is run: no definition is called, and a hint block is
documentation.

The units: `never_eqv` and `nil_eqv` are the names that stand for `never` and
`nil` while the source is read (VibaReflectConfig's parameters,
viba/core/data/viba-reflect.md). None means the language's own set, reflect.language_config;
a list given is that list, and an empty one adds no name.

This layer says what to copy; it does not judge the design. A decision that
cannot be made in the source keeps every file that could take the application,
so a generic whose body applies another generic to a parameter still extracts
to a complete document.
"""


from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Set

from viba import markdown, viba_ast
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

__all__ = ["extract_sub_repo_into_viba_markdown", "extract_sub_viba_markdown"]

# The language's own units (reflect.language_config): a name that stands for the
# sum's unit or the product's unit. `Object`/`nil`/`never`/`Oneof` resolve as
# builtin names by themselves; the config adds the names that only a document
# block gives (`Assert`, `Appendix`, `Hint`).
DEFAULT_NEVER_EQV = ("Oneof",)
DEFAULT_NIL_EQV = ("Object", "Assert", "Appendix", "Hint")

CONTAINER_NAMES = ("list", "set", "dict")
UNIT_NAMES = ("Object", "nil", "void", "None", "never", "Oneof")


def extract_sub_repo_into_viba_markdown(repo, viba_path=(),
                                        extract_source_viba_path=(),
                                        type_names=(), never_eqv=None,
                                        nil_eqv=None) -> Result:
    """Result[(VibaMarkdown, VibaMdToRepoSourceMap)]: the sub-repo as one document.

    `repo` is a directory; `viba_path` the directories an import is looked up
    on, in order; `extract_source_viba_path` the directories of `repo` whose
    definitions are the ones to copy; `type_names` the full names to start from
    (`xxx.yyy.zzz`). `never_eqv` / `nil_eqv` are the unit names, or None for the
    language's own (`reflect.language_config`).

    The answer is the document the sub-repo is (`viba/core/data/viba-data.md`:
    one section per file, its path the section's name), and a map from each of
    the document's code lines to where it came from — the module it is a piece
    of, and the line in that module.

    A file of the repo may be a document (`.viba.md`): its sections are files
    too, and a definition reached in one is extracted like any other — it comes
    out as a section of its own, at the path its module has.

    Err says why no document was made, one sentence, the first thing that does
    not hold.
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
        document, origin = slicer.markdown()
    except _SliceErr as exc:
        return Err(str(exc))
    return Ok((document, {at: (module, line) for at, module, line in origin}))


def extract_sub_viba_markdown(source_viba_md, type_names=(), never_eqv=None,
                              nil_eqv=None) -> Result:
    """Result[(VibaMarkdown, VibaMdToVibaMdSourceMap)]: a document in, a document out.

    `source_viba_md` is a document's content: one section per file, the section's
    name the file's path in the repo the document stands for. The sub-document
    answers the definitions `type_names` reach, cut the way the repo entry cuts
    them; the map says, for each code line of the answer, the line of the source
    document it came from.

    A document is a repo whose files are documents in turn; this entry takes the
    one it is given as the whole source, so the only files it can reach are the
    ones the document holds. The section names are files of the document's
    directory: a name outside it (an absolute path, `..`) is refused. Every code
    line of the answer is placed in the source document; a line that cannot be
    placed is an Err.
    """
    if not isinstance(source_viba_md, str):
        return Err(f"source_viba_md is a document's text, not "
                   f"{type(source_viba_md).__name__}")
    try:
        files = markdown.files_of(source_viba_md)
    except SyntaxError as exc:
        return Err(str(exc))
    problem = _sections_problem([name for name, _body in files])
    if problem is not None:
        return Err(problem)
    with tempfile.TemporaryDirectory(prefix="viba-extract-") as scratch:
        root = Path(scratch)
        for name, body in files:
            path = root / name
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
            except OSError as exc:
                return Err(f"cannot write the section {name!r}: {exc}")
        problem = _argument_problem(str(root), [str(root)], [str(root)],
                                    type_names, never_eqv, nil_eqv)
        if problem is not None:
            return Err(problem)
        slicer = _Slicer(
            str(root), [str(root)], [str(root)],
            [str(one) for one in type_names or ()],
            None if never_eqv is None else [str(one) for one in never_eqv],
            None if nil_eqv is None else [str(one) for one in nil_eqv],
        )
        try:
            document, origin = slicer.markdown()
        except _SliceErr as exc:
            return Err(str(exc))
    # where a line of a module is in the source document: its section's lines
    places = {}
    for name, lines in markdown.lines_of(source_viba_md):
        for index, (_text, at) in enumerate(lines, 1):
            places[(name, index)] = at
    texts = dict(files)

    def source_line_of(module: str, module_line: int):
        """The source document's line a module line is, or None when it is not one.

        The places a module can come from are the loader's, in its order: the
        document beside its directory first (a section that is a document of its
        own), then the file at its dotted path, then one file named with its dots.
        """
        parts = module.split(".")
        if len(parts) > 1:
            outer = "/".join(parts[:-1]) + ".viba.md"
            if outer in texts:
                try:
                    inner = dict(markdown.lines_of(texts[outer]))
                except SyntaxError:
                    return None
                lines = inner.get(parts[-1] + ".viba")
                if lines is None or not 1 <= module_line <= len(lines):
                    return None
                return places.get((outer, lines[module_line - 1][1]))
        for section in ("/".join(parts) + ".viba", module + ".viba"):
            at = places.get((section, module_line))
            if at is not None:
                return at
        return None

    answer = {}
    for document_line, module, module_line in origin:
        at = source_line_of(module, module_line)
        if at is None:
            return Err(
                f"line {document_line} of the answer is from {module!r}, and the "
                f"source document has no section for it")
        answer[document_line] = at
    return Ok((document, answer))


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


def _sections_problem(names: Sequence[str]) -> Optional[str]:
    """Why the sections cannot be the files of one directory, or None when they can.

    A section is a file of the document's directory, so its name is a relative
    path without `.` or `..`; and no section may name a file that another
    section needs as a directory (`a` beside `a/b.viba`).
    """
    known = set(names)
    for name in names:
        parts = name.split("/")
        if (not name or name.startswith("/") or "\x00" in name
                or any(part in ("", ".", "..") for part in parts)):
            return (f"a section names a file outside the document's directory: "
                    f"{name!r}")
        for cut in range(1, len(parts)):
            prefix = "/".join(parts[:cut])
            if prefix in known:
                return (f"a section names the file {prefix!r}, and another section "
                        f"needs it to be a directory")
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
        # (file, definition, bindings) already walked.
        self.visited = set()

    # ---- the entry ----

    def markdown(self):
        """The sub-repo as one document, and where each of its code lines came from."""
        if not self.repo.is_dir():
            raise _SliceErr(f"no such repo directory: {self.repo}")
        for name in self.type_names:
            self._start(name)
        return self._render()

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
        if self._is_pattern_file(module):
            raise _SliceErr(
                f"{name!r} is a definition in a pattern file: a generic's files "
                f"are kept whole by an application of it, not by a name")
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

    def _keep_whole(self, module: CustomModuleType, document=None) -> None:
        """Keep the whole file, and walk every definition it has.

        A generic's pattern file is kept whole: which definitions the file needs
        is the file's own business (its `pattern` lines and `__decl__` decide
        what the file answers). What the file's own definitions mention is
        walked with no bindings — a name no decision bound there is a pattern
        parameter, and a pattern parameter is a placeholder.
        """
        file_path = module.file_path
        name = self._whole_name(module, document)
        if not file_path and document is None:
            return
        if name in self.whole_walked:
            return
        section = None
        if file_path is None:
            # A section of a document: the document is the file, the section its text.
            file_path = document
            section = name.rsplit(".", 1)[-1] + ".viba"
        self.whole_walked.add(name)
        self.whole[name] = (file_path, section)
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
            return
        marker = os.path.join(directory, GENERIC_FILE)
        is_document = str(generic.directory).endswith(".viba.md")
        self.whole[f"{generic.name}.__generic__"] = (
            str(generic.directory) if is_document else marker,
            GENERIC_FILE if is_document else None)
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
        if not self._is_source_file(entry.path):
            return
        document = str(Path(entry.path).parent)
        self._keep_whole(module, document if document.endswith(".viba.md") else None)

    def _kept_file(self, module: CustomModuleType) -> _KeptFile:
        entry = self.kept.get(module.file_path)
        if entry is None:
            entry = _KeptFile(module)
            self.kept[module.file_path] = entry
        return entry

    # ---- the walk ----

    def _walk(self, node, module: ModuleType, bindings: dict,
              loose: bool = False) -> None:
        if node is None:
            return
        if isinstance(node, (viba_ast.Constant, viba_ast.Nil, viba_ast.Never,
                             viba_ast.Any, viba_ast.Ellipsis, viba_ast.HintBlock,
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
            if isinstance(argument, viba_ast.HintBlock):
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

    def _is_pattern_file(self, module) -> bool:
        """Is this module one pattern file: a file beside a marker, or a section
        of a document that holds a marker?

        A pattern file is no module of its own — its definitions mean something
        only under a decision — so a name that lands on one is refused.
        """
        file_path = module.file_path
        if not file_path:
            return False
        path = Path(file_path)
        if path.name == GENERIC_FILE:
            return False
        if path.name.endswith(".viba") and self._is_generic_file(path):
            return True
        if not path.name.endswith(".viba.md"):
            return False
        try:
            sections = dict(markdown.files_of(path.read_text(encoding="utf-8")))
        except (OSError, SyntaxError):
            return False
        section = ((getattr(module, "name", "") or "").rsplit(".", 1)[-1]
                   + ".viba")
        return (GENERIC_FILE in sections and section in sections
                and section != GENERIC_FILE)

    # ---- the answer ----

    def _render(self):
        """The kept files as one document's sections, and the line map.

        A section is a file: its path is the section's name, and its body is the
        file's text as the source has it — the imports it uses and the
        definitions kept, in the order the source has them. A file kept whole (a
        generic's) is its own text. The map says, for each body line, the module
        it is a piece of and the line in that module.
        """
        entries = []
        for file_path, entry in self.kept.items():
            module_name = self._module_name(entry, file_path)
            entries.append((_path_of_module(module_name),
                            self._kept_lines(entry, module_name)))
        for module_name, (path, section) in self.whole.items():
            lines = _lines_at(path, section)
            entries.append((_path_of_module(module_name),
                            [(text, module_name, index)
                             for index, text in enumerate(lines, 1)]))
        entries.sort(key=lambda pair: pair[0])
        lines, origin = [], []
        for path, parts in entries:
            lines.append(f"## `{path}`")
            lines.append("")
            lines.append(f"```viba {path}")
            for text, module, module_line in parts:
                lines.append(text)
                origin.append((len(lines), module, module_line))
            lines.append("```")
            lines.append("")
        document = "\n".join(lines).rstrip("\n") + "\n" if lines else ""
        return document, origin

    def _whole_name(self, module: CustomModuleType, file_path=None) -> str:
        """The name a file kept whole answers to: its module's, or its path taken as one."""
        name = getattr(module, "name", "") or ""
        if name:
            return name
        return _module_of_path(self._relative(file_path or module.file_path))

    def _module_name(self, entry: "_KeptFile", file_path) -> str:
        """The module's own name: what it was loaded as, or its path taken as one."""
        name = getattr(entry.module, "name", "") or ""
        if name:
            return name
        return _module_of_path(self._relative(file_path))

    def _kept_lines(self, entry: "_KeptFile", module_name: str):
        """The file's kept statements, line by line, with where each came from."""
        source = self._module_lines(entry, module_name)
        statements = [statement for statement in entry.module.module.body
                      if getattr(statement, "lineno", None)]
        out = []
        for statement in statements:
            if isinstance(statement, viba_ast.Import):
                if (statement.alias or statement.module) not in entry.imports:
                    continue
            elif not (is_definition(statement) and statement.name in entry.names):
                continue
            start = statement.lineno
            after = _lineno_after(statements, statement)
            end = after if after is not None else len(source) + 1
            out.extend((text, module_name, index)
                       for index, text in enumerate(source[start - 1:end - 1], start))
        while out and not out[-1][0].strip():
            out.pop()                     # the file ends at its last line of text
        return out

    def _module_lines(self, entry: "_KeptFile", module_name: str):
        """The module's own lines: the file's, or the section's of a document."""
        return self._module_lines_of(entry.module, module_name)

    def _module_lines_of(self, module, module_name: str):
        file_path = str(module.file_path or "")
        if file_path.endswith(".viba.md"):
            section = module_name.rsplit(".", 1)[-1] + ".viba"
            for name, lines in markdown.lines_of(
                    Path(file_path).read_text(encoding="utf-8")):
                if name == section:
                    return [text for text, _at in lines]
            return []
        return Path(file_path).read_text(encoding="utf-8").splitlines()

    def _relative(self, file_path) -> str:
        return os.path.relpath(Path(file_path).resolve(), self.repo_root)


def _lines_at(path, section=None):
    """The lines of a file, or of one section of a document."""
    text = Path(path).read_text(encoding="utf-8")
    if section is None:
        return text.splitlines()
    for name, lines in markdown.lines_of(text):
        if name == section:
            return [part for part, _at in lines]
    return []


def _path_of_module(module_name: str) -> str:
    """The file a module is: its name's dots become directories, `.viba` at the end."""
    return module_name.replace(".", "/") + ".viba"


def _module_of_path(path: str) -> str:
    """The module a file is: the path without `.viba`, its separators dots."""
    name = path[: -len(".viba")] if path.endswith(".viba") else path
    return name.replace("/", ".")


def _lineno_after(statements, statement):
    """The line the next statement starts on, or None when this is the last."""
    for index, other in enumerate(statements):
        if other is statement:
            return statements[index + 1].lineno if index + 1 < len(statements) else None
    return None

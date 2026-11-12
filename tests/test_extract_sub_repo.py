"""extract_sub_repo: a sub-repo as one document, and where each line came from.

Two entries. `extract_sub_repo_into_viba_markdown` takes a repo — its `.viba`
files, and the `.viba.md` documents inside it — and answers the sub-repo as one
document (one section per file, its path the section's name) plus a map from
every code line to the module and the line it came from.
`extract_sub_viba_markdown` takes a document and answers a document, mapping to
the source document's lines; idempotence is its law for a document that holds
every module it imports.

The map is checked twice over on every answer: a mapped line's text is the
line's text in the source, and every code line of the answer is in the map. The
case files are `tests/data/extract_sub_repo/<case>/` — a repo to cut — and the
documents are this project's own `.viba.md` files.

An error check asserts the message too, not only that the answer is an Err: the
words tell which refusal it is, so two cases cannot pass for each other.

    python3 tests/test_extract_sub_repo.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import markdown
from viba.extract_sub_repo import (extract_sub_repo_into_viba_markdown,
                                   extract_sub_viba_markdown)
from viba.type import Err, Ok

DATA = Path(__file__).resolve().parent / "data" / "extract_sub_repo"
ERRORS = DATA / "errors" / "kb"
DOCUMENTS = Path(__file__).resolve().parent / "data" / "pattern_custom_types"

PASS = 0
FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}\n  expected: {want!r}\n  got:      {got!r}")


def _refused(name: str, answer, words: str) -> None:
    """The answer is an Err, and the message says which refusal it is."""
    check(name, isinstance(answer, Err) and words in answer.msg, True)


def _run(name, type_names, extract=("kb",), viba=("kb",), **kwargs):
    """One extraction on a copy of a case file: (the answer, the case directory)."""
    scratch = Path(tempfile.mkdtemp(prefix="viba-extract-"))
    case = scratch / name
    shutil.copytree(DATA / name, case)
    answer = extract_sub_repo_into_viba_markdown(
        str(case / "kb"),
        [str(case / path) for path in viba],
        [str(case / path) for path in extract],
        list(type_names), **kwargs)
    return answer, case


def _files(document: str) -> dict:
    return dict(markdown.files_of(document))


def _code_lines(document: str):
    """(document line, text) for every line of the answer's code."""
    out = []
    for _name, lines in markdown.lines_of(document):
        out.extend((at, text) for text, at in lines)
    return out


def _source_line(root: Path, module: str, line: int):
    """A module's line in the source: its file, or its document's section."""
    parts = module.split(".")
    path = root.joinpath(*parts).with_suffix(".viba")
    if path.is_file():
        return path.read_text(encoding="utf-8").splitlines()[line - 1]
    for cut in range(1, len(parts)):
        document = root.joinpath(*parts[:cut]).with_suffix(".viba.md")
        if not document.is_file():
            continue
        section = ".".join(parts[cut:]) + ".viba"
        for name, lines in markdown.lines_of(document.read_text(encoding="utf-8")):
            if name == section:
                return lines[line - 1][0]
    return None


def _check_map(document: str, origin: dict, root: Path, label: str) -> None:
    """Every code line is mapped, and each mapped line is the source's line."""
    code = dict(_code_lines(document))
    check(f"{label}: every code line is in the map", sorted(origin), sorted(code))
    wrong = [(at, origin[at], text) for at, text in code.items()
             if at in origin and _source_line(root, *origin[at]) != text]
    check(f"{label}: a mapped line is the source's line", wrong, [])


def run_basic() -> None:
    answer, case = _run("basic", ["report.Report"], viba=("kb", "base"))
    check("basic: answered a document", isinstance(answer, Ok), True)
    document, origin = answer.ok_value
    files = _files(document)
    check("basic: the files the closure reaches", sorted(files),
          ["item.viba", "report.viba"])
    check("basic: the imports the definitions use, and the definition",
          files["report.viba"],
          "import item as it\nimport meta as mt\n"
          "Report = Object * $items list[it.Item] * $meta mt.Meta\n")
    check("basic: a definition is written as the source has it",
          files["item.viba"], "Item = Object * $id int * $name str\n")
    check("basic: a definition that was not reached is not a section",
          "Extra = bool" in files["item.viba"], False)
    check("basic: a leaf primitive's definition is not a section",
          "base/meta.viba" in files, False)
    check("basic: nothing is written next to the repo",
          sorted(path.name for path in case.iterdir()), ["base", "kb"])
    _check_map(document, origin, case / "kb", "basic")

    one, _case = _run("basic", ["item.Item"], viba=("kb", "base"))
    check("basic: one definition, one file", sorted(_files(one.ok_value[0])),
          ["item.viba"])

    empty, _case = _run("basic", [], viba=("kb", "base"))
    check("basic: no type names, no sections", _files(empty.ok_value[0]), {})


def run_generics() -> None:
    chosen, case = _run("generic_chosen", ["use.Gauge"])
    document, origin = chosen.ok_value
    files = _files(document)
    check("generic: the marker and the file the decision chose", sorted(files),
          ["G/100.viba", "G/__generic__.viba", "use.viba"])
    check("generic: a file kept whole is as the source has it",
          files["G/__generic__.viba"], "# G: one parameter, kept whole.\n")
    check("generic: the pattern file too", files["G/100.viba"],
          "pattern int\n\n__decl__ = int\n")
    check("generic: the application stands in its own file",
          files["use.viba"], "import G\n\nGauge = G[int]\n")
    _check_map(document, origin, case / "kb", "generic")

    both, _case = _run("generic_chosen", ["use.Gauge", "use.Named"])
    check("generic: two applications keep two files",
          sorted(_files(both.ok_value[0])),
          ["G/100.viba", "G/200.viba", "G/__generic__.viba", "use.viba"])

    fallback, _case = _run("generic_fallback", ["use.Only"])
    check("generic: a decision the source cannot make keeps every file",
          sorted(_files(fallback.ok_value[0])),
          ["G/100.viba", "G/__generic__.viba", "use.viba"])

    builtin, _case = _run("builtin_generic", ["colour.Colour"])
    check("generic: the builtin vocabulary is not copied",
          sorted(_files(builtin.ok_value[0])), ["colour.viba"])

    primitive, _case = _run("primitive_generic", ["use.Wrapped"],
                            viba=("kb", "base"))
    check("generic: a generic outside the repo is a leaf primitive",
          sorted(_files(primitive.ok_value[0])), ["use.viba"])


def run_walks() -> None:
    member, _case = _run("member", ["use.Wrapped"])
    check("member: the file the decision chose",
          sorted(_files(member.ok_value[0])),
          ["M/100.viba", "M/__generic__.viba", "use.viba"])
    check("member: the member take is as the source has it",
          _files(member.ok_value[0])["use.viba"],
          "import M\n\nWrapped = M[int].value\n")

    call, _case = _run("call", ["step.Call"])
    check("call: the head, its arguments and what they use",
          sorted(_files(call.ok_value[0])), ["step.viba"])
    check("call: only the reached definitions are kept",
          _files(call.ok_value[0])["step.viba"],
          "Step = Object * $n int\n\nRun = Step <- $env Env <- $step Step\n\n"
          "Call = Run << $env Env << $step Step\n")

    primitives, _case = _run("primitives", ["use.Use"],
                             viba=("kb", "kind", "other"))
    check("primitives: nothing but the reachable file is written",
          sorted(_files(primitives.ok_value[0])), ["use.viba"])
    check("primitives: the leaf primitives' imports stay",
          _files(primitives.ok_value[0])["use.viba"],
          "import late as l\nimport early as e\n\n"
          "Use = Object * $y e.Early * $x l.Late\n")

    units, _case = _run("units", ["doc.Doc"])
    check("units: Assert is a unit by default",
          sorted(_files(units.ok_value[0])), ["doc.viba"])
    check("units: the unit is not a section",
          _files(units.ok_value[0])["doc.viba"],
          "Doc = str * Assert[{ the text is a path }]\n")
    strict, _case = _run("units", ["doc.Doc"], nil_eqv=[])
    _refused("units: an empty nil_eqv leaves Assert unresolved", strict,
             "cannot resolve 'Assert'")
    own, _case = _run("units", ["doc.Doc"], nil_eqv=["Assert", "Hint"])
    check("units: a list given is the list used",
          sorted(_files(own.ok_value[0])), ["doc.viba"])


def run_deep_and_multi() -> None:
    deep, case = _run("deep", ["top.Top"])
    document, origin = deep.ok_value
    files = _files(document)
    check("deep: every file at its own path", sorted(files),
          ["deep/helper.viba", "pkg/a/b/choice/100.viba",
           "pkg/a/b/choice/__generic__.viba", "pkg/a/b/leaf.viba",
           "pkg/a/mid.viba", "top.viba"])
    check("deep: the top keeps both imports and only what it reached",
          files["top.viba"],
          "import pkg.a.mid as mid\nimport pkg.a.b.leaf as leaf\n\n"
          "Top = Object * $mid mid.Mid * $leaf leaf.Leaf\n")
    check("deep: a nested file keeps only its own imports",
          files["pkg/a/b/leaf.viba"],
          "import deep.helper as h\nimport pkg.a.b.choice as ch\n\n"
          "Leaf = Object * $n int * $h h.Helper * $c ch[int]\n")
    check("deep: a nested generic keeps the file the decision chose",
          files["pkg/a/b/choice/100.viba"], "pattern A\n\n__decl__ = A\n")
    check("deep: an unreached directory is not written",
          "unused/x.viba" in files, False)
    _check_map(document, origin, case / "kb", "deep")

    both, _case = _run("deep", ["top.Top", "unused.x.Unused"])
    check("deep: a second, disjoint name brings its own file",
          "unused/x.viba" in _files(both.ok_value[0]), True)

    leaf, _case = _run("deep", ["pkg.a.b.leaf.Leaf"])
    check("deep: a type named in a deep module reaches only its own files",
          sorted(_files(leaf.ok_value[0])),
          ["deep/helper.viba", "pkg/a/b/choice/100.viba",
           "pkg/a/b/choice/__generic__.viba", "pkg/a/b/leaf.viba"])

    multi, _case = _run("multi_extract", ["one.a.A"],
                        extract=("kb/one", "kb/two"))
    check("multi extract: two roots, the rest of the repo a leaf primitive",
          sorted(_files(multi.ok_value[0])),
          ["one/a.viba", "one/inner/extra.viba", "two/sub/b.viba"])
    check("multi extract: a leaf primitive's import stays",
          _files(multi.ok_value[0])["one/a.viba"],
          "import one.inner.extra as ex\nimport two.sub.b as b\n"
          "import three.deep.c as c\n\n"
          "A = Object * $ex ex.Extra * $b b.B * $c c.C\n")

    multi_path, _case = _run("multi_viba_path", ["use.Use"],
                             viba=("kb", "vp1", "vp2", "vp3"))
    check("multi viba_path: only the repo's file is written",
          sorted(_files(multi_path.ok_value[0])), ["use.viba"])
    check("multi viba_path: the use-site order does not decide it",
          _files(multi_path.ok_value[0])["use.viba"],
          "import one as o1\nimport two as o2\nimport three as o3\n\n"
          "Use = Object * $c o3.Three * $b o2.Two * $a o1.One\n")


def run_documents() -> None:
    scratch = Path(tempfile.mkdtemp(prefix="viba-extract-doc-"))
    kb = scratch / "pattern_custom_types"
    shutil.copytree(DOCUMENTS, kb)
    answer = extract_sub_repo_into_viba_markdown(
        str(kb), [str(kb)], [str(kb)], ["ask_red.__impl__"])
    check("documents: answered a document", isinstance(answer, Ok), True)
    document, origin = answer.ok_value
    files = _files(document)
    check("documents: a definition reached inside a .viba.md is a section",
          "types/colors.viba" in files, True)
    check("documents: its text is the document's section",
          files["types/colors.viba"], "Red   = tagged[__type__, Object]\n")
    check("documents: a generic written as a document keeps its marker",
          files["is_red/__generic__.viba"], "# 标记：这份文档是一个泛型。\n")
    check("documents: and the file the decision chose",
          files["is_red/100.viba"],
          "import types.colors as colors\n\npattern colors.Red\n\nvalue = true\n")
    check("documents: the design using it", files["ask_red.viba"],
          "import is_red as is_red\nimport types.colors as colors\n\n"
          "__impl__ = is_red[colors.Red].value\n")
    _check_map(document, origin, kb, "documents")

    again = extract_sub_viba_markdown(document, ["ask_red.__impl__"])
    check("document entry: answered a document", isinstance(again, Ok), True)
    second, places = again.ok_value
    check("document entry: the same document", second, document)
    code = dict(_code_lines(second))
    check("document entry: every code line is in the map",
          sorted(places), sorted(code))
    source = document.splitlines()
    check("document entry: a mapped line is the source document's line",
          [at for at, text in code.items()
           if at in places and source[places[at] - 1] != text], [])
    third = extract_sub_viba_markdown(second, ["ask_red.__impl__"])
    check("document entry: idempotent",
          isinstance(third, Ok) and third.ok_value[0], second)

    # a document is the whole source: a leaf primitive is not in it, so an
    # answer that still imports one cannot be cut again on its own
    basic, _case = _run("basic", ["report.Report"], viba=("kb", "base"))
    once, _source_map = basic.ok_value
    twice = extract_sub_viba_markdown(once, ["report.Report"])
    _refused("document entry: a leaf primitive is outside the document", twice,
             "cannot resolve 'mt.Meta'")

    # a definition in a pattern file is kept by an application, not by a name
    inside = extract_sub_repo_into_viba_markdown(
        str(kb), [str(kb)], [str(kb)], ["is_red.100.value"])
    _refused("documents: a name inside a pattern file", inside,
             "is a definition in a pattern file")

    # a section may be a document of its own: its files are the document's, and
    # a line of the answer is a line of the section that document is
    nested = extract_sub_viba_markdown(
        "## `docs/things.viba.md`\n\n"
        "````viba docs/things.viba.md\n"
        "## `colors.viba`\n\n```viba colors.viba\nRed = Object\n```\n"
        "````\n", ["docs.things.colors.Red"])
    check("document entry: a section that is a document",
          isinstance(nested, Ok), True)
    files = _files(nested.ok_value[0])
    check("document entry: its files come out at their own paths",
          sorted(files), ["docs/things/colors.viba"])
    check("document entry: its text", files["docs/things/colors.viba"],
          "Red = Object\n")
    check("document entry: its line is the outer section's line",
          nested.ok_value[1], {4: 7})

    # a file named with the dots of its module is the module too
    dotted = extract_sub_viba_markdown(
        "## `a.b.c.viba`\n\n```viba a.b.c.viba\nX = int\n```\n", ["a.b.c.X"])
    check("document entry: a file named with its dots",
          dotted.ok_value[1], {4: 4})


def run_errors() -> None:
    # Walking the names
    missing, _case = _run("basic", ["nope.Nope"], viba=("kb", "base"))
    _refused("error: a module the path cannot find", missing,
             "cannot load the module 'nope' of 'nope.Nope'")
    outside, _case = _run("basic", ["meta.Meta"], viba=("kb", "base"))
    _refused("error: a definition outside the extraction source", outside,
             "'meta.Meta' is not in the extraction source")
    short, _case = _run("errors", ["Fine"])
    _refused("error: a name without a module", short,
             "'Fine' is no type name: it is a module and a definition")
    absent, _case = _run("basic", ["report.Unrelated2"], viba=("kb", "base"))
    _refused("error: a name the module does not define", absent,
             "no type 'report.Unrelated2'")
    builtin_name, _case = _run("basic", ["report.int"], viba=("kb", "base"))
    _refused("error: a name that is no definition", builtin_name,
             "'report.int' is no definition")
    broken, _case = _run("errors", ["broken.Broken"])
    _refused("error: a name used in a definition does not resolve", broken,
             "cannot resolve 'Missing'")
    generic_name, _case = _run("generic_chosen", ["G.value"])
    _refused("error: a name that lands on a generic directory", generic_name,
             "'G.value' names a generic")
    pattern_name, _case = _run("generic_chosen", ["G.100.__decl__"])
    _refused("error: a name that lands in a pattern file", pattern_name,
             "is a definition in a pattern file")

    # Walking the definitions
    broken_pattern, _case = _run("broken_pattern", ["use.Only"])
    _refused("error: a pattern file that does not parse", broken_pattern,
             "cannot parse")
    no_member, _case = _run("member_missing", ["use.Wrapped"])
    _refused("error: a member the chosen file does not define", no_member,
             "no definition 'other' in")
    bare, _case = _run("cannot_take", ["use.Bare"])
    _refused("error: an application whose import cannot load", bare,
             "cannot take 'nope.G' in")
    taken, _case = _run("cannot_take", ["use.Taken"])
    _refused("error: a member whose import cannot load", taken,
             "cannot take 'nope.G' in")

    # The arguments
    no_repo = extract_sub_repo_into_viba_markdown("/no/such/repo", [], [], ["a.B"])
    _refused("error: no such repo", no_repo, "no such repo directory")
    wrong_repo = extract_sub_repo_into_viba_markdown(3, [], [], ["a.B"])
    _refused("error: the repo is a path", wrong_repo, "repo is a path, not int")
    wrong_path = extract_sub_repo_into_viba_markdown(
        str(ERRORS), "kb", [], ["a.B"])
    _refused("error: the viba path is a list", wrong_path,
             "viba_path is a list of strings, not one string")
    no_path = extract_sub_repo_into_viba_markdown(str(ERRORS), 3, [], ["a.B"])
    _refused("error: the viba path is not one value", no_path,
             "viba_path is a list of strings, not int")
    wrong_source = extract_sub_repo_into_viba_markdown(
        str(ERRORS), [], "kb", ["a.B"])
    _refused("error: the extraction source is a list", wrong_source,
             "extract_source_viba_path is a list of strings, not one string")
    no_names = extract_sub_repo_into_viba_markdown(str(ERRORS), [], [], None)
    _refused("error: the type names are a list", no_names,
             "type_names is a list of strings")
    one_name = extract_sub_repo_into_viba_markdown(str(ERRORS), [], [], "a.B")
    _refused("error: the type names are not one string", one_name,
             "type_names is a list of strings, not one string")
    bad_name = extract_sub_repo_into_viba_markdown(str(ERRORS), [], [], [1])
    _refused("error: a type name is not a string", bad_name,
             "type_names holds a string, not int")
    bad_unit = extract_sub_repo_into_viba_markdown(
        str(ERRORS), [], [], [], never_eqv=[1])
    _refused("error: a unit name is not a string", bad_unit,
             "never_eqv holds a string, not int")
    not_list = extract_sub_repo_into_viba_markdown(
        str(ERRORS), [], [], [], nil_eqv="Oneof")
    _refused("error: the nil units are a list", not_list,
             "nil_eqv is a list of strings, not one string")

    # The document entry
    wrong_text = extract_sub_viba_markdown(3, ["a.B"])
    _refused("document entry: the source is a document's text", wrong_text,
             "source_viba_md is a document's text, not int")
    unclosed = extract_sub_viba_markdown(
        "## `x.viba`\n\n```viba x.viba\nX = int\n", ["x.X"])
    _refused("document entry: a block that never closes", unclosed,
             "a fenced code block is not closed at the end of the document")
    unnamed = extract_sub_viba_markdown(
        "## `x.viba`\n\n```viba x.viba\nX = int\n```\n", ["nope.Nope"])
    _refused("document entry: a type the document does not name", unnamed,
             "cannot load the module 'nope'")
    no_name = extract_sub_viba_markdown("##\n\n```viba\nX = int\n```\n", ["x.X"])
    _refused("document entry: a section with no name", no_name,
             "a section needs a file name")
    two_names = extract_sub_viba_markdown(
        "## `x.viba`\n\n```viba x.viba\nX = int\n```\n\n"
        "## `x.viba`\n\n```viba x.viba\nY = int\n```\n", ["x.X"])
    _refused("document entry: two sections naming one file", two_names,
             "two sections name the file 'x.viba'")
    doc_names = extract_sub_viba_markdown(
        "## `x.viba`\n\n```viba x.viba\nX = int\n```\n", "x.X")
    _refused("document entry: the type names are not one string", doc_names,
             "type_names is a list of strings, not one string")

    # A section is a file of the document's directory, and nothing else
    escaped = Path(tempfile.gettempdir()) / f"viba-escape-{os.getpid()}.viba"
    if escaped.exists():
        escaped.unlink()
    escaped_answer = extract_sub_viba_markdown(
        f"## `../{escaped.name}`\n\n```viba ../{escaped.name}\nX = int\n```\n",
        [f"{escaped.stem}.X"])
    _refused("document entry: a section above the directory", escaped_answer,
             "a section names a file outside the document's directory")
    check("document entry: nothing is written above the directory",
          escaped.exists(), False)
    absolute = extract_sub_viba_markdown(
        "## `/tmp/viba-absolute.viba`\n\n"
        "```viba /tmp/viba-absolute.viba\nX = int\n```\n", ["viba-absolute.X"])
    _refused("document entry: a section that is an absolute path", absolute,
             "a section names a file outside the document's directory")
    collision = extract_sub_viba_markdown(
        "## `a`\n\n```viba a\ntext\n```\n\n"
        "## `a/b.viba`\n\n```viba a/b.viba\nX = int\n```\n", ["a.b.X"])
    _refused("document entry: a section where another needs a directory",
             collision, "and another section needs it to be a directory")


def main() -> int:
    run_basic()
    run_generics()
    run_walks()
    run_deep_and_multi()
    run_documents()
    run_errors()
    print(f"extract_sub_repo: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

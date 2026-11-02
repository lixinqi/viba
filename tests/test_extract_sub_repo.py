"""Tests for extract_sub_repo: one repo in, its sub-repo out.

Every case is a small repo under data/extract_sub_repo, copied into a temporary
directory so the sub-repo is written somewhere disposable and the source tree is
never touched:

    data/extract_sub_repo/basic/            kb/ + base/: a primitive, an unused
                                            definition, an unused import
    data/extract_sub_repo/generic_chosen/   one generic, two applications
    data/extract_sub_repo/generic_fallback/ a decision the source cannot make
    data/extract_sub_repo/builtin_generic/  Either, which is the interpreter's
    data/extract_sub_repo/primitive_generic/a generic outside the repo
    data/extract_sub_repo/member/           g[T].value
    data/extract_sub_repo/call/             a call, and the definitions it uses
    data/extract_sub_repo/units/            Assert, a name that stands for nil
    data/extract_sub_repo/deep/             three levels of modules, six files
    data/extract_sub_repo/multi_extract/    two extraction roots, the rest of the
                                            repo being a primitive
    data/extract_sub_repo/multi_viba_path/  three search roots, in reverse use order
    data/extract_sub_repo/errors/           a name that does not resolve

    python3 tests/test_extract_sub_repo.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.extract_sub_repo import SubRepo, extract_sub_repo
from viba.type import Ok, VibaProgramErr

DATA = Path(__file__).resolve().parent / "data" / "extract_sub_repo"

PASS = 0
FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}\n  expected: {want!r}\n  got:      {got!r}")


def check_true(name: str, got) -> None:
    check(name, bool(got), True)


def content(path: Path) -> dict:
    """Every file under `path`, by its path relative to it, with its text."""
    found = {}
    if not path.is_dir():
        return found
    for one in sorted(path.rglob("*")):
        if one.is_file():
            found[str(one.relative_to(path))] = one.read_text(encoding="utf-8")
    return found


def slice_case(name: str, type_names, extract=("kb",), viba=("kb", "base"),
               never_eqv=None, nil_eqv=None):
    """Slice a copy of one case: the answer, the files, and both trees' content."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "case"
        shutil.copytree(DATA / name, root)
        repo = root / "kb"
        before = content(repo)
        answer = extract_sub_repo(
            str(repo),
            [str(root / one) for one in viba],
            [str(root / one) for one in extract],
            list(type_names),
            never_eqv=never_eqv,
            nil_eqv=nil_eqv,
        )
        result = {"answer": answer, "before": before, "after": content(repo)}
        if isinstance(answer, Ok):
            sub = Path(answer.ok_value.repo)
            result["repo"] = answer.ok_value.repo
            result["viba_path"] = answer.ok_value.viba_path
            result["files"] = content(sub)
            result["sub"] = answer.ok_value.repo
            result["names"] = [Path(one).name for one in answer.ok_value.viba_path]
        return result


def slice_twice(name: str, type_names, extract=("kb",), viba=("kb", "base")):
    """Slice, then slice what came out; the two contents and the two answers."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "case"
        shutil.copytree(DATA / name, root)
        first = extract_sub_repo(
            str(root / "kb"), [str(root / one) for one in viba],
            [str(root / one) for one in extract], list(type_names))
        if not isinstance(first, Ok):
            return first, None, None
        second = extract_sub_repo(
            first.ok_value.repo, first.ok_value.viba_path,
            first.ok_value.viba_path, list(type_names))
        if not isinstance(second, Ok):
            return first, second, None
        return (first, second,
                (content(Path(first.ok_value.repo)),
                 content(Path(second.ok_value.repo))))


def run_basic() -> None:
    answer = slice_case("basic", ["report.Report"])
    sub = answer["answer"]
    check_true("basic: answered a sub-repo", isinstance(sub, Ok))
    check("basic: the answer carries the repo and its path",
          isinstance(sub.ok_value, SubRepo), True)
    check("basic: the sub-repo is beside the repo",
          answer["names"], ["kb-sub", "base"])
    check("basic: the answer's viba_path starts at the sub-repo",
          answer["viba_path"][0], answer["repo"])
    check("basic: the source tree is untouched", answer["after"], answer["before"])
    check("basic: the files", sorted(answer["files"]),
          ["item.viba", "report.viba"])
    check("basic: report keeps its definitions and the imports it uses",
          answer["files"]["report.viba"],
          "import item as it\n"
          "import meta as mt\n"
          "\n"
          "Report =\n"
          "  Object * $items list[it.Item] * $meta mt.Meta\n")
    check("basic: item keeps only what was reached",
          answer["files"]["item.viba"],
          "Item =\n  Object * $id int * $name str\n")

    one = slice_case("basic", ["item.Item"])
    check("basic: one definition, one file", sorted(one["files"]), ["item.viba"])
    check("basic: nothing outside was reached", one["names"], ["kb-sub"])

    empty = slice_case("basic", [])
    check("basic: no type names keeps nothing", empty["files"], {})


def run_generics() -> None:
    chosen = slice_case("generic_chosen", ["use.Gauge"])
    check("generic: only the file the decision chose",
          sorted(chosen["files"]),
          ["G/100.viba", "G/__generic__.viba", "use.viba"])
    check("generic: the marker is kept as it is",
          chosen["files"]["G/__generic__.viba"], "# G: one parameter, kept whole.\n")
    check("generic: the pattern file is kept as it is",
          chosen["files"]["G/100.viba"], "pattern int\n\n__decl__ = int\n")

    both = slice_case("generic_chosen", ["use.Gauge", "use.Named"])
    check("generic: two applications keep two files",
          sorted(both["files"]),
          ["G/100.viba", "G/200.viba", "G/__generic__.viba", "use.viba"])

    fallback = slice_case("generic_fallback", ["use.Only"])
    check("generic: a decision the source cannot make keeps every file",
          sorted(fallback["files"]),
          ["G/100.viba", "G/__generic__.viba", "use.viba"])

    builtin = slice_case("builtin_generic", ["colour.Colour"])
    check("generic: the builtin vocabulary is not copied",
          sorted(builtin["files"]), ["colour.viba"])
    check("generic: and not named on the viba_path", builtin["names"], ["kb-sub"])

    primitive = slice_case("primitive_generic", ["use.Wrapped"])
    check("generic: a generic outside the repo is a primitive",
          sorted(primitive["files"]), ["use.viba"])
    check("generic: its root is on the viba_path", primitive["names"],
          ["kb-sub", "base"])


def run_walks() -> None:
    member = slice_case("member", ["use.Wrapped"])
    check("member: the chosen file's definition is kept", sorted(member["files"]),
          ["M/100.viba", "M/__generic__.viba", "use.viba"])
    check("member: the member take is spelled again",
          member["files"]["use.viba"], "import M\n\nWrapped =\n  M[int].value\n")

    call = slice_case("call", ["step.Call"])
    check("call: the head, its arguments and what they use",
          sorted(call["files"]), ["step.viba"])
    check("call: only the reached definitions are kept",
          call["files"]["step.viba"],
          "Step =\n"
          "  Object * $n int\n"
          "\n"
          "Run =\n"
          "  (Step <- $env Env) <- $step Step\n"
          "\n"
          "Call =\n"
          "  Run << $env Env << $step Step\n")

    primitives = slice_case("primitives", ["use.Use"],
                            viba=("kb", "kind", "other"))
    check("primitives: the roots follow the viba_path", primitives["names"],
          ["kb-sub", "kind", "other"])
    check("primitives: nothing but the reachable file is copied",
          sorted(primitives["files"]), ["use.viba"])

    units = slice_case("units", ["doc.Doc"])
    check("units: Assert is a unit by default", sorted(units["files"]), ["doc.viba"])
    strict = slice_case("units", ["doc.Doc"], nil_eqv=[])
    check_true("units: an empty nil_eqv leaves Assert unresolved",
               isinstance(strict["answer"], VibaProgramErr))
    own = slice_case("units", ["doc.Doc"], nil_eqv=["Assert", "Hint"])
    check("units: a list given is the list used", sorted(own["files"]), ["doc.viba"])


def run_shapes() -> None:
    """Deep directories, several files, several roots on each side."""
    deep = slice_case("deep", ["top.Top"])
    check("deep: every file at its own path", sorted(deep["files"]),
          ["deep/helper.viba", "pkg/a/b/choice/100.viba",
           "pkg/a/b/choice/__generic__.viba", "pkg/a/b/leaf.viba",
           "pkg/a/mid.viba", "top.viba"])
    check("deep: the top keeps both imports and only what it reached",
          deep["files"]["top.viba"],
          "import pkg.a.mid as mid\n"
          "import pkg.a.b.leaf as leaf\n"
          "\n"
          "Top =\n"
          "  Object * $mid mid.Mid * $leaf leaf.Leaf\n")
    check("deep: a nested file keeps only its own imports",
          deep["files"]["pkg/a/b/leaf.viba"],
          "import deep.helper as h\n"
          "import pkg.a.b.choice as ch\n"
          "\n"
          "Leaf =\n"
          "  Object * $n int * $h h.Helper * $c ch[int]\n")
    check("deep: a nested generic keeps the file the decision chose",
          deep["files"]["pkg/a/b/choice/100.viba"],
          "pattern A\n\n__decl__ = A\n")
    check("deep: an unreached directory is not copied",
          "unused/x.viba" in deep["files"], False)
    check("deep: nothing outside the repo", deep["names"], ["kb-sub"])

    both = slice_case("deep", ["top.Top", "unused.x.Unused"])
    check("deep: a second, disjoint name brings its own file",
          "unused/x.viba" in both["files"], True)

    leaf = slice_case("deep", ["pkg.a.b.leaf.Leaf"])
    check("deep: a type named in a deep module reaches only its own files",
          sorted(leaf["files"]),
          ["deep/helper.viba", "pkg/a/b/choice/100.viba",
           "pkg/a/b/choice/__generic__.viba", "pkg/a/b/leaf.viba"])

    multi = slice_case("multi_extract", ["one.a.A"],
                       extract=("kb/one", "kb/two"), viba=("kb",))
    check("multi extract: two roots, the rest of the repo a primitive",
          multi["names"], ["kb-sub", "kb"])
    check("multi extract: the files", sorted(multi["files"]),
          ["one/a.viba", "one/inner/extra.viba", "two/sub/b.viba"])
    check("multi extract: the primitive's import stays",
          multi["files"]["one/a.viba"],
          "import one.inner.extra as ex\n"
          "import two.sub.b as b\n"
          "import three.deep.c as c\n"
          "\n"
          "A =\n"
          "  Object * $ex ex.Extra * $b b.B * $c c.C\n")

    multi_path = slice_case("multi_viba_path", ["use.Use"],
                            viba=("kb", "vp1", "vp2", "vp3"))
    check("multi viba_path: the answer follows the path's order",
          multi_path["names"], ["kb-sub", "vp1", "vp2", "vp3"])
    check("multi viba_path: only the repo's file is copied",
          sorted(multi_path["files"]), ["use.viba"])
    check("multi viba_path: the use-site order does not decide it",
          multi_path["files"]["use.viba"],
          "import one as o1\n"
          "import two as o2\n"
          "import three as o3\n"
          "\n"
          "Use =\n"
          "  Object * $c o3.Three * $b o2.Two * $a o1.One\n")


def run_idempotence() -> None:
    for name, type_names, extract, viba in (
            ("basic", ["report.Report"], ("kb",), ("kb", "base")),
            ("generic_chosen", ["use.Gauge"], ("kb",), ("kb",)),
            ("primitive_generic", ["use.Wrapped"], ("kb",), ("kb", "base")),
            ("deep", ["top.Top"], ("kb",), ("kb",)),
            ("multi_extract", ["one.a.A"], ("kb/one", "kb/two"), ("kb",)),
            ("multi_viba_path", ["use.Use"], ("kb",),
             ("kb", "vp1", "vp2", "vp3"))):
        first, second, contents = slice_twice(name, type_names, extract, viba)
        check_true(f"idempotence: {name} answered twice",
                   isinstance(first, Ok) and isinstance(second, Ok))
        if contents is None:
            continue
        check(f"idempotence: {name} keeps the same content",
              contents[1], contents[0])


def run_errors() -> None:
    answer = extract_sub_repo("/no/such/repo", [], [], ["a.B"])
    check_true("error: no such repo", isinstance(answer, VibaProgramErr))

    short = slice_case("errors", ["Fine"])
    check_true("error: a name without a module", isinstance(short["answer"],
                                                            VibaProgramErr))

    outside = slice_case("basic", ["meta.Meta"], viba=("kb", "base"))
    check_true("error: a name outside the extraction source",
               isinstance(outside["answer"], VibaProgramErr))

    broken = slice_case("errors", ["broken.Broken"])
    check_true("error: a name that does not resolve",
               isinstance(broken["answer"], VibaProgramErr))
    if isinstance(broken["answer"], VibaProgramErr):
        check_true("error: the message names the name",
                   "Missing" in broken["answer"].msg)

    wrong_repo = extract_sub_repo(3, [], [], ["a.B"])
    check_true("error: the repo is a path", isinstance(wrong_repo, VibaProgramErr))
    wrong_names = extract_sub_repo(str(DATA / "errors" / "kb"), [], [], "a.B")
    check_true("error: the type names are a list",
               isinstance(wrong_names, VibaProgramErr))
    wrong_unit = extract_sub_repo(str(DATA / "errors" / "kb"), [], [], [],
                                  never_eqv=[1])
    check_true("error: the unit names are strings",
               isinstance(wrong_unit, VibaProgramErr))


def main() -> int:
    run_basic()
    run_generics()
    run_walks()
    run_shapes()
    run_idempotence()
    run_errors()
    print(f"extract_sub_repo: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for is_complete: take the corpora as designs and ask if they are complete.

All material comes from the repo, nothing new is invented:

    data/type_descriptor/case_081/     deep entry (import util as base) + util.viba
    data/is_sub_type/sub030.viba       recursive Chain = $head int * $tail Chain | nil
    data/is_sub_type/sup025.viba       X = ...

    python3 tests/test_is_complete.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.is_complete import is_complete

DATA = Path(__file__).resolve().parent / "data"
CASES = DATA / "type_descriptor" / "case_081"
TYPES = DATA / "is_sub_type"

PASS = FAIL = 0


def check(name: str, got, want):
    global PASS, FAIL
    if got is want:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}: expected {want}, got {got}")


def read(path: Path) -> str:
    return path.read_text()


def entry() -> str:
    """The deep entry of case_081: import util as base."""
    return read(CASES / "lv1" / "lv2" / "lv3" / "leaf.viba")


def main():
    # Where the dependencies come from: viba_paths, library, or nowhere
    check("deep entry + viba_paths", is_complete(entry(), [], [str(CASES)], set()), True)
    check("deep entry, no dependencies",
          is_complete(entry(), [], [], set()), False)
    check("deep entry, dependencies through library",
          is_complete(entry(), [("util.viba", read(CASES / "util.viba"))], [], set()), True)

    # A design with documentation blocks: the terminators are the caller's words
    demo = ("Note[T] = $text T\n"
            "Report = Object * $len Note[int] * $check Checked[{len under 50}]\n")
    stops = {"Note", "Checked"}
    check("a design with a code block, no terminators",
          is_complete(demo, [], [], set()), False)
    check("the same design with the caller's terminators",
          is_complete(demo, [], [], stops), True)
    check("only one of them (the wrapper over the code block is still there)",
          is_complete(demo, [], [], {"Note"}), False)

    # Recursive definitions: coinduction lets them through
    check("recursive Chain", is_complete(read(TYPES / "sub030.viba"), [], [], set()), True)

    # Ellipsis: incomplete by default, ends a walk once it is a terminator
    ellipsis = read(TYPES / "sup025.viba")
    check("ellipsis, no terminators", is_complete(ellipsis, [], [], set()), False)
    check("ellipsis + \"...\"", is_complete(ellipsis, [], [], {"..."}), True)

    # A name that does not resolve: no terminator can rescue it
    broken = "X = $a Missing\n"
    check("unresolved reference", is_complete(broken, [], [], set()), False)
    check("unresolved reference + \"...\"", is_complete(broken, [], [], {"..."}), False)

    # Edges: an entry that does not compile, and an empty one
    check("entry does not compile", is_complete("X = ", [], [], set()), False)
    check("empty file", is_complete("", [], [], set()), True)
    check("a lexical error is a compile error",
          is_complete("X = A @ B", [], [], set()), False)
    check("an unterminated code block is a compile error",
          is_complete("X = {never closed", [], [], set()), False)
    check("a source written with CRLF",
          is_complete("X =\r\n  int\r\n", [], [], set()), True)
    check("a byte-order mark is a compile error",
          is_complete("\ufeffX = int", [], [], set()), False)

    corners()

    print(f"is_complete: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


def corners():
    """边角：顶类型、内建定义、终结符、泛型实参、依赖从哪来。"""
    # The top type has no members, and no leaf reads out of it either — unlike
    # never, which admits no material at all. So a design resting on it does
    # not walk through.
    check("Any alone", is_complete("X = Any", [], [], set()), False)
    check("Any as a member", is_complete("X = int * Any", [], [], set()), False)
    check("Any under a container", is_complete("X = list[Any]", [], [], set()), False)

    # Builtin definitions count: builtin.viba's own names resolve too.
    check("a builtin definition behind a member",
          is_complete("X = $env Environment", [], [], set()), True)
    check("a design carrying a doc block, behind a member",
          is_complete("Block[T] = $text T * $note {what it means}\nX = $b Block[int]\n",
                      [], [], set()), False)

    # A code block ends a walk only when the name wrapped around it is a
    # terminator, and the terminator is asked about the application.
    check("Hint without the terminator",
          is_complete("X = Hint[$python_code {a}]", [], [], set()), False)
    check("Hint with the terminator",
          is_complete("X = Hint[$python_code {a}]", [], [], {"Hint"}), True)
    check("Predicate without the terminator",
          is_complete("X = Predicate[{a}, $python_code {b}]", [], [], set()), False)
    check("Predicate with the terminator",
          is_complete("X = Predicate[{a}, $python_code {b}]", [], [], {"Predicate"}), True)

    # A generic parameter is walked through the argument an application binds
    # it to; unbound, it is a placeholder and decides nothing.
    check("an unbound parameter alone", is_complete("W[T] = T\n", [], [], set()), True)
    check("a parameter bound to a leaf",
          is_complete("W[T] = T\nA = W[int]\n", [], [], set()), True)
    check("a parameter bound to a code block",
          is_complete("W[T] = T\nA = W[{code}]\n", [], [], set()), False)
    check("a parameter bound behind a member",
          is_complete("W[T] = $a T\nA = W[int]\nW2[T] = $b T\n", [], [], set()), True)
    check("too many arguments",
          is_complete("W[T] = T\nA = W[int, str]\n", [], [], set()), False)
    check("too few arguments",
          is_complete("W[T, U] = T * U\nA = W[int]\n", [], [], set()), False)

    # A dependency that does not compile is skipped; the name it would have
    # defined then does not resolve.
    entry = "import util\nX = util.M\n"
    check("a dependency that does not compile",
          is_complete(entry, [("util.viba", "M = ")], [], set()), False)
    check("a library directory that is not there",
          is_complete(entry, [], ["/no/such/directory"], set()), False)
    check("the first library entry keeps the module name",
          is_complete(entry, [("util.viba", "M = int\n"),
                              ("util.viba", "M = {code}\n")], [], set()), True)

    # A dotted module name is the path under the directory, and a dotted
    # import binds its whole name (`import pkg.mod` binds `pkg.mod`).
    with tempfile.TemporaryDirectory() as directory:
        package = Path(directory) / "pkg"
        package.mkdir()
        (package / "mod.viba").write_text("M = int\n")
        check("a dotted import with no alias",
              is_complete("import pkg.mod\nX = pkg.mod.M\n", [], [directory], set()), True)
        check("a dotted import with an alias",
              is_complete("import pkg.mod as m\nX = m.M\n", [], [directory], set()), True)
        check("a dotted name through the parent module",
              is_complete("import pkg\nX = pkg.mod.M\n", [], [directory], set()), True)
        check("a dotted name that is not there",
              is_complete("import pkg\nX = pkg.mod.Nope\n", [], [directory], set()), False)
        check("the same directory twice",
              is_complete("import pkg.mod\nX = pkg.mod.M\n", [], [directory, directory], set()),
              True)

    # A terminator may be a plain name, not only an application.
    check("a name as a terminator", is_complete("X = Foo", [], [], {"Foo"}), True)
    check("and the same name without it", is_complete("X = Foo", [], [], set()), False)

    # Arguments to a definition that has no parameters: no parameter to bind.
    check("arguments to a plain definition",
          is_complete("Box = int\nA = Box[int]\n", [], [], set()), False)

    # The builtin library walks like any other design, and its definitions may
    # name each other (Environment recurses through its own members).
    check("a builtin that is complete", is_complete("X = Environment", [], [], set()), True)
    check("the same builtin behind a member",
          is_complete("X = $env Environment", [], [], set()), True)

    # The entry's own file name is taken, so a library file under that name is
    # skipped and the entry still walks through.
    check("a library file that takes the entry's name",
          is_complete("X = int\n", [("entry.viba", "Y = str\n")], [], set()), True)

    # A directory named like a module: reading it fails, so the module is not
    # there (and the walk says so instead of crashing).
    with tempfile.TemporaryDirectory() as directory:
        (Path(directory) / "pkg.viba").mkdir()
        check("a directory where a module should be",
              is_complete("import pkg\nX = pkg.M\n", [], [directory], set()), False)

    # The chain helper also has to read the binary form the parser builds
    # before canonicalisation (nothing walks those through the entry).
    from viba import viba_ast as nodes
    from viba.is_complete import _chain_elements

    binary = nodes.Product(
        nodes.Product(nodes.TypeRef("A"), nodes.TypeRef("B")), nodes.TypeRef("C"))
    same_names = [n.name for n in _chain_elements(binary)]
    check(f"a binary product flattens in order ({same_names})",
          same_names == ["A", "B", "C"], True)
    # A right-nested run is a branch, not part of the main chain: the walk
    # reads it as one element, the way the chain form keeps it.
    binary_sum = nodes.Sum(nodes.TypeRef("A"), nodes.Sum(nodes.TypeRef("B"), nodes.TypeRef("C")))
    sum_elements = _chain_elements(binary_sum)
    check("a right-nested sum stays a branch",
          len(sum_elements) == 2 and isinstance(sum_elements[1], nodes.Sum), True)
    binary_exp = nodes.Exponent(
        nodes.Exponent(nodes.TypeRef("A"), nodes.TypeRef("B")), nodes.TypeRef("C"))
    check("a binary exponent flattens", len(_chain_elements(binary_exp)) == 3, True)
    check("anything else has no chain", _chain_elements(nodes.TypeRef("A")) is None, True)


if __name__ == "__main__":
    sys.exit(main())

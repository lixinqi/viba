"""Tests for check_tag_and_inline: the tags a design's products end up with,
once the inline chains are spread, and whether those chains end.

Cases are small designs written here, then the generated descriptor corpus is
run as designs — a file the repo already trusts must come back Ok(None). The
corpora that belong to a judgement sit in that judgement's own suite, which
reviews its designs through this check.

    python3 tests/test_check_tag_and_inline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.check_tag_and_inline import check_tag_and_inline
from viba.reflect import Config
from viba.type import Ok
from viba.viba_type_descriptor import empty_pool, parse_viba_file, pool_add_file

DATA = Path(__file__).resolve().parent / "data"
DESCRIPTORS = DATA / "type_descriptor"

PASS = FAIL = 0


def check(label: str, got, want):
    global PASS, FAIL
    text = "Ok" if isinstance(got, Ok) else str(got)
    ok = (want == "Ok" and isinstance(got, Ok)) or (want != "Ok" and want in text)
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}: wanted {want!r}, got {text!r}")


def check_empty(label: str, wrong):
    """A list of mismatches; empty is what we want."""
    global PASS, FAIL
    if wrong:
        FAIL += 1
        print(f"FAIL: {label}:")
        for line in wrong:
            print(f"  {line}")
    else:
        PASS += 1


def design(source: str):
    """check_tag_and_inline on one file's source: Ok, or the reason the file
    could not even be compiled (which the descriptor layer reports first)."""
    pool = empty_pool()
    parsed = parse_viba_file(pool, source, "design.viba", "design")
    if not isinstance(parsed, Ok):
        return f"does not parse: {parsed.err_msg}"
    built = pool_add_file(pool, parsed.ok_value)
    if not isinstance(built, Ok):
        return f"does not compile: {built.err_msg}"
    return check_tag_and_inline(built.ok_value)


# ----------------------------------------------------------------------
# 摊开以后才对得出来的两件事
# ----------------------------------------------------------------------

CLEAN = {
    "plain product": "Box = Object * $a int * $b str\n",
    "a base with tags": "A = $x int * $y str\nB = A * $z bool\n",
    "a chain three deep": "A = $x int\nB = A * $y int\nC = B * $z int\n",
    "one base, two products": "A = $x int\nB = A * $y int\nC = A * $w int\n",
    "units are not members": "Box = Object * Object * $a int\n",
    "a name over the unit": "U = Object\nBox = U * $a int\n",
    "a parameter of a tag": "Box[T] = $x T\nB = Box[int] * $y int\n",
    "a parameter that is a product": (
        "Pair = $p int * $q str\nBox[T] = T * $a int\nB = Box[Pair] * $z bool\n"),
    "a positional member": "Pos = int * $a int\n",
    "recursion through a tag": "Chain = $head int * $tail Chain\n",
    "recursion through two tags": "A = $a B\nB = $b A\n",
    "recursion behind a tag's body": "A = $p (A * $x int)\n",
    "a recursive generic": "Tree[T] = $leaf T * $kids list[Tree[T]]\nA = Tree[int] * $s str\n",
    "a generic that only grows": "W[T] = W[list[T]]\nA = W[int] * $s str\n",
    "a product inside a sum": "S = ($x int * $y str) | nil\n",
    "a product inside a tuple": "Box = $p ($x int, $y str)\n",
    "two instantiations, both fine": (
        "Base = $a int\nPair = Base * $b str\nBox[T] = T * $x int\n"
        "A = $p Box[int] * $q Box[Pair]\n"),
    "one body, one instantiation clean": (
        "Pair = $z int\nBox[T] = $p (T * $y int)\n"
        "A = $q Box[int] * $r Box[Pair]\n"),
    "nested generics": (
        "Inner[T] = $i T\nOuter[T] = $o (Inner[list[T]])\nA = Outer[int] * $y int\n"),
    "a source written with CRLF": "Box = int\r\nBox2 = str\r\n",
}

MALFORMED = {
    "the same tag twice, written": ("Box = $x int * $x str\n", "does not compile"),
    "the same tag twice, inlined": ("A = $x int\nB = A * $x str\n", "written twice"),
    "one base inlined twice": ("A = $x int * $y int\nB = A * A\n", "written twice"),
    "through an alias": ("A = $x int\nAlias = A\nB = Alias * $x str\n", "written twice"),
    "through a generic": ("Box[T] = $x T\nB = Box[int] * $x str\n", "written twice"),
    "inside a tagged body": ("Box = $p ($x int * $x str)\n", "written twice"),
    "inside a container": ("Box = $p list[$x int * $x str]\n", "written twice"),
    "inside a tuple": ("Box = $p ($x int * $x str, int)\n", "written twice"),
    "inside a sum branch": ("Box = $p (($x int * $x str) | nil)\n", "written twice"),
    "one instantiation is bad": (
        "Base = $a int\nPair = Base * $a str\nBox[T] = T * $x int\n"
        "A = $p Box[int] * $q Box[Pair]\n", "written twice"),
    "the same application twice": (
        "Box[T] = $x T\nA = Box[int] * Box[str] * $z bool\n", "written twice"),
    "one instantiation of a body": (
        "Pair = $y int\nBox[T] = $p (T * $y int)\nA = $q Box[Pair]\n", "written twice"),
    "the good instantiation first": (
        "Pair = $y int\nBox[T] = $p (T * $y int)\n"
        "A = $q Box[int] * $r Box[Pair]\n", "written twice"),
    "the bad instantiation first": (
        "Pair = $y int\nBox[T] = $p (T * $y int)\n"
        "A = $q Box[Pair] * $r Box[int]\n", "written twice"),
    "an instantiation behind a name": (
        "Pair = $y int\nBox[T] = $p (T * $y int)\nAlias = Box[Pair]\n"
        "A = $q Alias\n", "written twice"),
    "an inline cycle": ("A = A * $x int\n", "comes back to 'A'"),
    "an inline cycle through an alias": ("A = B * $x int\nB = A\n", "comes back to 'B'"),
    "an inline cycle of two": ("A = B * $x int\nB = A * $y int\n", "comes back to 'B'"),
    "an inline ring of three": (
        "A = B * $x int\nB = C * $y float\nC = A * $z bool\n", "comes back to 'B'"),
    "a generic that inlines itself": (
        "Gen[T] = Gen[T] * $g int\nA = Gen[int] * $x int\n", "comes back to 'Gen'"),
}


# 连词法都过不去的源：先报编译不了，轮不到 tag 与内联
LEXICAL = {
    "an illegal character": "Box = A @ B\n",
    "a minus sign": "Box = -1\n",
    "an unterminated code block": "Box = {never closed\n",
    "a builtin container as a definition": "list = int\n",
    "a builtin literal as a parameter": "W[ListLiteral] = int\n",
    "a byte-order mark": "\ufeffBox = int\n",
}


def run_cases():
    for label, source in CLEAN.items():
        check(label, design(source), "Ok")
    for label, (source, want) in MALFORMED.items():
        check(label, design(source), want)
    for label, source in LEXICAL.items():
        check(label, design(source), "does not parse")


def pool_design(files):
    """几份文件编进一个池子，再当设计审一遍。"""
    pool = empty_pool()
    for file_name, module_name, source in files:
        parsed = parse_viba_file(pool, source, file_name, module_name)
        if not isinstance(parsed, Ok):
            return f"does not parse: {parsed.err_msg}"
        built = pool_add_file(pool, parsed.ok_value)
        if not isinstance(built, Ok):
            return f"does not compile: {built.err_msg}"
        pool = built.ok_value
    return check_tag_and_inline(pool)


# 跨模块：名字按它自己那个模块的 import 表解析，摊进来的 tag 与环也跨文件
BASE = ("base.viba", "base", "A = $x int * $y str\nU = Object\n")
CROSS = {
    "a product from another module is inlined": (
        [BASE, ("main.viba", "main", "import base\nB = base.A * $z bool\n")], "Ok"),
    "an import of an import is inlined": (
        [BASE, ("mid.viba", "mid", "import base as b\nM = b.A\n"),
         ("main.viba", "main", "import mid as m\nB = m.M * $z bool\n")], "Ok"),
    "a tag repeated across modules": (
        [BASE, ("main.viba", "main", "import base\nDup = base.A * $x int\n")],
        "written twice"),
    "an inline ring across two files": (
        [("other.viba", "other", "import ring\nCyc = ring.Ring * $c int\n"),
         ("ring.viba", "ring", "import other\nRing = other.Cyc * $r int\n")],
        "comes back to"),
    "a unit from another module": (
        [BASE, ("main.viba", "main", "import base\nUBox = base.U * $w int\n")], "Ok"),
}


def run_cross_module_cases():
    for label, (files, want) in CROSS.items():
        check(label, pool_design(files), want)


def run_config_case():
    """换个词汇问同一件事：单位元是调用方点的名，写对的仍然 Ok，写错的仍然 Err——
    单位没有 tag，也不参与内联，所以 tag 的答案不由它决定。"""
    source = ("Base = $x Box[int]\n"
              "Bad = Unit * Base * $x Box[str]\n"
              "Fine = Unit * $x Box[int] * $y Box[str]\n")
    pool = empty_pool()
    parsed = parse_viba_file(pool, source, "units.viba", "units")
    built = pool_add_file(pool, parsed.ok_value).ok_value
    check("an inlined tag repeats (language names)",
          check_tag_and_inline(built), "written twice")
    unit_config = Config(never_eqv={"Oneof"}, nil_eqv={"Object", "Unit", "Box"})
    check("the same, asked with the caller's own units",
          check_tag_and_inline(built, unit_config), "written twice")

    clean = "Fine = Unit * $x Box[int] * $y Box[str]\n"
    pool = empty_pool()
    parsed = parse_viba_file(pool, clean, "fine.viba", "fine")
    built = pool_add_file(pool, parsed.ok_value).ok_value
    check("a clean one is clean under that vocabulary too",
          check_tag_and_inline(built, unit_config), "Ok")


# ----------------------------------------------------------------------
# 语料：仓库里现成的设计
# ----------------------------------------------------------------------

def file_design(path: Path):
    pool = empty_pool()
    parsed = parse_viba_file(pool, path.read_text(), path.name, path.stem)
    if not isinstance(parsed, Ok):
        return f"does not parse: {parsed.err_msg}"
    built = pool_add_file(pool, parsed.ok_value)
    if not isinstance(built, Ok):
        return f"does not compile: {built.err_msg}"
    return check_tag_and_inline(built.ok_value)


def run_descriptor_corpus():
    """生成语料：一份都该是干净的。"""
    wrong = []
    files = 0
    for path in sorted(DESCRIPTORS.rglob("*.viba")):
        files += 1
        got = file_design(path)
        if not isinstance(got, Ok):
            wrong.append(f"{path.name}: {got}")
    check_empty(f"type_descriptor corpus: {files} files", wrong)


def run_package_sources():
    """包自己的 .viba 文件也要能编、能过审：语言单位（viba/type.viba）、
    描述符形状（viba/viba_type_descriptor.viba）、内建词汇（viba/builtin.viba）。"""
    root = Path(__file__).resolve().parent.parent / "viba"
    paths = [root / "type.viba", root / "viba_type_descriptor.viba",
             root / "builtin.viba"]
    wrong = []
    for path in paths:
        got = file_design(path)
        if not isinstance(got, Ok):
            wrong.append(f"{path.relative_to(root.parent)}: {got}")
    check_empty(f"package sources: {len(paths)} files", wrong)


def run():
    run_cases()
    run_cross_module_cases()
    run_config_case()
    run_descriptor_corpus()
    run_package_sources()
    print(f"check_tag_and_inline: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())

"""Tests for check_tag_and_inline: the tags a design's products end up with,
once the inline chains are spread, and whether those chains end.

Cases are small designs written here; then the repo's own corpora are run as
designs — a file the repo already trusts must come back Ok(None), and the files
the corpus marks as malformed must come back Err.

    python3 tests/test_check_tag_and_inline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.check_tag_and_inline import check_tag_and_inline
from viba.rule import reflect as rule_reflect
from viba.type import Ok
from viba.viba_type_descriptor import empty_pool, parse_viba_file, pool_add_file

DATA = Path(__file__).resolve().parent / "data"
TYPES = DATA / "is_sub_type"
DESCRIPTORS = DATA / "type_descriptor"
RULES = DATA / "rule_coding_style_check"

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
    "plain product": "Box := Object * $a int * $b str\n",
    "a base with tags": "A := $x int * $y str\nB := A * $z bool\n",
    "a chain three deep": "A := $x int\nB := A * $y int\nC := B * $z int\n",
    "one base, two products": "A := $x int\nB := A * $y int\nC := A * $w int\n",
    "units are not members": "Box := Object * Object * $a int\n",
    "a name over the unit": "U := Object\nBox := U * $a int\n",
    "a parameter of a tag": "Box[T] := $x T\nB := Box[int] * $y int\n",
    "a parameter that is a product": (
        "Pair := $p int * $q str\nBox[T] := T * $a int\nB := Box[Pair] * $z bool\n"),
    "a positional member": "Pos := int * $a int\n",
    "recursion through a tag": "Chain := $head int * $tail Chain\n",
    "recursion through two tags": "A := $a B\nB := $b A\n",
    "recursion behind a tag's body": "A := $p (A * $x int)\n",
    "a recursive generic": "Tree[T] := $leaf T * $kids list[Tree[T]]\nA := Tree[int] * $s str\n",
    "a generic that only grows": "W[T] := W[list[T]]\nA := W[int] * $s str\n",
    "a product inside a sum": "S := ($x int * $y str) | nil\n",
    "a product inside a tuple": "Box := $p ($x int, $y str)\n",
    "two instantiations, both fine": (
        "Base := $a int\nPair := Base * $b str\nBox[T] := T * $x int\n"
        "A := $p Box[int] * $q Box[Pair]\n"),
    "one body, one instantiation clean": (
        "Pair := $z int\nBox[T] := $p (T * $y int)\n"
        "A := $q Box[int] * $r Box[Pair]\n"),
    "nested generics": (
        "Inner[T] := $i T\nOuter[T] := $o (Inner[list[T]])\nA := Outer[int] * $y int\n"),
}

MALFORMED = {
    "the same tag twice, written": ("Box := $x int * $x str\n", "does not compile"),
    "the same tag twice, inlined": ("A := $x int\nB := A * $x str\n", "written twice"),
    "one base inlined twice": ("A := $x int * $y int\nB := A * A\n", "written twice"),
    "through an alias": ("A := $x int\nAlias := A\nB := Alias * $x str\n", "written twice"),
    "through a generic": ("Box[T] := $x T\nB := Box[int] * $x str\n", "written twice"),
    "inside a tagged body": ("Box := $p ($x int * $x str)\n", "written twice"),
    "inside a container": ("Box := $p list[$x int * $x str]\n", "written twice"),
    "inside a tuple": ("Box := $p ($x int * $x str, int)\n", "written twice"),
    "inside a sum branch": ("Box := $p (($x int * $x str) | nil)\n", "written twice"),
    "one instantiation is bad": (
        "Base := $a int\nPair := Base * $a str\nBox[T] := T * $x int\n"
        "A := $p Box[int] * $q Box[Pair]\n", "written twice"),
    "the same application twice": (
        "Box[T] := $x T\nA := Box[int] * Box[str] * $z bool\n", "written twice"),
    "one instantiation of a body": (
        "Pair := $y int\nBox[T] := $p (T * $y int)\nA := $q Box[Pair]\n", "written twice"),
    "the good instantiation first": (
        "Pair := $y int\nBox[T] := $p (T * $y int)\n"
        "A := $q Box[int] * $r Box[Pair]\n", "written twice"),
    "the bad instantiation first": (
        "Pair := $y int\nBox[T] := $p (T * $y int)\n"
        "A := $q Box[Pair] * $r Box[int]\n", "written twice"),
    "an instantiation behind a name": (
        "Pair := $y int\nBox[T] := $p (T * $y int)\nAlias := Box[Pair]\n"
        "A := $q Alias\n", "written twice"),
    "an inline cycle": ("A := A * $x int\n", "comes back to 'A'"),
    "an inline cycle through an alias": ("A := B * $x int\nB := A\n", "comes back to 'B'"),
    "an inline cycle of two": ("A := B * $x int\nB := A * $y int\n", "comes back to 'B'"),
    "an inline ring of three": (
        "A := B * $x int\nB := C * $y float\nC := A * $z bool\n", "comes back to 'B'"),
    "a generic that inlines itself": (
        "Gen[T] := Gen[T] * $g int\nA := Gen[int] * $x int\n", "comes back to 'Gen'"),
}


def run_cases():
    for label, source in CLEAN.items():
        check(label, design(source), "Ok")
    for label, (source, want) in MALFORMED.items():
        check(label, design(source), want)


def run_rule_layer_case():
    """规则层用自己那份 accessor 问同一件事：写得对的设计两边都 Ok，写错的两边
    都 Err（哪个名字算单位元是那一层的事，tag 不由它决定）。"""
    source = ("Base := $x Metric[int]\n"
              "Bad := RuleObject * Base * $x Metric[str]\n"
              "Fine := RuleObject * $x Metric[int] * $y Metric[str]\n")
    pool = empty_pool()
    parsed = parse_viba_file(pool, source, "rules.viba", "rules")
    built = pool_add_file(pool, parsed.ok_value).ok_value
    check("a rule's inlined tag repeats (language names)",
          check_tag_and_inline(built), "written twice")
    check("the same, asked with the rule accessor",
          check_tag_and_inline(built, rule_reflect.access), "written twice")

    clean = "Fine := RuleObject * $x Metric[int] * $y Metric[str]\n"
    pool = empty_pool()
    parsed = parse_viba_file(pool, clean, "fine.viba", "fine")
    built = pool_add_file(pool, parsed.ok_value).ok_value
    check("a clean rule is clean for the rule accessor",
          check_tag_and_inline(built, rule_reflect.access), "Ok")


# ----------------------------------------------------------------------
# 语料：仓库里现成的设计
# ----------------------------------------------------------------------

# 语料里本来就写错的那几份：重标签（sub071 写在同层、sub112 摊进来、sup113 写在
# 同层）与内联成环（sub117、sub120-sub123、sub125）。语料按"对/错"标的是子类型
# 判定，不是写法；这两边恰好一致，所以这里能拿它当验收。
MALFORMED_CORPUS = {"sub071", "sub112", "sup113", "sub117",
                    "sub120", "sub121", "sub122", "sub123", "sub125"}


def file_design(path: Path):
    pool = empty_pool()
    parsed = parse_viba_file(pool, path.read_text(), path.name, path.stem)
    if not isinstance(parsed, Ok):
        return f"does not parse: {parsed.err_msg}"
    built = pool_add_file(pool, parsed.ok_value)
    if not isinstance(built, Ok):
        return f"does not compile: {built.err_msg}"
    return check_tag_and_inline(built.ok_value)


def run_is_sub_type_corpus():
    """sub/sup 两份语料：写对的 Ok，语料标成写错的 Err。"""
    wrong = []
    files = 0
    pairs = sorted(TYPES.glob("sub*.viba"))
    for path in sorted(TYPES.glob("*.viba")):
        if not path.stem.startswith(("sub", "sup")):
            continue
        files += 1
        want_clean = path.stem not in MALFORMED_CORPUS
        got = file_design(path)
        if isinstance(got, Ok) is not want_clean:
            wrong.append(f"{path.name}: {'Ok' if isinstance(got, Ok) else got}")
    check_empty(f"is_sub_type corpus: {len(pairs)} pairs, {files} files, "
                f"{len(MALFORMED_CORPUS)} of them malformed", wrong)


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


def run_rule_corpus():
    """仓库里现成的规则：自己写的字段都摊得开、tag 不重。"""
    wrong = []
    files = 0
    for path in sorted(RULES.rglob("*.viba")):
        files += 1
        got = file_design(path)
        if not isinstance(got, Ok) and "broken" not in path.name:
            wrong.append(f"{path.name}: {got}")
    check_empty(f"rule corpus: {files} files (the broken ones excepted)", wrong)


def run():
    run_cases()
    run_rule_layer_case()
    run_is_sub_type_corpus()
    run_descriptor_corpus()
    run_rule_corpus()
    print(f"check_tag_and_inline: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())

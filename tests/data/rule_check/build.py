"""Build the rule_check rule corpus: tests/data/rule_check/rules/ruleNN.viba.

Each rule file defines helper Metric types plus one rule whose body
is a product of Metric fields and at least ten Assert fields carrying
$python_code predicators. Existing files are never overwritten —
extend by adding new specs, never by rewriting old ones.

Usage: python3 tests/data/rule_check/build.py
"""

import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "rules"

# (metric name, type expr, kind, field tag)
POOL = [
    ("CodeLength", "int", "int", "code_length"),
    ("MaxLines", "int", "int", "max_lines"),
    ("CyclomaticComplexity", "int", "int", "cyclomatic_complexity"),
    ("LoopCount", "int", "int", "loop_count"),
    ("ParamCount", "int", "int", "param_count"),
    ("CoverageRatio", "float", "float", "coverage_ratio"),
    ("TimeBudget", "float", "float", "time_budget"),
    ("HasTests", "bool", "bool", "has_tests"),
    ("IsPublic", "bool", "bool", "is_public"),
    ("ModuleName", "str", "str", "module_name"),
    ("AuthorName", "str", "str", "author_name"),
    ("IndentWidth", "$spaces int", "prod1", "indent_width"),
    ("DocCoverage", "$documented_lines int * $total_lines int", "prod2", "doc_coverage"),
    ("BudgetSplit", "$spent float * $cap float", "prod3", "budget_split"),
    ("Keywords", "list[str]", "list", "keywords"),
    ("Imports", "list[str]", "list", "imports"),
    ("Tags", "set[str]", "set", "tags"),
    ("Scores", "dict[str, int]", "dict", "scores"),
    ("TraceLevel", "$quiet () | $verbose ()", "sum", "trace_level"),
    ("IntPair", "$a int * $b int", "prod_ab", "int_pair"),
    ("NameSize", "$name str * $size int", "prod_ns", "name_size"),
    ("Triplet", "$x int * $y int * $z int", "prod_xyz", "triplet"),
    ("IntList", "list[int]", "intlist", "int_list"),
    ("Lookup", "dict[str, str]", "dictss", "lookup"),
]

# kind -> list of (description, predicate body); {t} is the field tag
BANK = {
    "int": [
        ("{t} at most 24", "return self.{t}.value <= 24"),
        ("{t} under 100", "return self.{t}.value < 100"),
    ],
    "cross_int": [
        ("{t1} not above {t2}", "return self.{t1}.value <= self.{t2}.value"),
    ],
    "float": [
        ("{t} within 50.5", "return self.{t}.value <= 50.5"),
        ("{t} non-negative", "return self.{t}.value >= 0.0"),
    ],
    "bool": [
        ("{t} required", "return self.{t}.value"),
        ("{t} forbidden", "return not self.{t}.value"),
    ],
    "str": [
        ("{t} short", "return len(self.{t}.value) <= 8"),
        ("{t} not empty", "return len(self.{t}.value) >= 1"),
    ],
    "list": [
        ("{t} non-empty", "return len(self.{t}.value) >= 1"),
        ("{t} bounded", "return len(self.{t}.value) <= 4"),
    ],
    "set": [
        ("{t} bounded", "return len(self.{t}.value) <= 3"),
        ("{t} non-empty", "return len(self.{t}.value) >= 1"),
    ],
    "dict": [
        ("{t} non-empty", "return len(self.{t}.value) >= 1"),
        ("{t} bounded", "return len(self.{t}.value) <= 4"),
    ],
    "prod1": [("{t} non-negative", "return self.{t}.spaces >= 0")],
    "prod2": [
        ("{t} documented within total", "return self.{t}.documented_lines <= self.{t}.total_lines"),
    ],
    "prod3": [("{t} spent within cap", "return self.{t}.spent <= self.{t}.cap")],
    "intlist": [
        ("{t} non-empty", "return len(self.{t}.value) >= 1"),
        ("{t} bounded", "return len(self.{t}.value) <= 4"),
    ],
    "dictss": [
        ("{t} non-empty", "return len(self.{t}.value) >= 1"),
        ("{t} bounded", "return len(self.{t}.value) <= 4"),
    ],
    "prod_ab": [
        ("{t} ordered", "return self.{t}.a <= self.{t}.b"),
        ("{t} sum bounded", "return self.{t}.a + self.{t}.b <= 100"),
    ],
    "prod_ns": [("{t} name fits size", "return len(self.{t}.name) <= self.{t}.size")],
    "prod_xyz": [("{t} x plus y within z", "return self.{t}.x + self.{t}.y <= self.{t}.z")],
}

# 20 field combinations; "sum"-kind fields exercise generator branch choice
COMBOS = [
    ["CodeLength", "MaxLines", "CoverageRatio", "HasTests", "Keywords"],
    ["CodeLength", "LoopCount", "CyclomaticComplexity", "TimeBudget", "AuthorName"],
    ["MaxLines", "ParamCount", "CoverageRatio", "IsPublic", "Imports"],
    ["CodeLength", "CoverageRatio", "ModuleName", "Tags", "DocCoverage", "HasTests"],
    ["LoopCount", "MaxLines", "TimeBudget", "HasTests", "Scores"],
    ["CyclomaticComplexity", "ParamCount", "CoverageRatio", "AuthorName", "Keywords"],
    ["CodeLength", "IndentWidth", "DocCoverage", "BudgetSplit", "HasTests", "Keywords", "AuthorName"],
    ["MaxLines", "CoverageRatio", "IsPublic", "ModuleName", "Imports", "Tags"],
    ["CodeLength", "LoopCount", "HasTests", "Keywords", "Scores", "TraceLevel"],
    ["ParamCount", "TimeBudget", "AuthorName", "Tags", "DocCoverage", "IsPublic"],
    ["CyclomaticComplexity", "CoverageRatio", "ModuleName", "Imports", "HasTests", "TraceLevel"],
    ["CodeLength", "MaxLines", "CoverageRatio", "TimeBudget", "IsPublic", "Keywords", "AuthorName"],
    ["LoopCount", "ParamCount", "IndentWidth", "DocCoverage", "HasTests", "Scores", "Tags"],
    ["CodeLength", "CyclomaticComplexity", "CoverageRatio", "BudgetSplit", "ModuleName", "Imports", "IsPublic"],
    ["MaxLines", "TimeBudget", "AuthorName", "Keywords", "DocCoverage", "Tags", "TraceLevel"],
    ["CodeLength", "LoopCount", "ParamCount", "CoverageRatio", "HasTests", "ModuleName", "Scores", "Imports"],
    ["CyclomaticComplexity", "IndentWidth", "BudgetSplit", "CoverageRatio", "IsPublic", "AuthorName", "Keywords", "Tags"],
    ["CodeLength", "MaxLines", "LoopCount", "TimeBudget", "HasTests", "DocCoverage", "Imports", "Scores", "TraceLevel"],
    ["ParamCount", "CyclomaticComplexity", "CoverageRatio", "IndentWidth", "IsPublic", "ModuleName", "Keywords", "BudgetSplit", "AuthorName"],
    ["CodeLength", "MaxLines", "LoopCount", "ParamCount", "CoverageRatio", "TimeBudget", "HasTests", "IsPublic", "ModuleName", "AuthorName"],
    # --- rules 20-39: wider variance, assert counts from ASSERT_TARGETS ---
    ["CodeLength", "MaxLines", "CoverageRatio", "HasTests", "ModuleName"],
    ["CodeLength", "LoopCount", "ParamCount", "TimeBudget", "AuthorName", "Keywords"],
    ["CodeLength", "MaxLines", "CoverageRatio", "TimeBudget", "HasTests", "ModuleName", "Tags", "Scores"],
    ["CodeLength", "MaxLines", "LoopCount", "CoverageRatio", "HasTests", "ModuleName", "Keywords", "IntList", "Scores", "Lookup", "IntPair"],
    ["CodeLength", "MaxLines", "CoverageRatio", "TimeBudget", "HasTests", "ModuleName", "Keywords", "Tags", "Scores", "NameSize", "Triplet", "IntPair"],
    ["CodeLength", "MaxLines", "CoverageRatio", "TimeBudget", "HasTests", "ModuleName", "Keywords", "Tags", "Scores", "Lookup", "IntPair", "NameSize", "Triplet", "DocCoverage"],
    ["CodeLength", "CoverageRatio", "HasTests", "Keywords", "IntPair", "ModuleName"],
    ["CodeLength", "MaxLines", "TimeBudget", "AuthorName", "Tags", "Scores", "DocCoverage"],
    ["CodeLength", "MaxLines", "CoverageRatio", "HasTests", "ModuleName", "Keywords", "Tags", "IntPair"],
    ["CodeLength", "MaxLines", "CoverageRatio", "TimeBudget", "ModuleName", "Keywords", "Tags", "Scores", "NameSize"],
    ["CodeLength", "MaxLines", "LoopCount", "CoverageRatio", "HasTests", "AuthorName", "Keywords", "Tags", "Lookup", "IntPair", "Triplet"],
    ["CodeLength", "MaxLines", "CoverageRatio", "HasTests", "ModuleName", "Keywords", "IntList", "Scores", "NameSize", "Triplet", "DocCoverage"],
    ["CodeLength", "MaxLines", "CoverageRatio", "TimeBudget", "HasTests", "ModuleName", "Keywords", "Tags", "Scores", "Lookup", "IntPair", "NameSize", "Triplet", "DocCoverage", "BudgetSplit"],
    ["CodeLength", "LoopCount", "CoverageRatio", "HasTests", "Keywords", "Scores", "IntPair"],
    ["MaxLines", "TimeBudget", "IsPublic", "AuthorName", "IntList", "Tags"],
    ["CyclomaticComplexity", "ParamCount", "CoverageRatio", "ModuleName", "Imports", "Lookup", "IndentWidth"],
    ["CodeLength", "MaxLines", "CoverageRatio", "HasTests", "IsPublic", "ModuleName", "Keywords", "IntPair", "Triplet"],
    ["CodeLength", "MaxLines", "CoverageRatio", "HasTests", "ModuleName", "Keywords", "Tags", "Scores", "NameSize", "Triplet"],
    ["CodeLength", "MaxLines", "LoopCount", "CoverageRatio", "TimeBudget", "HasTests", "AuthorName", "Keywords", "IntList", "IntPair", "DocCoverage"],
    ["CodeLength", "MaxLines", "CoverageRatio", "TimeBudget", "HasTests", "ModuleName", "Keywords", "Tags", "Scores", "Lookup", "IntPair", "NameSize", "Triplet", "IndentWidth", "BudgetSplit"],
]

# Assert-count targets for rules 20-39 (rules 0-19 keep 10).
ASSERT_TARGETS = [
    11, 13, 17, 21, 22, 24, 12, 14, 16, 18,
    19, 20, 23, 15, 10, 12, 14, 16, 18, 20,
]


def _kind_preds(by_kind):
    out = []
    for kind, preds in BANK.items():
        if kind in ("int", "cross_int"):
            continue
        for tag in by_kind.get(kind, []):
            out.extend((d.format(t=tag), b.format(t=tag)) for d, b in preds)
    return out


def _predicates(fields, limit=10):
    """Assemble up to `limit` (description, body) pairs for the fields."""
    by_kind = {}
    for _, _, kind, tag in fields:
        by_kind.setdefault(kind, []).append(tag)
    out = []
    ints = by_kind.get("int", [])
    for tag in ints:
        for desc, body in BANK["int"]:
            out.append((desc.format(t=tag), body.format(t=tag)))
    if len(ints) >= 2:
        desc, body = BANK["cross_int"][0]
        out.append((desc.format(t1=ints[0], t2=ints[1]), body.format(t1=ints[0], t2=ints[1])))
    out.extend(_kind_preds(by_kind))
    return out[:limit]


def _assert_field(index, desc, body):
    slug = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in desc)
    slug = "_".join(filter(None, slug.split()))
    lines = [
        f"  * $assert_{index:02d}_{slug}",
        f"      Assert[{{{desc}}}, $python_code {{",
        "def handler(self):",
        f"    {body}",
        "}]",
    ]
    return "\n".join(lines)


def _assert_target(number):
    return ASSERT_TARGETS[number - 20] if number >= 20 else 10


def _render(number):
    names = COMBOS[number]
    lookup = {name: spec for spec in POOL for name in [spec[0]]}
    fields = [lookup[name] for name in names]
    defs = "\n".join(f"{name} := {expr}" for name, expr, _, _ in fields)
    metrics = "\n".join(f"  * ${tag} Metric[{name}]" for name, _, _, tag in fields)
    preds = _predicates(fields, _assert_target(number))
    asserts = [_assert_field(i, d, b) for i, (d, b) in enumerate(preds)]
    body = "\n".join([f"Rule{number:02d} :=", "  RuleObject", metrics, *asserts])
    return f"{defs}\n\n{body}\n"


def main():
    written = []
    for number in range(len(COMBOS)):
        path = DATA / f"rule{number:02d}.viba"
        if path.exists():
            continue
        path.write_text(_render(number))
        written.append(path.name)
    (DATA / "expected.txt").write_text(
        "".join(f"{n:02d} ok\n" for n in range(len(COMBOS)))
    )
    print(f"built {len(written)} rule files: {', '.join(written) or 'none (all present)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

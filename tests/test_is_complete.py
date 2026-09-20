"""Tests for is_complete: take the corpora as designs and ask if they are complete.

All material comes from the repo, nothing new is invented:

    data/type_descriptor/case_081/     deep entry (import util as base) + util.viba
    data/rule_coding_style_check/demo.viba   a rule: Predicate[{...}, $python_code {...}]
    data/rule_coding_style_check/broken_rules.viba   $x ... and $x Missing
    data/is_sub_type/sub030.viba       recursive Chain := $head int * $tail Chain | nil
    data/is_sub_type/sup025.viba       X := ...

    python3 tests/test_is_complete.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.is_complete import is_complete

DATA = Path(__file__).resolve().parent / "data"
CASES = DATA / "type_descriptor" / "case_081"
RULES = DATA / "rule_coding_style_check"
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

    # A rule: the rule layer hands in its own names as terminators
    demo = read(RULES / "demo.viba")
    rule_stops = {"Metric", "Predicate", "PredicationFailed",
                  "RuleObject", "Oneof"}
    check("rule file, no terminators", is_complete(demo, [], [], set()), False)
    check("rule file + the rule layer's terminators",
          is_complete(demo, [], [], rule_stops), True)
    check("rule file + Metric only (Predicate still has code)",
          is_complete(demo, [], [], {"Metric"}), False)

    # Recursive definitions: coinduction lets them through
    check("recursive Chain", is_complete(read(TYPES / "sub030.viba"), [], [], set()), True)

    # Ellipsis: incomplete by default, ends a walk once it is a terminator
    ellipsis = read(TYPES / "sup025.viba")
    check("ellipsis, no terminators", is_complete(ellipsis, [], [], set()), False)
    check("ellipsis + \"...\"", is_complete(ellipsis, [], [], {"..."}), True)

    # A name that does not resolve: no terminator can rescue it
    broken = read(RULES / "broken_rules.viba")
    check("unresolved reference", is_complete(broken, [], [], set()), False)
    check("unresolved reference + \"...\"", is_complete(broken, [], [], {"..."}), False)

    # Edges: an entry that does not compile, and an empty one
    check("entry does not compile", is_complete("X := ", [], [], set()), False)
    check("empty file", is_complete("", [], [], set()), True)

    print(f"is_complete: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

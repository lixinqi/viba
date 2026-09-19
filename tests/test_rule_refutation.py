"""规则层的否证判据：not[...] 的每支都得有终止子。

`viba.rule.is_compliant` 把 `PredicationFailed` 这个本层的词交给核心
（`is_sub_type(..., terminators=...)`），于是"没否证任何一支的呈证不算合规"
这条判据住在规则层。语料 100 对 sub/sup（data/rule_refutation/not）。

    python3 tests/test_rule_refutation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import viba_ast
from viba.type import AstNodeType, Err, Ok, custom_module
from viba.rule.is_compliant import is_compliant

CASES = Path(__file__).resolve().parent / "data" / "rule_refutation" / "not"
PASS = FAIL = 0


def check_result(result, want, label: str):
    global PASS, FAIL
    if want == "error":
        ok, got = isinstance(result, Err), type(result).__name__
    elif isinstance(result, Ok):
        ok, got = result.ok_value is want, repr(result.ok_value)
    else:
        ok, got = False, f"Err({result.err_msg!r})"
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}: expected {want}, got {got}")


def parse_want(text: str):
    text = text.strip()
    if text.startswith("true"):
        return True
    if text.startswith("false"):
        return False
    return "error"


def _entry_node(text: str):
    tree = viba_ast.parse(text)
    defs = [n for n in tree.body if not isinstance(n, viba_ast.Import)]
    node = defs[-1]
    return node.body if isinstance(node, viba_ast.TypeDefinition) else node


def load_entry(text: str):
    """The module's last definition, as the judgment's entry type."""
    module = custom_module(text)
    return AstNodeType(_entry_node(text), module)


def run_not_cases():
    """100 not[...] cases, one sub/sup .viba pair each."""
    expected = {}
    for line in (CASES / "expected.txt").read_text().splitlines():
        num, _, want = line.partition(" ")
        if num:
            expected[num] = parse_want(want)
    for sup_path in sorted(CASES.glob("sup*.viba")):
        num = sup_path.stem[3:]
        sub_e = load_entry((CASES / f"sub{num}.viba").read_text())
        sup_e = load_entry(sup_path.read_text())
        check_result(is_compliant(sub_e, sup_e), expected[num], f"not case {num}")
    print(f"rule_refutation: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run_not_cases())

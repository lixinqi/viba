"""viba_lint：类型语句的版式。

每条用例是 `tests/data/viba_lint/` 下的一份文件；`document.viba.md` 是一份文档，诊断要落在
文档自己的行号上。

    python3 tests/test_lint.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks

from viba.lint import (fix_file, fix_markdown, fix_source, lint_file,
                                 lint_markdown, lint_paths, lint_source, main as lint_main)

checks = Checks("viba_lint")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "viba_lint"


def rules_of(name: str) -> list:
    return [violation.rule for violation in lint_file(CASES / name)]


def fixed(name: str) -> str:
    return fix_source((CASES / name).read_text(), name)


def run() -> None:
    check(lint_file(CASES / "good.viba") == [],
          "the standard lays out without violations")

    check("statement-fold" in rules_of("inline_sum.viba"),
          "a one-line sum is not folded")
    check("sum-head" in rules_of("field_head_sum.viba"),
          "a sum whose first line is not Oneof is reported")
    check("sum-head" in rules_of("misaligned_oneof.viba"),
          "Oneof and the alternatives must share a column")

    check("statement-fold" in rules_of("inline_product.viba"),
          "a one-line multi-member product is not folded")
    check("statement-fold" in rules_of("inline_box.viba"),
          "a one-line single-member product is not folded")
    check("product-head" in rules_of("field_head_product.viba"),
          "a product block whose first line is a field is reported")
    check("product-head" in rules_of("misaligned_product.viba"),
          "the product's head and fields must share a column")

    check("chain-indent" in rules_of("chain_indent_pipe.viba"),
          "a `|` line at four spaces is reported")
    check("chain-indent" in rules_of("chain_indent_star.viba"),
          "a `*` line at four spaces is reported")
    check("chain-indent" in rules_of("chain_indent_arrow.viba"),
          "a `<-` line at four spaces is reported")
    check("chain-indent" in rules_of("chain_indent_lshift.viba"),
          "a `<<` line at four spaces is reported")

    check("exponent-result" in rules_of("exponent_result.viba"),
          "an exponent's result type must share the arguments' column")
    check("partial-function" in rules_of("partial_function.viba"),
          "a partial application's function must share the arguments' column")

    check("field-comments" in rules_of("field_comments.viba"),
          "field comments must share a column")
    check("field-comments" not in rules_of("good.viba"),
          "aligned field comments are not reported")

    check("parse" in rules_of("parse_error.viba"),
          "a source that does not parse is reported")

    document = lint_file(CASES / "document.viba.md")
    check([violation.line for violation in document] == [6],
          f"a document's diagnostic points at its own line: {document!r}")

    found = lint_paths([str(CASES)])
    check(any(v.path.endswith("inline_sum.viba") and v.rule == "statement-fold"
              for v in found),
          "a directory's .viba files are read")
    check(any(v.path.endswith("document.viba.md") and v.line == 6 for v in found),
          "a directory's .viba.md document is read")
    check(not any(v.path.endswith("good.viba") for v in found),
          "a directory's file without violations is not reported")

    check(lint_source(fixed("sums.viba"), "fixed.viba") == [],
          "the fixer adds Oneof to a sum whose first branch has no |")
    check("Oneof" in fixed("sums.viba"),
          "the fixed sum has its Oneof head")
    check(lint_source(fixed("product_without_head.viba"), "fixed.viba") == [],
          "the fixer adds Object to a product whose first line is a field")
    check("Object" in fixed("product_without_head.viba"),
          "the fixed product has its Object head")
    check(fixed("good.viba") == (CASES / "good.viba").read_text(),
          "a file already laid out is left as it is")
    for name in ("inline_sum.viba", "inline_product.viba", "inline_box.viba",
                 "misaligned_oneof.viba", "misaligned_product.viba",
                 "field_head_sum.viba", "field_head_product.viba",
                 "chain_indent_pipe.viba", "chain_indent_star.viba",
                 "chain_indent_arrow.viba", "chain_indent_lshift.viba",
                 "exponent_result.viba", "partial_function.viba",
                 "field_comments.viba"):
        check(lint_source(fixed(name), "fixed.viba") == [],
              f"the fixer lays out {name}")

    document = (CASES / "document.viba.md").read_text()
    fixed_document = fix_markdown(document)
    check(lint_markdown(fixed_document, "fixed.viba.md") == [],
          f"the fixer lays out a document's sections: {fixed_document!r}")
    check("    Oneof" in fixed_document,
          "the fixed document carries the Oneof head")

    with tempfile.TemporaryDirectory(prefix="viba-lint-") as scratch:
        root = Path(scratch)
        box = root / "box.viba"
        box.write_text((CASES / "inline_box.viba").read_text())
        check(fix_file(box) is True, "fix_file writes a fixable file")
        check(lint_file(box) == [], "the written file has no violations")
        check(fix_file(box) is False, "fix_file leaves a fixed file alone")

        document_path = root / "doc.viba.md"
        document_path.write_text((CASES / "document.viba.md").read_text())
        check(fix_file(document_path) is True, "fix_file writes a document's section")
        check(lint_file(document_path) == [], "the written document has no violations")

        cli = root / "cli.viba"
        cli.write_text((CASES / "inline_sum.viba").read_text())
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = lint_main(["--fix", str(cli)])
        check(code == 0, "the --fix command exits 0 when nothing is left")
        check(printed.getvalue().startswith("fixed "), "the --fix command says what it changed")
        check(lint_file(cli) == [], "the --fix command wrote the layout")

    for name in ("expression_product.viba", "expression_exponent.viba",
                 "expression_partial.viba", "expression_nested.viba",
                 "expression_deep.viba"):
        check("expression-fold" in rules_of(name),
              f"a one-line compound expression is reported: {name}")
        check(lint_source(fixed(name), "fixed.viba") == [],
              f"the fixer folds the expression of {name}")
    check(fixed("expression_product.viba").startswith("A = $foo (\n"),
          "the fixed expression opens with the parenthesis")
    check("        Object\n" in fixed("expression_product.viba"),
          "a product expression gets its Object head at base + 4")
    check("         int\n" in fixed("expression_exponent.viba"),
          "an exponent's result is three spaces past <-")
    check("      <- $env Env\n" in fixed("expression_exponent.viba"),
          "the <- of an exponent stays two spaces past the base")
    check("         f\n" in fixed("expression_partial.viba"),
          "a partial application's function is three spaces past <<")
    check("      << $a 1\n" in fixed("expression_partial.viba"),
          "the << of a partial application stays two spaces past the base")
    check(fixed("expression_tag.viba") == (CASES / "expression_tag.viba").read_text(),
          "a pure tag chain is left on one line")
    document_with_expression = (CASES / "expression_document.viba.md").read_text()
    check(lint_markdown(fix_markdown(document_with_expression), "fixed.viba.md") == [],
          "the fixer folds an expression inside a document")
    check("        Object\n" in fix_markdown(document_with_expression),
          "the document's expression gets its fold")

    hint = "Distance =\n  int\n  <- $env Environment\n  <- { a hint }\n"
    check(fix_source(hint) == hint, "a statement with a hint is left as it is")
    check("exponent-result" in [v.rule for v in lint_source(hint)],
          "a statement with a hint is still reported")


def main() -> int:
    run()
    return checks.report()


if __name__ == "__main__":
    sys.exit(main())

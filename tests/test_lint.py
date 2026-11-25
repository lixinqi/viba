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

from viba.lint import (fix_file, fix_markdown, fix_markdown_report, fix_source,
                       lint_file, lint_markdown, lint_paths, lint_source,
                       main as lint_main)

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

    check("statement-fold" in rules_of("long_sum.viba"),
          "a long one-line sum is not folded")
    for name in ("inline_sum.viba", "short_sum_expression.viba"):
        check(lint_source((CASES / name).read_text(), name) == [],
              f"a short sum stays on one line: {name}")
        check(fixed(name) == (CASES / name).read_text(),
              f"the fixer leaves a short sum alone: {name}")
    check("expression-fold" in rules_of("long_sum_expression.viba"),
          "a long sum expression is folded")
    check("Oneof" in fixed("long_sum_expression.viba"),
          "the folded sum expression gets its Oneof head")
    check("expression-unfold" in rules_of("folded_short_sum.viba"),
          "a folded short sum expression is put back on one line")
    folded_sum = fixed("folded_short_sum.viba")
    check("(int | nil)" in folded_sum and "list[(int | nil)]" in folded_sum,
          f"the fixer puts a short sum back on one line: {folded_sum!r}")
    check(lint_source(folded_sum, "fixed.viba") == [],
          "the one-line short sums have no violations")
    check("expression-unfold" not in rules_of("short_sum_expression.viba"),
          "a one-line sum expression is not reported")
    for name in ("misindented_expression_sum.viba", "misindented_expression.viba"):
        check("expression-fold" in rules_of(name),
              f"a folded expression in the wrong columns is reported: {name}")
        laid_out = fixed(name)
        check(lint_source(laid_out, "fixed.viba") == [],
              f"the fixer lays out the expression block: {name}")
        check(fix_source(laid_out, "fixed.viba") == laid_out,
              f"the laid-out expression block is a fixed point: {name}")
    misindented_sum = fixed("misindented_expression_sum.viba")
    check("       Oneof\n"
          "     | LTX.prompt_encoder\n"
          "     | MAGI.get_txt_embeddings\n"
          "     | SN.t2i_prefix_forward\n"
          "     )\n" in misindented_sum,
          "a block opened after `<- (` lines up with the chain's `$name` column")
    check("expression-fold" in rules_of("headless_sum_block.viba"),
          "a folded sum without a `Oneof` head is reported")
    headless = fixed("headless_sum_block.viba")
    check("        Oneof\n"
          "      | AlphaComponent\n"
          "      | BetaComponent\n" in headless,
          "the fixer adds the `Oneof` head and makes the first line a `|` member")
    check(lint_source(headless, "fixed.viba") == [],
          "the headed sum block has no violations")
    check(fix_source(headless, "fixed.viba") == headless,
          "the headed sum block is a fixed point")
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

    check("statement-fold" in rules_of("inline_chain.viba"),
          "a chain with several operators on one line must fold")
    check([violation.line for violation in lint_file(CASES / "inline_chain.viba")]
          == [3], "the chain diagnostic is on the line that holds them")
    folded_chain = fixed("inline_chain.viba")
    check(lint_source(folded_chain, "fixed.viba") == [],
          "the fixer folds the chain one operator per line")
    check(folded_chain.count("  << ") == 5,
          "each argument of the chain stands on its own line")
    check(fix_source(folded_chain, "fixed.viba") == folded_chain,
          "the folded chain is a fixed point")
    short_chain = "A =\n     int\n  <- $env Env <- $x int\n"
    check(lint_source(short_chain) == [],
          "a short chain may stay on one line")
    check(fix_source(short_chain) == short_chain,
          "the fixer leaves a short chain alone")
    long_sum = "A =\n    Oneof\n  | a | b | c | d\n"
    check("statement-fold" in [violation.rule
                               for violation in lint_source(long_sum)],
          "a long sum written on one line must fold")
    check(lint_source(fix_source(long_sum)) == [],
          "the fixer folds the sum one branch per line")
    check("field-comments" in rules_of("field_comments.viba"),
          "field comments must share a column")
    check("field-comments" not in rules_of("good.viba"),
          "aligned field comments are not reported")

    check("parse" in rules_of("parse_error.viba"),
          "a source that does not parse is reported")

    document = lint_file(CASES / "document.viba.md")
    check([(violation.rule, violation.line) for violation in document
           if violation.rule == "statement-fold"] == [("statement-fold", 6)],
          f"a document's diagnostic points at its own line: {document!r}")
    check(not any(violation.rule == "document-title" for violation in document),
          "a title heading over `.viba` sections is accepted")

    found = lint_paths([str(CASES)])
    check(any(v.path.endswith("long_sum.viba") and v.rule == "statement-fold"
              for v in found),
          "a directory's .viba files are read")
    check(any(v.path.endswith("document.viba.md") and v.line == 6 for v in found),
          "a directory's .viba.md document is read")
    check(not any(v.path.endswith("good.viba") for v in found),
          "a directory's file without violations is not reported")
    excluded = lint_paths([str(CASES)], [str(CASES / "long_sum.viba")])
    check(not any(v.path.endswith("long_sum.viba") for v in excluded),
          "an excluded file is left alone")
    check(any(v.path.endswith("inline_product.viba") for v in excluded),
          "a file beside the excluded one is still read")

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
    for name in ("long_sum.viba", "long_sum_expression.viba",
                 "folded_short_sum.viba", "inline_product.viba", "inline_box.viba",
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

    hinted = (CASES / "document_with_hint.viba.md").read_text()
    hinted_fixed, hinted_skipped = fix_markdown_report(hinted)
    check("说明文字一行。" in hinted_fixed and "# 说明文字一行。" not in hinted_fixed,
          "the document's prose is left as it was")
    check("    Object\n  * $x str" in hinted_fixed,
          "the fixable product in the document is laid out")
    check("    Object\n  * $w (" in hinted_fixed,
          "the product with a one-line hint is laid out too")
    check("{ 宽度 }" in hinted_fixed,
          "the field and its hint text are kept")
    check(hinted_skipped == [],
          f"a one-line hint is not skipped: {hinted_skipped!r}")
    check(lint_markdown(hinted_fixed, "doc.viba.md") == [],
          "both products are laid out")

    long_hint = (CASES / "document_with_long_hint.viba.md").read_text()
    long_fixed, long_skipped = fix_markdown_report(long_hint)
    check("    Object\n  * $w Hint[{" in long_fixed,
          "a multi-line hint still gets its head and member columns")
    check("      宽度" in long_fixed and "    }]" in long_fixed,
          "the hint text lines are kept byte for byte")
    check(long_skipped == [], "a head that is already there is fixed, not skipped")
    check(lint_markdown(long_fixed, "doc.viba.md") == [],
          "the fixed document has no violations")
    long_again, long_skipped_again = fix_markdown_report(long_fixed)
    check(long_again == long_fixed and long_skipped_again == [],
          "a second --fix changes nothing")

    bad = (CASES / "document_with_field_head_hint.viba.md").read_text()
    bad_fixed, bad_skipped = fix_markdown_report(bad)
    check(bad_fixed == bad, "a headless product with a hint is left as it is")
    check(bad_skipped == [7],
          f"the field line is the skipped head line: {bad_skipped!r}")
    check([(v.rule, v.line) for v in lint_markdown(bad_fixed, "doc.viba.md")]
          == [("product-head", 7)],
          "the headless product is still reported at its first field")

    text_doc = (CASES / "document_with_text_block.viba.md").read_text()
    text_fixed, text_skipped = fix_markdown_report(text_doc)
    check("说明文字一行。" in text_fixed and "# 说明文字一行。" not in text_fixed,
          "the prose next to the blocks is left as it was")
    check("X =\n  Y" in text_fixed, "a non-viba block is not rewritten")
    check("    Object\n  * $x str" in text_fixed, "the viba block is laid out")
    check(text_skipped == [], "the viba block has nothing to skip")

    once, skipped_once = fix_markdown_report(hinted)
    twice, skipped_twice = fix_markdown_report(once)
    check(twice == once and skipped_once == [] and skipped_twice == [],
          "the second --fix of the fixed document changes nothing")

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
        check(lint_markdown(document_path.read_text(), str(document_path)) == [],
              "the written document's sections have no layout violations")

        cli = root / "cli.viba"
        cli.write_text((CASES / "long_sum.viba").read_text())
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = lint_main(["--fix", str(cli)])
        check(code == 0, "the --fix command exits 0 when nothing is left")
        check(printed.getvalue().startswith("fixed "), "the --fix command says what it changed")
        check(lint_file(cli) == [], "the --fix command wrote the layout")

        hinted = root / "hinted.viba.md"
        hinted.write_text((CASES / "document_with_hint.viba.md").read_text())
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = lint_main(["--fix", str(hinted)])
        check(code == 1 and "document-title" in printed.getvalue(),
              "the --fix command reports the document structure it cannot fix")
        check("skipped " not in printed.getvalue(), "nothing is skipped for a one-line hint")
        check("说明文字一行。" in hinted.read_text()
              and "# 说明文字一行。" not in hinted.read_text(),
              "the --fix command leaves the document's prose")

        long_path = root / "long.viba.md"
        long_path.write_text((CASES / "document_with_long_hint.viba.md").read_text())
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = lint_main(["--fix", str(long_path)])
        check("document-title" in printed.getvalue() and code == 1,
              "the --fix command leaves the document structure reported")
        check("skipped " not in printed.getvalue(),
              "a head that is already there is not skipped")
        check("    Object\n  * $w Hint[{" in long_path.read_text(),
              "the multi-line hint statement is laid out")

        bad_path = root / "bad.viba.md"
        bad_path.write_text((CASES / "document_with_field_head_hint.viba.md").read_text())
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = lint_main(["--fix", str(bad_path)])
        check(code == 1, "the --fix command exits 1 for the headless product")
        check(f"skipped {bad_path}:7 " in printed.getvalue(),
              "the --fix command says which field line it skipped")
        check(bad_path.read_text() == bad,
              "the headless product with a hint is unchanged")

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
    for name in ("short_exponent.viba", "short_partial.viba", "short_expression.viba"):
        check(lint_source((CASES / name).read_text(), name) == [],
              f"a short <-/<< chain stays on one line: {name}")
        check(fixed(name) == (CASES / name).read_text(),
              f"the fixer leaves a short chain alone: {name}")
    for name in ("long_exponent.viba", "long_partial.viba"):
        check("statement-fold" in rules_of(name),
              f"a long <-/<< statement must fold: {name}")
        check(lint_source(fixed(name), "fixed.viba") == [],
              f"the long statement folds into a clean layout: {name}")
    document_with_expression = (CASES / "expression_document.viba.md").read_text()
    check(lint_markdown(fix_markdown(document_with_expression), "fixed.viba.md") == [],
          "the fixer folds an expression inside a document")
    check("        Object\n" in fix_markdown(document_with_expression),
          "the document's expression gets its fold")

    hint = "Distance =\n  int\n  <- $env Environment\n  <- { a hint }\n"
    check(fix_source(hint) == ("Distance =\n     int\n  <- $env Environment\n"
                               "  <- { a hint }\n"),
          "a statement with a one-line hint is laid out and the hint is kept")
    check(lint_source(fix_source(hint)) == [],
          "the laid-out statement has no violations")


def main() -> int:
    run()
    return checks.report()


if __name__ == "__main__":
    sys.exit(main())

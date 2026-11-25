"""viba_lint：`.viba.md` 这种写法的 viba 代码。

`.viba.md` 按扩展名自动认：代码块照版式规则查，另外查这份写法自己的结构。结构那四条只有
检查，没有 `--fix`。用例是 `tests/data/viba_lint_document/` 下的文档。

    python3 tests/test_lint_document.py
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks

from viba.lint import (check_document, fix_markdown, lint_file, lint_paths,
                       lint_markdown, main as lint_main)

checks = Checks("viba_lint_document")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "viba_lint_document"


def violations_of(name: str):
    return lint_file(CASES / name)


def rules_of(name: str) -> list:
    return [violation.rule for violation in violations_of(name)]


def run() -> None:
    check(violations_of("good.viba.md") == [],
          "a document that follows the rules has no violations")
    check(rules_of("bad_title.viba.md") == ["document-title"],
          "a first-level heading that is not a `.viba` path is reported")
    check(violations_of("bad_title.viba.md")[0].line == 1,
          "the title diagnostic is on the heading's line")
    check(rules_of("bad_equals.viba.md") == ["document-equals"],
          "a block with two `=` is reported")
    check(violations_of("bad_equals.viba.md")[0].line == 7,
          "the equals diagnostic is on the second `=`'s line")
    check(rules_of("bad_name.viba.md") == ["document-type-name"],
          "a definition whose name is not in the heading is reported")
    check(violations_of("bad_name.viba.md")[0].line == 6,
          "the name diagnostic is on the definition's line")
    check(rules_of("bad_field.viba.md") == ["document-field"],
          "a field that is not `$name ($field_type * { comment })` is reported")
    check(violations_of("bad_field.viba.md")[0].line == 8,
          "the field diagnostic is on the field's line")
    check(violations_of("hint_equals.viba.md") == [],
          "an `=` inside a hint does not count against the block")

    check([v.rule for v in lint_file(CASES / "bad_equals.viba.md",
                                     one_assignment=False)] == [],
          "the one-assignment switch leaves the `=` count alone")
    check([v.rule for v in lint_file(CASES / "bad_name.viba.md",
                                     one_assignment=False)] == [],
          "the switch leaves the heading name alone")
    split = ("# a.viba\n\n## `Session`\n\n```viba\n"
             "import a.b as b\nimport c.d as d\n```\n\n"
             "```viba\nSession = int\n```\n")
    check(check_document(split) == [], "several imports keep their own block")
    mixed = ("# a.viba\n\n## `Session`\n\n```viba\n"
             "import a.b as b\nSession = int\n```\n")
    check([(v.rule, v.line) for v in check_document(mixed)]
          == [("document-imports", 7)],
          "an import and an assignment in one block are reported")
    check(check_document(mixed, split_imports_and_assignments=False) == [],
          "the split switch leaves the import and the assignment together")
    check([v.rule for v in lint_file(CASES / "mixed_import.viba.md")]
          == ["document-imports"],
          "the mixed block is reported through the file path too")
    check([v.rule for v in lint_file(CASES / "bad_title.viba.md",
                                     one_assignment=False)] == ["document-title"],
          "the title rule is checked with the switch off")
    check([v.rule for v in lint_file(CASES / "bad_field.viba.md",
                                     one_assignment=False)] == ["document-field"],
          "the field rule is checked with the switch off")
    picture = (CASES / "big_picture.viba.md").read_text()
    check([v.rule for v in lint_file(CASES / "big_picture.viba.md")]
          == ["document-order", "document-order"],
          "a definition written above the definitions it uses is reported")
    laid_out = fix_markdown(picture)
    check(check_document(laid_out) == [] and lint_markdown(laid_out) == [],
          "the fixer puts the big picture first")
    check(laid_out.index("## `DTensor`") < laid_out.index("## `Mesh`"),
          "the section of the big concept moves above its dependency")
    check(fix_markdown(laid_out) == laid_out,
          "the reordered document is a fixed point")
    check(check_document(picture, present_big_picture_first=False) == [],
          "the big-picture switch turns the check off")
    check(fix_markdown(picture, present_big_picture_first=False) == picture,
          "the switch keeps the fixer from moving the sections")
    cycle = ("# c.viba\n\n## `A`\n\n```viba\nA = B\n```\n\n"
             "## `B`\n\n```viba\nB = A\n```\n")
    check([v.rule for v in check_document(cycle)] == ["document-order"],
          "a cycle is reported")
    check(fix_markdown(cycle) == cycle,
          "a cycle cannot be ordered, so the fixer leaves it")

    found = lint_paths([str(CASES)])
    check(any(violation.rule == "document-field" for violation in found),
          "the extension alone brings the structure rules in")
    with tempfile.TemporaryDirectory(prefix="viba-document-") as scratch:
        source = Path(scratch) / "plain.viba"
        source.write_text("A = int\nB = str\n")
        check(lint_file(source) == [],
              "a `.viba` file is not held to the document structure")

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = lint_main([str(CASES / "good.viba.md")])
    check(status == 0 and output.getvalue() == "",
          "the command answers 0 for a document that follows the rules")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = lint_main([str(CASES / "bad_field.viba.md")])
    check(status == 1 and "document-field" in output.getvalue(),
          "the command answers 1 and names the rule")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = lint_main(["--disable-document-code-block-num-assignments-le-one",
                            str(CASES / "bad_equals.viba.md")])
    check(status == 0 and output.getvalue() == "",
          "the switch turns the one-assignment rule off")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = lint_main(["--disable-document-code-block-num-assignments-le-one",
                            str(CASES / "bad_field.viba.md")])
    check(status == 1 and "document-field" in output.getvalue(),
          "the field rule stays on with the switch off")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = lint_main(["--no-split-imports-and-assignments",
                            str(CASES / "mixed_import.viba.md")])
    check(status == 0 and output.getvalue() == "",
          "`--no-split-imports-and-assignments` leaves them together")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = lint_main(["--no-present-big-picture-first",
                            str(CASES / "big_picture.viba.md")])
    check(status == 0 and output.getvalue() == "",
          "`--no-present-big-picture-first` leaves the order alone")
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = lint_main([str(CASES / "mixed_import.viba.md")])
    check(status == 1 and "document-imports" in output.getvalue(),
          "the mixed block is reported by default")

    with tempfile.TemporaryDirectory(prefix="viba-document-") as scratch:
        document = Path(scratch) / "doc.viba.md"
        document.write_text((CASES / "bad_name.viba.md").read_text())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = lint_main(["--fix", str(document)])
        check(status == 1 and "document-type-name" in output.getvalue(),
              "`--fix` lays out the code and leaves the structure rule reported")
        check(lint_file(document) != [], "the structure rule is still there")

    with tempfile.TemporaryDirectory(prefix="viba-document-") as scratch:
        document = Path(scratch) / "big.viba.md"
        document.write_text((CASES / "big_picture.viba.md").read_text())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = lint_main(["--fix", str(document)])
        check(status == 0 and "fixed " in output.getvalue(),
              "`--fix` reorders the sections and then reports nothing")
        check(document.read_text().index("## `DTensor`")
              < document.read_text().index("## `Mesh`"),
              "the big concept's section is first after `--fix`")


def main() -> int:
    run()
    return checks.report()


if __name__ == "__main__":
    sys.exit(main())

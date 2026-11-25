"""viba_lint: report type statements whose layout is not the two-space block.

A statement is a top-level `Name = ...` definition or a `pattern ...`; a block's
body is the lines after it. The layout this tool enforces:

- A sum statement is a block led by `Oneof`, its alternatives starting with
  `|`; `Oneof` is in the column of the alternative types (the text after `| `),
  not the column of the `|` itself.
- A product statement is a block led by `Object`, `tagged[...]` or another base;
  the head is in the column of the field types (the text after `* `), not the
  column of the `*` itself.
- Every line that starts with `|`, `*`, `<-` or `<<` is indented exactly two
  spaces.
- An exponent statement's result type is in the column of the parameter types
  (the text after `<- `).
- A partial application's function is in the column of the arguments (the text
  after `<< `).
- The trailing comments of a statement's fields start in the same column.

A statement whose body is a sum, product, exponent or partial application must
fold: the body is on the lines after `Name =` (or `pattern`). Only a type
expression may stay on one line; `Box = Object * $value T` is a statement and
folds, while `UserId = int` or the `T` of `* $value T` may stay inline. A short
chain is the exception: a sum, exponent or partial application with fewer than
four operators on a line shorter than 50 characters may stay on that line.

A compound type expression in parentheses folds too: `$foo ($bar int * $nice
str)` becomes a block, its base two spaces past the line that opened it; an
exponent's result and a partial application's function sit three spaces past
`<-` / `<<`, and the closing `)` stands on a line of its own in the base column.
A pure tag chain (`$foo ($bar ($foobar int))`) stays on one line,
and so does a short sum (`(int | nil)`); a short sum folded into a block is
reported (`expression-unfold`) and the fixer puts it back on one line. A folded
sum is led by `Oneof`, its first branch a `|` member, and a misindented block is
put back in its columns; the `)` stands on a line of its own.

Both `.viba` files and the `viba` sections of `.viba.md` documents are linted;
in a document a diagnostic points at the document's own line. A `.viba.md` file
is viba code in its own written form, so it is also checked for the structure
that form asks for (`document-title`, `document-assignments`, `document-type-name`,
`document-field`, `document-imports`, `document-order`). The order rule is
fixed by moving sections; the rest are check only and stay reported.

    python -m viba.lint viba tests
    python -m viba.lint --help
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from viba import viba_ast
from viba.partial import product_elements
from viba.viba_ast.nodes import (
    Exponent,
    ExponentChain,
    GenericDefinition,
    HintBlock,
    Import,
    Partial,
    Pattern,
    Product,
    ProductChain,
    Sum,
    SumChain,
    Tagged,
    TypeDefinition,
    TypeRef,
)

INDENT = 2
OPERATORS = ("|", "*", "<-", "<<")
# A one-line sum, exponent or partial application this short may stay on one
# line: `(int | nil)` is a sum of two, not a block to fold.
SHORT_CHAIN_OPERATORS = 4
SHORT_CHAIN_CHARS = 50
SHORT_CHAIN_OPERATOR_BY_KIND = {"sum": "|", "exponent": "<-", "partial": "<<"}

# The generated pattern corpora and the test fixtures are not designs to lint.
DEFAULT_EXCLUDES = (
    "tests/corpus",
    "tests/data",
    "viba/apply_impl",
    "viba/sequential_impl",
    "viba/builtin",
)


@dataclass(frozen=True)
class Violation:
    """One place whose layout breaks the standard."""

    path: str
    line: int
    column: int
    rule: str
    message: str


@dataclass
class _Line:
    text: str
    number: int


@dataclass
class _Code:
    """A line's code part, its trailing comment, and its indentation."""

    text: str                # the code before the comment, right-stripped
    comment: str             # `#...`, or ""
    comment_column: int      # the 0-based index of `#`, -1 when there is none
    indent: int              # leading spaces
    number: int
    index: int               # index in the line list

    @property
    def stripped(self) -> str:
        return self.text.strip()


@dataclass
class _Statement:
    start: int               # index in the line list
    end: int                 # exclusive
    inline: Optional[str]    # the body on the start line, None when it is a block
    body: List[_Code]        # the code lines after the start


_START_DEFINITION = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*|__[A-Za-z0-9_]+__)(?:\[[^\]]*\])?[ \t]*=")
_START_PATTERN = re.compile(r"^pattern\b")
_START_IMPORT = re.compile(r"^import\b")


def _scan_line(text: str) -> Tuple[str, str, Optional[str]]:
    """(code, comment, open triple quote) for one line.

    A `#` inside a string is not a comment; a triple-quoted string that does not
    close on this line leaves the third item set to its delimiter.
    """
    i = 0
    quote: Optional[str] = None
    while i < len(text):
        if quote is not None:
            if text.startswith(quote, i):
                i += len(quote)
                if len(quote) == 3:
                    quote = None
                else:
                    quote = None
                continue
            i += 1
            continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            delimiter = text[i:i + 3]
            end = text.find(delimiter, i + 3)
            if end < 0:
                return text[:i], "", delimiter
            i = end + 3
            continue
        char = text[i]
        if char in "\"'":
            quote = char
            i += 1
            continue
        if char == "#":
            return text[:i], text[i:], None
        i += 1
    return text, "", None


def _code_lines(lines: Sequence[_Line], start_index: int = 0) -> List[_Code]:
    """The code lines, leaving out comments, hint text and multi-line strings."""
    out: List[_Code] = []
    hints = 0
    triple: Optional[str] = None
    for index, line in enumerate(lines):
        text = line.text
        if triple is not None:
            end = text.find(triple)
            if end < 0:
                continue
            text = text[end + len(triple):]
            triple = None
            if not text.strip():
                continue
        if hints > 0:
            if "}" in text:
                hints -= 1
            continue
        stripped = text.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        code, comment, opened = _scan_line(text)
        comment_column = len(code)
        code = code.rstrip()
        if not code:
            continue
        indent = len(code) - len(code.lstrip(" "))
        out.append(_Code(code, comment.rstrip(), comment_column, indent,
                          line.number, start_index + index))
        if opened is not None:
            triple = opened
            continue
        opens = code.count("{")
        closes = code.count("}")
        if opens > closes:
            hints = opens - closes
    return out


def _statements(lines: Sequence[_Line]) -> List[_Statement]:
    """The top-level definitions and patterns, with their body lines."""
    starts: List[Tuple[int, bool]] = []   # (line index, is a pattern)
    for index, line in enumerate(lines):
        text = line.text
        if not text or text[0] in " \t":
            continue
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _START_IMPORT.match(text):
            continue
        if _START_PATTERN.match(text):
            starts.append((index, True))
            continue
        if _START_DEFINITION.match(text):
            starts.append((index, False))
    statements: List[_Statement] = []
    for position, (index, is_pattern) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        text = lines[index].text
        if is_pattern:
            rest = text[len("pattern"):].strip()
        else:
            rest = text.split("=", 1)[1].strip()
        if rest.startswith("#"):
            rest = ""
        inline = rest or None
        body = _code_lines(lines[index + 1:end], index + 1)
        statements.append(_Statement(index, end, inline, body))
    return statements


def _kind(node) -> Optional[str]:
    if isinstance(node, (Sum, SumChain)):
        return "sum"
    if isinstance(node, (Product, ProductChain)):
        return "product"
    if isinstance(node, (Exponent, ExponentChain)):
        return "exponent"
    if isinstance(node, Partial):
        return "partial"
    return None


def _statement_kind(block: Sequence[_Line]) -> Optional[str]:
    """The kind of the one statement in `block`: sum, product, exponent, partial."""
    source = "\n".join(line.text for line in block) + "\n"
    try:
        tree = viba_ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (TypeDefinition, GenericDefinition)):
            return _kind(node.body)
        if isinstance(node, Pattern):
            return _kind(node.pattern)
    return None


def _short_chain(kind: str, text: str, line_text: str) -> bool:
    """A `|` / `<-` / `<<` chain short enough to stay on one line."""
    operator = SHORT_CHAIN_OPERATOR_BY_KIND.get(kind)
    if operator is None:
        return False
    return (text.count(operator) < SHORT_CHAIN_OPERATORS
            and len(line_text) < SHORT_CHAIN_CHARS)


def _first_token(text: str) -> str:
    stripped = text.lstrip()
    for operator in ("<-", "<<"):
        if stripped.startswith(operator):
            return operator
    return stripped[:1] if stripped else ""


def _check_chain_indent(lines: Sequence[_Line], path: str) -> List[Violation]:
    found: List[Violation] = []
    documented = _documentation_mask([line.text for line in lines])
    blocks = [block for block in _folded_blocks(lines)
              if not any(documented[block[0]:block[2] + 1])]
    for code in _code_lines(lines):
        if documented[code.index]:
            continue                        # a hint's example is prose
        token = _first_token(code.stripped)
        if token not in OPERATORS:
            continue
        owner = None
        for block in blocks:
            if block[0] < code.index <= block[2]:
                if owner is None or block[0] > owner[0]:
                    owner = block
        expected = INDENT if owner is None else _block_columns(
            lines, owner[0], owner[1], owner[2])[0]
        if code.indent != expected:
            found.append(Violation(
                path, code.number, code.indent + 1, "chain-indent",
                f"`{token}` must be indented {INDENT} spaces past its block "
                f"({expected}), not {code.indent}"))
    return found


def _check_sum(statement: _Statement, body: Sequence[_Code], path: str,
               start_line: int) -> List[Violation]:
    if statement.inline is not None:
        return []
    if not body:
        return []
    head = body[0]
    found: List[Violation] = []
    if head.stripped.startswith("|"):
        found.append(Violation(path, head.number, head.indent + 1, "sum-head",
                               "a sum statement is missing its `Oneof` head"))
    elif not head.stripped.startswith("Oneof"):
        found.append(Violation(path, head.number, head.indent + 1, "sum-head",
                               "a sum statement must be led by `Oneof`"))
    elif head.stripped != "Oneof":
        found.append(Violation(path, head.number, head.indent + 1, "sum-head",
                               "the `Oneof` head must be on a line of its own"))
    else:
        pipes = [code for code in body[1:]
                 if code.stripped.startswith("|") and code.indent == INDENT]
        if pipes and head.indent != pipes[0].indent + 2:
            found.append(Violation(
                path, head.number, head.indent + 1, "sum-head",
                "`Oneof` must share the column of the alternative types"))
    return found


def _check_product(statement: _Statement, body: Sequence[_Code], path: str,
                   start_line: int) -> List[Violation]:
    if statement.inline is not None:
        return []
    if not body:
        return []
    head = body[0]
    found: List[Violation] = []
    token = _first_token(head.stripped)
    if (head.stripped.startswith("Oneof") or token in ("|", "*", "<-", "<<")
            or head.stripped.startswith("$")):
        found.append(Violation(path, head.number, head.indent + 1, "product-head",
                               "a product statement must be led by `Object`, "
                               "`tagged[...]` or another base"))
    else:
        stars = [code for code in body[1:]
                 if code.stripped.startswith("*") and code.indent == INDENT]
        if stars and head.indent != stars[0].indent + 2:
            found.append(Violation(
                path, head.number, head.indent + 1, "product-head",
                "the product's head must share the column of the field types"))
    if "*" in head.text:
        found.append(Violation(path, head.number, head.indent + 1, "product-head",
                               "each product field must be on its own `*` line"))
    return found


def _check_exponent(statement: _Statement, body: Sequence[_Code], path: str,
                    start_line: int) -> List[Violation]:
    if statement.inline is not None or not body:
        return []
    result = body[0]
    if _first_token(result.stripped) == "<-":
        return [Violation(path, result.number, result.indent + 1, "exponent-result",
                          "an exponent statement is missing its result type")]
    arrows = [code for code in body
              if _first_token(code.stripped) == "<-" and code.indent == INDENT]
    if not arrows:
        return []
    if result.indent != arrows[0].indent + 3:
        return [Violation(path, result.number, result.indent + 1, "exponent-result",
                          "the result type must share the column of the parameter "
                          "types")]
    return []


def _check_partial(statement: _Statement, body: Sequence[_Code], path: str,
                   start_line: int) -> List[Violation]:
    if statement.inline is not None or not body:
        return []
    function = body[0]
    if _first_token(function.stripped) == "<<":
        return [Violation(path, function.number, function.indent + 1, "partial-function",
                          "a partial application is missing its function")]
    arrows = [code for code in body
              if _first_token(code.stripped) == "<<" and code.indent == INDENT]
    if not arrows:
        return []
    if function.indent != arrows[0].indent + 3:
        return [Violation(path, function.number, function.indent + 1, "partial-function",
                          "the function must share the column of the arguments")]
    return []


def _check_field_comments(body: Sequence[_Code], path: str) -> List[Violation]:
    columns = [(code.number, code.comment_column + 1) for code in body
               if code.comment and code.indent == INDENT
               and _first_token(code.stripped) in ("|", "*")]
    if len(columns) < 2:
        return []
    wanted = columns[0][1]
    for number, column in columns[1:]:
        if column != wanted:
            return [Violation(path, number, column, "field-comments",
                              "the fields' comments must share a column")]
    return []


def lint_source(source: str, path: str = "<source>") -> List[Violation]:
    """The layout violations of one viba source."""
    lines = [_Line(text, index + 1) for index, text in enumerate(source.splitlines())]
    return _lint_lines(lines, path)


def _lint_lines(lines: Sequence[_Line], path: str) -> List[Violation]:
    try:
        viba_ast.parse("\n".join(line.text for line in lines) + "\n")
    except SyntaxError as exc:
        return [Violation(path, exc.lineno or 1, exc.offset or 1, "parse",
                          f"cannot parse: {exc.msg}")]
    found: List[Violation] = _check_chain_indent(lines, path)
    found.extend(_check_expression_folds(lines, path))
    for statement in _statements(lines):
        block = lines[statement.start:statement.end]
        kind = _statement_kind(block)
        body = statement.body
        start_line = lines[statement.start].number
        if kind in ("sum", "product", "exponent", "partial") and statement.inline is not None:
            short = _short_chain(kind, statement.inline,
                                 lines[statement.start].text)
            if not short:
                found.append(Violation(
                    path, start_line, 1, "statement-fold",
                    "a type statement must fold; only a type expression may stay on one line"))
        if kind == "sum":
            found.extend(_check_sum(statement, body, path, start_line))
        elif kind == "product":
            found.extend(_check_product(statement, body, path, start_line))
        elif kind == "exponent":
            found.extend(_check_exponent(statement, body, path, start_line))
        elif kind == "partial":
            found.extend(_check_partial(statement, body, path, start_line))
        if kind in ("sum", "product"):
            found.extend(_check_field_comments(body, path))
    return found


_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})[ \t]*(.*)$")


def _viba_blocks(source: str) -> List[Tuple[int, int, List[Tuple[str, int]]]]:
    """(first line index, end index, [(text, document line)]) for each `viba` block."""
    lines = source.splitlines()
    blocks: List[Tuple[int, int, List[Tuple[str, int]]]] = []
    index = 0
    while index < len(lines):
        opened = _FENCE.match(lines[index])
        if opened is None:
            index += 1
            continue
        marker = opened.group(2)
        info = opened.group(3).split()
        if not info or info[0] != "viba":
            index += 1
            continue
        first = index + 1
        end = first
        while end < len(lines):
            closed = _FENCE.match(lines[end])
            if (closed is not None and closed.group(2)[0] == marker[0]
                    and closed.group(2).startswith(marker) and not closed.group(3)):
                break
            end += 1
        content = [(text, number + 1) for number, text in
                   enumerate(lines[first:end], first)]
        blocks.append((first, end, content))
        index = end + 1
    return blocks


def lint_markdown(source: str, path: str = "<markdown>") -> List[Violation]:
    """The layout violations of the `viba` blocks of a `.viba.md` document."""
    found: List[Violation] = []
    for _first, _end, content in _viba_blocks(source):
        lines = [_Line(text, number) for text, number in content]
        found.extend(_lint_lines(lines, path))
    return found


_HEADING = re.compile(r"^(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")


def _closes_fence(text: str, marker: str) -> bool:
    """A line that ends the fenced block opened with `marker`."""
    matched = _FENCE.match(text)
    return (matched is not None
            and matched.group(2)[0] == marker[0]
            and matched.group(2).startswith(marker)
            and not (matched.group(3) or "").strip())


def _code_name(text: str) -> str:
    """The name a heading carries: a code span's backticks are no part of it."""
    stripped = text.strip()
    if (len(stripped) >= 2 and stripped.startswith("`")
            and stripped.endswith("`")):
        return stripped.strip("`").strip()
    return stripped


def _document_blocks(source: str):
    """(the first-level headings, the `viba` blocks with the heading above each).

    A heading is `(line, text)`; a block is `[(text, line), ...]`. Only `viba`
    fenced blocks count; the prose and the other blocks are not looked at.
    """
    headings: List[Tuple[int, str]] = []
    blocks: List[Tuple[Optional[Tuple[int, str]], List[Tuple[str, int]]]] = []
    heading: Optional[Tuple[int, str]] = None
    fence = None
    block = None
    for index, text in enumerate(source.splitlines()):
        number = index + 1
        if fence is not None:
            if _closes_fence(text, fence):
                fence = None
                block = None
            elif block is not None:
                block.append((text, number))
            continue
        opened = _FENCE.match(text)
        if opened is not None:
            fence = opened.group(2)
            if (opened.group(3) or "").split()[:1] == ["viba"]:
                block = []
                blocks.append((heading, block))
            continue
        matched = _HEADING.match(text)
        if matched is not None:
            name = (matched.group(2) or "").strip()
            heading = (number, name)
            if len(matched.group(1)) == 1:
                headings.append(heading)
    return headings, blocks


def _top_level(block: Sequence[Tuple[str, int]]) -> Tuple[List[int], List[int]]:
    """(the lines with an `=` outside hints, strings and comments,
    the lines that begin with an import at the top level)."""
    equals: List[int] = []
    imports: List[int] = []
    depth = 0
    triple = None
    for text, number in block:
        if triple is not None:
            if triple in text:
                triple = None
            continue
        code, _comment, opened = _scan_line(text)
        if opened is not None:
            triple = opened
            continue
        quote = None
        index = 0
        while index < len(code):
            char = code[index]
            if quote is not None:
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                if depth:
                    depth -= 1
            elif char == "=" and depth == 0:
                equals.append(number)
            index += 1
        stripped = code.strip()
        if depth == 0 and (stripped == "import"
                           or stripped.startswith("import ")):
            imports.append(number)
    return equals, imports


def _product_fields(tree) -> List[Tagged]:
    """The tagged members of every product in the definitions of the tree."""
    fields: List[Tagged] = []
    for statement in getattr(tree, "body", ()):
        if not isinstance(statement, (TypeDefinition, GenericDefinition)):
            continue
        for node in viba_ast.walk(statement.body):
            if not isinstance(node, (Product, ProductChain)):
                continue
            for element in product_elements(node):
                if isinstance(element, Tagged):
                    fields.append(element)
    return fields


def _field_documented(field: Tagged) -> bool:
    """Whether a field is written `$name ($field_type * { comment })`."""
    body = field.type
    if not isinstance(body, (Product, ProductChain)):
        return False
    elements = product_elements(body)
    if len(elements) < 2:
        return False
    hint = elements[-1]
    return isinstance(hint, HintBlock) and bool(hint.code.strip())


def _field_line(block: Sequence[Tuple[str, int]], tag: str) -> Optional[int]:
    pattern = re.compile(r"\*\s+" + re.escape(tag) + r"(?:\s|$)")
    for text, number in block:
        if pattern.search(text):
            return number
    return None


def _document_definitions(blocks) -> List[Tuple[str, int, Set[str]]]:
    """(name, document line, the names it refers to) for a document's definitions."""
    found: List[Tuple[str, int, Set[str]]] = []
    for _heading, block in blocks:
        code = "\n".join(line for line, _at in block) + "\n"
        try:
            tree = viba_ast.parse(code)
        except SyntaxError:
            continue
        for statement in tree.body:
            if not isinstance(statement, (TypeDefinition, GenericDefinition)):
                continue
            line = (block[statement.lineno - 1][1]
                    if 1 <= statement.lineno <= len(block) else block[0][1])
            params = set(getattr(statement, "generic_params", ()) or ())
            refs = {node.name for node in viba_ast.walk(statement.body)
                    if isinstance(node, TypeRef)}
            found.append((statement.name, line, refs - params - {statement.name}))
    return found


def _headings(lines: Sequence[str]) -> List[Tuple[int, int]]:
    """(line index, level) of the headings outside fenced blocks."""
    found: List[Tuple[int, int]] = []
    fence = None
    for index, text in enumerate(lines):
        if fence is not None:
            if _closes_fence(text, fence):
                fence = None
            continue
        opened = _FENCE.match(text)
        if opened is not None:
            fence = opened.group(2)
            continue
        matched = _HEADING.match(text)
        if matched is not None:
            found.append((index, len(matched.group(1))))
    return found


def _section_tree(lines: Sequence[str]):
    """The headings as a tree: (the root, every section)."""
    headings = _headings(lines)
    root = {"start": -1, "level": 0, "end": len(lines),
            "intro_end": len(lines), "content_end": len(lines), "children": []}
    sections = []
    stack = [root]
    for index, level in headings:
        section = {"start": index, "level": level, "end": len(lines),
                   "intro_end": len(lines), "content_end": len(lines),
                   "children": []}
        while stack[-1]["level"] >= level:
            stack.pop()
        stack[-1]["children"].append(section)
        stack.append(section)
        sections.append(section)
    for position, section in enumerate(sections):
        end = len(lines)
        for later_index, later_level in headings[position + 1:]:
            if later_level <= section["level"]:
                end = later_index
                break
        section["end"] = end
        section["intro_end"] = (section["children"][0]["start"]
                                if section["children"] else end)
        section["content_end"] = _content_end(lines, section["start"], end)
    root["intro_end"] = (root["children"][0]["start"]
                         if root["children"] else len(lines))
    root["content_end"] = _content_end(lines, 0, len(lines))
    return root


def _content_end(lines: Sequence[str], start: int, end: int) -> int:
    """The line after a section's last non-blank line."""
    position = end - 1
    while position > start and not lines[position].strip():
        position -= 1
    return position + 1


def _big_first_order(count: int, edges) -> List[int]:
    """A stable order of the units with `(first, second)` meaning first before second."""
    before = {index: set() for index in range(count)}
    for first, second in edges:
        before[second].add(first)
    remaining = set(range(count))
    order: List[int] = []
    while remaining:
        ready = [index for index in range(count)
                 if index in remaining and not (before[index] & remaining)]
        if not ready:
            ready = [min(remaining)]     # a cycle: leave it in its written order
        pick = ready[0]
        order.append(pick)
        remaining.remove(pick)
    return order


def _order_sections(section, definitions) -> bool:
    """Put the sections a definition needs after the one that uses them.

    Answers whether any section changed places.
    """
    children = section["children"]
    changed = False
    assigned = list(children)
    units = [child for child in children
             if any(child["start"] <= line < child["end"]
                    for _name, line, _refs in definitions)]
    if len(units) > 1:
        unit_defs = [[entry for entry in definitions
                      if child["start"] <= entry[1] < child["end"]]
                     for child in units]
        edges = set()
        for first, entries in enumerate(unit_defs):
            refs = set()
            for _name, _line, used in entries:
                refs |= used
            for second, other in enumerate(unit_defs):
                if first == second:
                    continue
                names = {name for name, _line, _refs in other}
                if refs & names:
                    edges.add((first, second))
        order = _big_first_order(len(units), edges)
        ordered = [units[index] for index in order]
        starts = {unit["start"] for unit in units}
        slot = 0
        for position, child in enumerate(assigned):
            if child["start"] in starts:
                assigned[position] = ordered[slot]
                slot += 1
        changed = assigned != children
    section["assigned_children"] = assigned
    for child in assigned:
        if _order_sections(child, definitions):
            changed = True
    return changed


def _render_section(lines: Sequence[str], section, limit: int) -> List[str]:
    """The section's lines, children in their assigned order, gaps in place."""
    out: List[str] = []
    children = section["children"]
    if section["start"] >= 0:
        out.append(lines[section["start"]])
    if not children:
        out.extend(lines[(section["start"] + 1 if section["start"] >= 0 else 0):limit])
        return out
    first = children[0]
    out.extend(lines[(section["start"] + 1 if section["start"] >= 0 else 0):first["start"]])
    for slot, child in zip(children, section["assigned_children"]):
        out.extend(_render_section(lines, child, child["content_end"]))
        gap_end = min(slot["end"], limit)
        gap_start = min(slot["content_end"], gap_end)
        out.extend(lines[gap_start:gap_end])
    return out


def _reorder_sections(source: str) -> str:
    """The same document with the sections in big-picture-first order."""
    lines = source.splitlines()
    definitions = _document_definitions(_document_blocks(source)[1])
    if not definitions:
        return source
    root = _section_tree(lines)
    if not _order_sections(root, definitions):
        return source
    out = _render_section(lines, root, len(lines))
    return "\n".join(out) + ("\n" if source.endswith("\n") else "")


def check_document(source: str, path: str = "<markdown>",
                   one_assignment: bool = True,
                   split_imports_and_assignments: bool = True,
                   present_big_picture_first: bool = True
                   ) -> List[Violation]:
    """The rules a `.viba.md` design document follows; check only, no fix.

    The first-level heading is the viba file's path name; a `viba` block holds
    one `=` at most; the name on the left of `=` is part of the heading above
    the block; a field is written `$name ($field_type * { comment })`; an import
    and an assignment keep out of one block; a definition stands before the
    definitions it uses.

    `one_assignment` is the switch for the middle two: a block as one assignment
    under a heading that names it. It is on by default; a document written as a
    set of files (several assignments in one block) turns it off, and the title
    and field rules are still checked. Imports are no assignments, so several of
    them may share a block either way. `split_imports_and_assignments`, on by
    default, keeps an import and an assignment out of one block.
    `present_big_picture_first`, on by default, asks that a definition stand
    before the definitions it uses.
    """
    found: List[Violation] = []
    headings, blocks = _document_blocks(source)
    if len(headings) != 1:
        found.append(Violation(
            path, headings[0][0] if headings else 1, 1, "document-title",
            "a document has one first-level heading, the viba file it describes"))
    else:
        number, name = headings[0]
        if not _code_name(name).endswith(".viba"):
            found.append(Violation(
                path, number, 1, "document-title",
                "the first-level heading is the viba file's path name "
                "(ending `.viba`)"))
    for heading, block in blocks:
        equals, imports = _top_level(block)
        if one_assignment and len(equals) > 1:
            found.append(Violation(
                path, equals[1], 1, "document-assignments",
                "a viba block holds at most one assignment"))
        if one_assignment and equals:
            number = equals[0]
            line_text = next(line for line, at in block if at == number)
            name = line_text.split("=", 1)[0].strip().split("[", 1)[0].strip()
            if heading is None:
                found.append(Violation(
                    path, number, 1, "document-type-name",
                    f"the definition {name!r} stands under no heading"))
            else:
                heading_name = _code_name(heading[1])
                if not name or name not in heading_name:
                    found.append(Violation(
                        path, number, 1, "document-type-name",
                        f"the name {name!r} is not part of the heading above "
                        f"the block ({heading_name!r})"))
        if split_imports_and_assignments and imports and equals:
            line = max(imports[0], equals[0])
            found.append(Violation(
                path, line, 1, "document-imports",
                "an import and an assignment do not share a block"))
        code = "\n".join(line for line, _at in block) + "\n"
        try:
            tree = viba_ast.parse(code)
        except SyntaxError:
            continue                    # the layout lint reports the parse error
        for field in _product_fields(tree):
            if _field_documented(field):
                continue
            line = _field_line(block, field.tag) or (
                block[0][1] if block else 1)
            found.append(Violation(
                path, line, 1, "document-field",
                f"the field {field.tag} is written "
                f"`$name ($field_type * {{ comment }})`"))
    if present_big_picture_first:
        definitions = _document_definitions(blocks)
        lines_of = {}
        for name, line, _refs in definitions:
            lines_of.setdefault(name, line)
        for name, line, refs in definitions:
            for used in sorted(refs):
                if (used in lines_of and lines_of[used] < line):
                    found.append(Violation(
                        path, line, 1, "document-order",
                        f"the big picture comes first: {name} uses {used}, "
                        f"which stands above it"))
    return found


def lint_file(path: Path, one_assignment: bool = True,
              split_imports_and_assignments: bool = True,
              present_big_picture_first: bool = True) -> List[Violation]:
    text = path.read_text()
    if path.name.endswith(".viba.md"):
        # A `.viba.md` file is viba code in its own written form: the layout of
        # its blocks, and the structure that form asks for.
        return (lint_markdown(text, str(path))
                + check_document(text, str(path), one_assignment,
                                 split_imports_and_assignments,
                                 present_big_picture_first))
    return lint_source(text, str(path))


def _files(paths: Iterable[str], excludes: Iterable[str] = ()) -> List[Path]:
    skip = [str(Path(ex).as_posix()).rstrip("/") for ex in excludes]
    found: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if "__pycache__" in child.parts or ".git" in child.parts:
                    continue
                text = child.as_posix()
                if any(text == ex or text.startswith(ex + "/") for ex in skip):
                    continue
                if child.is_file() and (child.name.endswith(".viba")
                                        or child.name.endswith(".viba.md")):
                    found.append(child)
    return found


def lint_paths(paths: Iterable[str], excludes: Iterable[str] = (),
               one_assignment: bool = True,
               split_imports_and_assignments: bool = True,
               present_big_picture_first: bool = True) -> List[Violation]:
    found: List[Violation] = []
    for path in _files(paths, excludes):
        found.extend(lint_file(path, one_assignment,
                               split_imports_and_assignments,
                               present_big_picture_first))
    return found


def _opens_hint_or_string(text: str) -> bool:
    return "{" in text or "}" in text or '"""' in text or "'''" in text


def _spans_hint_or_string(lines: Sequence[_Line]) -> bool:
    """A hint or triple-quoted string that runs past one line."""
    depth = 0
    for line in lines:
        _code, _comment, opened = _scan_line(line.text)
        if opened is not None:
            return True
        depth += line.text.count("{") - line.text.count("}")
        if depth > 0:
            return True
    return False


def _split_top_level(text: str, operator: str) -> List[str]:
    """The pieces `text` splits into at the top-level `operator`."""
    pieces: List[str] = []
    start = 0
    depth = 0
    quote = None
    i = 0
    while i < len(text):
        char = text[i]
        if quote is not None:
            if text.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
            i += 1
            continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            quote = text[i:i + 3]
            i += 3
            continue
        if char in "\"'":
            quote = char
            i += 1
            continue
        if char in "([{":
            depth += 1
            i += 1
            continue
        if char in ")]}":
            depth -= 1
            i += 1
            continue
        if depth == 0 and text.startswith(operator, i):
            pieces.append(text[start:i].strip())
            i += len(operator)
            start = i
            continue
        i += 1
    pieces.append(text[start:].strip())
    return pieces


def _member_content(code: _Code, operator: str) -> str:
    return code.stripped[len(operator):].strip()


def _inline_elements(kind: str, inline: str):
    """(head, [(content, comment)]) for an inline statement body, or None."""
    if kind == "sum":
        parts = _split_top_level(inline, "|")
        if parts and parts[0] == "Oneof":
            return "Oneof", [(part, "") for part in parts[1:]]
        return "Oneof", [(part, "") for part in parts]
    if kind == "product":
        parts = _split_top_level(inline, "*")
        if len(parts) < 2:
            return None
        head = parts[0]
        members = [(part, "") for part in parts[1:]]
        if head.startswith("$"):
            members = [(head, "")] + members
            head = "Object"
        return head, members
    if kind == "exponent":
        parts = _split_top_level(inline, "<-")
        if len(parts) < 2:
            return None
        return parts[0], [(part, "") for part in parts[1:]]
    if kind == "partial":
        parts = _split_top_level(inline, "<<")
        if len(parts) < 2:
            return None
        return parts[0], [(part, "") for part in parts[1:]]
    return None


def _block_elements(kind: str, body: Sequence[_Code]):
    """(head, head comment, [(content, comment)]) for a block body, or None."""
    head = body[0]
    if kind == "sum":
        if head.stripped == "Oneof":
            members = [(_member_content(code, "|"), code.comment)
                       for code in body[1:]]
            return "Oneof", head.comment, members
        members = [(head.stripped, head.comment)]
        members += [(_member_content(code, "|"), code.comment) for code in body[1:]]
        return "Oneof", "", members
    if kind == "product":
        parts = _split_top_level(head.stripped, "*")
        members = []
        if len(parts) >= 2:
            head_text = parts[0]
            members += [(part, "") for part in parts[1:]]
        else:
            head_text = head.stripped
        members += [(_member_content(code, "*"), code.comment) for code in body[1:]]
        if head_text.startswith("$"):
            members = [(head_text, head.comment)] + members
            return "Object", "", members
        return head_text, head.comment, members
    operator = "<-" if kind == "exponent" else "<<"
    parts = _split_top_level(head.stripped, operator)
    members = []
    if len(parts) >= 2:
        result = parts[0]
        members += [(part, "") for part in parts[1:]]
    else:
        result = head.stripped
    members += [(_member_content(code, operator), code.comment)
                for code in body[1:]]
    return result, head.comment, members


def _render(kind: str, head: str, head_comment: str,
            members: Sequence[Tuple[str, str]]) -> List[str]:
    if kind == "sum":
        prefix = "  | "
        first = "    Oneof"
    elif kind == "product":
        prefix = "  * "
        first = "    " + head
    elif kind == "exponent":
        prefix = "  <- "
        first = "     " + head
    else:
        prefix = "  << "
        first = "     " + head
    if head_comment:
        first = first + "  " + head_comment
    lines = [first]
    commented = [prefix + content for content, comment in members if comment]
    column = max((len(line) for line in commented), default=0) + 2
    for content, comment in members:
        line = prefix + content
        if comment:
            line = line + " " * (column - len(line)) + comment
        lines.append(line)
    return lines


def _fix_statement_prefixes(lines: Sequence[_Line],
                            statement: _Statement) -> Optional[List[str]]:
    """Re-indent the head and the statement's own members; keep the rest."""
    block = lines[statement.start:statement.end]
    kind = _statement_kind(block)
    if kind not in ("sum", "product", "exponent", "partial"):
        return None
    body = statement.body
    if not body:
        return None
    head = body[0]
    stripped = head.stripped
    if kind == "sum":
        if stripped != "Oneof":
            return None
    elif kind == "product":
        if (stripped.startswith("$") or stripped == "Oneof"
                or stripped.startswith(("|", "*", "<-", "<<"))):
            return None
        head_column = 4
    else:
        if stripped.startswith(("|", "*", "<-", "<<")) or stripped == "Oneof":
            return None
        head_column = 5
    if kind in ("sum", "product"):
        head_column = 4
    blocks = _folded_blocks(lines)
    new = [line.text for line in block]
    for code in body:
        if code is head:
            column = head_column
        elif _first_token(code.stripped) in OPERATORS:
            inside = any(open_index < code.index <= close_index
                         for open_index, _base, close_index, _close_col in blocks)
            if inside:
                continue
            column = INDENT
        else:
            continue
        new[code.index - statement.start] = " " * column + block[code.index - statement.start].text.lstrip()
    return new


def _fix_statement(lines: Sequence[_Line], statement: _Statement) -> Optional[List[str]]:
    text = lines[statement.start].text
    if text.lstrip().startswith("pattern") or "=" not in text:
        return None
    left = text.split("=", 1)[0].rstrip()
    block = lines[statement.start:statement.end]
    kind = _statement_kind(block)
    if kind not in ("sum", "product", "exponent", "partial"):
        return None
    if statement.inline is not None:
        if _short_chain(kind, statement.inline, lines[statement.start].text):
            return None
        if "#" in statement.inline:
            return None
        elements = _inline_elements(kind, statement.inline)
        if elements is None:
            return None
        head, members = elements
        body = _render(kind, head, "", members)
        trailing = [line.text for line in lines[statement.start + 1:statement.end]]
        return [left + " ="] + body + trailing
    if _spans_hint_or_string(block) or any("(" in line.text for line in block):
        return _fix_statement_prefixes(lines, statement)
    body = statement.body
    if not body:
        return None
    first, last = body[0].index, body[-1].index
    if any(line.text.strip().startswith("#") for line in lines[first:last + 1]):
        return None
    if any(line.text.strip().startswith("#")
           for line in lines[statement.start + 1:first]):
        return None
    elements = _block_elements(kind, body)
    if elements is None:
        return None
    head, head_comment, members = elements
    body_lines = _render(kind, head, head_comment, members)
    trailing = [line.text for line in lines[last + 1:statement.end]]
    return [left + " ="] + body_lines + trailing


def _expression_base(text: str) -> int:
    """Two spaces past the start of the expression the line holds."""
    i = 0
    while i < len(text) and text[i] == " ":
        i += 1
    stripped = text[i:]
    equals = stripped.find("=")
    if equals >= 0 and not stripped.startswith(("|", "*", "<")):
        rest = stripped[equals + 1:]
        j = 0
        while j < len(rest) and rest[j] == " ":
            j += 1
        return i + equals + 1 + j
    for operator in ("<- ", "<< ", "| ", "* "):
        if stripped.startswith(operator):
            return i + len(operator)
    return i


def _folded_blocks(lines: Sequence[_Line]) -> List[Tuple[int, int, int, int]]:
    """(opening line index, base, closing line index, the closing `)` column).

    A block is an expression whose `(` ends a line. The `)` may sit on a line of
    its own or trail the last member (`| nil)`); both are the same block, and the
    second is reported so the fixer can put the `)` on its own line.
    """
    blocks: List[Tuple[int, int, int, int]] = []
    stack: List[Tuple[int, int, int]] = []    # (line, base, depth before its `(`)
    depth = 0
    for index, line in enumerate(lines):
        code, _comment, opened = _scan_line(line.text)
        if opened is not None:
            continue
        for column, char in enumerate(code):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if stack and depth == stack[-1][2]:
                    open_index, base, _before = stack.pop()
                    blocks.append((open_index, base, index, column))
        if code.rstrip().endswith("("):
            stack.append((index, _expression_base(line.text), depth - 1))
    return blocks


def _block_columns(lines: Sequence[_Line], open_index: int, base: int,
                   close_index: int) -> Tuple[int, int, int]:
    """(member column, head column, closing column) for a folded expression block.

    A block opened right after a chain operator — `<- (`, `* (` — and holding a
    sum or product puts its `|` / `*` in that operator's type column, the column
    of the `$name` tags, so the block lines up with the chain it belongs to; the
    head is two past and the `)` closes in the same column. Other blocks keep
    the base's columns: the head four (five for `<-` / `<<`) past the base, the
    members two, the `)` at the base.
    """
    member_lines = [line for line in lines[open_index + 1:close_index + 1]]
    tokens = {_first_token(_scan_line(line.text)[0].strip())
              for line in member_lines}
    is_product = bool(tokens & {"|", "*"})
    opening = _scan_line(lines[open_index].text)[0].rstrip()
    before = opening[:-1].strip() if opening.endswith("(") else ""
    if before in ("<-", "<<", "*", "|") and is_product:
        column = _expression_base(lines[open_index].text)
        return column, column + INDENT, column
    return base + INDENT, base + (4 if is_product else 5), base


def _paren_pairs(text: str) -> List[Tuple[int, int, str]]:
    """(open index, close index, content) for the parentheses of one line."""
    pairs: List[Tuple[int, int, str]] = []
    stack: List[int] = []
    quote = None
    i = 0
    while i < len(text):
        char = text[i]
        if quote is not None:
            if text.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
            i += 1
            continue
        if text.startswith('"""', i) or text.startswith("'''", i):
            quote = text[i:i + 3]
            i += 3
            continue
        if char in "\"'":
            quote = char
            i += 1
            continue
        if char == "(":
            stack.append(i)
        elif char == ")" and stack:
            start = stack.pop()
            pairs.append((start, i, text[start + 1:i]))
        i += 1
    return pairs


def _content_kind(content: str) -> Optional[str]:
    """The kind of a type expression, or None when it is a leaf or a tag."""
    try:
        tree = viba_ast.parse("X = " + content + "\n")
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, TypeDefinition):
            return _kind(node.body)
    return None


def _documentation_mask(lines: Sequence[str]) -> List[bool]:
    """Which lines sit inside a `{...}` hint or a multi-line string.

    A hint is prose: a `.viba` example in it is not this file's layout, so the
    linter leaves it alone (the fold checks have always done so line by line;
    the block checks need the whole span).
    """
    mask: List[bool] = []
    depth = 0
    triple = None
    for text in lines:
        if triple is not None:
            mask.append(True)
            if triple in text:
                triple = None
            continue
        _code, _comment, opened = _scan_line(text)
        mask.append(depth > 0 or opened is not None)
        if opened is not None:
            triple = opened
            continue
        depth += text.count("{") - text.count("}")
        if depth < 0:
            depth = 0
    return mask


def _collapsed_short_sum(lines: Sequence[_Line], open_index: int,
                         close_index: int, close_col: int) -> Optional[str]:
    """The one line a folded short sum expression becomes, or None.

    Only a plain sum goes back on one line — a `Oneof` head, `|` member lines,
    no comments, no nested block — and only while that line stays short. An
    expression like `(Oneof | int | nil)` is a sum of two written as a block;
    `(int | nil)` says the same thing.
    """
    opening, comment, quote = _scan_line(lines[open_index].text)
    if comment or quote is not None:
        return None
    opening = opening.rstrip()
    if not opening.endswith("("):
        return None
    closing, close_comment, close_quote = _scan_line(lines[close_index].text)
    if close_comment or close_quote is not None:
        return None
    closing = closing.rstrip()
    before = closing[:close_col].rstrip()
    suffix = closing[close_col + 1:].rstrip()
    inner = list(lines[open_index + 1:close_index])
    if before.strip():
        inner.append(_Line(before, lines[close_index].number))
    if not inner:
        return None
    head, head_comment, head_quote = _scan_line(inner[0].text)
    if head_comment or head_quote is not None or head.strip() != "Oneof":
        return None
    members = []
    for line in inner[1:]:
        code, code_comment, code_quote = _scan_line(line.text)
        stripped = code.strip()
        if (code_comment or code_quote is not None
                or not stripped.startswith("| ")):
            return None
        members.append(stripped[2:])
    if not members:
        return None
    joined = " | ".join(members)
    collapsed = opening[:-1] + "(" + joined + ")" + suffix
    if not _short_chain("sum", joined, collapsed):
        return None
    return collapsed


def _unfold_short_sums(lines: Sequence[str]) -> Optional[List[str]]:
    """The same lines with one folded short sum on one line, or None."""
    texts = list(lines)
    documented = _documentation_mask(texts)
    wrapped = [_Line(text, index + 1) for index, text in enumerate(texts)]
    for open_index, _base, close_index, close_col in _folded_blocks(wrapped):
        if any(documented[open_index:close_index + 1]):
            continue
        collapsed = _collapsed_short_sum(wrapped, open_index, close_index,
                                         close_col)
        if collapsed is not None:
            return (list(lines[:open_index]) + [collapsed]
                    + list(lines[close_index + 1:]))
    return None


def _reindented_lines(lines: Sequence[_Line], open_index: int, base: int,
                      close_index: int, close_col: int) -> Optional[List[str]]:
    """The lines a folded expression block becomes, or None when it is already
    right or cannot be put right (a comment inside, a nested block, no head).

    The head shares the column four past the base (`<-` / `<<` blocks five), the
    members two, and the `)` stands on a line of its own in the base column.
    """
    opening, comment, quote = _scan_line(lines[open_index].text)
    if comment or quote is not None or not opening.rstrip().endswith("("):
        return None
    closing, close_comment, close_quote = _scan_line(lines[close_index].text)
    if close_comment or close_quote is not None:
        return None
    closing = closing.rstrip()
    before = closing[:close_col].rstrip()
    suffix = closing[close_col + 1:].rstrip()
    inner = list(lines[open_index + 1:close_index])
    if before.strip():
        inner.append(_Line(before, lines[close_index].number))
    if not inner:
        return None
    head_code, head_comment, head_quote = _scan_line(inner[0].text)
    head = head_code.strip()
    if head_comment or head_quote is not None or not head:
        return None
    if _first_token(head) in OPERATORS:
        return None                     # the head is no head: a member first
    members = []
    tokens = set()
    for line in inner[1:]:
        code, code_comment, code_quote = _scan_line(line.text)
        stripped = code.strip()
        token = _first_token(stripped)
        if (code_comment or code_quote is not None
                or token not in OPERATORS):
            return None
        members.append(stripped)
        tokens.add(token)
    if not members:
        return None
    if not tokens <= {"|", "*"} and not tokens <= {"<-", "<<"}:
        return None
    if tokens <= {"|"} and head != "Oneof":
        # a sum block is led by `Oneof`: the line standing first is a member
        members = ["| " + head] + members
        head = "Oneof"
    member_column, head_column, closing_column = _block_columns(
        lines, open_index, base, close_index)
    rendered = ([" " * head_column + head]
                + [" " * member_column + member for member in members]
                + [" " * closing_column + ")" + suffix])
    original = [line.text.rstrip()
                for line in lines[open_index + 1:close_index + 1]]
    if rendered == original:
        return None                     # already the block the rules ask for
    return rendered


def _reindent_expression_blocks(lines: Sequence[str]) -> List[str]:
    """The same lines, with every folded expression block in its columns."""
    out = list(lines)
    for _ in range(32):
        wrapped = [_Line(text, index + 1) for index, text in enumerate(out)]
        documented = _documentation_mask(out)
        changed = False
        for open_index, base, close_index, close_col in _folded_blocks(wrapped):
            if any(documented[open_index:close_index + 1]):
                continue
            block = _reindented_lines(wrapped, open_index, base, close_index, close_col)
            if block is None:
                continue
            out[open_index + 1:close_index + 1] = block
            changed = True
            break
        if not changed:
            break
    return out


def _check_expression_folds(lines: Sequence[_Line], path: str) -> List[Violation]:
    """A compound expression in `(...)` must fold; a pure tag chain does not."""
    found: List[Violation] = []
    code_lines = _code_lines(lines)
    for code in code_lines:
        if _spans_hint_or_string([_Line(code.text, code.number)]):
            continue
        for open_col, _close_col, content in _paren_pairs(code.text):
            kind = _content_kind(content)
            if kind is None:
                continue
            if _short_chain(kind, content, code.text):
                continue
            found.append(Violation(
                path, code.number, open_col + 1, "expression-fold",
                "a compound type expression in parentheses must fold"))
    documented = _documentation_mask([line.text for line in lines])
    for open_index, base, close_index, _close_col in _folded_blocks(lines):
        if any(documented[open_index:close_index + 1]):
            continue
        inner = [code for code in code_lines
                 if open_index < code.index <= close_index]
        if not inner:
            continue
        head = inner[0]
        member_column, head_column, closing_column = _block_columns(
            lines, open_index, base, close_index)
        if head.indent != head_column:
            found.append(Violation(
                path, head.number, head.indent + 1, "expression-fold",
                f"the expression's head must be in column {head_column + 1}"))
        member_tokens = {_first_token(code.stripped) for code in inner[1:]
                         if not code.stripped.startswith(")")}
        head_text = _scan_line(head.text)[0].strip()
        if member_tokens == {"|"} and head_text != "Oneof":
            found.append(Violation(
                path, head.number, head.indent + 1, "expression-fold",
                "a sum expression's head is `Oneof`"))
        closing = lines[close_index]
        text = _scan_line(closing.text)[0].rstrip()
        stripped = text.lstrip(" ")
        if (not stripped.startswith(")")
                or len(text) - len(stripped) != closing_column):
            found.append(Violation(
                path, closing.number, 1, "expression-fold",
                f"the closing `)` must be in column {closing_column + 1}, "
                "on a line of its own"))
    for open_index, _base, close_index, close_col in _folded_blocks(lines):
        if any(documented[open_index:close_index + 1]):
            continue
        collapsed = _collapsed_short_sum(lines, open_index, close_index,
                                         close_col)
        if collapsed is not None:
            opening = _scan_line(lines[open_index].text)[0].rstrip()
            found.append(Violation(
                path, lines[open_index].number, len(opening), "expression-unfold",
                "a short sum expression stays on one line"))
    return found


def _expression_elements(kind: str, content: str):
    """(head, operator, [members]) for a folded expression's content, or None."""
    if kind == "sum":
        parts = _split_top_level(content, "|")
        if parts and parts[0] == "Oneof":
            return "Oneof", "|", parts[1:]
        return "Oneof", "|", parts
    if kind == "product":
        parts = _split_top_level(content, "*")
        if len(parts) < 2:
            return None
        head = parts[0]
        members = parts[1:]
        if head.startswith("$"):
            members = [head] + members
            head = "Object"
        return head, "*", members
    if kind == "exponent":
        parts = _split_top_level(content, "<-")
        if len(parts) < 2:
            return None
        return parts[0], "<-", parts[1:]
    if kind == "partial":
        parts = _split_top_level(content, "<<")
        if len(parts) < 2:
            return None
        return parts[0], "<<", parts[1:]
    return None


def _fold_line(text: str) -> Optional[List[str]]:
    """The lines a line with an unfolded compound expression becomes, or None."""
    code, comment, opened = _scan_line(text)
    if opened is not None or _spans_hint_or_string([_Line(code, 1)]):
        return None
    code = code.rstrip()
    pairs = [(open_col, close_col, content) for open_col, close_col, content
             in _paren_pairs(code) if _content_kind(content) is not None]
    chosen = None
    for pair in sorted(pairs, key=lambda candidate: candidate[0]):
        pair_kind = _content_kind(pair[2])
        if _short_chain(pair_kind, pair[2], text):
            continue
        chosen = pair
        break
    if chosen is None:
        return None
    open_col, close_col, content = chosen
    kind = _content_kind(content)
    elements = _expression_elements(kind, content)
    if elements is None:
        return None
    head, operator, members = elements
    base = _expression_base(text)
    head_column = base + (4 if kind in ("sum", "product") else 5)
    lines = [code[:open_col] + "("]
    lines.append(" " * head_column + head)
    for member in members:
        lines.append(" " * (base + INDENT) + operator + " " + member)
    last = " " * base + ")" + code[close_col + 1:]
    if comment:
        last = last + "  " + comment
    lines.append(last)
    return lines


def _fold_expressions(lines: Sequence[str]) -> List[str]:
    """Fold the compound expressions of the lines, leaving the rest."""
    out: List[str] = []
    hints = 0
    triple = None
    for text in lines:
        if triple is not None:
            if triple in text:
                triple = None
            out.append(text)
            continue
        if hints > 0:
            if "}" in text:
                hints -= 1
            out.append(text)
            continue
        code, _comment, opened = _scan_line(text)
        if opened is not None:
            triple = opened
            out.append(text)
            continue
        if _spans_hint_or_string([_Line(text, 1)]):
            opens = text.count("{") - text.count("}")
            if opens > 0:
                hints = opens
            out.append(text)
            continue
        folded = _fold_line(text)
        if folded is None:
            out.append(text)
        else:
            out.extend(folded)
    return out


def _fix_source_report(source: str, path: str = "<source>") -> Tuple[str, List[int]]:
    """Rewrite the fixable statements; answer the lines left alone for a hint."""
    lines = [_Line(text, index + 1) for index, text in enumerate(source.splitlines())]
    while True:
        unfolded = _unfold_short_sums([line.text for line in lines])
        if unfolded is None:
            break
        lines = [_Line(text, index + 1) for index, text in enumerate(unfolded)]
    out: List[str] = []
    skipped: List[int] = []
    cursor = 0
    for statement in _statements(lines):
        out.extend(line.text for line in lines[cursor:statement.start])
        fixed = _fix_statement(lines, statement)
        if fixed is None:
            block = lines[statement.start:statement.end]
            if _spans_hint_or_string(block):
                head = statement.body[0].number if statement.body \
                    else lines[statement.start].number
                skipped.append(head)
            out.extend(line.text for line in lines[statement.start:statement.end])
        else:
            out.extend(fixed)
        cursor = statement.end
    out.extend(line.text for line in lines[cursor:])
    fixed = "\n".join(out)
    trailing = "\n" if source.endswith("\n") else ""
    for _ in range(8):
        reindented = _reindent_expression_blocks(fixed.splitlines())
        folded = _fold_expressions(reindented)
        updated = "\n".join(folded) + trailing
        if updated == fixed:
            break
        fixed = updated
    return fixed, skipped


def fix_source(source: str, path: str = "<source>") -> str:
    """Rewrite the fixable statements of a viba source to the layout."""
    return _fix_source_report(source, path)[0]


def fix_markdown_report(source: str,
                        present_big_picture_first: bool = True
                        ) -> Tuple[str, List[int]]:
    """Rewrite the `viba` blocks of a document; the prose and fences stay."""
    if present_big_picture_first:
        source = _reorder_sections(source)
    document = source.splitlines()
    skipped: List[int] = []
    for first, end, content in reversed(_viba_blocks(source)):
        text = "\n".join(line for line, _at in content) + "\n"
        fixed, block_skipped = _fix_source_report(text)
        for line in block_skipped:
            if 1 <= line <= len(content):
                skipped.append(content[line - 1][1])
        document[first:end] = fixed.splitlines()
    return ("\n".join(document) + ("\n" if source.endswith("\n") else ""),
            sorted(skipped))


def fix_markdown(source: str, present_big_picture_first: bool = True) -> str:
    """Rewrite the fixable sections of a `.viba.md` document."""
    return fix_markdown_report(source, present_big_picture_first)[0]


def fix_file_report(path: Path,
                    present_big_picture_first: bool = True
                    ) -> Tuple[bool, List[int]]:
    """Rewrite one file; answer whether it changed and the hint-block lines left."""
    text = path.read_text()
    if path.name.endswith(".viba.md"):
        fixed, skipped = fix_markdown_report(text, present_big_picture_first)
    else:
        fixed, skipped = _fix_source_report(text)
    if fixed != text:
        path.write_text(fixed)
        return True, skipped
    return False, skipped


def fix_file(path: Path, present_big_picture_first: bool = True) -> bool:
    """Rewrite one file; answer whether it changed."""
    return fix_file_report(path, present_big_picture_first)[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m viba.lint",
        description="Report type statements whose layout is not the two-space block.")
    parser.add_argument("paths", nargs="*", default=["."],
                        help="files or directories to lint (default: .)")
    parser.add_argument("--fix", action="store_true",
                        help="rewrite the fixable layout, then report what is left")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATH",
                        help="a path prefix to leave alone (repeatable)")
    parser.add_argument("--no-default-excludes", action="store_true",
                        help="also read the generated corpora and the fixtures")
    parser.add_argument("--document-code-block-num-assignments-le-one",
                        dest="document_code_block_num_assignments_le_one",
                        action="store_true", default=True,
                        help="a `.viba.md` code block holds one assignment at "
                             "most (default)")
    parser.add_argument("--no-document-code-block-num-assignments-le-one",
                        dest="document_code_block_num_assignments_le_one",
                        action="store_false",
                        help="a code block may hold more than one assignment")
    parser.add_argument("--split-imports-and-assignments",
                        dest="split_imports_and_assignments",
                        action="store_true", default=True,
                        help="a `.viba.md` block holds either imports or an "
                             "assignment, not both (default)")
    parser.add_argument("--no-split-imports-and-assignments",
                        dest="split_imports_and_assignments",
                        action="store_false",
                        help="a block may hold imports and an assignment")
    parser.add_argument("--present-big-picture-first", action="store_true",
                        default=True,
                        help="a `.viba.md` definition stands before the "
                             "definitions it uses (default)")
    parser.add_argument("--no-present-big-picture-first",
                        dest="present_big_picture_first",
                        action="store_false",
                        help="the definitions need not be in that order")
    args = parser.parse_args(argv)
    excludes = ([] if args.no_default_excludes
                else list(DEFAULT_EXCLUDES)) + args.exclude
    if args.fix:
        for path in _files(args.paths, excludes):
            changed, skipped = fix_file_report(path,
                                               args.present_big_picture_first)
            if changed:
                print(f"fixed {path}")
            for line in skipped:
                print(f"skipped {path}:{line} (hint block)")
    violations = sorted(
        lint_paths(args.paths, excludes,
                   args.document_code_block_num_assignments_le_one,
                   args.split_imports_and_assignments,
                   args.present_big_picture_first),
        key=lambda v: (v.path, v.line, v.column, v.rule))
    for violation in violations:
        print(f"{violation.path}:{violation.line}:{violation.column}: "
              f"{violation.rule}: {violation.message}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())

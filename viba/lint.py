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
folds, while `UserId = int` or the `T` of `* $value T` may stay inline.

A compound type expression in parentheses folds too: `$foo ($bar int * $nice
str)` becomes a block, its base two spaces past the line that opened it; an
exponent's result and a partial application's function sit three spaces past
`<-` / `<<`. A pure tag chain (`$foo ($bar ($foobar int))`) stays on one line.

Both `.viba` files and the `viba` sections of `.viba.md` documents are linted;
in a document a diagnostic points at the document's own line.

    python -m viba.lint viba tests
    python -m viba.lint --help
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from viba import viba_ast
from viba.viba_ast.nodes import (
    Exponent,
    ExponentChain,
    GenericDefinition,
    Import,
    Partial,
    Pattern,
    Product,
    ProductChain,
    Sum,
    SumChain,
    TypeDefinition,
)
from viba.wiki import markdown

INDENT = 2
OPERATORS = ("|", "*", "<-", "<<")


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


def _first_token(text: str) -> str:
    stripped = text.lstrip()
    for operator in ("<-", "<<"):
        if stripped.startswith(operator):
            return operator
    return stripped[:1] if stripped else ""


def _check_chain_indent(lines: Sequence[_Line], path: str) -> List[Violation]:
    found: List[Violation] = []
    blocks = _folded_blocks(lines)
    for code in _code_lines(lines):
        token = _first_token(code.stripped)
        if token not in OPERATORS:
            continue
        base = 0
        for open_index, block_base, close_index in blocks:
            if open_index < code.index < close_index and block_base > base:
                base = block_base
        expected = base + INDENT
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


def lint_markdown(source: str, path: str = "<markdown>") -> List[Violation]:
    """The layout violations of the sections of a `.viba.md` document."""
    found: List[Violation] = []
    try:
        sections = markdown.lines_of(source)
    except SyntaxError as exc:
        return [Violation(path, 1, 1, "parse", f"cannot parse the document: {exc}")]
    for name, numbered in sections:
        lines = [_Line(text, number) for text, number in numbered]
        found.extend(_lint_lines(lines, path))
    return found


def lint_file(path: Path) -> List[Violation]:
    text = path.read_text()
    if path.name.endswith(".viba.md"):
        return lint_markdown(text, str(path))
    return lint_source(text, str(path))


def _files(paths: Iterable[str]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if "__pycache__" in child.parts or ".git" in child.parts:
                    continue
                if child.is_file() and (child.name.endswith(".viba")
                                        or child.name.endswith(".viba.md")):
                    found.append(child)
    return found


def lint_paths(paths: Iterable[str]) -> List[Violation]:
    found: List[Violation] = []
    for path in _files(paths):
        found.extend(lint_file(path))
    return found


def _opens_hint_or_string(text: str) -> bool:
    return "{" in text or "}" in text or '"""' in text or "'''" in text


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


def _fix_statement(lines: Sequence[_Line], statement: _Statement) -> Optional[List[str]]:
    text = lines[statement.start].text
    if text.lstrip().startswith("pattern") or "=" not in text:
        return None
    left = text.split("=", 1)[0].rstrip()
    block = lines[statement.start:statement.end]
    if any(_opens_hint_or_string(line.text) for line in block):
        return None
    kind = _statement_kind(block)
    if kind not in ("sum", "product", "exponent", "partial"):
        return None
    if statement.inline is not None:
        if "#" in statement.inline:
            return None
        elements = _inline_elements(kind, statement.inline)
        if elements is None:
            return None
        head, members = elements
        body = _render(kind, head, "", members)
        trailing = [line.text for line in lines[statement.start + 1:statement.end]]
        return [left + " ="] + body + trailing
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


def _folded_blocks(lines: Sequence[_Line]) -> List[Tuple[int, int, int]]:
    """(opening line index, base, closing line index) for folded expressions."""
    blocks: List[Tuple[int, int, int]] = []
    stack: List[Tuple[int, int]] = []
    for index, line in enumerate(lines):
        code, _comment, opened = _scan_line(line.text)
        if opened is not None:
            continue
        code = code.rstrip()
        stripped = code.strip()
        if stack and stripped.startswith(")"):
            open_index, base = stack.pop()
            blocks.append((open_index, base, index))
            continue
        if code.endswith("("):
            base = _expression_base(line.text)
            stack.append((index, base))
    return blocks


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


def _check_expression_folds(lines: Sequence[_Line], path: str) -> List[Violation]:
    """A compound expression in `(...)` must fold; a pure tag chain does not."""
    found: List[Violation] = []
    code_lines = _code_lines(lines)
    for code in code_lines:
        if _opens_hint_or_string(code.text):
            continue
        for open_col, _close_col, content in _paren_pairs(code.text):
            if _content_kind(content) is None:
                continue
            found.append(Violation(
                path, code.number, open_col + 1, "expression-fold",
                "a compound type expression in parentheses must fold"))
    for open_index, base, close_index in _folded_blocks(lines):
        inner = [code for code in code_lines
                 if open_index < code.index < close_index]
        if not inner:
            continue
        head = inner[0]
        stripped = head.stripped
        head_expected = base + 4 if stripped in ("Object", "Oneof") else base + 5
        operators = ("|", "*") if stripped in ("Object", "Oneof") else ("<-", "<<")
        if head.indent != head_expected:
            found.append(Violation(
                path, head.number, head.indent + 1, "expression-fold",
                f"the expression's head must be in column {head_expected + 1}"))
        for code in inner[1:]:
            token = _first_token(code.stripped)
            if token not in operators or code.indent != base + INDENT:
                continue
        closing = lines[close_index]
        text = _scan_line(closing.text)[0].rstrip()
        if len(text) - len(text.lstrip(" ")) != base:
            found.append(Violation(
                path, closing.number, 1, "expression-fold",
                "the closing `)` must be in the expression's base column"))
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
    if opened is not None or _opens_hint_or_string(code):
        return None
    code = code.rstrip()
    pairs = [(open_col, close_col, content) for open_col, close_col, content
             in _paren_pairs(code) if _content_kind(content) is not None]
    if not pairs:
        return None
    open_col, close_col, content = min(pairs, key=lambda pair: pair[0])
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
        if _opens_hint_or_string(text):
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


def fix_source(source: str, path: str = "<source>") -> str:
    """Rewrite the fixable statements of a viba source to the layout."""
    lines = [_Line(text, index + 1) for index, text in enumerate(source.splitlines())]
    out: List[str] = []
    cursor = 0
    for statement in _statements(lines):
        out.extend(line.text for line in lines[cursor:statement.start])
        fixed = _fix_statement(lines, statement)
        if fixed is None:
            out.extend(line.text for line in lines[statement.start:statement.end])
        else:
            out.extend(fixed)
        cursor = statement.end
    out.extend(line.text for line in lines[cursor:])
    fixed = "\n".join(out)
    trailing = "\n" if source.endswith("\n") else ""
    for _ in range(8):
        folded = _fold_expressions(fixed.splitlines())
        updated = "\n".join(folded) + trailing
        if updated == fixed:
            break
        fixed = updated
    return fixed


def fix_markdown(source: str) -> str:
    """Rewrite the fixable sections of a `.viba.md` document."""
    try:
        sections = markdown.lines_of(source)
    except SyntaxError:
        return source
    document = source.splitlines()
    for _name, numbered in reversed(sections):
        if not numbered:
            continue
        first, last = numbered[0][1], numbered[-1][1]
        if last - first + 1 != len(numbered):
            continue
        text = "\n".join(line for line, _at in numbered) + "\n"
        fixed = fix_source(text).splitlines()
        document[first - 1:last] = fixed
    return "\n".join(document) + ("\n" if source.endswith("\n") else "")


def fix_file(path: Path) -> bool:
    """Rewrite one file; answer whether it changed."""
    text = path.read_text()
    fixed = fix_markdown(text) if path.name.endswith(".viba.md") else fix_source(text)
    if fixed != text:
        path.write_text(fixed)
        return True
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m viba.lint",
        description="Report type statements whose layout is not the two-space block.")
    parser.add_argument("paths", nargs="*", default=["."],
                        help="files or directories to lint (default: .)")
    parser.add_argument("--fix", action="store_true",
                        help="rewrite the fixable layout, then report what is left")
    args = parser.parse_args(argv)
    if args.fix:
        for path in _files(args.paths):
            if fix_file(path):
                print(f"fixed {path}")
    violations = sorted(lint_paths(args.paths),
                        key=lambda v: (v.path, v.line, v.column, v.rule))
    for violation in violations:
        print(f"{violation.path}:{violation.line}:{violation.column}: "
              f"{violation.rule}: {violation.message}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())

"""A directory written as one Markdown document.

A directory of files can be given as a single Markdown file — `<dir>.viba.md`,
next to `<dir>/`: each heading names one file of the directory, whatever its
level. The file's code is every code block under it that is marked `viba` (the
label may name the file, as in ```viba 100.viba), in the order they stand,
glued into one source; the prose between them and any other code block become
comments (`# ...`), so the section reads as the file it names and parses as one.
A section with no `viba` block names a file of comments or nothing at all, and
the prose above the first heading names no file at all:

    ## `point.viba`

    ```viba
    Point = Object * $x int * $y int
    ```

The reading takes no notice of what the files are for: it answers the (name,
text) pairs a directory listing and the files themselves would. `viba.interpret`
resolves names this way (`<dir>.viba.md` is read as `<dir>/`, and wins over it
when both are there), and anything else with a directory in its head can do the
same with `files_of`.
"""

import re
from typing import List, Tuple

__all__ = ["files_of", "module_files"]

# A section: `#` to `######` then the file's name; `##` alone names nothing.
_HEADING = re.compile(r"^(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")

# A fenced code block: up to three spaces of indent, three or more backticks or
# tildes, and the rest of the line as its info string.
_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$")

# A name written as a code span: its backticks are no part of the name.
_CODE_SPAN = re.compile(r"^(`+)(.*?)\1$")


def files_of(source: str) -> List[Tuple[str, str]]:
    """The files a Markdown document names: (name, text), in the order given.

    The name is the heading's text as written — `point.viba`,
    `__generic__.viba` — so the document reads as the directory it stands for.
    The heading's level is not looked at: `#`, `##`, `###` and `####` all name a
    file, and there is no separate title. Two sections naming one file are an
    error: a directory holds a name once. A fenced code block that is never
    closed is an error too: the file's text would be cut at the end of the
    document.

    Raises SyntaxError, the way viba's own parser does, with the line it
    happened on.
    """
    return [(name, _text([part for part, _at in lines]))
            for name, lines in _read(source)]


def lines_of(source: str) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """The files a document names, line by line: (name, [(text, document line)]).

    The same reading `files_of` gives, with where each line came from: a tool
    that answers a document of its own says which line of the source a line of
    its answer stands for.
    """
    return [(name, list(lines)) for name, lines in _read(source)]


def _read(source: str) -> List[Tuple[str, List[Tuple[str, int]]]]:
    """The files a document names, each line with the document line it came from."""
    files: List[Tuple[str, List[Tuple[str, int]]]] = []
    names = set()
    current: List[Tuple[str, int]] = None   # the file being read: text and line
    fence = None                        # (indent, marker, lines, is a viba block)
    for number, line in enumerate(source.splitlines(), 1):
        if fence is None:
            heading = _HEADING.match(line)
            if heading is not None:
                name = _heading_name(heading.group(2) or "")
                if not name:
                    raise SyntaxError(
                        f"line {number}: a section needs a file name")
                if name in names:
                    raise SyntaxError(
                        f"line {number}: two sections name the file {name!r}")
                names.add(name)
                current = []
                files.append((name, current))
                continue
        if current is None:
            continue
        if fence is not None:
            indent, marker, lines, viba = fence
            closed = _FENCE.match(line)
            if (closed is not None and closed.group(2)[0] == marker[0]
                    and closed.group(2).startswith(marker)
                    and not closed.group(3)):
                # The block's lines join the file: a viba block is the code,
                # any other block is documentation and becomes comments.
                current.extend(lines if viba else
                               [(_comment(part), at) for part, at in lines])
                fence = None
                continue
            lines.append((_unindent(line, indent), number))
            continue
        opened = _FENCE.match(line)
        if opened is None:
            current.append((_comment(line), number))
            continue
        info = opened.group(3).split()
        fence = (len(opened.group(1)), opened.group(2), [],
                 bool(info) and info[0] == "viba")
    if fence is not None:
        raise SyntaxError(
            "a fenced code block is not closed at the end of the document")
    return [(name, list(_trimmed(lines))) for name, lines in files]


def module_files(module_name: str, source: str):
    """(file name, module name, text) for the files a document holds.

    A generic's directory names its files under the generic: the marker is
    `pkg.g.__generic__`, and `pkg/g/100.viba` is `pkg.g.100`. A document read as
    that directory is named the same way, so a layer that takes files by module
    name (a descriptor pool) takes the document and the directory alike.
    """
    out = []
    for file_name, text in files_of(source):
        if not file_name.endswith(".viba"):
            continue
        out.append((file_name,
                    f"{module_name}.{file_name[: -len('.viba')]}",
                    text))
    return out


def _heading_name(text: str) -> str:
    """The file name a heading spells: a code span's content, or the text as written.

    A name may be written as code — `## `__generic__.viba`` — because Markdown
    reads `__x__` as emphasis: the rendered heading has to show the name, and the
    backticks keep it whole.
    """
    span = _CODE_SPAN.match(text)
    if span is not None:
        return span.group(2).strip()
    return text


def _unindent(line: str, indent: int) -> str:
    """One line of a fenced block, the fence's own indent taken off."""
    return line[indent:] if line[:indent].strip() == "" else line


def _comment(line: str) -> str:
    """A line that is not the file's code, as a comment in the file."""
    return f"# {line}" if line.strip() else ""


def _trimmed(lines):
    """The lines without the blank ones at either end: (text, line) pairs."""
    first, last = 0, len(lines)
    while first < last and not lines[first][0].strip():
        first += 1
    while last > first and not lines[last - 1][0].strip():
        last -= 1
    return lines[first:last]


def _text(lines: List[str]) -> str:
    """A file's text: its lines, and a newline after the last one."""
    return "".join(f"{line}\n" for line in lines)

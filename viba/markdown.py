"""A directory written as one Markdown document.

A directory of files can be given as a single Markdown file: each section names
one file of the directory, and the first fenced code block under it is that
file's text. A section with no code block names an empty file, and the
document's own title — a level-one heading, and the prose under it — names no
file at all:

    # is_base_type

    ## __generic__.viba

    ## 100.viba

    ```viba
    pattern bool | int | str

    value = true
    ```

The reading takes no notice of what the files are for: it answers the (name,
text) pairs a directory listing and the files themselves would. `viba.interpret`
reads a generic this way when its name is a Markdown file rather than a
directory (`viba-pattern.md`), and anything else with a directory in its head can
do the same with `files_of`.
"""

import re
from typing import List, Tuple

__all__ = ["files_of", "module_files"]

# A section: `## name`, up to six levels. A level-one heading is the title, and
# `##` alone is a section that names nothing.
_HEADING = re.compile(r"^(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")

# A fenced code block: up to three spaces of indent, three or more backticks or
# tildes, and the rest of the line as its info string.
_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})[ \t]*(.*?)[ \t]*$")

# How far a fence's own indent is taken off the lines inside it.
_INDENT = re.compile(r"^ {0,3}")


def files_of(source: str) -> List[Tuple[str, str]]:
    """The files a Markdown document names: (name, text), in the order given.

    The name is the heading's text as written — `__generic__.viba`, `100.viba` —
    so the document reads as the directory it stands for. A level-one heading is
    the title, not a file. Two sections naming one file are an error: a
    directory holds a name once. A fenced code block that is never closed is an
    error too: the file's text would be cut at the end of the document.

    Raises SyntaxError, the way viba's own parser does, with the line it
    happened on.
    """
    files: List[Tuple[str, List[str]]] = []
    names = set()
    current: List[str] = None          # the lines of the file being read
    taken = False                       # its first code block is already in
    fence = None                        # (indent, marker, lines) of the open block
    for number, line in enumerate(source.splitlines(), 1):
        if fence is None:
            heading = _HEADING.match(line)
            if heading is not None:
                if len(heading.group(1)) == 1:
                    current = None
                    continue
                name = heading.group(2) or ""
                if not name:
                    raise SyntaxError(
                        f"line {number}: a section needs a file name")
                if name in names:
                    raise SyntaxError(
                        f"line {number}: two sections name the file {name!r}")
                names.add(name)
                current = []
                taken = False
                files.append((name, current))
                continue
        if current is None:
            continue
        if fence is not None:
            indent, marker, lines = fence
            closed = _FENCE.match(line)
            if (closed is not None and closed.group(2)[0] == marker[0]
                    and closed.group(2).startswith(marker)
                    and not closed.group(3)):
                current.extend(lines)
                fence = None
                continue
            lines.append(_unindent(line, indent))
            continue
        if taken:
            continue
        opened = _FENCE.match(line)
        if opened is None:
            continue
        fence = (len(opened.group(1)), opened.group(2), [])
        taken = True
    if fence is not None:
        raise SyntaxError(
            "a fenced code block is not closed at the end of the document")
    return [(name, _text(lines)) for name, lines in files]


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


def _unindent(line: str, indent: int) -> str:
    """One line of a fenced block, the fence's own indent taken off."""
    return line[indent:] if line[:indent].strip() == "" else line


def _text(lines: List[str]) -> str:
    """A file's text: its lines, and a newline after the last one."""
    return "".join(f"{line}\n" for line in lines)

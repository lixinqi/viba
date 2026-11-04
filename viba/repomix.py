"""viba.repomix — a directory of viba code, packed into one Markdown document.

    from viba.repomix import viba_repomix
    viba_repomix("kb")          # the directory's code as one document's text

The text is the one `viba.markdown` reads: a section per file, the section's
name is the file's path in the directory, and the first fenced code block is the
file's text ([`viba-markdown-as-directory.md`](../viba-markdown-as-directory.md)).
What counts as code:

- a `.viba` file, unless the directory holding it is written as `<dir>.viba.md`:
  the document wins over the directory, so the file is not read
  (viba-markdown-as-directory.md);
- a `.viba.md` document: its sections are packed under the paths its directory
  holds (`g.viba.md`'s `100.viba` is packed as `g/100.viba`).

Everything else is no code and is left out: prose documents, notes, binaries,
the `.viba.md` documents themselves.

A `.viba` file's comments become the section's documentation, as far as that
loses nothing: the comment lines above the file's first line of code and below
its last one are lifted out of the code block and written as prose around it —
they only ever said something about the file, and the code means the same
without them. A comment *between* two lines of code stays where it is: it
belongs to the line it sits with, and moving it would take it away from that
line.

A directory of files and no subdirectories packs to exactly the document that
stands for it, so `files_of` reads the pack back as the same files. A tree packs
to a reading copy: the loader's documents are one directory deep, and a path in
a section name lets one document hold a whole tree.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

from viba import markdown

__all__ = ["viba_repomix"]


def viba_repomix(viba_code_dir: str) -> str:
    """A directory of viba code as one Markdown document's content.

    The document is the one the loader reads for a directory: one section per
    file, its name the file's path in `viba_code_dir`, its first code block the
    file's code, and the file's leading and trailing comments as prose around
    it. The design's type has no error branch (`viba/repomix.viba`): a path that
    is no directory holds no code, so the pack is empty.
    """
    sections = _files_of(viba_code_dir)
    parts = []
    for path in sorted(sections):
        before, code, after = sections[path]
        part = f"## `{path}`\n"
        if before:
            part += f"\n{before}\n"
        if code:
            part += "\n```viba\n" + code + "```\n"
        if after:
            part += f"\n{after}\n"
        parts.append(part)
    return "\n".join(parts)


def _files_of(viba_code_dir: str) -> Dict[str, Tuple[str, str, str]]:
    """The code a directory holds: relative path -> (before, code, after)."""
    root = Path(viba_code_dir)
    if not root.is_dir():
        return {}
    found: Dict[str, Tuple[str, str, str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.endswith(".viba.md"):
            _pack_document(found, relative, path)
            continue
        if not relative.endswith(".viba"):
            continue
        if _written_as_document(path):
            continue                    # the document is the directory: it wins
        found[relative] = _lifted(_text(path))
    return found


def _pack_document(found: Dict[str, Tuple[str, str, str]], relative: str,
                   path: Path) -> None:
    """A directory written as one document: its sections, under the paths its
    directory would hold them."""
    directory = relative[: -len(".viba.md")]
    try:
        sections = markdown.files_of(_text(path))
    except SyntaxError:
        return                          # a document that does not parse holds no code
    for file_name, text in sections:
        code = text if text.endswith("\n") or not text else f"{text}\n"
        found[f"{directory}/{file_name}"] = ("", code, "")


def _written_as_document(path: Path) -> bool:
    """True when this file's directory is written as one document."""
    return Path(f"{path.parent}.viba.md").is_file()


def _text(path: Path) -> str:
    """One file's text, with a newline after the last line."""
    text = path.read_text(encoding="utf-8")
    return text if text.endswith("\n") else f"{text}\n"


def _lifted(text: str) -> Tuple[str, str, str]:
    """(before, code, after): a file's code, and the comments around it lifted.

    Only the comments above the first line of code and below the last one are
    lifted; a comment between two lines of code stays with them.
    """
    lines = text.splitlines()

    def code_line(line: str) -> bool:
        return bool(line.strip()) and not line.lstrip().startswith("#")

    first = next((index for index, line in enumerate(lines) if code_line(line)),
                 None)
    if first is None:
        return _prose(lines), "", ""
    last = max(index for index, line in enumerate(lines) if code_line(line))
    code = "".join(f"{line}\n" for line in lines[first:last + 1])
    return _prose(lines[:first]), code, _prose(lines[last + 1:])


def _prose(lines: List[str]) -> str:
    """Comment lines as prose: the `#` gone, the empty ones kept as breaks."""
    out: List[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        text = line.lstrip()[1:]
        if text.startswith(" "):
            text = text[1:]
        text = text.rstrip()
        if text.startswith("#"):
            text = "\\" + text      # a heading would name a file of the document
        out.append(text)
    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)

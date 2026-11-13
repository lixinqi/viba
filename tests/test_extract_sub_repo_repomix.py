"""the two entries are one: pack a repo, then cut the document.

    ExtractSubVibaMarkdown * VibaRepoMix == ExtractSubRepoIntoVibaMarkdown

For a repo that holds every module it imports, with `viba_path` and
`extract_source_viba_path` the repo itself, packing it with `viba_repomix` and
cutting that document answers the same document as cutting the repo, and the
document's line map, read through the pack's lines, is the repo entry's map.

The pack is what makes the two sides see the same files: a section per file,
its code the file's code, and the comments around the code as prose that reads
back as the same lines.

    python3 tests/test_extract_sub_repo_repomix.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import markdown
from viba.extract_sub_repo import (extract_sub_repo_into_viba_markdown,
                                   extract_sub_viba_markdown)
from viba.repomix import viba_repomix
from viba.type import Ok

DATA = Path(__file__).resolve().parent / "data"
DOCUMENTS = DATA / "pattern_custom_types"

PASS = 0
FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}\n  expected: {want!r}\n  got:      {got!r}")


def packed_places(pack: str) -> dict:
    """(module, line) for every line of the pack: its section's module, its index."""
    places = {}
    for name, lines in markdown.lines_of(pack):
        module = name[: -len(".viba")].replace("/", ".")
        for index, (_text, at) in enumerate(lines, 1):
            places[at] = (module, index)
    return places


def equation(repo: Path, names, label: str) -> None:
    """Cut the repo directly, and cut the pack: the answers are one."""
    pack = viba_repomix(str(repo))
    left = extract_sub_viba_markdown(pack, names)
    right = extract_sub_repo_into_viba_markdown(
        str(repo), [str(repo)], [str(repo)], names)
    check(f"{label}: both answer a document",
          isinstance(left, Ok) and isinstance(right, Ok), True)
    if not (isinstance(left, Ok) and isinstance(right, Ok)):
        return
    document, to_source = left.ok_value
    same, to_module = right.ok_value
    check(f"{label}: the same document", document, same)
    places = packed_places(pack)
    check(f"{label}: the same lines",
          {at: places.get(line) for at, line in to_source.items()}, to_module)
    check(f"{label}: every code line is in the map",
          sorted(to_source),
          sorted(at for _name, lines in markdown.lines_of(document)
                 for _text, at in lines))


def run() -> None:
    cases = [
        (DATA / "extract_sub_repo/generic_chosen/kb", ["use.Gauge"],
         "a generic"),
        (DATA / "extract_sub_repo/member/kb", ["use.Wrapped"], "a member take"),
        (DATA / "extract_sub_repo/deep/kb", ["top.Top"], "a tree"),
        (DATA / "extract_sub_repo/call/kb", ["step.Call"], "a call"),
        (DATA / "extract_sub_repo/units/kb", ["doc.Doc"], "a unit"),
        (DATA / "extract_sub_repo/comments/kb", ["use.X"],
         "comments around the code"),
        (DATA / "repomix/kb", ["report.Point"], "a file"),
        (DATA / "repomix/kb", ["box.a.A"], "a directory's document"),
        (DOCUMENTS, ["ask_red.__impl__"], "a repo of documents"),
    ]
    for repo, names, label in cases:
        equation(repo, names, label)


def main() -> int:
    run()
    print(f"extract_sub_repo_repomix: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

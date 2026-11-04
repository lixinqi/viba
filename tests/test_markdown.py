"""A directory written as one Markdown document: the files it names.

    python3 tests/test_markdown.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.markdown import files_of, module_files

PASS = 0
FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}\n  expected: {want!r}\n  got:      {got!r}")


def refused(name: str, source: str) -> None:
    global PASS, FAIL
    try:
        files_of(source)
    except SyntaxError:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}: it was accepted")


DOCUMENT = """\
# report

报表用到的两个类型。文档的标题不是文件，这一整段也不是。

## point.viba

一个二维点。

```viba
Point = Object * $x int * $y int
```

## note.viba

```
Note = Object * $text str
```

## notes.txt

没有代码块的一章是空文件。
"""


def run() -> None:
    files = files_of(DOCUMENT)
    check("one file per section, in the order given",
          [name for name, _text in files],
          ["point.viba", "note.viba", "notes.txt"])
    check("the title names no file", "# report" in [name for name, _ in files],
          False)
    check("the first code block is the file's text",
          files[0][1], "Point = Object * $x int * $y int\n")
    check("the info string is no part of the text",
          files[1][1], "Note = Object * $text str\n")
    check("a section with no code block is an empty file", files[2][1], "")

    two_blocks = """\
## point.viba

```viba
X = int
```

```viba
Y = str
```
"""
    check("a second code block is no file either",
          files_of(two_blocks), [("point.viba", "X = int\n")])

    tilde = """\
## point.viba

~~~
X = int
~~~
"""
    check("a tilde fence counts too", files_of(tilde),
          [("point.viba", "X = int\n")])

    indented = """\
## point.viba

  ```viba
  X = int
    Y = str
  ```
"""
    check("the fence's own indent comes off",
          files_of(indented), [("point.viba", "X = int\n  Y = str\n")])

    labelled = """\
## point.viba

```text
X = int
```
"""
    check("the label may be anything, or nothing",
          files_of(labelled), [("point.viba", "X = int\n")])

    no_title = """\
## point.viba

```viba
X = int
```
"""
    check("a document may have no title", files_of(no_title),
          [("point.viba", "X = int\n")])

    named = module_files("report", DOCUMENT)
    check("a file is named under the module the document stands for",
          [module for _file, module, _text in named],
          ["report.point", "report.note"])
    check("a file that is no .viba is left out",
          [file_name for file_name, _name, _text in named],
          ["point.viba", "note.viba"])

    refused("two sections naming one file", """\
## point.viba

## point.viba
""")
    refused("a section with no name", """\
##
""")
    refused("a code block that never closes", """\
## point.viba

```viba
X = int
""")


if __name__ == "__main__":
    run()
    print(f"markdown: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)

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
报表用到的两个类型。这一整段不是任何文件。

# report

## point.viba

一个二维点。

```viba
Point = Object * $x int * $y int
```

### note.viba

```
Note = Object * $text str
```

#### notes.txt

没有代码块的一章是空文件。
"""


def run() -> None:
    files = files_of(DOCUMENT)
    check("one file per heading, in the order given",
          [name for name, _text in files],
          ["report", "point.viba", "note.viba", "notes.txt"])
    check("the level of a heading is not looked at", files[0], ("report", ""))
    check("the first code block is the file's text",
          files[1][1], "Point = Object * $x int * $y int\n")
    check("the info string is no part of the text",
          files[2][1], "Note = Object * $text str\n")
    check("a section with no code block is an empty file", files[3][1], "")
    check("prose outside a heading names no file",
          files_of("just a paragraph\n\nand another\n"), [])

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

    deep = """\
###### deep.viba

```viba
X = int
```
"""
    as_code = """\
## `__generic__.viba`

## point.viba

```viba
X = int
```
"""
    check("a name written as code is the name inside the backticks",
          files_of(as_code), [("__generic__.viba", ""),
                             ("point.viba", "X = int\n")])
    check("a name written as it is needs no backticks",
          files_of("## __generic__.viba\n"), [("__generic__.viba", "")])

    check("six hashes name a file too", files_of(deep),
          [("deep.viba", "X = int\n")])

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

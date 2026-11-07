# repomix：把一个目录的 viba 代码收成一份 Markdown

一个目录（一个 repo）里的 viba 代码要一次读完：把每个文件收进一份 Markdown——一章一个文件，
章名是它在目录里的路径，章里那个代码块是它的代码；文件里写在代码前后的注释变成这一章的说明。

## 1. 类型

```viba
VibaRepoMix =
    (str * { markdown file content })
  <- $viba_code_dir (str * { directory containing viba code })
```

答的是那份文档的内容，没有错误这一支：给一个不存在（或者不是目录）的路径，目录里没有代码，
收出来就是空串。

## 2. 收成什么

- **一章一个文件**：章名是文件相对这个目录的路径，包在反引号里——``## `g/100.viba` ``；平铺目录里就是文件名。
  包成代码是为了名字不在 Markdown 里被吃掉（`__generic__.viba` 的 `__` 会被当成强调）。
- **代码块就是文件的内容**：围栏要标 `viba`（标后面可以带文件名），读的时候只认这种块；
  章里别的文字读回来是这个文件里的注释。
- **注释变说明**：文件第一行代码之前、最后一行代码之后的注释提出来，写成这一章代码块前后的
  说明。夹在两行代码之间的注释留在原处——它挨着那一行，挪走就离开了那一行。
- **注释照原文**：不翻译、不改写；注释文字以 `#` 开头时加一个反斜杠转义，不然它会在文档里变成
  一章（标题就是文件名）。
- **只收代码**：说明文档、`notes.txt` 这类东西、以及 `.viba.md` 文档本身都不收。

## 3. 什么算代码

- `.viba` 文件；但如果它所在的目录写成了 `<目录>.viba.md`，文档说了算，目录里的文件不收
  （[`viba-markdown-as-directory.md`](viba-markdown-as-directory.md)）。
- `.viba.md` 文档：它的章按它代表的目录收——`g.viba.md` 里的 `100.viba` 收成 `g/100.viba`。
- 其它一律不收。

## 4. 读回来

读它的是 [`viba/markdown.py`](../markdown.py) 的 `files_of`：一份文档读出（名字，内容）的一串，
章名原样当名字。

- **平铺目录**（只有文件、没有子目录）：收出来的就是它的 `<目录>.viba.md`——`files_of` 读回来是
  同一串（名字，内容）。
- **有子目录**：章名里带路径，比加载器的一层文档多；收出来的是给人读的副本，不是一份能被加载器
  当目录读的文档（加载器的文档一章一个文件，树由各层的文档表达）。

## 5. 例子

`tests/data/repomix/kb/` 收出来是这样：

````markdown
## `box/a.viba`

```viba
A = int
```

## `g/100.viba`

```viba
pattern bool

value = true
```

a trailing note

## `g/200.viba`

```viba
pattern A

# this one sits with the line, so it stays
value = false
```

## `g/__generic__.viba`

__generic__.viba

## `report.viba`

The point and the note.

```viba
Point = Object * $x int * $y int
```
````

`box/a.viba` 来自目录 `box/` 的文档（`box.viba.md`），`box/` 里那份同名文件被文档替掉，没有收；
`g/100.viba` 的尾注释成了它下面那段说明；`g/200.viba` 的注释夹在代码中间，留在原处；
`g/__generic__.viba` 只有一行注释，收出来是一章带说明、没有代码——文件读回来是空的。

## 6. 测试

[`tests/test_repomix.py`](../../tests/test_repomix.py)：章名与内容、前导与尾部注释变说明、中间注释留下、
文档优先于目录、非代码不收、平铺目录读回来是同一串、不存在的路径收成空串。

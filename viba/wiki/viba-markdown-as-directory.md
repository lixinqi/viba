# Markdown 当作目录

## 1. 场景

一个目录放着一组 `.viba` 文件时，可以把它写成一份 Markdown 文档：每一章是目录下的一个文件，
章名就是文件名，章里标了 `viba` 的代码块按顺序拼起来就是那个文件的代码，其余文字变成这个文件里的注释。
文件的内容不变，名字不变，只是把一组
散开的文件收进一份能读的文档——说明和代码挨在一起，不用在文件之间跳。

**任意目录**都行：仓库里的一个普通目录、一个泛型目录、更深的子目录。文档写在哪里都一样：和目录
并列，名字是目录名加 `.viba.md`（`report.viba.md` 是 `report/` 的另一种写法）。泛型目录
（[`viba-pattern.md`](../core/viba-pattern.md)）里的文件叫 `__generic__.viba` 和编号文件，写法没有区别。

## 2. 一个普通的例子

目录 `report/`，两个普通模块：

- `point.viba`：`Point = Object * $x int * $y int`
- `note.viba`：`Note = Object * $text str`

写成一份 `report.viba.md`：

````markdown
报表用到的两个类型。

## `point.viba`

一个二维点：两个整数成员。

```
Point = Object * $x int * $y int
```

## `note.viba`

一段文字，只有一个成员。

```
Note = Object * $text str
```
````

读出来是两个文件，和目录里那两份一模一样：

```text
("point.viba", "Point = Object * $x int * $y int\n")
("note.viba", "Note = Object * $text str\n")
```

代码块前后的说明文字不进文件：章节正文是给人读的，文件的内容只有那段代码。原来写在文件里的
注释也可以提到代码块上面，写成这一章的说明——文件的内容就只剩代码。

## 3. 文档优先

`report.viba.md` 和 `report/` 同时在时，**文档说了算**：`report/` 里的文件不再读，`report` 下
一个名字解析到的每个文件都来自文档的章；文档里没有的，就当没有，不退回目录。同一份设计只会看到
一种写法，不会一半来自文档、一半来自目录。

子目录仍然是目录：它们可以有自己的文档，写在它们里面（`report/notes.viba.md` ↔
`report/notes/`）。

查找一个名字 `a.b.c` 时按这个顺序问，先命中的算：

1. 父目录的文档 `a/b.viba.md` 里的 `c.viba` 一章；
2. 文件 `a/b/c.viba`；
3. 文档 `a/b/c.viba.md`（目录 `a/b/c/` 的另一种写法）；
4. 目录 `a/b/c/__generic__.viba`；
5. 带点的单个文件 `a.b.c.viba`。

第 1 条命中时就不再问第 2 条：父目录的文档赢过父目录。同理，第 3 条命中时不再问第 4 条。

## 4. 规则

- **名字**：文档是 `<目录>.viba.md`，与目录同名、并列。名字里的 `.viba` 说的是“里面装的是 viba
  文件”，`.md` 说的是形式——所以它不会和讲这个目录的说明文档（`foo.md`）混起来。
- **标题**：任何一级标题（`# …` 到 `###### …`）都是文件，层级不看——`#`、`##`、`###` 一样；
  标题文字原样当文件名，连后缀一起。名字一律包成代码——``## `point.viba` ``：反引号不算名字的一部分，
  而 `__generic__.viba` 这种名字不包的话，Markdown 会把 `__` 当成强调，标题就显示成加粗的
  `generic.viba`。文档没有单独的标题这一说：第一个标题也是文件，标题上方的段落不命名任何文件。
- **内容**：每一章里标了 `viba` 的代码块，按顺序拼起来是这个文件的代码；标后面可以带文件名
  （`` ```viba 100.viba ``）。它们之间的文字、以及别的代码块（比如 `` ```python ``），都变成这个
  文件里的注释（`# …`）——一节读下来就是一个文件，而且照样能编译。围栏自己的缩进会从内容里去掉；
  这一章没有 `viba` 代码块，这个文件就只有注释（或者什么都没有）。
- **顺序**：章在文档里的先后就是文件在目录里的先后；哪一支先被用到由文件名自己定（编号、名字），
  与章的顺序无关。
- **报错**：两个章同名、章名是空的、围栏没有闭合，都当语法错误报，带行号。

## 5. 读法在哪用

[`viba/markdown.py`](../markdown.py) 提供两件事：

- `files_of(source)`：一份文档读出（名字，内容）的一串，是这一页说的全部规则；
- `module_files(module_name, source)`：同一串，再把名字按目录的命名挂到模块名下——
  `pkg/report.viba.md` 里的 `point.viba` 是 `pkg.report.point`。

解析一个名字时，按第 3 节那个顺序：`<目录>.viba.md` 在，就在它的章里找。从文件建描述符池的入口
同样按这个顺序——`is_complete` 扫 `viba_paths` 时先扫 `*.viba.md`，目录里被文档替掉的文件跳过；
判定与转换工具建池时也把文档的章按模块名收进去。

于是凡是“一个目录、里面一组文件”的地方，都能用一份文档顶替。

## 6. 测试

- [`tests/test_markdown.py`](../../tests/test_markdown.py)：读法本身——章与文件、代码块的拼接、注释、空文件、
  围栏缩进、同名与未闭合的报错，以及名字写成代码的情况。
- [`tests/test_markdown_directory.py`](../../tests/test_markdown_directory.py)：一个普通目录的两种写法同时
  存在——文档说了算（`Point` 是文档里的两个整数，不是目录里的两个字符串）、文档里没有的名字找不到、
  `is_complete` 与两个 JSON 工具都认它。
- [`tests/test_pattern_markdown.py`](../../tests/test_pattern_markdown.py)：拿泛型做的对比测试。
  `tests/data/pattern/demo/` 那十二个短泛型各写了一份
  `tests/data/pattern_markdown/demo/<名字>.viba.md`，同一批设计跑两遍——目录一次、文档一次——
  答案必须一样；`is_complete` 两种写法都要判成完整。
- 判定与转换的用例里有 [`elem.viba.md`](../../tests/data/check_convertable_to_json_schema/kb/elem.viba.md)：
  一个用文档写成的泛型，[`uses_elem.viba`](../../tests/data/check_convertable_to_json_schema/kb/uses_elem.viba)
  应用它，判定答 `Ok`，schema 与目录写法逐字相同。

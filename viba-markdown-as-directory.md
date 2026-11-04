# Markdown 当作目录

## 1. 场景

一个目录放着一组 `.viba` 文件时，可以把它写成一份 Markdown 文档：每一章是目录下的一个文件，
章名就是文件名，章里第一个围栏代码块就是那个文件的内容。文件的内容不变，名字不变，只是把一组
散开的文件收进一份能读的文档——说明和代码挨在一起，不用在文件之间跳。

泛型目录（[`viba-pattern.md`](viba-pattern.md)）就是这个写法的一个用法：那里一个目录装的是
一个标记加若干模式文件。

## 2. 一个普通的例子

目录 `report/`，两个普通模块：

- `point.viba`：`Point = Object * $x int * $y int`
- `note.viba`：`Note = Object * $text str`

写成一份 `report.md`：

````markdown
报表用到的两个类型。

## point.viba

一个二维点：两个整数成员。

```
Point = Object * $x int * $y int
```

## note.viba

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

## 3. 规则

- **位置与名字**：文档写在 `<名字>.md`，与目录同名、同位置。查找一个名字时依次问：
  `<名字>.viba` 文件、`<名字>/__generic__.viba`（目录）、`<名字>.md`（本文档）、带点的单个
  `<名字>.viba`。
- **标题**：任何一级标题（`# …` 到 `###### …`）都是文件，层级不看——`#`、`##`、`###` 一样；
  标题文字原样当文件名，连后缀一起（`point.viba`、`100.viba`）。文档没有单独的标题这一说：第一个
  标题也是文件，标题上方的段落不命名任何文件。名字里有 `__` 这种会被 Markdown 当成强调的字符时
  （标题会显示成 `generic.viba`），把名字包成代码——``## `__generic__.viba` `` 这样，反引号不算
  名字的一部分。
- **内容**：每一章的第一个围栏代码块是这个文件的内容；围栏前面标不标 `viba` 都一样，标了是给
  人看的，读的时候不看；围栏自己的缩进会从内容里去掉；这一章没有代码块就是空文件；同一章里第二
  个及以后的代码块不进这个文件。
- **顺序**：章在文档里的先后就是文件在目录里的先后；哪一支先被用到由文件名自己定（编号、名字），
  与章的顺序无关。
- **报错**：两个章同名、章名是空的、围栏没有闭合，都当语法错误报，带行号。

## 4. 读法在哪用

[`viba/markdown.py`](viba/markdown.py) 提供两件事：

- `files_of(source)`：一份文档读出（名字，内容）的一串，是这一页说的全部规则；
- `module_files(module_name, source)`：同一串，再把名字按目录的命名挂到模块名下——`pkg/report/`
  里的 `point.viba` 是 `pkg.report.point`。

读它的有两处：

- 解释器把名字解析到目录时（[`viba/interpret.py`](viba/interpret.py) 的 `_Runner`）：`<名字>.md`
  在，就按目录读它；
- 从文件建描述符池的入口：`is_complete` 扫 `viba_paths` 时也扫 `*.md`，判定与转换工具建池时也把
  文档的章按模块名收进去。

于是凡是“一个目录、里面一组文件”的地方，都能用一份文档顶替。

## 5. 测试

- [`tests/test_markdown.py`](tests/test_markdown.py)：读法本身——章与文件、第一个代码块、空文件、
  围栏缩进、同名与未闭合的报错。
- [`tests/test_pattern_markdown.py`](tests/test_pattern_markdown.py)：拿泛型做的对比测试。
  `tests/data/pattern/demo/` 那十二个短泛型各写了一份 `tests/data/pattern_markdown/demo/<名字>.md`，
  同一批设计跑两遍——目录一次、文档一次——答案必须一样；`is_complete` 两种写法都要判成完整。
- 判定与转换的用例里有 [`elem.md`](tests/data/check_convertable_to_json_schema/kb/elem.md)：一个用
  文档写成的泛型，[`uses_elem.viba`](tests/data/check_convertable_to_json_schema/kb/uses_elem.viba)
  应用它，判定答 `Ok`，schema 与目录写法逐字相同。

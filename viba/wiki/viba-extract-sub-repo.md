# extract_sub_repo：从一个 repo 抽出子文档

一个 repo 越写越大，而一个任务只要其中一部分。这里有两件事：把 repo 的子集写成**一份文档**
（`.viba.md`），以及从一份文档里再抽它的子集。两次都给出**行号来源**：文档里每个代码行来自哪个模块的
哪一行、或源文档的第几行——后读的人据此回到出处。

## 1. 两个入口

```viba
ExtractSubRepoIntoVibaMarkdown =
    Result[VibaMarkdown, VibaMdToRepoSourceMap]
  <- $repo Repo
  <- $viba_path list[str]
  <- $extract_source_viba_path list[str]
  <- $type_names list[TypeName]
  <- $never_eqv (list[str] | nil)
  <- $nil_eqv (list[str] | nil)

VibaMarkdown =
    str * { 符合 .viba.md 文件内容格式，所有类型能自描述，不能自描述的部分全是叶子，不用继续展开 }

VibaMdToRepoSourceMap =
    dict[
      $lineno_in_viba_md int,
      ($importable_dot_splited_module_path str, $lineno_in_module int)]

ExtractSubVibaMarkdown =
    Result[VibaMarkdown, VibaMdToVibaMdSourceMap]
  <- $source_viba_md VibaMarkdown
  <- $type_names list[TypeName]
  <- $never_eqv (list[str] | nil)
  <- $nil_eqv (list[str] | nil)

VibaMdToVibaMdSourceMap =
    dict[$lineno_in_dst_viba_md int, $lineno_in_src_viba_md int]

Repo =
    str * Assert[{ this is a file path }]

TypeName =
    str * { a module and a definition, like `pkg.mod.Name` }
```

`$never_eqv`、`$nil_eqv` 是单位元的名字，读源码时不算定义；给 `nil` 就用语言默认那份
（[`viba/core/data/viba-reflect.md`](../core/data/viba-reflect.md)）。

## 2. 抽什么

- 类型名是 `xxx.yyy.zzz`：模块 `xxx.yyy` 按 import 的找法在 `viba_path` 上找，`zzz` 是模块里的一个定义；
- 从这些名字出发走一遍：文件在 `$repo` 里、又在 `$extract_source_viba_path` 下的定义**抽出来**，连同
  它走到的定义；其余定义是**叶子原语**（抽出来的部分走到它为止，不再展开——`VibaMarkdown` 的说明里
  「不能自描述的部分全是叶子」就是这个意思）：它们的定义不进文档，引用它们的文件保留 `import`；
- 泛型的文件整份留下：`__generic__.viba` 标记与决断选中的那个 pattern 文件（决断做不出时，所有能接
  那么多实参的文件）；
- **`$repo` 里的 `.viba.md` 文档一样抽**：文档的节就是文件，其中的定义被走到时照样抽出来——文档还是
  目录，只是写在一块儿（[`viba-markdown-as-directory.md`](viba-markdown-as-directory.md)）。

## 3. 抽成什么

一份文档：一节一个文件，节名是它在 repo 里的路径（`report.viba`、`G/100.viba`），正文是**源文件自己
的那些行**——用到的 `import` 与留下的定义，原样搬，注释与空行也在。所以：

- 文档读回来就是这个子 repo（`files_of`）；
- 拿它回答「这一行从哪来」是精确的：`VibaMdToRepoSourceMap` 把文档的每个代码行对到（模块的点分名，
  模块里的行号）。

叶子原语不写成文件：它们的 `import` 留在用它们的文件里，由调用方自己的 `viba_path` 找到。

## 4. 从文档到文档

`ExtractSubVibaMarkdown` 收一份文档，按同一套走法抽它的子文档；节名就是源文档里的文件路径，所以文档
里写着的模块仍然按 `xxx.yyy` 找。答的是子文档 + `VibaMdToVibaMdSourceMap`（子文档的行号 → 源文档的
行号）。

一节也可以是一份文档（`docs/things.viba.md`）：它里面的节是 `docs/things/` 下的文件，抽出来的节按它们
的路径写（`docs/things/colors.viba`），行号仍指回外层文档里那一行。

## 5. 幂等

只有 `ExtractSubVibaMarkdown` 上验这一条：源文档的每个 `import` 都指向它自己的一个节（或语言内置模块）
时，抽出来的文档再抽一遍，文本完全一样。

带叶子原语的文档不算：叶子原语的定义不在文档里，再切会在解析它时被拒绝（第 8 节）。
`ExtractSubRepoIntoVibaMarkdown` 的输入是目录，它的幂等要先把答案落回目录才谈得上，不在这一层的承诺里。

## 6. 例

```python
from viba.extract_sub_repo import (extract_sub_repo_into_viba_markdown,
                                   extract_sub_viba_markdown)

answer = extract_sub_repo_into_viba_markdown(
    "kb", ["kb", "base"], ["kb"], ["report.Report"])
document, source_map = answer.ok_value
source_map[4]                      # (模块, 行号)：文档第 4 行来自哪
```

`kb/report.viba` 里被走到的定义与它用到的 `import` 进文档；`base/` 下的定义是叶子原语，不进来。

## 7. 测试

[`tests/test_extract_sub_repo.py`](../../tests/test_extract_sub_repo.py)：走到的定义与文件集合、叶子原语的定义不
进文档（引用它们的 `import` 保留）、泛型的标记与选中的文件、`.viba.md` 里的定义（本项目
`pattern_custom_types` 那份文档）、两个入口的行号来源——**每个映射行的文字与源行逐字相同**、**答案的每个
代码行都有映射**——以及文档入口的幂等、文档里嵌一份文档（那一节的行号指回外层）、两个入口的报错（第
8 节）。报错不只断言它是 `Err`，也断言消息点到了出事的名字或参数。

## 8. 报错

两个入口都答 `Err`，消息一句话，说第一件不成立的事。

仓库入口拒绝：

- 名字不是 `xxx.yyy.zzz`（`Fine` 这样不带模块的）；
- 模块在 `viba_path` 上找不到；
- 名字落在泛型目录上——泛型等实参来应用，不能当定义名；
- 定义的文件在 `$repo` 里，但不在 `$extract_source_viba_path` 下；
- 模块里没有这个定义；
- 定义体里引用的名字解析不到——泛型 pattern 文件里的裸名字不算，那是 pattern 参数；
- 泛型目录里的文件取不出来：`pattern` 行解析不了，或者决断做不出时要取遍能接那么多实参的文件，其中一个是
  坏的；
- 名字落在 pattern 文件里的定义上——pattern 文件靠对它的泛型的一次应用整份留下，不靠名字；
- 泛型带实参取成员，选中的文件里没有这个定义；
- 参数不对：`repo` 不是路径，`viba_path`、`extract_source_viba_path`、`type_names` 不是字符串列表，
  单位元名字不是字符串。

文档入口拒绝 `type_names` 与单位元名字那几条参数错误，另有：

- 源不是文本；
- 某个代码围栏没闭合；
- 某个章节没有名字；
- 两个章节同名；
- 章节的名字不是文档目录里的文件名（绝对路径、`..`、空的一段）；
- 两个章节把同一个名字一个当文件、一个当目录（`a` 与 `a/b.viba`）；
- 名字在文档里找不到。文档只认自己那些节：叶子原语的 `import` 虽然还在，它的定义不在文档里，所以带
  叶子原语的答案不能再当源切（第 5 节）。

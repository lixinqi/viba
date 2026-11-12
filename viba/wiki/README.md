# viba/wiki：目录与文档、JSON Schema、检查、工具

viba 用纯类型描述设计：一份设计是一组类型与定义，逻辑由实现者补上。这里放 viba 的几类说明：

- **目录与文档**：一个目录怎么写成一份 Markdown，一个目录的代码又怎么收成一份 Markdown——
  [`viba-markdown-as-directory.md`](viba-markdown-as-directory.md)、[`viba-repomix.md`](viba-repomix.md)。
- **JSON Schema**：一个类型能不能写成 JSON Schema、怎么写——
  [`viba-check-convertable-to-json-schema.md`](viba-check-convertable-to-json-schema.md)、[`viba-convert-to-json-schema.md`](viba-convert-to-json-schema.md)。
- **检查**：一份设计自己的错（一个积里同一个 tag 两次、内联成环）——
  [`viba-check-tag-and-inline.md`](viba-check-tag-and-inline.md)。
- **工具**：从一个 repo 里抽出子 repo、把多个 handler 拼成一个函数——
  [`viba-extract-sub-repo.md`](viba-extract-sub-repo.md)、[`viba-make-dispatcher.md`](viba-make-dispatcher.md)。

## 章节

| 文档 | 讲什么 |
|------|---------|
| [`viba-markdown-as-directory.md`](viba-markdown-as-directory.md) | 一个目录写成一份 Markdown：一节一个文件，章里的 `viba` 代码块拼成它的代码 |
| [`viba-repomix.md`](viba-repomix.md) | `repomix`：一个目录的 viba 代码收成一份 Markdown，注释变成说明 |
| [`viba-check-tag-and-inline.md`](viba-check-tag-and-inline.md) | `check_tag_and_inline`：摊开内联位之后，一个积里同一个 tag 不能出现两次，内联链不能绕回正在摊的定义 |
| [`viba-check-convertable-to-json-schema.md`](viba-check-convertable-to-json-schema.md) | `check_convertable_to_json_schema`：一个类型能不能写成 JSON Schema |
| [`viba-convert-to-json-schema.md`](viba-convert-to-json-schema.md) | `convert_to_json_schema`：类型写成 JSON Schema |
| [`viba-extract-sub-repo.md`](viba-extract-sub-repo.md) | `extract_sub_repo`：把 repo 的子集写成一份文档，以及从文档再抽子集（带行号来源） |
| [`viba-make-dispatcher.md`](viba-make-dispatcher.md) | `make_dispatcher`：多个 handler 拼成一个函数，以及它们一起构成的类型 |
| [`banned-words.md`](banned-words.md) | 写文档时不许用的词，以及各自的替代词 |

## 例：抽一个子 repo

一个 repo 越写越大，而一个任务只要其中一部分。`extract_sub_repo` 把那部分写成一份文档：给定的
类型名走得到的定义，每个文件一节，节名是它在 repo 里的路径；同时给出每个代码行的来源。

```python
from viba.extract_sub_repo import extract_sub_repo_into_viba_markdown

answer = extract_sub_repo_into_viba_markdown("kb", ["kb", "base"], ["kb"], ["report.Report"])
document, source_map = answer.ok_value
source_map[4]        # (模块的点分名, 模块里的行号)：文档第 4 行来自哪
```

`extract_source_viba_path`（这里的 `kb`）下的定义才抽；它们走到它外面的部分（`base`）是**叶子原语**：
叶子原语的定义不进文档，引用它们的文件保留 `import`，由调用方自己的 `viba_path` 找到。repo 里的
`.viba.md` 文档一样抽。从一份文档再抽子文档用 `extract_sub_viba_markdown`，幂等只在那里成立。
整份说明在 [`viba-extract-sub-repo.md`](viba-extract-sub-repo.md)。

## 这里的代码

| 模块 | 干什么 |
|------|--------|
| `markdown.py` | 一个目录写成一份 Markdown——见 `viba-markdown-as-directory.md` |
| `repomix.py` | 一个目录收成一份 Markdown——见 `viba-repomix.md` |
| `check_tag_and_inline.py` | 查一份设计自己的错：摊开内联位之后，一个积里同一个 tag 不能出现两次，内联链不能绕回来——见 `viba-check-tag-and-inline.md` |
| `design_walk.py` | 几处共用的读法：链、调用、类型名、绑定 |
| `extract_sub_repo.py` | 把 repo 的子集写成一份文档，以及从文档再抽子集——见 `viba-extract-sub-repo.md` |
| `check_convertable_to_json_schema.py` | 能不能写成 JSON Schema——见 `viba-check-convertable-to-json-schema.md` |
| `convert_to_json_schema.py` | 类型写成 JSON Schema——见 `viba-convert-to-json-schema.md` |



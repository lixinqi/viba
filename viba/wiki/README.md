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
| [`viba-markdown-as-directory.md`](viba-markdown-as-directory.md) | 一个目录写成一份 Markdown：一节一个文件，第一个代码块是文件内容 |
| [`viba-repomix.md`](viba-repomix.md) | `repomix`：一个目录的 viba 代码收成一份 Markdown，注释变成说明 |
| [`viba-check-tag-and-inline.md`](viba-check-tag-and-inline.md) | `check_tag_and_inline`：摊开内联位之后，一个积里同一个 tag 不能出现两次，内联链不能绕回正在摊的定义 |
| [`viba-check-convertable-to-json-schema.md`](viba-check-convertable-to-json-schema.md) | `check_convertable_to_json_schema`：一个类型能不能写成 JSON Schema |
| [`viba-convert-to-json-schema.md`](viba-convert-to-json-schema.md) | `convert_to_json_schema`：类型写成 JSON Schema |
| [`viba-extract-sub-repo.md`](viba-extract-sub-repo.md) | `extract_sub_repo`：从一个 repo 里抽出子 repo |
| [`viba-make-dispatcher.md`](viba-make-dispatcher.md) | `make_dispatcher`：多个 handler 拼成一个函数，以及它们一起构成的类型 |
| [`banned-words.md`](banned-words.md) | 写文档时不许用的词，以及各自的替代词 |

## 例：抽一个子 repo

一个 repo 越写越大，而一个任务只要其中一部分。`extract_sub_repo` 把那部分写成它自己的 repo：给
定的类型名走得到的定义，每个文件按原来的路径，连同仍能找到留在原处的东西的 `viba_path`。

```python
from viba.extract_sub_repo import extract_sub_repo

answer = extract_sub_repo("kb", ["kb", "base"], ["kb"], ["report.Report"])
answer.ok_value.repo         # "kb-sub"
answer.ok_value.viba_path    # ["kb-sub", "base"]
```

`extract_source_viba_path`（这里的 `kb`）下的定义才复制；它们走到它外面的部分（`base`）是原语——
不复制，import 留着，根挂在返回值的 `viba_path` 上。模块名从相对 repo 的路径来，所以全名不变
（`kb/report.viba` 是模块 `report`，抽之前之后都一样），同一批类型名在子 repo 里照样解析得到。
再抽一次同样的子 repo，内容一样。整份说明在 [`viba-extract-sub-repo.md`](viba-extract-sub-repo.md)。

## 这里的代码

| 模块 | 干什么 |
|------|--------|
| `markdown.py` | 一个目录写成一份 Markdown——见 `viba-markdown-as-directory.md` |
| `repomix.py` | 一个目录收成一份 Markdown——见 `viba-repomix.md` |
| `check_tag_and_inline.py` | 查一份设计自己的错：摊开内联位之后，一个积里同一个 tag 不能出现两次，内联链不能绕回来——见 `viba-check-tag-and-inline.md` |
| `design_walk.py` | 几处共用的读法：链、调用、类型名、绑定 |
| `extract_sub_repo.py` | 抽子 repo——见 `viba-extract-sub-repo.md` |
| `check_convertable_to_json_schema.py` | 能不能写成 JSON Schema——见 `viba-check-convertable-to-json-schema.md` |
| `convert_to_json_schema.py` | 类型写成 JSON Schema——见 `viba-convert-to-json-schema.md` |



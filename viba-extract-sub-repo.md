# extract_sub_repo：从一个 viba repo 里抽出一个子 repo

知识库会一直累积、一直重构，而一次任务只用得上其中一小片。`extract_sub_repo` 把这一小片单独
抽出来，成为一个能自己编译、自己运行的子 repo：原 repo 里别处的改动进不到它里面来。

抽出来的子 repo 有两条性质：

- **闭**：子 repo 里每个定义提到的名字，要么在子 repo 里解析得到，要么是不抽走的那些定义（下面叫
  **原语**，靠返回值里的 `$viba_path` 找到），要么是解释器给的内建词汇。
- **小**：只有从给定的 type 走得到的定义才留下；同一个文件里别的定义不会跟着走。

抽的粒度是定义：一个文件里只有几个定义被用到，子 repo 里就只带这几个。

## 1. 类型

```viba
ExtractSubRepo =
    Result[$repo Repo * $viba_path list[str]]
  <- $repo Repo
  <- $viba_path list[str]
  <- $extract_source_viba_path list[str]
  <- $type_names list[TypeName]
  <- $never_eqv (list[str] | nil)
  <- $nil_eqv (list[str] | nil)
  <- Hint[{
      从 $type_names 出发，把 $repo 里、$extract_source_viba_path 下的那些定义抽成一个能自己编译的子 repo：
      路径、模块名、定义名不变，一个文件只带留下的定义实际用到的 import。
      其余定义是原语：不复制，留着 import，它所在的根进返回值的 $viba_path。
      $never_eqv、$nil_eqv 是单位元的名字，读源码时不算定义；给 nil 就用语言默认那份。
    }]
  <- Assert[{
      幂等：抽出来的子 repo 再抽一遍，两个目录的内容完全一样。
      对任何 $repo、$viba_path、$extract_source_viba_path、$type_names、$never_eqv、$nil_eqv：
      first = ExtractSubRepo
        << $repo repo
        << $viba_path viba_path
        << $extract_source_viba_path extract_source_viba_path
        << $type_names type_names
        << $never_eqv never_eqv
        << $nil_eqv nil_eqv
      second = ExtractSubRepo
        << $repo first.ok.repo
        << $viba_path first.ok.viba_path
        << $extract_source_viba_path first.ok.viba_path
        << $type_names type_names
        << $never_eqv never_eqv
        << $nil_eqv nil_eqv
      assert content_of(first) === content_of(second)
      content_of(x) 是 x.ok.repo 这个目录的内容：同样的相对路径，同样的文件原文。
      两边都得答 $ok；first 答 $ok 而 second 答 $err 也违反这条。
    }]

Repo =
    str * Assert[{ this is a file path }]

TypeName =
    str * Assert[{ xxx.yyy.zzz：xxx.yyy 在 $viba_path 下 import 得到，zzz 是它里面的定义 }]
```

Python 里是这个函数（[`viba/extract_sub_repo.py`](viba/extract_sub_repo.py)）：

```python
from viba.extract_sub_repo import extract_sub_repo

answer = extract_sub_repo("kb", ["kb", "base"], ["kb"], ["report.Report"])
answer.ok_value.repo         # "kb-sub"
answer.ok_value.viba_path    # ["kb-sub", "base"]
```

答 `Ok(SubRepo(repo, viba_path))` 或 `Err(...)`（一句话，说的是第一处不成立的地方）。

## 2. 例子

```
kb/
  report.viba
  item.viba
base/
  meta.viba
```

```viba
# kb/report.viba
import item as it
import meta as mt

Report = Object * $items list[it.Item] * $meta mt.Meta
```

```viba
# kb/item.viba
Item = Object * $id int * $name str
```

```viba
# base/meta.viba
Meta = Object * $author str
```

`kb` 是 repo；`kb/report.viba` 的模块是 `report`，`kb/item.viba` 的模块是 `item`。`report.viba` 里
`item` 在它旁边找得到，`meta` 不在 `kb` 里——它落在 `base`，按 `$viba_path` 找到。

```python
extracted = extract_sub_repo("kb", ["kb", "base"], ["kb"], ["report.Report"])
```

`never_eqv` / `nil_eqv` 不给就用语言默认那份（签名里给 `nil` 是同一件事）。

抽取源是 `kb`；`Report` 的 `$meta` 字段是 `mt.Meta`，用得到 `meta`，但 `meta` 不在抽取源里，
所以它是原语——用得到，但不抽走。返回值：

```viba
# kb-sub/report.viba
import item as it
import meta as mt

Report = Object * $items list[it.Item] * $meta mt.Meta
```

```viba
# kb-sub/item.viba
Item = Object * $id int * $name str
```

```
extracted.ok_value.repo        "kb-sub"
extracted.ok_value.viba_path   ["kb-sub", "base"]
```

`base/meta.viba` 不复制，但 `report.viba` 里那行 `import meta as mt` 得留下——`Report` 的
`$meta` 字段就是 `mt.Meta`，没有这行它解析不了。`base` 进返回值的 `$viba_path`，子 repo 靠它
解析到 `meta`，所以不复制也能编译、能跑。

## 3. 参数

**`$repo`**：要切的目录。它里面每个 `.viba` 文件按相对路径定模块名：去掉 `.viba`，`/` 换成 `.`
（`kb/report.viba` 是模块 `report`，`kb/pkg/a.viba` 是模块 `pkg.a`）。只有 `.viba` 文件参与，repo 里
别的文件既不读也不复制。

**`$viba_path`**：`import` 的查找顺序，和解释器里的规矩一样：先看给出这个 `import` 的那个文件旁边，
再按这张表的顺序，最后是内建词汇。`$type_names` 里的 `xxx.yyy` 只按这张表找，它没有“旁边”。

**`$extract_source_viba_path`**：抽取源在 `$repo` 里的范围，可以给多个目录。文件在 `$repo` 里、
且落在其中一个目录下，它的定义就是要抽走的；其余定义都是原语——包括 `$repo` 里没落进这些目录的文件，
和 `$repo` 外、按 `$viba_path` 找到的模块。

**`$type_names`**：每个是 `xxx.yyy.zzz`。`xxx.yyy` 按 `$viba_path` 找到模块，`zzz` 是它里面的定义；
这个名字要在抽取源里。泛型目录本身不是定义，不能点名：它由用到它的应用带进来。

**`$never_eqv`**：`(list[str] | nil)`。算作 `never`（和的单位元）的名字；给 `nil` 就用语言默认那份
（`Oneof`）。

**`$nil_eqv`**：`(list[str] | nil)`。算作 `nil`（积的单位元）的名字；给 `nil` 就用语言默认那份
（`Object`、`Assert`、`Appendix`、`Hint`）。

这两份和 [`viba-reflect.md`](viba-reflect.md) 里 `VibaReflectConfig` 的同名参数是同一份：读源码时，
这些名字不是定义，不抽，也不算“解析不到”的错。给了清单就用清单里的那些，不往上加语言默认。

## 4. 依赖怎么算

- 从 `$type_names` 对应的定义出发。
- 定义体里出现名字的三处算依赖：类型位置（名字、应用、元组、和/积/指数链）、`<<` 的头、
  `g[T].member`。代码块里的字不算。
- 落到抽取源里的定义：留下，接着走。
- 落到原语：不留下，也不往它里面走；它所在的根记进返回值的 `$viba_path`。原语不是“用不到”的
  定义，它是用得到、但不抽走的那些定义。
- 落到内建词汇：什么也不做——标量、容器、`Any`、`builtin` 的成员、`viba/builtin/` 下的模块
  （`Either`、`make_dispatcher`），以及 `$never_eqv` / `$nil_eqv` 说的单位元名字。

## 5. 泛型

`G[...]` 是 `G` 那个目录：`__generic__.viba` 必留；编号文件按闭包里对 `G` 的每一次应用来留：

- 这次应用的实参在源码里定得下来：只留决断选中的那一份；
- 定不下来（实参还是泛型参数）：留下所有能收下这么多实参的编号文件。

留下的编号文件整份留下、原样复制：里面有 `pattern`、`__decl__`、`__impl__` 和它们用到的定义，少一样
这个决断就不成立，所以这些定义连同它们依赖的名字一起留下。`G` 的目录不在抽取源里，整个 `G` 是
原语。

## 6. import

- 留下的定义用到哪个 `import` 的本地名，那行 `import` 就留；没用到的 `import` 不留。
- 原语的模块要能从返回值的 `$viba_path` 找到：它所在的根进返回值的 `$viba_path`。

## 7. 返回值

- **`$repo`**：新子 repo 的路径。建在 `$repo` 旁边，目录名是原目录名加 `-sub`（`kb` → `kb-sub`）；
  已经存在就整个替换（先建好新的再换上去）。原 repo 不动。
- **`$viba_path`**：新 repo 的根排第一，其余是原语实际所在的根，按它们在原 `$viba_path` 里的顺序，
  去重；导入文件旁边找到的根不在原表里，排在它们后面。新根排第一，所以子 repo 自己的模块优先于原
  repo 里的同名模块。
- **文件**：路径相对根不变（多深都一样），模块名、定义名不变；一个文件只带被依赖的定义和它们用到的
  `import`。

## 8. 幂等

第 1 节 `<- Assert[{...}]` 里写的就是这条。把第一次的返回值再喂进去——`$repo` 给 `first.ok.repo`，
`$viba_path` 和 `$extract_source_viba_path` 都给 `first.ok.viba_path`——第二次抽出来的目录与第一次
的内容完全一样：同样的相对路径，同样的文件原文。两次落在两个目录上，路径不同，所以比的是内容，不是
路径。

这条 law 也钉住了 `$extract_source_viba_path` 的范围：它只能圈 `$repo` 里的目录。要是能圈到 `$repo`
外，第二次就会把原语也抽进来，两边内容就不等了。例子里的 `base` 就是这样：第二次它在
`$extract_source_viba_path` 里，但在 `$repo`（`kb-sub`）外，所以 `meta` 还是原语、还是只留
import，内容不变。

## 9. 错误

- `$type_names` 里的名字按 `$viba_path` 解析不出来，或者解析到的定义不在抽取源里；
- 留下的定义引用了既不在抽取源、也不在 `$viba_path`、也不是内建词汇、也不是单位元名字的东西；
- 抽取走到的文件编不过。

答 `$err`，说的是第一处。

## 10. 这一版不做的

- 普通文件重新排版：`#` 行注释和空行不保留；带代码块的成员（`Comments[{...}]`、`Hint[{...}]`）是
  定义体的一部分，跟着走。泛型的 `__generic__.viba` 和留下的编号文件原样复制，注释和排版都在。
- 不改模块名，不合并文件，不做任何重命名。
- 只处理 `.viba` 文件；repo 里的其它文件不复制。
- 不替你跑测试：闭是静态保证的，跑不跑得通由子 repo 的调用方决定。

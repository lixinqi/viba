# viba_lint：类型语句的版式

`viba_lint` 检查一个类型语句写下来之后的样子。一个语句是顶层的 `名字 = ...` 或
`pattern ...`。**语句必须折行**：体放在 `=` 后面那些行，一行写完的语句报 `statement-fold`；
只有类型表达式可以留在一行（`UserId = int`、`* $value T` 里的 `T`），短链例外（见下）。其余规则是：

1. **和用 `Oneof` 引导，和可选项的类型同一列。** 和语句是一个块，第一行是 `Oneof`，其余
   每个可选项以 `|` 起头；`Oneof` 在 `|` 后面那一列（可选项类型的第一列），不是和 `|`
   同一列。一行写完的和没有块，报——短的和例外：`Placement = Shard | Replicate | Partial`
   留在一行，不补 `Oneof`。
2. **积有引导，和字段的类型同一列。** 积语句是一个块，第一行是 `Object`、`tagged[...]`
   或别的基类，其余每个字段以 `*` 起头；引导在 `*` 后面那一列（字段类型的第一列），不是
   和 `*` 同一列。单成员的积也折行：`Box = Object * $value T` 是语句，不是表达式。
3. **`|`、`*`、`<-`、`<<` 起头的行总是缩进 2 个空白。**
4. **指数的结果类型和参数类型同一列**（`<-` 后面那一列）。一行写完的指数没有第二行可对，
   不查。
5. **部分应用的函数和实参同一列**（`<<` 后面那一列）。一行写完的部分应用不查。
6. **同一个块里，字段的注释在同一列。**

表达式也一样：`(...)` 里如果是和、积、指数或部分应用，就折成块；内层块的基是**这一行里那个表达式
的起点本身**（`$foo` 在第 4 列，基就是 4）。`|` / `*` / `<-` / `<<` 在基后 2 列；`Object` /
`Oneof` 在基后 4 列；`<-` 的结果和 `<<` 的函数在基后 5 列（比运算符多 3 列）；`)` 自己一行，在基。纯 tag
嵌套（`$foo ($bar ($foobar int))`）留在一行。括号里的和比这更严一点：短到能一行写完，就必须
一行写完（已经折成块的报 `expression-unfold`，`--fix` 放回一行）；语句里的和两种写法都收。
已经折成块的表达式缩进不对时也报 `expression-fold`：`--fix` 把头、成员、`)` 放回各自的列，
`| SN.t2i_prefix_forward )` 这种把 `)` 贴在前面的写法，会把 `)` 拆到自己一行。直接在
`<- (` / `* (` 后面开的和/积块，`|` / `*` 和这条链的 `$name` 同列（头再往右 2 列），`)` 也回到
这一列——它跟着外面那条链走，不是跟着表达式起点再往里缩。折开的和块由 `Oneof` 领头：头的位置
上原来是分支的那份，会补出 `Oneof`，那一行变成第一个 `|` 成员。

`|` / `<-` / `<<` 的链本身是一行、运算符少于 4 个、整行不到 50 个字符时，可以留在一行不折：
语句和括号里的表达式都一样；超过才折。所以 `(int | nil)`、`(torch.DTensor | nil)` 不折，
也不补 `Oneof`；长的和才折成 `Oneof` 块。括号里的和再短也按这一条来（`expression-unfold`），
语句里的和两种写法都收。

```viba
A = $foo (
        Object
      * $bar int
      * $nice str
    )

B = $foo (
         int
      <- $env Env
    )
```

```viba
Result =
    Oneof
  | $ok Entry
  | $missing UserId

Placement = Shard | Replicate | Partial

Entry =
    Object
  * $name str        # 名字
  * $state State     # 状态

Distance =
     int
  <- $env Environment
  <- $a int

Guess =
     compare
  << $a 1

Box =
    Object
  * $value T
```

`.viba` 文件和 `.viba.md` 文档都查：文档里每一节（`## `x.viba`` 那一节）按同一套规则检查，
诊断的行号是文档自己的行号。

```bash
python -m viba.lint viba tests
```

有违反时退出码是 1，每条形如 `路径:行:列: 规则: 说明`。要查的路径可以是文件，也可以是
目录；目录会递归找 `.viba` 和 `.viba.md`。

加 `--fix` 就直接改，不再只报：

```bash
python -m viba.lint --fix viba tests
```

它能改的是版式：语句折行、和补 `Oneof`、积补 `Object`、`|` / `*` / `<-` / `<<` 的缩进、结果
和函数的列、字段注释的列，以及括号里的和/积/指数/部分应用折行；`.viba.md` 只改 `viba` 围栏块里的
代码，文档的标题、正文和围栏不动。语句的成员已经各占一行时，块头和运算符的列会改，字段内容和提示
文字原样保留；缺块头、成员行还要折行的语句不改，单独输出 `skipped 路径:行`（行是块头或第一个成员
那一行，和诊断同一行）；其它改不了的（整行注释等）原样保留，照常报出来。

默认跳过生成语料与用例文件：`tests/corpus`、`tests/data`、`viba/apply_impl`、
`viba/sequential_impl`、`viba/builtin`。要再加，用 `--exclude`；要连默认清单一起来读，用
`--no-default-excludes`。

## `.viba.md` 是另一种写法的 viba 代码

`.viba.md` 不是说明文档，是 viba 代码的另一种写法：标了 `viba` 的围栏块就是代码，代码块之间的
文字和标题是它的说明。`viba_lint` 按扩展名自动认它——代码块照上面那套版式检查，另外查这份写法
自己的六条结构：

1. **一级标题是这份文件描述的 viba 文件的路径名**，以 `.viba` 结尾；文档只有一个一级标题。
2. **每个 `viba` 围栏块最多一次赋值（一个 `=`）**：`=` 在提示块、字符串、注释里不算；
   `import` 没有 `=`，不算赋值，几条 import 可以写在一个块里。
3. **`=` 左边的类型名是它上面那个标题的一部分**；标题几级都行，`##`、`###` 一样。
4. **字段的类型是 `$field_name ($field_type * { field comments })`**：字段的类型和它的说明是一个
   积，说明是字段自己的提示块，不能为空。这个积在版式上折成块，和别的积表达式一样。
5. **import 和赋值不放在一个块里**：import 几条可以写在一起，但和一个 `=` 不同块。
6. **大概念先呈现**：一份文档里，定义排在它用到的定义前面；`|`、`*`、`<-` 里引用到的、同一份
   文档里定义的名字都算。fix 把整节（标题 + 说明 + 代码块）挪到正确位置；成环的顺序改不了，只报。

前五条只有检查，没有 `--fix`；第 6 条有 fix（挪整节）。`--fix` 还会照旧改代码的版式，改不了的
照常报出来。规则名分别是 `document-title`、`document-equals`、`document-type-name`、
`document-field`、`document-imports`、`document-order`。
第 2、3 条是一个开关，默认开着；一份文档本来就是一个块里写好几份赋值、标题只是个文件名的，用
`--disable-document-code-block-num-assignments-le-one` 关掉它，`document-title` 和 `document-field`
照查。第 5 条也有开关 `--split-imports-and-assignments`，默认开着；要允许 import 和赋值同块，用
`--no-split-imports-and-assignments`。第 6 条的开关是 `--present-big-picture-first`，默认开着；
`--no-present-big-picture-first` 关掉后，检查和 fix 都不管顺序。

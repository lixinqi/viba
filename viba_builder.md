# Viba.Builder 使用

## 1. 引言

`viba.builder` 让你用 Python 的表达式写 Viba 源码。你写定义，`str(vb)` 给出整份文件。

它只做两件事：

- **组装**：Python 的算子直接对应语言的算子，写出来的东西进的是同一套 AST，再由 `viba.viba_ast.unparse` 打印成**规范写法**（跟打印器互为定点）。
- **把守**：写错的地方在赋值那一行就抛异常，不会等你落盘才发现。

它不做：不重排既有文件、不校验语义（类型是否成立是判断层的事）、不猜你的意思。

## 2. 先看一段

```python
import viba.builder

vb = viba.builder.Builder("import store_core as sc")
tag = viba.builder.tag

vb.UserName = str

vb.Optional[vb.T] = vb.T | None

vb.List[vb.T] = (
    vb.Oneof
    | vb.Object * tag.head(vb.T) * tag.tail(vb.List[vb.T])
    | None
)

vb.latest_file[vb.Ctx] = (
    vb.sc.Result[vb.sc.FileState]
    ** tag.ctx(vb.Ctx)
    ** tag.file(vb.sc.FileId)
)

print(str(vb))
```

```viba
import store_core as sc

UserName :=
  str

Optional[T] :=
  T
  | nil

List[T] :=
  Oneof
  | Object
    * $head T
    * $tail List[T]
  | nil

latest_file[Ctx] :=
  sc.Result[sc.FileState]
  <- $ctx Ctx
  <- $file sc.FileId
```

`Builder(…)` 里那串是**文件的起点**：只有 import 时它就是个头；给一整份既有文件时，后面的定义追加在它后面（第 10 节）。

## 3. 对照表

| Python | Viba |
|---|---|
| `vb.Name = body` | `Name := body` |
| `vb.Name[T, U] = body` | `Name[T, U] := body` |
| `vb.Name` | `Name` |
| `vb.a.b.Name` | `a.b.Name` |
| `vb.Name[arg0, arg1]` | `Name[arg0, arg1]` |
| `vb.Name[()]` | `Name[]` |
| `A \| B` | `A \| B`（和） |
| `A * B` | `A * B`（积） |
| `A ** B` | `A <- B`（指数；`B` 必须是 tag） |
| `tag.name(body)` | `$name body` |
| `tag(body)` | 一个分组（跟括号等价；`**` 右边要嵌套时只能用这个） |
| `None` | `nil`（`vb.nil`、`vb.void` 同义） |
| `vb.never` | `never` |
| `...` | `...` |
| `True` / `False` | `true` / `false` |
| `42` / `1.5` / `"x"` | `42` / `1.5` / `"x"` |
| `str` / `int` / `float` / `bool` / `list` / `set` / `dict` | 同名类型 |
| `int \| str` | `int \| str`（Python 自己会先算成联合类型，照样认） |
| `list[vb.A]` / `set[vb.A]` / `dict[str, int]` | `list[A]` / `set[A]` / `dict[str, int]` |
| `list[int] \| None` | `list[int] \| nil` |
| `[a, b]` / `[]` | `ListLiteral[a, b]` / `ListLiteral[]` |
| `{a, b}` | `SetLiteral[a, b]` |
| `{k: v}` | `DictLiteral[(k, v)]` |
| `(A, B)` / `(A,)` / `()` | `(A, B)` / `(A,)` / `()` |
| `builder.code(text)` | `{ text }` |
| `builder.comment(vb, text)` | `# text` |
| `builder.add_import(vb, m, a)` | `import m as a` |
| `builder.check(vb)` | 让解析器把写出来的源码读一遍：返回解析出来的 Module，读不过去就抛 ValueError |

## 4. 定义

```python
vb.UserName = str                 # UserName := str
vb.Map[vb.K, vb.V] = vb.V         # Map[K, V] := V
```

- 定义名要是一个普通标识符：不能是关键字（`nil` / `never` / `true` / `false` / `void` / `None` / `import` / `as`），不能带点、不能带横杠，`vb.a.b = …` 也拦（定义名是一个名字，不是一个点分路径）。
- 同一个名字可以写多次，顺序就是写的顺序（第 10 节的"既有文件里已有这个名字"是另一回事）。
- 没有别的定义形式：Viba 的定义只有"带形参"和"不带形参"两种。

## 5. 和 / 积 / 指数

```python
vb.S = vb.A | vb.B | vb.C          # A | B | C
vb.P = vb.A * vb.B * vb.C          # A * B * C
vb.E = vb.A ** tag.p(vb.B)         # A <- $p B
```

**结合性与分组**跟语言一致：

- `|` 与 `*` 两边都是左结合：`A | B | C` 长成一条主链；`A | (B | C)` 括号里那条是**支链**，会带着括号写出来。要分组，括号和 `tag(…)` 都行（见下）。
- `**` 在 Python 里右结合、语言的 `<-` 是左结合，所以 `A ** tag.x(B) ** tag.y(C)` 到手是 `A ** (tag.x(B) ** tag.y(C))`，builder 会把它摊平成书写顺序的一条链 `A <- $x B <- $y C`。**括号在这里不分组**，要嵌套就写 `tag(…)`：

```python
vb.Flat  = vb.A ** tag.p(vb.B) ** tag.q(vb.C)   # A <- $p B <- $q C
vb.Nest  = vb.A ** tag(vb.B ** tag.p(vb.C))     # A <- (B <- $p C)
```

- 优先级也是语言的：`**` 比 `*` 紧、`*` 比 `|` 紧，所以 `vb.A * vb.B ** tag.p(vb.C)` 就是 `A * (B <- $p C)`。

**分组**有两种写法：括号，或者 `tag(…)`。在 `|` 和 `*` 的位置上两者一模一样——`vb.A | tag(vb.B | vb.C)` 就是 `vb.A | (vb.B | vb.C)`，裹一个名字等于没裹；在 `**` 右边则**只能**用 `tag(…)`，因为那里的括号会被摊平。

**`**` 的右边只能是两种东西**，别的当场抛：

- `tag.name(body)`（一个 tagged 字段）；
- `tag(body)`（一个分组，里面通常是另一条指数链）。

```python
vb.X = vb.A ** vb.B            # TypeError：右边是名字
vb.X = vb.A ** tag.head        # TypeError：忘了调用 tag
vb.X = vb.A ** (vb.B | vb.C)   # TypeError：右边是和
```

链可以接着长：`vb.A ** tag.p(vb.B) ** tag.q(vb.C)` 合法——右边那一环是 `tag.q(…)`，已经查过了。

## 6. 名字与应用

```python
vb.X = vb.some.deep.Name       # some.deep.Name
vb.Y = vb.F[vb.A]              # F[A]
vb.Z = vb.Map[vb.K, vb.V]      # Map[K, V]
vb.W = vb.F[()]                # F[]
```

`vb.F` 是**引用**，`vb.F[()]` 是**应用**（零实参）：Python 没有空下标，所以零实参只能这么写。实参可以是名字、应用、元组、字面量、tagged 字段。

Python 自己那套下标类型也认，落成语言里的写法：

```python
vb.ListA = list[vb.A]              # list[A]
vb.SetA  = set[vb.A]               # set[A]
vb.Map   = dict[str, int]          # dict[str, int]
vb.Maybe = list[int] | None        # list[int] | nil
vb.Typed = typing.List[vb.A]       # list[A]（typing 那套也落成 list）
vb.Opt   = typing.Optional[vb.A]   # A | nil
```

## 7. 单位元与字面量

```python
vb.A = None        # nil
vb.B = vb.nil      # nil
vb.C = vb.void     # nil（nil 的别名）
vb.D = vb.never    # never
vb.E = ...         # ...
vb.F = vb.Object | vb.Oneof      # Object | Oneof（这两个是名字，不是关键字）
vb.G = True        # true
vb.H = False       # false
vb.I = 42          # 42
vb.J = 1.5         # 1.5
vb.K = "x"         # "x"
vb.L = str         # str（类型名）
vb.M = int | str   # int | str（Python 的联合类型就是和）
vb.N = int | None  # int | nil
```

`vb.nil` / `vb.never` 建出来的就是 `Nil` / `Never` 节点本身，不只是打印成 `nil` / `never`。`vb.true` / `vb.false` 是写错（Python 里该写 `True` / `False`），当场抛。

## 8. 容器、元组、代码块

```python
vb.Items  = [1, "x"]                 # ListLiteral[1, "x"]
vb.Empty  = []                       # ListLiteral[]
vb.Seen   = {"c", "a", "b"}          # SetLiteral["a", "b", "c"]
vb.Table  = {"k": 1, "m": vb.V}      # DictLiteral[("k", 1), ("m", V)]
vb.Nested = [[1], {"k": [2]}]        # 容器套容器
vb.Pair   = (vb.A, vb.B)             # (A, B)
vb.One    = (vb.B,)                  # (B,)
vb.Unit   = ()                       # ()
vb.Code   = builder.code("return 1") # {return 1}
```

- `list` 顺序照写；`dict` 顺序照写，键值成对；**`set` 按打印出来的文本排序**——集合本身无序，这样同一个集合每次给出的源码一样。
- 一元组必须留逗号：`(B)` 读回来只是 `B`，`(B,)` 才是元组。
- 元素跟别处一样走同一套包装：字面量、名字、表达式都行。

## 9. 文件级的三件事

```python
vb = viba.builder.Builder()
vb.Answer = 42

viba.builder.add_import(vb, "store_core", "sc")   # 排在最前，哪怕定义先写
viba.builder.comment(vb, "先放一条注释")             # 写在它站的位置
module = viba.builder.check(vb)                     # 让解析器读一遍 str(vb)
print(str(vb))
```

```viba
import store_core as sc

Answer :=
  42

# 先放一条注释
```

`check(vb)` 做的事就一件：把 `str(vb)` 交给 `viba_ast.parse`。写出来的是合法 Viba 就返回那棵 Module（AST），不是就抛 `ValueError`——等于落盘前先自己读一遍。

这几个都是模块函数，不是 `Builder` 的方法——`Builder` 上没有任何公开方法，所以 `vb.<名字>` 永远只可能是定义。

`str(vb)` 的收尾永远只有一个换行；什么也没有的 builder 写出空串。

## 10. 续写既有文件

`Builder(…)` 的参数是文件的起点，所以续写就是把它交进去：

```python
vb = viba.builder.Builder(Path("store.viba").read_text())

viba.builder.comment(vb, "加上一条")
vb.Added = vb.Object * tag.x(vb.T)

Path("store.viba").write_text(str(vb))
```

**只能追加新定义**，三道拦截：

| 你做的事 | 结果 |
|---|---|
| 起点不是合法 Viba 源码 | `ValueError`（当场） |
| 定义文件里已有的名字 | `TypeError: 'DemoRule' is already defined: only new definitions can be appended` |
| 给续写的 builder 加 import | `TypeError`（import 得站在最上面，那是插入，不是追加） |

既有内容原样带在最前，不会被重排、不会掉注释。

## 11. 会当场报错的地方

| 写法 | 报什么 |
|---|---|
| `vb.X = vb.A ** vb.B` | 右边是名字：只收 tag |
| `vb.X = vb.A ** tag.head` | `tag.head is a tag with no body: write tag.head(body)` |
| `vb.X = vb.Object * tag.float_value` | 同上（积里也一样） |
| `vb.X = vb.A ** (vb.B \| vb.C)` | 右边是和/积/字面量：只收 tag |
| `vb.X = vb.true` | `write True, not vb.true` |
| `vb.nil = 1` | `'nil' is not a definition name` |
| `setattr(vb, "a-b", 1)` | 同上（定义名得是标识符） |
| `vb.a.b = 1` | 定义名是一个名字，不是一个点分路径 |
| `vb.List[1] = vb.A` | 泛型形参只能是名字 |
| `vb.Map[vb.T, vb.T] = vb.A` | 泛型形参不能重复 |
| `vb.X = object()` | 写不出来的 Python 值 |

## 12. 边界

- **Python 没有空下标**：零实参应用写 `vb.F[()]`（写成 `vb.F[]` 是 Python 语法错）。
- **最左边那个操作数得是 builder 出来的**：`vb.A | 1` 可以，`1 | vb.A` 也可以（走 `__ror__`），但 `"a" | 1` 跟 builder 无关，是 Python 自己不让。
- **注释只在你写的位置**：起点里原有注释原样保留；新注释用 `builder.comment`，落在调用它的位置。
- **不重排**：builder 只往后面接，既有文件哪怕写法不"规范"也照原样留着——要统一格式，自己拿 `viba_ast.unparse(viba_ast.parse(text))` 走一遍。
- **不管语义**：`vb.X = vb.Y * vb.Y` 这种重复、未定义的名字、单位元的用法，builder 不查，那是判断层（`viba.is_sub_type`）的事。

## 13. 测试

```bash
PYTHONPATH=/tmp/viba-deps python3 tests/test_builder.py
```

`tests/test_builder.py` 里有 108 个用例，一个细节一个：

- **上层写法** 11：整份文件逐字、起点、定义名与顺序、普通/带形参定义、`|`/`*` 到树、应用、点名、平链、能被解析器读回来；
- **算子** 16：和、积、指数的各种结合与分组；
- **单位元与字面量** 14：`None` / `vb.nil` / `vb.void` / `vb.never` / `...` / 布尔 / 数值 / 字符串 / 类型名 / Python 联合 / 字面量进和；
- **名字、应用、定义** 15（含 Python 的下标类型与 typing 那套）；
- **容器** 11：list / set / dict / 嵌套 / 空；
- **元组与代码块** 5；
- **文件级** 10：import / comment / check / 输出；
- **续写既有文件** 9；
- **写错拦截** 13；
- **规范写法** 4：两份源码都是打印器定点、打印→解析→打印稳定、续写出来的能读回来。

# is_complete：这份设计能反射到底吗

反射要求类型里列得出的每一条数据路径都落得到叶子（[`viba-reflect.md`](../core/viba-reflect.md) 第 9 节）。
`is_complete` 在还没有任何数据的时候先问一遍：把这份设计（`.viba` 源码）反射到底，走得通吗？
它答 `true` 或 `false`，不跑任何程序。

## 1. 类型

```viba
IsComplete =
    bool
  <- $entry_file_content str
  <- $library list[(str, str)]
  <- $viba_paths list[str]
  <- $terminators set[str]
  <- Hint[{
      $entry_file_content 是入口文件的全部内容，当成这份设计。
      $library 是 (文件路径, 文件内容) 的一串，解析引用用；模块名从路径来：去掉 .viba，/ 换成 .。
      $viba_paths 里的目录递归扫 *.viba 收进 library；内置目录不扫。
      $terminators 是一组类型名：走到它就停（TypeRef 的名字，或泛型应用的构造子），默认空。
      答 true / false，没有错误这一支——走不通就是 false。
    }]
```

## 2. 一次走到底

走法是“列得出的每一条路径”，所以只要有一处没有成员、又停不下来，答案就是 `false`：

- **叶子**：字面量、`nil`、`never`、内置标量（`bool`、`int`、`float`、`str`）到这里停。
- **链**：和、积、指数链的成员是主链的元素（单位元头不算），每个元素接着走；元组按位置走；
  带 tag 的一项走它标的东西。
- **和**：选项靠值的种类或 tag 分开。名字不同的选项不要求 tag；两个选项共用一个名字
  （`list[str]` 和 `list[int]` 都是 `list`，两个 `int` 都是 `int`），或者一个选项的形式这一遍还
  说不出来（泛型形参单独当选项）时，两边都要 tag。字面量按种类和值认：`"a"` 和 `"b"` 是两个选项、
  不用 tag，两个 `7` 算一个。两个选项带同一个 tag 是错的。一个选项本身是和时，跟它所在的和摊平了
  算，摊平后不能有重复的名字或 tag。单位元（`Object`、`nil`、`Oneof`、`never`）不是选项：它们没有
  内容，也不要 tag。
- **名字与应用**：经 import 和描述符池解析到定义，接着走它的 body。语言自己给的内置词表
  （`viba/builtin.viba` 里那些标量、容器、算子）算数；内置目录里的文件不读、不扫。名字解析不到
  就是 `false`。泛型形参按应用绑上的实参走；没绑上时它是占位符，不影响答案。`tagged[...]` 的符号
  是这种占位符时同理：只走它标的东西。
- **泛型应用**：按源码里给的实参当场决断，选中的那个文件走一遍（它的形参名换成实参里对应的部分，
  [`viba-pattern.md`](../core/viba-pattern.md)）。决断失败就是 `false`。
- **提示块**：`Hint[...]` 这类应用、裸的 `{...}`、`...`，还有别的没有成员又不在 `terminators` 里的
  东西，都算走不通。想在某处停下，把包着它的名字放进 `terminators`（第 4 节）。
- **递归是共归纳的**：同一个（模块、定义、实参）在走的过程中再遇到，算走通。

## 3. 参数

- `entry_file_content`：入口文件的全部内容，当成这份设计。
- `library`：`(文件路径, 文件内容)` 的一串。模块名从路径来——去掉 `.viba`、把 `/` 换成 `.`
  （`pkg/mod.viba` 是 `pkg.mod`）——路径要和源码里的 import 对得上。
- `viba_paths`：再递归扫这些目录里的 `*.viba`，模块名相对目录。别的都不扫，尤其不扫内置目录。
- `terminators`：一组类型名，默认空。空是最严的：只有叶子和单位元能停。

## 4. terminators 怎么用

`terminators` 里的名字有两种：`TypeRef` 的名字，和泛型应用的构造子。走到它就停，不再往下展开。
想在一个提示块上停，放的是包着它的那个名字：

```python
is_complete("V = Object * $a int * Hint[{ 说明 }]\n")                  # False
is_complete("V = Object * $a int * Hint[{ 说明 }]\n",
            terminators={"Hint"})                                       # True
```

裸的 `{...}` 没有名字可放（它不是应用），带这种提示的设计总是 `false`。

## 5. 例子

走得通的：

```viba
U = Object * $a int

Z =
  Oneof
  | $a int
  | $b str

R = Object * $r R
```

走不通的（每一份 `is_complete` 都答 `false`）：

```viba
A = Object * $a int * { 说明 }

B = Any
```

`B` 里 `Any` 没有成员，也不在 `terminators` 里；`A` 的提示块同理，只是 `terminators={"Hint"}` 能
让第一种停下来。

## 6. 这一版不做的

- **不说错在哪**：答案只有一个 `bool`。要指出第一处错，用
  [`check_tag_and_inline`](viba-check-tag-and-inline.md)，或者拿这份设计去跑别的检查。
- **不看数据**：只走设计，没有实例、没有 viba data。
- **不扫内置目录**：`viba/` 下那些定义（标量、容器、算子）由内置词表回答，目录里的文件不读；
  要用内置目录里的模块，得自己在 `library` 或 `viba_paths` 里交出来。
- **裸的 `{...}`**：没有名字可以放进 `terminators`，带它的设计总是 `false`。

## 7. 测试

[`tests/test_is_complete.py`](../../tests/test_is_complete.py)：叶子与单位元、和与积、泛型决断、命名、
`terminators`、递归，以及 `.viba` 用例文件。

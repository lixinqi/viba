# check_convertable_to_json_schema：这份类型能写成 JSON Schema 吗

知识库里的类型要写成 JSON Schema，交给只认 JSON 的工具（校验器、模型的 tool schema、审计脚本）。
schema 要写得准，前提是每一种 JSON 值只落到类型的一支：JSON 只有六种东西（null、布尔、数字、字符串、
数组、对象），一个和的两个选项如果在这六种里分不开，schema 也就有歧义。这个判定查的就是这件事——这份
设计能不能写成一份没有歧义的 JSON Schema。

```viba
Good = $a int | $b int     # 能写成 schema：两支各带一个 tag

B = int
C = int
Bad = B | C                # 写不准：两支都是数字
```

## 1. 类型

```viba
CheckConvertableToJsonSchema =
    Result[void]
  <- $viba_path list[str]
  <- $entry ($module_name ImportableTypeName | $type_names list[ImportableTypeName])
  <- $strict ((bool | nil) * { 默认为 False，表示允许给定 module_name 内部可以包含 <- 和 <<，函数内部会选择性忽略他们 })
  <- Hint[{
      $entry 给一个模块名：查这个模块里每个定义；给一组 type 名：查这几个定义。
      从定义出发把类型走一遍，每个和的选项要能和 JSON 互转：写出去有表示，读回来要么带不同的 tag，
      要么 JSON 凭种类分得开（null、布尔、数字、字符串、数组、对象）。数字不分 int 和 float；
      一个值和同一种类的类型也分不开（7 和 int）。产品那一侧用 check_tag_and_inline，先查。
      函数、Any、...、调用没有 JSON 表示；泛型参数没有实参就没有表示。
      `{...}` 和 `Hint[{...}]`、`Assert[{...}]`、`Appendix[{...}]` 一样是单位元：不算成员、不算选项，单独出现读成 null。
      $strict 给 True：函数（`<-`）和调用（`<<`）不跳过，按规则拒掉；
      给 nil 或 False（默认）：模块里的这两种定义跳过，剩下的照常判。
      答 Ok(nil)；不成立就答 $err，说的是第一处。
    }]

ImportableTypeName =
    str * Assert[{ 形如 xxx.yyy 等可被 import 的字符串 }]
```

## 2. 两个入口

`$entry` 是两支合一：给模块名，查这个模块里的每个定义；给一组 type 名，查这几个定义。type 名是
`模块.定义`，模块按 `$viba_path` 找——和 `import` 找到它的是同一张表。

```viba
ByModule = check_convertable_to_json_schema
  << $viba_path ListLiteral["kb"]
  << $entry ($module_name "report")
```

```viba
ByTypes = check_convertable_to_json_schema
  << $viba_path ListLiteral["kb"]
  << $entry ($type_names ListLiteral["report.Report"])
```

上面是按签名写；Python 里是这样（[`viba/check_convertable_to_json_schema.py`](../check_convertable_to_json_schema.py)）：

```python
from viba.check_convertable_to_json_schema import check_convertable_to_json_schema

check_convertable_to_json_schema(["kb"], ("module_name", "report"))
check_convertable_to_json_schema(["kb"], ("type_names", ["report.Report"]))
```

答 `Ok(None)`（互转都行）、或者一个 `Err(...)`（一句话，说的是第一处不成立的地方）。

第三个实参 `strict` 说函数（`A <- B`）和调用（`f << ...`）怎么处理：

- `None` 或 `False`（默认）：模块里的这两种定义跳过——程序模块只看它的数据类型；
- `True`：不跳过，按规则拒掉（`a function is no JSON value`）。

```python
check_convertable_to_json_schema(["kb"], ("module_name", "demo"), strict=True)
```

点名给的类型不受这个开关影响：给一个函数类型，两种模式都答错。

## 3. 规则

写出去这一头，每条类型都要有 JSON 表示（第 4 节那些没有）；读回来这一头，一个和要读得回来，
当且仅当它的选项两两分得开：

- **tag**：带 tag 的选项读成一个对象，键就是那个 tag。两个选项的 tag 一样就分不开；tag 不同就分得开。
- **种类**：不带 tag 的选项，全靠 JSON 的种类分开——null、布尔、数字、字符串、数组、对象。两支种类
  一样就分不开。
  - 数字不分 `int` 和 `float`：`int | float` 读不回来，两支都是数字。
  - `list[T]`、`set[T]`、元组都读成数组；积、`dict` 都读成对象。
- **字面量**：同种类的字面量凭值分开（`"a" | "b"` 可以）；字面量和同种类的类型分不开（`7 | int` 不行，
  值 7 两边都说得通）。
- **摊平了算**：一个选项是名字、走到的是一串和（`A = B | C`，`B`、`C` 又是别的），摊平了跟外面的
  选项一起算；名字走到的是别的种类，就按那个种类算。
- **产品那一侧**：摊开之后一个积里没有重复的 tag、内联链不会绕回来——这是 `check_tag_and_inline`，本判定先拿同一份设计
  跑一遍；它不过，就直接答它的错。
- **积读成什么**：积读成对象；单位元成员（`Object`、`nil`、`Hint[{...}]`）不算成员——一个不剩读成
  null，只剩一个不带 tag 的成员时读成那一个（`"S0" * Hint[{...}]` 就是一个字符串常量）。
- **提示就是单位元**：`{...}` 和 `Hint[{...}]`、`Assert[{...}]`、`Appendix[{...}]` 是一回事，都不带值；
  它单独出现时读成 null。

## 4. 读不成 JSON 值的

这些不进“和”的比较，直接答错：

- **函数**（`A <- $b B`）：JSON 没有函数。
- **`Any`**：没有表示，读回来是一份没有约束的值。
- **`...`**：开着的和，不知道以后还会加哪些选项。
- **调用**（`f << ...`）：没有表示。
- **没有实参的泛型参数**：还没有表示。

## 5. 例子

读得回来的：

```viba
A = $a int | $b int                  # 两个 tag
B = int | str                        # 数字对字符串
C = $a list[int] | $b dict[str, int] | nil
D = $a int | ($b str | $c bool)      # 摊平了三个 tag
E = "a" | "b"                        # 同种类，凭值分开
F = list[str] * { 说明 }             # 提示是单位元：就是 list[str]
```

读不回来的：

```viba
A = B | C                            # B = int，C = int：都读成数字
N = int | float                      # 都读成数字
L = 7 | int                          # 7 两边都说得通
Same = "x" | "x"                     # 同一个值
Objects = Object * $x int | Object * $y int   # 都读成对象
```

## 6. 错误

答 `$err`，说的是第一处：

- 名字（模块或定义）按 `$viba_path` 解析不到；
- 泛型的决断在源码里做不出来；
- 类型读不成 JSON 值（上一节那些）；
- 一个和里两个选项读成同一个东西；
- `strict` 不是 bool 或 nil；模块里没有可判的类型（全是函数和调用，`$strict` 没开）。

## 7. 这一版不做的

- **值域的约束**不管：提示块里写的前提（`{ n 大于 0 }`）不进判定；这里只回答"JSON 值能不能落到某一
  支"。
- **tag 和字段名撞名**不查：带 tag 的选项和对象选项之间，靠“对象里那个键是不是 tag”分开；一个对象
  选项的字段名恰好和某个 tag 同名时，这一版不报。
- **集合的顺序**不管：`set[T]` 和 `list[T]` 在 JSON 里都是数组。

# Viba 编写规范

[`README.md`](README.md) 是语言参考：语法、算子、例子。这一份讲**怎么把定义写对、写清楚**——写给
写 viba 的人，也写给照着写 viba 的 agent。三段：先想清楚结构，再落笔，最后自检。

## 1. 结构能说的，不要留给断言

能由**类型结构**保证的事（有哪几个字段、什么类型、是不是可重复、和有几个分支），就写在结构里；
不要退化成一句提示（`Assert[{...}]`、`{...}`）或一个运行时判断。

`Assert` / `Hint` / `Appendix` 这类名字在类型层是**单位元**：`Assert[{no cycles}]` 只是挂了一段
文字，判定会把它当单位跳过去，它约束不了任何东西。典型的多余写法是给"叶子"再加一个字段：

```viba
# 叶子就是 children 为空——不需要一个 $is_leaf bool 去追着结构跑
Node =
  Object
  * $children list[Node]
```

结构说不了的**全局**不变量（不成环、单父、整体连通）不要散成断言，写成一份**规则程序**，让
`is_compliant` 判：规则读呈证、答 `bool`（见 [`viba-compliance.md`](viba-compliance.md)）。

## 2. 只被用一次的中间类型不要造

只被引用一次、又不承载独立语义的包装类型一律摊平到使用处：别名、单字段 `Object`、`XxxRequest`。
判据两条：**有没有第二处复用？名字有没有提供额外信息？** 都没有就别造。

函数参数尤其不要包成一个请求类型——读者得来回跳。参数直接列在函数链上：

```viba
# 不要 RegroupRequest = Object * $id Id * $target Target
#     Regroup = Result <- $req RegroupRequest <- { ... }
Regroup =
    Result
  <- $env Environment
  <- $id Id
  <- $target Target
  <- { move the node that id names under target }
```

## 3. 递归结构长成它该有的样子

- **子节点是 `list[T]`，不是单个 `T`**：单个孩子把树退化成了链表，将来要加第二个孩子就得改结构。
- **叶子由 `list` 自己的 `nil` 分支表达**：空列表就是那个 `nil`，不要再去元素类型里写 `| nil`
  （那会把"这一层没有孩子"和"孩子这个值可以缺失"混成一件事）。

```viba
List[T] =
  Oneof
  | Object
    * $head T
    * $tail List[T]
  | nil
```

## 4. 概念落在问题域，别名要诚实

- **先确认是真实概念，再建模。** 问题域里真实存在的术语直接用；你自己拍出来的抽象结构是**设计
  选择**，不要写成领域事实。分清"领域共识"和"我的推断"。
- **别名要诚实。** `UserId = int`、`StateName = str` 能让字段自解释、将来能收紧成枚举；但它在类型
  系统里与裸 `int` / `str` **等价**，并不阻止错填。别为了对称硬造一对没有区分度的别名。

## 5. 标签就是语义

积的每个成员、函数的每个实参，都写 tag；tag 要**自解释**——读 tag 就知道装的是什么，不要留一个
含糊的短名让人猜：

```viba
Point = $x float * $y float
Handler = Response <- $request Request
```

不要写成 `Point = float * float`（读的人不知道哪个是横、哪个是竖）、`Handler = Response <-
Request`（这个参数是什么）或 `$d int`（哪个 d？）。

tag 写在链头时，它命中的是**第一个参数的成员**：`$sub_env << environ << "x"` 就是
`environ.sub_env << environ << "x"`（见 [`viba-interpreter.md`](viba-interpreter.md)）。tag 不是值，
`method = $sub_env` 这种写法编不过。

不带标签的成员只留给"本来就没有名字"的东西：

- **元组**：`(A, B)`——顺序本身就是语义；
- **空**：`nil` 这样的分支（这一支什么都不是）；
- **裸值分支**：`int | str`——按值的种类区分的叶子。

## 6. 表达式里的绑定：`:=`

一个表达式想要中间量，就在括号里写绑定：绑定在前、结果最后，结果就是整个括号的值。

```viba
Distance =
  (
    dx := sub << $env environ << $a x1 << $b x2
    dy := sub << $env environ << $a y1 << $b y2
    add
    << $env environ
    << $a (square << $env environ << $x dx)
    << $b (square << $env environ << $x dy)
  )
```

六条规矩：

- **只在表达式的右边**。它是表达式，不是语句：顶层写 `X := int` 是错的（定义只有 `=`），
  `A = (x := 3  x)` 才对。
- **一条绑定一行**。只有一个绑定的块可以挤在一行（`A = (a := 7  a)`）；两条以上就摊开，
  每条一行，结果最后一行——一行里连着好几个 `:=`，读的人得自己去找块从哪开始、到哪结束。
- **它是计算的写法，不是类型的写法**。值要被算出来的地方都能用（定义体由解释器读的时候也一样）；
  类型层只判类型，读到绑定块答 `a binding is computation, not a type`。类型里要起中间名，那是
  另一条定义。
- **绑定按书写顺序求值，用不用都算**（像 `let`，不像懒参数）；名字可以遮蔽外层的定义、import、
  甚至 `environ`，遮蔽只在这个括号里有效。
- **名字出不去**。块结束后 `x` 就不存在了；块的结果、以及块交给别人的东西（按需的实参的 getter）
  仍然看得见它。
- **可序列化数据里不能放绑定**。`$point (p := …  p)` 这种写法会被拒："a binding belongs in a value, not
  inside viba_data"——可序列化数据只按写下来的样子读，里面不跑表达式。要让可序列化数据里出现一个算出来的值，就让一个函数
  把它答出来。

同一份定义里可以套多层：内层看得见外层，内层的遮蔽不影响外层。

## 7. 一行不写头，多行块头在第一行

`Object` 是积的单位元（就是 `nil`），`Oneof` 是和的单位元（就是 `never`）。写不写都是同一个类型
（`Object * $x int` 与 `$x int` 互为子类型），所以这是**版式**上的事：

- **一行**：不写头。整份定义一眼看全，头是多余的：写 `Parent = nil | $parent_of Node`，**不是**
  `Parent = Oneof | nil | $parent_of Node`。
- **摊成多行块**：头写第一行。块的第一行就是"这一块是积还是和"的说明书；块以字段或分支开头，等于
  让读的人自己去猜。

```viba
Option[T] = $some T | nil
Config = $mode "fast" * $threads 42 * $ratio 3.14
Parent = nil | $parent_of Node
UserId = int

Entry =
  Object
  * $name str
  * $state State

Result =
  Oneof
  | $ok Entry
  | $missing UserId
```

链**内部**的单位仍写 `nil` / `never`——`Object` / `Oneof` 只是块头的拼法。**指数永远不带头**：
它的第一个元素就是结果，每个实参带自己的 tag。

递归结构是**类型**在指自己，不是执行在绕回来：`List[T]` 那样写没问题。执行上的递归（函数又调到
自己）在一份文件里不出现——文件里的定义不许绕回自己，见 [`viba-interpreter.md`](viba-interpreter.md)。

## 8. 一支不成和

和的链只有一支时，头和这一层都是多余的：`Oneof | T` 就是 `T`，`Oneof | $x T` 就是 `$x T`。所以：

- 要表达"二选一"，至少两支：`Result = $ok Entry | $missing UserId`；
- 只有一种情况就别写和：直接写那个成员，或者（想给它一个名字）写成一个带标签的积
  `Box = Object * $value T`。

**也别把单个 `Object` 再包进 `Oneof`**：`Oneof | Object * $x int * $y int` 与
`Object * $x int * $y int` 互为子类型，多包一层只是让读者多找一次"和在哪"。

积这边有个对称的坑：**不带标签的单成员不是那个成员**。`Object * int` 与 `int` 不是同一个类型
（前者是"一个位置成员的积"），而且读不出这个成员是什么——这正是第 5 条要求成员带 tag 的原因。

## 9. 容器：`list` / `set` / `dict`

三个容器是内建的，用法就三件事：**写法、字面量、寻址**。

```viba
Items = $items list[int]
Tags = $tags set[str]
Table = $table dict[str, int]
Nested = $nested list[dict[str, int]]
```

- **写法**：`list[T]` 有序、按位置寻址；`set[T]` 无序（枚举顺序由实现定）；`dict[K, V]` 按键寻址。
  键写 `str`——访问协议里按键那一步就是 `$at_key str`。
- **没有 `repeated` 这回事**：一个字段可重复就把类型写成 `list[T]`，不要再包一层。
- **字面量有自己的写法**：`ListLiteral[1, "x"]`、`SetLiteral[1, 2]`、`DictLiteral[("k", 1)]`，
  空的是 `ListLiteral[]`。它们是 `list[...]` 之类的居民，写在**实例**那一侧（呈证、Prepare）。
- **可以任意嵌套**：`list[dict[str, int]]`、`dict[str, list[$x int]]`。
- **这三个名字（连 `ListLiteral` / `SetLiteral` / `DictLiteral`）不能拿来定义**：定义名和泛型形参
  里出现它们，解析器当场拒；它们只在类型表达式里出现。

## 10. 函数：结果在前，实参带 tag，提示收尾

```viba
Distance =
    int
  <- $env Environment
  <- $a int
  <- $b int
  <- { how far apart the two were }
```

- 链的第一个元素是**结果**，然后是实参，按给的顺序排；每个实参带 tag。
- **一行放不下就折行**：每个实参占一行，续行以 `<-` / `<<` 起头。可打开的例子：
  [`tests/data/interpreter/add_demo.viba`](tests/data/interpreter/add_demo.viba)、
  [`tests/data/let/let_in_taken_branch.viba`](tests/data/let/let_in_taken_branch.viba)。
- **实参是表达式时把它括起来**：`$v (f << $env environ)`，不是 `$v f << $env environ`——`<<` 比
  tag 松，后者会被读成两个实参（先给 `$v f`，再给一个位置实参）。
- **实参要装得下那个参数**：`$a int` 那里写 `"x"` 是程序写错，`interpret` 当场报
  `VibaProgramErr`（不是等宿主崩了再报）。所以参数的类型写准，不要用 `Any` 顶替——写着 `Any`
  就等于说"这个参数什么都收"。
- **没有死参数**：一个实参不参与任何判断、不影响结果，就删掉它——读的人会一路找它在哪生效，找不到
  就是浪费。**唯一例外是 `$env Environment`**：它对 interpreter 是规矩（每个可执行函数都要依赖
  environ），不是死参数。
- `{...}` 是**提示**：写给照着它补实现的人（常常是 agent），说这一步要干什么。它不是参数，也不被
  执行——`<<` 会把说明跳过去。所以提示里可以写清"读哪些字段、怎么比、对不上返回什么"（顺着类型能
  一路读出字段路径），但不要指望它是代码。
- **可能会用不上的那个参数，写成 `CalledByNeed[T]`。** 标的是**实参**，不是函数：那个参数的实参不先算，
  宿主拿到的是"要用的时候再来拿"的东西，叫了才算——if/else 真正只走一支靠的就是它
  （见 [`viba-interpreter.md`](viba-interpreter.md)）。只在"确实有分支语义、且实参求值有代价或
  副作用"时用它；别的参数别标，标了就等于把求值时机的责任推给了每一个实现者。标记只改那个参数怎么
  给：判定层和描述符层读到的类型就是 `T`。**按需的那个参数存不进闭包**（实参没算过、不是可序列化数据），
  同一个调用里其它的实参照旧可以先给——`<<` 的偏应用因此只在那个参数上受限。

## 11. 命名、注释、提示说同一件事

- **函数名就是它真正做的事。** 名字、提示、将来的实现三者指向同一个动作；名字叫"拆分"、提示却在
  说"挂节点"这种相反的事，是错。
- **注释只讲语义，不写语法絮叨。** 不写"这是 product 块""head=Object""exponent 链"这类对读者没有
  价值的话；注释回答"这个类型／字段在问题域里是什么"，不回答"它在 DSL 里怎么拼"。

## 12. 其它

- **内建名字不用 import。** [`viba/builtin.viba`](viba/builtin.viba) 对每个模块可见（最低优先级，自己
  模块的同名定义优先）：`Environment` / `environ`、`Object` / `Oneof` / `nil` / `never` / `Any`、
  标量 `bool` / `int` / `float` / `str`、容器名 `list` / `set` / `dict`、以及按需标记
  `CalledByNeed`。所以 `$env Environment` 那个参数和 `CalledByNeed[...]` 都不需要写任何 import。
- **内建标量是 `bool` / `int` / `float` / `str`。** `string` 不是内建名：它什么都解析不到，而且
  语法层不会报——类型层会（`module_get_type`、`is_sub_type`）。
- **一份定义一个表达式**，各自一行：没有逗号，也没有语句分隔符。
- **有 `__ret__` 才是程序**：没有它的文件是类型，不是程序，跑它是错。
- **模块要收参数就声明 `__args__`**，写成一份积类型，成员就是实参，顺序就是给的顺序：

```viba
__args__ =
  Object
  * $a int
  * $b int

args = __args__
```

  调用时每个成员一个 `<<`（`square_sum << environ << 3 << 4`，或者按 tag 给 `<< $a 3`）。
  **给了环境就是要执行，那时必须给全**，少一个当场报错；不给环境的话它就是闭包，可以先存下来
  （`sg = square_sum << 3 << 4`），以后再 `sg << environ` 执行。零实参的模块执行时不用写 `()`
  （`lib << environ` 就是执行），`lib << ()` 只在要显式写出那份空实参时才写。模块体里 `args.a`
  是那个实参。别用 `__args__` 装"可能有也可能没有"的东西——那些是分支（第 8 条），不是实参。

## 13. 写完怎么查

语法检查器就是解析器，一次调用：

```python
from pathlib import Path
from viba import viba_ast

viba_ast.parse(Path("store.viba").read_text())   # SyntaxError: what is wrong, and which line
```

语法不认识的字符、没闭合的代码块、拿内建容器当定义名，都会报，而且一定给出行号。类型本身另有一份
检查（一个积里 tag 不重复、内联链要摊到底）：

```python
from viba.check_tag_and_inline import check_tag_and_inline
from viba.viba_type_descriptor import empty_pool, parse_viba_file, pool_add_file

pool = empty_pool()
file = parse_viba_file(pool, source, "store.viba", "store")
check_tag_and_inline(pool_add_file(pool, file.ok_value).ok_value)   # Ok(nil), or what is wrong
```

语言自己的用例是两条命令：

```bash
python -m viba.parser     # 语法：131 份能编过、13 份必须编不过
python -m viba.viba_ast   # 打印器写出来的，再解析回去是同一棵树
```

## 14. 照着写一份

这一节给出一份完整的定义，用到本章各节的规矩：叶子、一行不带头、块带头、函数、容器字段。它同样由
`python -m viba.parser` 检查。

```viba
UserId = int
State = "draft" | "live"          # told apart by the value: no tags to give

Parent = nil | $parent_of Node

Index = $by_name dict[str, UserId]

Entry =
  Object
  * $name str
  * $state State

Node =
  Object
  * $entry Entry
  * $parent Parent

Result =
  Oneof
  | $ok Entry
  | $missing UserId

Lookup =
    Result
  <- $env Environment
  <- $id UserId
  <- { find the entry that id names }
```

## 提交前自检

- [ ] 这个约束是结构能保证的，还是我又写了一句没人执行的断言？
- [ ] 这个类型是不是只被用了一次的中间包装（别名、单字段 `Object`、`XxxRequest`）？
- [ ] 递归结构的 children 是 `list`，还是退化成了单个 `T`？
- [ ] 叶子是 `list` 的 `nil` 分支，还是我在元素类型上又写了 `| nil`？
- [ ] 一行有没有多写头？多行块有没有漏头？
- [ ] `Oneof` / `Object` 用对了，没把单个 `Object` 包进 `Oneof`？
- [ ] 字段 tag 自解释吗？函数实参都带 tag 吗？
- [ ] 表达式里的中间量是就地 `:=` 绑定的吗？两条以上有没有一条一行？
- [ ] 绑定有没有写进可序列化数据里？
- [ ] 模块收参数是声明了 `__args__`（积类型），还是把参数藏进了模块名里？调用给全了吗？
- [ ] 注释有没有混进"这是 product 块"这类语法絮叨？
- [ ] 这个概念是问题域真实的，还是我臆测的？别名有没有装作它能阻止错填？
- [ ] 有没有死参数（`$env` 除外）？
- [ ] 函数名、提示、实现说的是不是同一件事？

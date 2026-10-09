# Viba 定义规范

[`README.md`](README.md) 是语言参考：语法、算子、例子。这一份讲**一份定义怎么才叫对、才叫清楚**——
给做出这些定义的人看。三段：先想清楚结构，再落笔，最后自检。

## 1. 结构能说的，不要留给断言

能由**类型结构**保证的事（有哪几个字段、什么类型、是不是可重复、和有几个分支），就放进结构里；
不要退化成一句提示（`Assert[{...}]`、`{...}`）或一个运行时判断。

`Assert` / `Hint` / `Appendix` 这类名字在类型层是**单位元**：`Assert[{no cycles}]` 只是挂了一段
文字，判定会把它当单位跳过去，它约束不了任何东西。典型的多余形式是给"叶子"再加一个字段：

```viba
# 叶子就是 children 为空——不需要一个 $is_leaf bool 去追着结构跑
Node =
  Object
  * $children list[Node]
```

结构说不了的**全局**不变量（不成环、单父、整体连通）不要散成断言：一句挂在类型上的文字约束不了
任何东西，而这类不变量类型层也表达不了 —— 要检查它们，只能是另一份程序，在宿主那一侧跑。

## 2. 只被用一次的中间类型不要造

只被引用一次、又不承载独立语义的包装类型一律摊平到使用处：别名、单字段 `Object`、`XxxRequest`。
先问两条：**有没有第二处复用？名字有没有提供额外信息？** 都没有就别造。

函数参数尤其不要包成一个请求类型——看的人得来回跳。参数直接列在函数链上：

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
- **叶子由 `list` 自己的 `nil` 分支表达**：空列表就是那个 `nil`，不要在元素类型里再放 `| nil`
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
  选择**，不要落成领域事实。分清"领域共识"和"我的推断"。
- **别名要诚实。** `UserId = int`、`StateName = str` 能让字段自解释、将来能收紧成枚举；但它在类型
  系统里与裸 `int` / `str` **等价**，并不阻止错填。别为了对称硬造一对没有区分度的别名。

## 5. 标签就是语义

积的每个成员、函数的每个实参，都带 tag；tag 要**自解释**——看 tag 就知道装的是什么，不要留一个
含糊的短名让人猜：

```viba
Point = $x float * $y float
Handler = Response <- $request Request
```

不要落成 `Point = float * float`（看的人不知道哪个是横、哪个是竖）、`Handler = Response <-
Request`（这个参数是什么）或 `$d int`（哪个 d？）。

tag 出现在链头时，它命中的是**第一个参数的成员**：`$sub_env << args.env << "x"` 就是
`args.env.sub_env << args.env << "x"`（见 [`viba-interpreter.md`](viba-interpreter.md)）。tag 不是值，
`method = $sub_env` 这种源码形式编不过。

tag 也可以在**字符串**里：`tagged["hello"] << X` 就是 `$hello << X`，`tagged["a", T]`
就是 `$a T` —— 名字本来就是字符串的时候这样用，见 [`viba-pattern.md`](viba-pattern.md) 第 3 节。

**点号取的是成员，定义左边只有一个名字。** 定义永远是 `A = …`：`a.b = A` 编不过。`a.b` 是在 `a`
上取成员 `b` —— 模块的成员、积的成员、泛型应用选中的那个模块的成员，都是一回事：

```viba
WeightPlan = $steps Step * $cost Cost
Handler = Response <- $env Env <- $request Request
```

一块 API 就是几个并列的定义，用的时候拿模块名一段一段取（`mooncake.reshard.weight.plan_storage_move`
这样一串只出现在**取**的时候：`mooncake` 是模块，后面每一段都是在上一段上取成员）。
**同一个名字不许出现两次**：一个模块里把同一个名字定义两次是程序错误。

不带标签的成员只留给"本来就没有名字"的东西：

- **元组**：`(A, B)`——顺序本身就是语义；
- **空**：`nil` 这样的分支（这一支什么都不是）；
- **裸值分支**：`int | str`——按值的种类区分的叶子。

## 6. 一行不带头，多行块的头放在第一行

`Object` 是积的单位元（就是 `nil`），`Oneof` 是和的单位元（就是 `never`）。带不带头都是同一个类型
（`Object * $x int` 与 `$x int` 互为子类型），所以这是**版式**上的事：

- **一行**：不带头。整份定义一眼看全，头是多余的：用 `Parent = nil | $parent_of Node`，**不是**
  `Parent = Oneof | nil | $parent_of Node`。
- **摊成多行块**：头放在第一行。块的第一行就是"这一块是积还是和"的说明书；块以字段或分支开头，等于
  让看的人自己去猜。

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

链**内部**的单位仍用 `nil` / `never`——`Object` / `Oneof` 只是块头的拼法。**指数永远不带头**：
它的第一个元素就是结果，每个实参带自己的 tag。

递归结构是**类型**在指自己，不是执行在绕回来：`List[T]` 那样定义没问题。执行上的递归（函数又调到
自己）在一份文件里不出现——文件里的定义不许绕回自己，见 [`viba-interpreter.md`](viba-interpreter.md)。

## 7. 一支不成和

和的链只有一支时，头和这一层都是多余的：`Oneof | T` 就是 `T`，`Oneof | $x T` 就是 `$x T`。所以：

- 要表达"二选一"，至少两支：`Result = $ok Entry | $missing UserId`；
- 只有一种情况就别用和：直接用那个成员，或者（想给它一个名字）做成一个带标签的积
  `Box = Object * $value T`。

**也别把单个 `Object` 再包进 `Oneof`**：`Oneof | Object * $x int * $y int` 与
`Object * $x int * $y int` 互为子类型，多包一层只是让看的人多找一次"和在哪"。

积这边有个对称的坑：**不带标签的单成员不是那个成员**。`Object * int` 与 `int` 不是同一个类型
（前者是"一个位置成员的积"），而且看不出这个成员是什么——这正是第 5 条要求成员带 tag 的原因。

## 8. 容器：`list` / `set` / `dict`

三个容器是内建的，用法就四件事：**源码形式、字面量、寻址、问在不在**。

```viba
Items = $items list[int]
Tags = $tags set[str]
Table = $table dict[str, int]
Nested = $nested list[dict[str, int]]
```

- **源码形式**：`list[T]` 有序、按位置寻址；`set[T]` 无序（枚举顺序由实现定）；`dict[K, V]` 按键寻址。
  键用 `str`——访问协议里按键那一步就是 `$at_key str`。
- **没有 `repeated` 这回事**：一个字段可重复就把类型落成 `list[T]`，不要再包一层。
- **字面量有它自己的源码形式**：`ListLiteral[1, "x"]`、`SetLiteral[1, 2]`、`DictLiteral[("k", 1)]`，
  空的是 `ListLiteral[]`。它们是 `list[...]` 之类的居民，出现在**实例**那一侧：放在值的位置上就是那个
  容器本身，所以能数、能按地址取、能当实参交给宿主、也能原样序列化回去。
- **寻址用 `$__getitem__`**：位置给 int，键给 str，两者各对应一条地址步子
  （`$at_index int` / `$at_key str`）。`$__getitem__ << xs << 1` 是 `xs[1]`，
  `$__getitem__ << table << "k"` 是 `table["k"]`；地址还没给时它是一个值，给了才走这一步取它。
  按名字取成员是另一个：`$__getattr__`（见 [`viba-interpreter.md`](viba-interpreter.md)）。
- **方括号是它的简式**：`xs[1]` 就是 `$__getitem__ << xs << 1`，`table["k"]` 就是
  `$__getitem__ << table << "k"`，容器可以是一条名字路径（`box.xs[1]`）。方括号只跟在名字后面，
  一层：`xs[0][1]` 要落成 `$__getitem__ << xs[0] << 1`。放在**值**的位置上它是这一步的取用；
  放在积的成员里时它跟别的成员一样是**源码里的数据**（不在这里取它，`viba-interpreter.md`）。
- **问在不在用 `$__in__`**：`$__in__ << xs << 3` 问元素在不在（list / set / tuple），
  `$__in__ << table << "k"` 问键在不在（dict）。答的是 `bool`；被问的不是容器、或者给 dict 的不是
  一个字符串键，都是程序错。
- **可以任意嵌套**：`list[dict[str, int]]`、`dict[str, list[$x int]]`。
- **这三个名字（连 `ListLiteral` / `SetLiteral` / `DictLiteral`）不能拿来定义**：定义名和泛型形参
  里出现它们，解析器当场拒。它们出现在类型表达式里是「这个容器的居民」，出现在值的位置上是那个容器。

## 9. 函数：结果在前，实参带 tag，提示收尾

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
  [`tests/data/closure/closure_with_module_call_argument.viba`](tests/data/closure/closure_with_module_call_argument.viba)。
- **实参是表达式时把它括起来**：`$get_v (f << $env args.env)`，不是 `$get_v f << $env args.env`——`<<`
  比 tag 松，后者会被判成两个实参（先给 `$get_v f`，再给一个位置实参）。**函数体是一次调用时也要括**：
  `<- (f << $env args.env)`，不然那次调用会被当成又一个参数。
- **参数的类型给准**：一次运行不核实参的类型（`$a int` 那里放 `"x"` 就照给，谁拿它做整数运算谁
  在那里崩），拦得住的是判定层 —— 把这份文件当类型判、给一个函数链取实参时要求 `3 <: int`。
  所以不要用一个说得清的类型去换 `Any`：是 `Any` 就等于说"这里什么都收"，判定层也就帮不上忙。
- **没有死参数**：一个实参不参与任何判断、不影响结果，就删掉它——看的人会一路找它在哪生效，找不到
  就是浪费。**唯一例外是 `$env Env`**：它对 interpreter 是规矩（每个要执行的函数都要依赖环境），
  不是死参数——所以它也不进类型：给一个函数的类型时不要这一个参数，它只出现在链上。
- **结果不是环境**：`Env` / `Environment` 只出现在参数位置上。只有内建函数给得了一个环境，
  别的函数在结果那里用它就当场报错；`__impl__ = args.env` 是值，声明是 `Any`。
- `{...}` 是**提示**：说这一步要干什么。它不是参数，也不被
  执行——`<<` 会把说明跳过去。所以提示里可以讲清"取哪些字段、怎么比、对不上返回什么"（顺着类型能
  一路取出字段路径），但不要指望它是代码。
- **可能会用不上的那个参数，用函数类型 `(T <- $env Environment)`。** 给的是**那个参数的类型**，
  不是函数：那个参数是惰性的（lazy，non-strict），实参不在这里求值，宿主拿到的是源码里的那次调用，
  叫了才求——if/else 真正只走一支靠的就是它（见 [`viba-interpreter.md`](viba-interpreter.md) 的
  「求值策略：按需求值（call-by-need）」一节）。宿主叫它时可以带上自己的实参
  （`f(env, 1, 2)`），源码里的那次调用按这些实参走完。只在"确实有分支语义、且实参
  求值有代价或副作用"时这样用；别的参数按值求值（call-by-value），给它们用函数类型就等于把
  求值时机的责任推给了每一个宿主函数。想给一个求好的值，用 `builtin.echo` 把它包成"给它一个环境
  就给出 V"的那个函数。

## 10. 命名、注释、提示说同一件事

- **函数名就是它真正做的事。** 名字、提示、将来的实现三者指向同一个动作；名字叫"拆分"、提示却在
  说"挂节点"这种相反的事，是错。
- **注释只讲语义，不要语法絮叨。** 不要留"这是 product 块""head=Object""exponent 链"这类对看的人没有
  价值的话；注释说的是"这个类型／字段在问题域里是什么"，不是"它在 DSL 里怎么拼"。

## 11. 其它

- **内建名字不用 import。** [`viba/builtin.viba`](viba/builtin.viba) 对每个模块可见（最低优先级，自己
  模块的同名定义优先）：`Environment` / `Env`、`Object` / `Oneof` / `nil` / `never` / `Any`、
  标量 `bool` / `int` / `float` / `str`、容器名 `list` / `set` / `dict`，以及 `builtin` 的成员。
  所以 `Environment` / `Env` 和 `builtin.echo` 都不需要任何 import。
- **内建库里的模块与泛型，名字也一样可见。** `viba/` 里 `builtin.viba` 旁边那几个模块
  （`Y.viba`、`apply.viba`、`sub_env_run.viba`、`sequential.viba`），以及 `viba/builtin/` 下的那些泛型
  （`is_closure/`、`unclosure/`、`sequential_step/`、`sequential_arg/`）对每个模块
  可见，优先级同样最低（自己模块的同名定义、import 优先）：用 `Y << step << …`、
  `apply << f << args`、`sub_env_run << $sub_env_name "low" << env << f << …`、
  `is_closure[add << $a 1].value`、`sequential << $x (…) << $y (…)` 都不用 import，
  带前缀的 `builtin.sub_env_run` 与 `builtin.is_closure` 叫的是同一个。用 `import` 也照旧。
- **`builtin` 的成员是内建算子，两种源码形式是同一次调用。** `builtin.add << $env args.env << $x 3 << $y 4`
  和 `add << $env args.env << $x 3 << $y 4` 叫的是同一个成员：带前缀的是它的全名，不带前缀的那个
  是它对每个模块可见的短名字。每个成员一张签名，实现跟别的函数一样在宿主手里 —— `get_func` 收到的
  名字是带前缀的那个（`builtin.add`）。名字说它管哪个类型：`int` 不带后缀，`float` 是 `_f`，
  `str` 是 `_str`，`bool` 是 `_bool`；`x_to_y` 是从 `x` 换算到 `y`。
- **内建标量是 `bool` / `int` / `float` / `str`。** `string` 不是内建名：它什么都解析不到，而且
  语法层不会报——类型层会（`module_get_type`、`is_sub_type`）。
- **一份定义一个表达式**，各自一行：没有逗号，也没有语句分隔符。
- **定义是绑定，不是语句**：viba 是声明式的，求值按需求值（call-by-need）——一个绑定被用到的时候
  才求值，而且只求值一次，所以源码里的次序不是求值的次序，出现在用之后的定义照样算（见
  [`viba-interpreter.md`](viba-interpreter.md) 的「求值策略：按需求值（call-by-need）」一节）。
- **有 `__impl__` 才是程序**：没有它的文件是类型，不是程序，跑它是错。
- **模块要收参数就用 `__decl__`**：它是模块当函数判时的整条链，结果在前，参数在后；`$env Env` 至多
  一个。用不用它，调用时给环境的方式一样：一个 `<<` 给一个环境值；用了，环境就给这个参数，模块体用
  `args.env` 取；没用，环境不进门，只用来跑这次调用，再缀在这次调用返回的那次调用末尾：

```viba
__decl__ =
    int
  <- $env Env
  <- $a int
  <- $b int

args = __get_args__ << __decl__
```

  调用时每个参数一个 `<<`（`square_sum << args.env << 3 << 4`，或者按 tag 给 `<< $a 3`）。
  **给了环境就是要执行，那时必须给全**，少一个当场报错；不给环境的话它就是闭包，可以先存下来
  （`sg = square_sum << $a 3 << $b 4`），以后再 `sg << args.env` 执行。参数不用 `()` 那一套：
  一个参数都不收的模块，执行就是 `add_demo << args.env`。模块体里 `args.a` 是那个实参，`args.env`
  是环境。别用 `__decl__` 的参数装"可能有也可能没有"的东西——那些是分支（第 8 条），不是参数。

## 12. 定义给出之后怎么查

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
python -m viba.parser     # 语法：135 份能编过、15 份必须编不过，外加文档里的例子
python -m viba.viba_ast   # 打印器输出的源码，再解析回去是同一棵树
```

## 13. 照着定义一份

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

- [ ] 这个约束是结构能保证的，还是我又留了一句没人执行的断言？
- [ ] 这个类型是不是只被用了一次的中间包装（别名、单字段 `Object`、`XxxRequest`）？
- [ ] 递归结构的 children 是 `list`，还是退化成了单个 `T`？
- [ ] 叶子是 `list` 的 `nil` 分支，还是我在元素类型上又放了 `| nil`？
- [ ] 一行有没有多带头？多行块有没有漏头？
- [ ] `Oneof` / `Object` 用对了，没把单个 `Object` 包进 `Oneof`？
- [ ] 字段 tag 自解释吗？函数实参都带 tag 吗？
- [ ] 模块的参数都在 `__decl__` 里了吗（含那个 `$env Env`）？还是把参数藏进了模块名里？调用给全了吗？
- [ ] 注释有没有混进"这是 product 块"这类语法絮叨？
- [ ] 这个概念是问题域真实的，还是我臆测的？别名有没有装作它能阻止错填？
- [ ] 有没有死参数（`$env` 除外）？
- [ ] 可能会用不上的那个实参，那个参数用函数类型了吗？别的实参有没有被顺手也标上？
- [ ] 函数名、提示、实现说的是不是同一件事？

# viba 教程

这份教程从最小的定义讲到能跑的模块，假设你会 Python，不假设你见过 viba。

文中的 viba 代码块都是完整的源码，`python -m viba.parser` 会把它们编一遍：**编得过才放得进这份文件**。
要动手跑，看第 4 节那份宿主；那一节之后，例子都能直接放进文件里跑。

## 1. 一个定义就是一个类型

```viba
Option = $some int | nil
```

等号左边是名字，右边是类型表达式。`|` 是**和**：`$some int` 或 `nil` 其中一个。`$some` 是这一支的
**tag**，给的时候带上它，用的时候也按它找。`nil` 是"什么都没有"的那一支。

`*` 是**积**：几个东西同时都在。

```viba
Point = $x float * $y float
Config = $host str * $port int * $debug bool
```

不带 tag 的成员只留给"本来就没有名字"的东西：`(A, B)` 是元组，顺序本身就是意思；`int | str` 是按值
的种类分开的叶子。

定义只依赖源码，不依赖谁来实现它。所以一份文件可以先当类型看：看它有哪些成员、每个成员是什么类型、
一个名字从哪来（[`viba-reflect.md`](viba-reflect.md)）。

```python
from viba import viba_ast

tree = viba_ast.parse("Option[T] = $some T | nil")
print(viba_ast.unparse(tree))
```

## 2. 单位元与底：`Object`、`nil`、`never`、`Any`

- `Object` 是积的单位元：`Object * $x int` 与 `$x int` 是同一个类型。
- `Oneof` 是和的单位元（一个都没有的和）。
- `never` 是底：没有值属于它。函数实参是 `never` 时表示"这个参数谁来都装不下"。
- `Any` 是顶：谁都装得下。

```viba
Nothing = Oneof
Anything = Any
NoSuchField = Object * $x never
```

一个积里同一个 tag 出现两遍是错的，判定的时候报。

## 3. 函数：结果在前，实参在后，tag 落在实参上

```viba
Add = int <- $a int <- $b int
Handler = Response <- $request Request
```

`Add` 是一条链：链头是结果 `int`，后面每一个 `<-` 添一个参数。它要两个 `int`，给出来一个 `int`。

**每个可执行的函数都得拿到环境**：`__decl__` 里可以带 `$env Environment` 这个参数，也可以不带。
带不带，调用时都是用一个 `<<` 给一个环境值：带了，环境就给这个参数，模块体用 `args.env` 取到它；
没带，环境不进门（`args.env` 取不到），它只用来跑这次调用，再缀在这次调用返回的那次调用末尾。
环境说这一步在哪个环境里做：

```viba
add =
    int
  <- $env Environment
  <- $a int
  <- $b int
  <- { 把两个整数加起来 }
```

`{ … }` 是**提示**：说这一步要实现什么。它不是参数，也不被执行，
`<<` 会跳过去。

`<<` 是给实参。给函数一个实参就是**一次调用**；给全了就是结果：

```viba
Call = add << $env args.env << $a 1 << $b 2
```

实参可以按位置给，也可以按 tag 给（顺序随意）；按位置给时，落在源码里第一个还没给的参数上。

## 4. 一份能跑的文件：`__impl__` 与环境

文件里有 `__impl__` 的那份定义就是这份程序的结果；**没有 `__impl__` 的文件是类型，不是程序**，跑它报
`VibaProgramErr`。

```viba
# add_demo.viba
__decl__ =
    void
  <- $env Env

args = __get_args__ << __decl__

add =
    int
  <- $env Env
  <- $a int
  <- $b int
  <- { 把两个整数加起来 }

__impl__ =
    add
    << args.env
    << $a 999999
    << $b 1
```

一份文件是一个模块，它只有两种意思：模块语义是"一份定义的映射"，顶层每份定义都是它的成员，按名字取
（`foo_module.Bar`，泛型应用选中那份文件时也一样：`g[T].value`）；函数语义要一份签名加一份结果 ——
没有 `__decl__` 就没有函数语义，它只是模块。

`__decl__` 是这份模块当函数判时的整条链：结果在前，参数在后；`$env Env` 至多一个（`Env` 是内建
的名字，就是 `Environment`）：带了，环境就给这个参数；不带，环境不进门，随这次调用的结果往后走。
`args = __get_args__ << __decl__` 取回这次调用收到的那份实参：`args.env` 是环境本身。

viba 是**声明式**的，不是命令式执行的：一份文件里的定义是**绑定**（binding），不是按源码里的次序
一条一条执行的语句。求值由需求驱动（demand-driven），一个绑定被用到的时候才求值，求一次就把结果
留下来（memoization）。这叫**按需求值**（call-by-need），也就是惰性求值。所以 `args` 什么时候求出来，
看谁用到它；源码里的次序不是求值的次序。整条策略见 [`viba-interpreter.md`](viba-interpreter.md) 的
「求值策略：按需求值（call-by-need）」一节。

结果不能是 `Env`：只有内建函数（`viba/builtin.viba` 里 `Environment` 的成员）给得了一个环境，
别的函数用了当场报错——环境是调用的规矩，不是能交出去的值。运行时的结果不受这条限制：
`__impl__ = args.env` 交回去的还是那个环境，只是声明成 `Any`。

宿主给两样东西：结果存在哪（`EnvironmentStorage`，默认一个临时目录）和每一步的实现
（`EnvironmentCompute`，里面一个 `get_func(module_path, func_name)`）。`interpret` 取文件、跑起来
（它的 viba 签名在 [`viba-interpreter.md`](viba-interpreter.md) 开头）。

```python
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret

def get_func(module_path, func_name):
    if func_name == "add":
        return lambda env, a, b: a.value + b.value
    return None

environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))

answer = interpret("add_demo.viba", environ)
print(answer)                       # $ok 1000000
print(answer.by_tag("$ok").value)   # 1000000
```

模块在手里、不从文件取时用 `exec`：参数形式和 `interpret` 一样，只是第一份不是路径而是那份模块本身 ——
它的文本（`exec(那份文本, environ)`），或者已经带着它的那份 viba 数据（把停下那一支里的 `$call` 交给它：
`$call` 本来就是节点，不用序列化出去再反序列化回来）。交给它的主模块没有文件、也没有名字：`import` 只按环境的搜索路径找。
`interpret` 与 `exec` 回答的都是一份 viba 数据（第 9 节）。

宿主收到的东西分两种：**可序列化数据**到手上是实例（`a` 是节点，`a.value` 是那个数），**环境**到手就是
环境自己。源码里的名字、字面量、调用，宿主看到的都是实例，按 [`viba-reflect.md`](viba-reflect.md) 一步步取。

`interpret` 自己不带任何实现：一次运行能碰到的实现，全部来自 `get_func` 交出来的那一个。

## 5. 模块：文件是函数，调用要给环境

一个 `.viba` 文件就是一个模块，而模块也是函数：输入是环境，输出是 `__impl__`。跨文件调用要给它一个
属于这次调用的环境：

```viba
# main.viba
import add_demo as demo

ret = demo << (args.env.sub_env << args.env << "add_demo")

__impl__ = demo.print << args.env << ret
```

`import` 到哪儿找是环境的事（先 import 旁边、再 `viba_path` 按顺序、最后内建目录 `viba/` 与
`viba/builtin/`）；内建库里的模块与泛型不用 import —— `Y`、`apply`、`sub_env_run`、
`sequential`、`is_closure`、`unclosure` 直接用名字就是它（加了 import 也一样），它们就在搜索路径的
最后一站。

`args.env.sub_env << args.env << "add_demo"` 拿一个子环境：**它带着父级的 compute**，数据路径是
`root/add_demo`。它也可以落成 `$sub_env << args.env << "add_demo"`——tag 在链头时命中的是第一个参数
的成员，两种源码形式是同一次调用（第 8 节）。这个模块只收环境这一个参数，所以给环境就是执行。
那条路径就是这次调用的**数据路径**（它是环境带着的；宿主拿到的 `module_path` 说的是这一步声明在哪个
模块，`func_name` 是它在那个模块里的名字），而一条数据路径上只该有一次调用：
同一条路径上正在跑就是环（报错并把那条链放在话里）；已经处理过、又是**同一个模块**，就是同一个子
计算被问了第二次，把它处理过的那份交回去；已经处理过、却是**另一个模块**，那是两条调用挤一条数据路径，
宿主分不出谁是谁，也报错。

不想起名字就用 `tmp_env`：它每次给一个新的子环境（路径是 `root/tmp_<随机>`），也可以落成
`$tmp_env << args.env`。

```viba
add_demo_call = add_demo << (args.env.tmp_env << args.env)
```

模块要收实参，就把它们放进 `__decl__`（结果在前，参数在后），再用 `args = __get_args__ << __decl__`：

```viba
# square_sum.viba
__decl__ =
    int
  <- $env Env
  <- $a int
  <- $b int

args = __get_args__ << __decl__

__impl__ =
    add
    << args.env
    << (mul << args.env << args.a << args.a)
    << (mul << args.env << args.b << args.b)
```

调用时每个参数一个 `<<`：按位置给（`<< 3 << 4`）或按 tag 给（`<< $b 4 << $a 3`）都行；**环境给了，
参数就得齐**，少一个当场报错，话里点名少了谁。环境那个参数：不带 tag 的实参跳过它，所以 `<< 3 << 4`
给的是 `$a` 和 `$b`。

## 6. 不给环境，链就是一个值

```viba
half = add << $a 40
same = add << $a 1 << $b 2
__impl__ = same
```

`add << $a 40` 是一个**闭包**：源码里的函数名加上已经算好的实参。它能存、能传、能当 `__impl__`、能序列
化，也能**换一个环境再执行一次**（`same << (args.env.tmp_env << args.env)`）。"必须给全"只对执行
成立，对闭包不成立。

闭包里只能装可序列化数据（只能取、能序列化出去）；环境从来不装在闭包里，它是执行那一刻才给的。所以同一个
闭包可以反复执行，每次的执行环境由给它的那个环境决定。

给环境的那一刻才是执行：`interpret` 去问宿主哪一步来实现，并按这次调用的数据路径把它认下来。

## 7. 可能会用不上的那个实参：用函数类型

有些实参可能根本用不上，算它要花代价或者有副作用。把那个参数用**函数类型** `(T <- $env Env)`，
那个实参就不在这里算：源码里的那次调用连同环境一起交给宿主，宿主叫它的时候才算，在宿主给的那个环境
里算。`branch.viba` 的两个开关就是这么用的：

```viba
echo_or_never =
    Any
  <- $env Env
  <- $cond bool
  <- $get_v (Any <- $env Env)
  <- { 条件成立就交回 get_v，否则交回 never }

never_or_echo =
    Any
  <- $env Env
  <- $cond bool
  <- $get_v (Any <- $env Env)
  <- { 条件成立就交回 never，否则交回 get_v }
```

```viba
# main.viba
import branch

ge =
    bool <- $env Env <- $x int <- $y int <- { x >= y }
tick =
    int <- $env Env <- { 算它会有副作用 }
tock =
    int <- $env Env <- { 算它也有副作用 }

condition = ge << $env args.env << $x 1 << $y 0
__impl__ =
    Oneof
  | (branch.echo_or_never
      << $env args.env << $cond condition
      << $get_v (tick << $env args.env))
  | (branch.never_or_echo
      << $env args.env << $cond condition
      << $get_v (tock << $env args.env))
```

`$get_v` 那个实参是一次调用（`tick << $env args.env`）；宿主拿到的是一个能带着环境调它自己的东西，
不叫它，那个实参就不求值（惰性求值，lazy）。所以没走的那一支里的副作用不会发生，它没有实现
也不会挡路。

几条规矩：

- **只给那一个参数**。别的实参按值求值（call-by-value），函数的 `$env` 也照旧按值给；
- **最多求值一次**（memoization）。第一次问出结果（值或停下），之后每一次问都拿同一个，问两遍不会
  做两遍；
- **想给一个已经算好的值**，用 `builtin.echo` 把它包成"给它一个环境就给出 V"的那个函数：
  `$get_v (builtin.echo << $x v)`；
- **半成品调用也能放进去**。一个还欠实参的调用（`add << $a 40`）本身就是个值；放在那一个参数里，
  宿主拿到的是它代表的那次调用，补上欠的那些实参就得到结果。

条件、值和开关本身都可以在别的文件里，用例只给一条链去调它们。这条路子的边角案例见
[`tests/test_interpreter_branch_switch.py`](tests/test_interpreter_branch_switch.py)。

这一节说的是求值策略落在**实参**上的那一半（那个参数惰性，别的参数按值求值）；落在**定义**上的
另一半是第 4 节那条：定义是按需求值的绑定。两半合起来见
[`viba-interpreter.md`](viba-interpreter.md) 的「求值策略：按需求值（call-by-need）」一节。

## 8. 调用方法：链头带 tag

`$tag << X << a` 就是 `X.tag << X << a`：**tag 在链头时，它命中的是第一个参数的成员，而那个值也接着
被交给成员当第一个实参**。环境成员因此有两种源码形式，一样长：

```viba
child = args.env.sub_env << args.env << "add_demo"   # 点号，环境本身就是那个值
```

```viba
child = $sub_env << args.env << "add_demo"           # 同一个调用，短一些
```

`viba/builtin.viba` 里 `Environment` 的成员就是这样声明的——环境是它的第一个参数：

```viba
Environment =
    Object
  * $viba_path str
  * $sub_env (Env <- $env Env <- $sub_env_name str)
  * $tmp_env (Env <- $env Env)
```

tag 本身不是值，`method = $sub_env` 编不过：标签只有在链头、后面跟着第一个参数时才成立。第一个参数
是别的值也一样，`$tag` 命中的是它的成员；成员出现在数据里时，它收的第一个参数是它所在的那个值。

## 9. 一次执行给出什么

`interpret` 与 `exec` 的结果是一份 viba 数据 —— 声明里那个 `InterpretResult` 的一个节点
（定义在 [`viba/interpret_result.viba`](viba/interpret_result.viba)）：`$ok` 是 `__impl__` 的值，
`$err` 是这次执行停下的方式，按 tag 往下取就行（`viba-interpreter.md`「一次执行会得到什么」）。
停下的方式有四种：

- `$viba_program_err ProgramErr`：**这份程序或环境不行**——编不过、文件不在、没有 `__impl__`、`$env` 没给，
  或者一次执行回答的是环境（环境是这次调用的规则，不是值）。它不说"哪一步"，所以不带步名；`$stack` 是这次
  执行走过的调用链，一帧是一次调用点（`$file_path` 是哪个文件、`$lineno` 是第几行）。
- `$underlying_viba_op_err UnderlyingOpErr`：**这一步坏了**——实现抛了、给出了没有叶子的东西，
  或者 `get_func` 自己坏了。
- `$not_implemented_err UnderlyingOpErr`：**这一步没有实现**——`get_func` 那里没有它（返回 `None`，
  或者交回了这一支）。`interpret` 不带库函数，所以"没有实现"很正常，不是错误。
- `$environment_api_invalid_argument_err EnvironmentApiInvalidArgumentErr`：**环境上的一个 api 收不下给它的
  东西**——`Environment` 的成员（`sub_env`、`tmp_env`、`get_relative_path`……），或者宿主挂在环境上的那些，
  把这次调用退回来了。它是 viba 自己这一侧的活，`get_func` 从来没有被问过它们；`$api_name` 是哪一个 api
  （`Environment.sub_env`），`$args` 是给它的实参（环境不在里面）。

后两支的字段是同一个 `UnderlyingOpErr`，靠 tag 分开；`$msg` 一句话说清是什么事，开头就是原因（`no implementation`、
`get_func raised`、`raised`、`no leaf`）。宿主自己要说"这一步没有实现"，交回那份数据就行
（`viba.interpret.not_implemented()`），不抛异常：缺的字段由 run 用它知道的这次调用补上。这一支里还带着是哪一步（`$module_path` 与 `$full_qualified_func_name`）
和这次调用本身（`$call`：`__dyn_call__` 加上名字和源码里的实参，名字是数据，环境不在里面 ——
`__dyn_call__ << "demo.add" << $a 1 << $b 2` 就是 `demo.add << $a 1 << $b 2`（名字是整名：模块加它在那儿
叫的名字，取回来按最后一个点切开）；成员是某份值的成员时保留成员那一层，
那份值里带调用的那一格按能跑的形式给，落成 `__dyn_method__ << "f" << ($f (__dyn_call__ << "inc") * $y 2) << 1`），
照着它就能把这次调用重新做一遍，不必重跑一次运行，也不必让给出它的那个模块在场（`viba-interpreter.md`
「把一次调用落成可执行的」）。

## 10. 要回放，就要有稳定的路径

一次执行里唯一不必重复自己的东西是宿主函数——它可能取时钟、掷骰子、调外部系统。所以由它自己负责让这次
运行可以重放：把结果快照下来，下次同一次调用直接回放。

```python
import random
from viba.interpret import replayed

def roll(env, n):
    return replayed(env, lambda: random.randint(1, 10 ** 6), f"roll-{n.value}")
```

`replayed(env, compute, name)` 取 `<这次调用的路径>/<name>.viba`；没有就算出来、存下来。快照是序列化
的 viba 数据，不是 pickle：人能看，类型侧也能当实例取。

路径必须稳定才回放得上：`args.env.sub_env << args.env << "一个稳定的名字"` 稳定；
`tmp_env` 每次都是新路径，挂在它底下的不纯步骤每次都重走——这是它给出的信号：这一步该显名保存了
（[`viba-interpreter.md`](viba-interpreter.md)）。

## 11. 序列化出去的东西可以再反序列化回来

两个方向都有工具：

- 取一份设计：`viba.reflect` 按类型把实例一步步取出来（问有没有、取一段、走到叶子）。
- 拼一份源码：`viba.builder` 用 Python 拼出 viba 源码，拼出来的还能再被解析回去。

```python
from viba import builder

vb = builder.Builder()
tag = builder.tag

vb.Point = vb.Object * tag.x(vb.float) * tag.y(vb.float)
print(str(vb))
```

```viba
Point =
    Object
  * $x float
  * $y float
```

一份设计也能被序列化回源码（`viba.serialize`），所以存下来的东西是能看的。

容器也一样：`ListLiteral[1, 2]` / `SetLiteral["a"]` / `DictLiteral[("k", 1)]` 放在值的位置上就是
那个 list / set / dict（空的用 `ListLiteral[]`、`DictLiteral[]`），是什么源码形式就取回什么；元素按
地址取，`$get_item << $env nil << xs << 0` 就是 `xs[0]`（用方括号就行），`$get_item << $env nil <<
table << "k"` 就是 `table["k"]`；问在不在用 `$in`（`$in << $env nil << xs << 0`）。这五个内建成员
跟别的调用一样把环境排在第一、而且总是要：不给环境时留着的是一条闭包，实现里不用这份环境，所以
解释器自己拼链时那一位给的 `$env nil`（简式就是这样）也照样当场跑。
（[`viba-style.md`](viba-style.md) 第 8 节、[`viba-interpreter.md`](viba-interpreter.md)）。

## 12. 泛型：一个目录，某个文件给出结果

给一张"什么类型配什么结果"的表时，用的是**模式**：泛型不是一份带形参的定义，而是一个目录，
里面每个文件是一个模式：文件名是它的决断顺序（一个数字），数字前面还可以带上它要几份
（`2_200.viba` 要两份），决断照着份数只取对得上的那几份（[`viba-pattern.md`](viba-pattern.md) 第 4.1 节）。

```
demo/is_base_type/__generic__.viba      标记：这个目录是一个泛型
demo/is_base_type/100.viba              pattern bool | int | float | str
demo/is_base_type/200.viba              pattern A
```

```viba
# demo/is_base_type/100.viba
pattern bool | int | float | str

__decl__ = true
```

```viba
# demo/is_base_type/200.viba
pattern A

__decl__ = false
```

`pattern` 一行管一个形参，按源码顺序。源码里的类型是**限定**（实参要落得进去），文件里没定义的
名字是**形参**（实参在那个位置是什么就萃取出来）。`__decl__` 是这个文件给出的类型。应用放在方括号里：

```viba
import demo.is_base_type as is_base_type

__decl__ = Any <- $env Env

__impl__ = $flag is_base_type[bool] * $n 1
```

决断按数字从小到大，第一个命中的赢；一个都没命中是程序错误，不是 `never`。结果是一条函数链时，
这个应用代表的就是那次调用；方括号里的实参本身也可以是一次调用。tag 也可以在字符串里：
`tagged["a", T]` 就是 `$a T`，`pattern tagged[name, T]` 还能把实参那个 tag 的名字
萃取成一个字符串。整个规矩、各种源码形式和报错，见 [`viba-pattern.md`](viba-pattern.md)。

## 13. 接下来看什么

- [`viba-style.md`](viba-style.md)：给设计时的规矩——tag 怎么起名、提示怎么给、什么时候用和、什么时候
  用积。
- [`viba-interpreter.md`](viba-interpreter.md)：执行这一层——环境、模块、闭包、函数类型的那个实参、一次执行会得到
  什么、幂等与快照。
- [`viba-pattern.md`](viba-pattern.md)：模式——泛型的目录、`pattern` 的形式、决断顺序。
- [`viba_builder.md`](viba_builder.md)：用 Python 拼 viba 源码。
- [`viba-reflect.md`](viba-reflect.md)：按类型取实例（数据路径、步子、访问函数）。
- `README.md`：文法、运算符、怎么装、怎么跑。

检查自己改的东西：

```bash
python -m viba.parser          # 文法自检 + 文档里的例子能不能编
python -m viba.viba_ast        # 打印出来的能不能解析回去
python3 tests/test_interpreter_member.py   # 也可以直接跑任何一个 tests/*.py
python3 tests/test_interpreter_branch_switch.py   # 独立文件 + 部分计算 + branch 的一百条边角案例
```

# viba 教程

这份教程从最小的定义写到能跑的模块，假设你会 Python，不假设你见过 viba。

文中的 viba 代码块都是完整的源码，`python -m viba.parser` 会把它们编一遍：**编得过才写得进这份文件**。
要动手跑，看第 4 节那份宿主；那一节之后，例子都能直接放进文件里跑。

## 1. 一个定义就是一个类型

```viba
Option = $some int | nil
```

等号左边是名字，右边是类型表达式。`|` 是**和**：`$some int` 或 `nil` 其中一个。`$some` 是这一支的
**tag**，写的时候带上它，读的时候也按它找。`nil` 是"什么都没有"的那一支。

`*` 是**积**：几个东西同时都在。

```viba
Point = $x float * $y float
Config = $host str * $port int * $debug bool
```

不带 tag 的成员只留给"本来就没有名字"的东西：`(A, B)` 是元组，顺序本身就是意思；`int | str` 是按值
的种类分开的叶子。

定义只依赖源码，不依赖谁来实现它。所以一份文件可以先当类型读：读它有哪些成员、每个成员是什么类型、
一个名字从哪来（[`viba-reflect.md`](viba-reflect.md)）。

```python
from viba import viba_ast

tree = viba_ast.parse("Option[T] = $some T | nil")
print(viba_ast.unparse(tree))
```

## 2. 单位元与底：`Object`、`nil`、`never`、`Any`

- `Object` 是积的单位元：`Object * $x int` 与 `$x int` 是同一个类型。
- `Oneof` 是和的单位元（一个都没有的和）。
- `never` 是底：没有值属于它。函数实参写成 `never` 表示"这个参数谁来都装不下"。
- `Any` 是顶：谁都装得下。

```viba
Nothing = Oneof
Anything = Any
NoSuchField = Object * $x never
```

一个积里同一个 tag 写两遍是错的，判定的时候报。

## 3. 函数：结果在前，实参在后，tag 落在实参上

```viba
Add = int <- $a int <- $b int
Handler = Response <- $request Request
```

`Add` 是一条链：链头是结果 `int`，后面每一个 `<-` 添一个参数。它要两个 `int`，给出来一个 `int`。

**每个可执行的函数都得拿到环境**：`__decl__` 里可以写 `$env Environment` 这个参数，也可以不写。
写不写，调用时都是用一个 `<<` 给一个环境值：写了，环境就给这个参数，模块体用 `args.env` 取到它；
没写，环境不进门（`args.env` 取不到），它只用来跑这次调用，再缀在这次调用答出来的那次调用末尾。
环境说这一步在哪个环境里做：

```viba
add =
    int
  <- $env Environment
  <- $a int
  <- $b int
  <- { 把两个整数加起来 }
```

`{ … }` 是**提示**：写给照着它补实现的人（常常是 agent），说这一步要干什么。它不是参数，也不被执行，
`<<` 会跳过去。

`<<` 是给实参。给函数一个实参就是**一次调用**；给全了就是结果：

```viba
Call = add << $env args.env << $a 1 << $b 2
```

实参可以按位置给，也可以按 tag 给（顺序随意）；按位置给时，落在写下来的第一个还没给的参数上。

## 4. 一份能跑的文件：`__impl__` 与环境

文件里写 `__impl__` 的那份定义就是这份程序的答案；**没有 `__impl__` 的文件是类型，不是程序**，跑它报
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
（`foo_module.Bar`，泛型应用选中那份文件时也一样：`g[T].value`）；函数语义要一份签名加一份答案 ——
不写 `__decl__` 就没有函数语义，它只是模块。

`__decl__` 是这份模块当函数读时的整条链：结果在前，参数在后；`$env Env` 至多写一个（`Env` 是内建
的名字，就是 `Environment`）：写了，环境就给这个参数；不写，环境不进门，随这次调用的答案往后走。
`args = __get_args__ << __decl__` 读回这次调用收到的那份实参：`args.env` 是环境本身。

viba 是**声明式**的，不是命令式执行的：一份文件里的定义是**绑定**（binding），不是按写下来的次序
一条一条执行的语句。求值由需求驱动（demand-driven），一个绑定被用到的时候才求值，求一次就把结果
留下来（memoization）。这叫**按需求值**（call-by-need），也就是惰性求值。所以 `args` 什么时候求出来，
看谁用到它；写下来的次序不是求值的次序。整条策略见 [`viba-interpreter.md`](viba-interpreter.md) 的
「求值策略：按需求值（call-by-need）」一节。

结果不能写 `Env`：只有内建函数（`viba/builtin.viba` 里 `Environment` 的成员）答得了一个环境，
别的函数写了当场报错——环境是调用的规矩，不是能交出去的值。运行时的答案不受这条限制：
`__impl__ = args.env` 交回去的还是那个环境，只是声明成 `Any`。

宿主给两样东西：答案存在哪（`EnvironmentStorage`，默认一个临时目录）和每一步的实现
（`EnvironmentCompute`，里面一个 `get_func(module_path, func_name)`）。`interpret` 读文件、跑起来。

```python
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret

def get_func(module_path, func_name):
    if func_name == "add":
        return lambda env, a, b: a.value + b.value
    return None

environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))

answer = interpret("add_demo.viba", environ)
print(answer)                  # Ok(VibaNode(root))
print(answer.ok_value.value)   # 1000000
```

宿主收到的东西分两种：**可序列化数据**到手上是实例（`a` 是节点，`a.value` 是那个数），**环境**到手就是
环境自己。写下来的名字、字面量、调用，宿主看到的都是实例，按 [`viba-reflect.md`](viba-reflect.md) 一步步读。

`interpret` 自己不带任何实现：一次运行能碰到的实现，全部来自 `get_func` 那一个答案。

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
`viba/builtin/`）；内建库里的模块与泛型不用写 import —— `Y`、`apply`、`sub_env_run`、
`sequential`、`is_closure`、`unclosure` 直接写名字就是它（写了 import 也一样），它们就在搜索路径的
最后一站。

`args.env.sub_env << args.env << "add_demo"` 拿一个子环境：**它带着父级的 compute**，storage 路径是
`root/add_demo`。它也可以写成 `$sub_env << args.env << "add_demo"`——tag 写在链头时命中的是第一个参数
的成员，两种写法是同一次调用（第 8 节）。这个模块只收环境这一个参数，所以给环境就是执行。
那条路径就是这次调用的身份——宿主拿到的 `module_path` 就是它，所以它是一次调用的**地址**：
同一条路径上正在跑就是环（报错并把写法写在话里）；已经答过、又是**同一个模块**，就是同一个子
计算被问了第二次，把它答过的那份交回去；已经答过、却是**另一个模块**，那是两条调用挤一个地址，
宿主分不出谁是谁，也报错。

不想起名字就用 `tmp_env`：它每次给一个新的子环境（路径是 `root/tmp_<随机>`），也可以写成
`$tmp_env << args.env`。

```viba
lib_call = lib << (args.env.tmp_env << args.env)
```

模块要收实参，就把它们写进 `__decl__`（结果在前，参数在后），再写 `args = __get_args__ << __decl__`：

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

`add << $a 40` 是一个**闭包**：写下来的函数名加上已经算好的实参。它能存、能传、能当 `__impl__`、能序列
化，也能**换一个环境再执行一次**（`same << (args.env.tmp_env << args.env)`）。"必须给全"只对执行
成立，对闭包不成立。

闭包里只能装可序列化数据（只读、能写下来）；环境从来不装在闭包里，它是执行那一刻才给的。所以同一个
闭包可以反复执行，每次的执行环境由给它的那个环境决定。

给环境的那一刻才是执行：`interpret` 去问宿主哪一步来实现，并按这次调用的 storage 路径把它认下来。

## 7. 可能会用不上的那个实参：写成函数类型

有些实参可能根本用不上，算它要花代价或者有副作用。把那个参数写成**函数类型** `(T <- $env Env)`，
那个实参就不在这里算：写下来的那次调用连同环境一起交给宿主，宿主叫它的时候才算，在宿主给的那个环境
里算。`branch.viba` 的两个开关就是这么写的：

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

`$get_v` 那个实参写成一次调用（`tick << $env args.env`）；宿主拿到的是一个能带着环境调它自己的东西，
不叫它，那个实参就不求值（惰性求值，lazy）。所以没走的那一支里的副作用不会发生，它没有实现
也不会挡路。

几条规矩：

- **只写那一个参数**。别的实参按值求值（call-by-value），函数的 `$env` 也照旧按值给；
- **最多求值一次**（memoization）。第一次问出结果（值或停下），之后每一次问都拿同一个，问两遍不会
  做两遍；
- **想给一个已经算好的值**，用 `builtin.echo` 把它包成"给它一个环境就答 V"的那个函数：
  `$get_v (builtin.echo << $x v)`；
- **半成品调用也能放进去**。一个还欠实参的调用（`add << $a 40`）本身就是个值；放在那一个参数里，
  宿主拿到的是它代表的那次调用，补上欠的那些实参就得到答案。

条件、值和开关本身都可以写在别的文件里，用例只写一条链去调它们。这条路子的边角案例见
[`tests/test_interpreter_branch_switch.py`](tests/test_interpreter_branch_switch.py)。

这一节说的是求值策略落在**实参**上的那一半（那个参数惰性，别的参数按值求值）；落在**定义**上的
另一半是第 4 节那条：定义是按需求值的绑定。两半合起来见
[`viba-interpreter.md`](viba-interpreter.md) 的「求值策略：按需求值（call-by-need）」一节。

## 8. 调用方法：链头写 tag

`$tag << X << a` 就是 `X.tag << X << a`：**tag 写在链头时，它命中的是第一个参数的成员，而那个值也接着
被交给成员当第一个实参**。环境成员因此有两种写法，一样长：

```viba
child = args.env.sub_env << args.env << "add_demo"   # 点号，环境自己写出来
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

tag 本身不是值，`method = $sub_env` 编不过：标签只有写在链头、后面跟着第一个参数时才成立。第一个参数
是别的值也一样，`$tag` 命中的是它的成员；成员写在数据里时，它收的第一个参数是它所在的那个值。

## 9. 一次执行给出什么

`interpret` 给的答案有四支：

- `Ok(VibaNode)`：`__impl__` 的值。
- `$viba_program_err str`（Python 侧 `VibaProgramErr`）：**这份程序或环境不行**——编不过、文件不在、
  没有 `__impl__`、`$env` 没给。它不说"哪一步的实现坏了"，所以不带步名。
- `$underlying_viba_op_failed Failure`（`UnderlyingVibaOpFailed`）：**某一步的实现坏了**，或者它答了
  没有叶子的东西。`$msg` 给人读，`$step`、`$reason` 给程序读。
- `$not_my_duty_exception Duty`（`NotMyDutyException`）：**这一步不在这台机器上作答**。这不是失败，是
  递延——程序停在那儿，等有实现的一方接着做。`interpret` 不带库函数，所以"没有实现"很正常。

递延里带着是哪一步（`$step` 的 `module_path` 与 `func_name`）和这一步拿到的实例（`$call`），所以拿着
它就能把下一步该做什么写出来，不必再跑一次（[`roadmap.md`](roadmap.md)）。

## 10. 要回放，就要有稳定的路径

一次执行里唯一不必重复自己的东西是宿主函数——它可能读时钟、掷骰子、调服务。所以由它自己负责让这次
运行可以重放：把答案快照下来，下次同一次调用直接回放。

```python
import random
from viba.interpret import replayed

def roll(env, n):
    return replayed(env, lambda: random.randint(1, 10 ** 6), f"roll-{n.value}")
```

`replayed(env, compute, name)` 读 `<这次调用的路径>/<name>.viba`；没有就算出来、存下来。快照是序列化
的 viba 数据，不是 pickle：人能读，类型侧也能当实例读。

路径必须稳定才回放得上：`args.env.sub_env << args.env << "一个稳定的名字"` 稳定；
`tmp_env` 每次都是新路径，挂在它底下的不纯步骤每次都重走——这是它给出的信号：这一步该显名保存了
（[`viba-interpreter.md`](viba-interpreter.md)）。

## 11. 写下来的东西可以再读回来

两个方向都有工具：

- 读一份设计：`viba.reflect` 按类型把实例一步步读出来（问有没有、取一段、走到叶子）。
- 写一份源码：`viba.builder` 用 Python 拼出 viba 源码，拼出来的还能再被解析回去。

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

一份设计也能被序列化回源码（`viba.serialize`），所以存下来的东西是可读的。

## 12. 泛型：一个目录，由某个文件作答

写一个"什么类型配什么答案"的表时，用的是**模式**：泛型不是一份带形参的定义，而是一个目录，
里面每个数字文件名是一个模式，数字就是决断顺序。

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

`pattern` 一行管一个形参，按书写顺序。写下来的类型是**限定**（实参要落得进去），文件里没定义的
名字是**形参**（实参在那个位置是什么就萃取出来）。`__decl__` 是这个文件答的类型。应用写在方括号里：

```viba
import demo.is_base_type as is_base_type

__decl__ = Any <- $env Env

__impl__ = $flag is_base_type[bool] * $n 1
```

决断按数字从小到大，第一个命中的赢；一个都没命中是程序错误，不是 `never`。答案写成函数链时，
这个应用代表的就是那次调用；方括号里的实参本身也可以写成一次调用。tag 也可以写在字符串里：
`tagged["a", T]` 就是 `$a T`，`pattern tagged[name, T]` 还能把实参那个 tag 的名字
萃取成一个字符串。整个规矩、各种写法和报错，见 [`viba-pattern.md`](viba-pattern.md)。

## 13. 接下来读什么

- [`viba-style.md`](viba-style.md)：写设计时的规矩——tag 怎么起名、提示怎么写、什么时候用和、什么时候
  用积。
- [`viba-interpreter.md`](viba-interpreter.md)：执行这一层——环境、模块、闭包、函数类型的那个实参、一次执行会得到
  什么、幂等与快照。
- [`viba-pattern.md`](viba-pattern.md)：模式——泛型的目录、`pattern` 的写法、决断顺序。
- [`viba_builder.md`](viba_builder.md)：用 Python 拼 viba 源码。
- [`viba-reflect.md`](viba-reflect.md)：按类型读实例（地址、步子、访问函数）。
- [`viba-compliance.md`](viba-compliance.md)：一次运行算不算数——规则、呈证、判定。
- [`roadmap.md`](roadmap.md)：这套东西打算接到哪里去。
- `README.md`：文法、运算符、怎么装、怎么跑。

检查自己改的东西：

```bash
python -m viba.parser          # 文法自检 + 文档里的例子能不能编
python -m viba.viba_ast        # 打印出来的能不能读回去
python3 tests/test_interpreter_member.py   # 也可以直接跑任何一个 tests/*.py
python3 tests/test_interpreter_branch_switch.py   # 独立文件 + 部分计算 + branch 的一百条边角案例
```

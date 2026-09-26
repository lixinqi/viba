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

**每个可执行的函数都要 `$env Environment` 这个参数**，它说这一步在哪个环境里做：

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
Call = add << $env environ << $a 1 << $b 2
```

实参可以按位置给，也可以按 tag 给（顺序随意）；按位置给时，落在写下来的第一个还没给的参数上。

## 4. 一份能跑的文件：`__ret__` 与环境

文件里写 `__ret__` 的那份定义就是这份程序的答案；**没有 `__ret__` 的文件是类型，不是程序**，跑它报
`VibaProgramErr`。

```viba
# add_demo.viba
add =
    int
  <- $env Environment
  <- $a int
  <- $b int
  <- { 把两个整数加起来 }

__ret__ =
    add
    << $env environ
    << $a 999999
    << $b 1
```

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

一个 `.viba` 文件就是一个模块，而模块也是函数：输入是环境，输出是 `__ret__`。跨文件调用要给它一个
属于这次调用的环境：

```viba
# main.viba
import add_demo as demo

ret = demo << (environ.sub_env << environ << "add_demo")

__ret__ = demo.print << environ << ret
```

`environ.sub_env << environ << "add_demo"` 拿一个子环境：**它带着父级的 compute**，storage 路径是
`root/add_demo`。它也可以写成 `$sub_env << environ << "add_demo"`——tag 写在链头时命中的是第一个参数
的成员，两种写法是同一次调用（第 8 节）。这个模块没有 `__args__`，所以给环境就是执行，不用再写一个 `()`。
那条路径就是这次调用的身份——宿主拿到的 `module_path` 就是它。所以一次运行里
**任何两次模块调用不许用同一条路径**，撞了就是 `VibaProgramErr`，并且把写法写在错误里。

不想起名字就用 `tmp_sub_env`：它每次给一个新的子环境（路径是 `root/tmp_<随机>`），也可以写成
`$tmp_sub_env << environ`。

```viba
lib_call = lib << (environ.tmp_sub_env << environ)
```

模块要收实参，就声明 `__args__`（一份积，成员就是实参）：

```viba
# square_sum.viba
__args__ =
  Object
  * $a int
  * $b int

args = __args__

__ret__ =
    add
    << environ
    << (mul << environ << args.a << args.a)
    << (mul << environ << args.b << args.b)
```

调用时按位置给（`<< 3 << 4`）或按 tag 给（`<< $b 4 << $a 3`）都行；**环境给了，成员就得齐**，少一个
当场报错，话里点名少了谁。

## 6. 不给环境，链就是一个值

```viba
half = add << $a 40
same = add << $a 1 << $b 2
__ret__ = same
```

`add << $a 40` 是一个**闭包**：写下来的函数名加上已经算好的实参。它能存、能传、能当 `__ret__`、能序列
化，也能**换一个环境再执行一次**（`same << (environ.tmp_sub_env << environ)`）。"必须给全"只对执行
成立，对闭包不成立。

闭包里只能装可序列化数据（只读、能写下来）；环境从来不装在闭包里，它是执行那一刻才给的。所以同一个
闭包可以反复执行，每次的执行环境由给它的那个环境决定。

给环境的那一刻才是执行：`interpret` 去问宿主哪一步来实现，并按这次调用的 storage 路径把它认下来。

## 7. 按需的实参：`CalledByNeed[T]`

有些实参可能根本用不上，算它要花代价或者有副作用。把那个参数写成 `CalledByNeed[T]`，它就不先算：

```viba
id_or_never =
    Any
  <- $env Environment
  <- $condition bool
  <- $v CalledByNeed[Any]
  <- { 条件成立交回 v，否则交回 never }

never_or_id =
    Any
  <- $env Environment
  <- $condition bool
  <- $v CalledByNeed[Any]
  <- { 条件成立交回 never，否则交回 v }
```

```viba
# main.viba
import branch

ge =
    bool <- $env Environment <- $x int <- $y int <- { x >= y }
tick =
    int <- $env Environment <- { 算它会有副作用 }
tock =
    int <- $env Environment <- { 算它也有副作用 }

condition = ge << $env environ << $x 1 << $y 0
__ret__ =
    Oneof
  | (branch.id_or_never << $env environ << $condition condition << $v (tick << $env environ))
  | (branch.never_or_id << $env environ << $condition condition << $v (tock << $env environ))
```

宿主拿到的是一个"要用的时候再来拿"的东西；不叫它，那个实参就一次都不算。所以没走的那一支里的副作用
不会发生，它没有实现也不会挡路。

几条规矩：标记只标那个参数，别的实参照旧先算；同一个实参最多算一次，问两遍不会做两遍；**按需的那个
参数存不进闭包**（它没算过，不是可序列化数据），"给一半"时只能先给别的参数。

## 8. 调用方法：链头写 tag

`$tag << X << a` 就是 `X.tag << X << a`：**tag 写在链头时，它命中的是第一个参数的成员，而那个值也接着
被交给成员当第一个实参**。环境成员因此有两种写法，一样长：

```viba
child = environ.sub_env << environ << "add_demo"   # 点号，环境自己写出来
```

```viba
child = $sub_env << environ << "add_demo"          # 同一个调用，短一些
```

`viba/builtin.viba` 里 `Environment` 的成员就是这样声明的——环境是它的第一个参数：

```viba
Environment =
    Object
  * $viba_path str
  * $sub_env (Environment <- $env Environment <- $sub_env_name str)
  * $tmp_sub_env (Environment <- $env Environment)
```

tag 本身不是值，`method = $sub_env` 编不过：标签只有写在链头、后面跟着第一个参数时才成立。第一个参数
是别的值也一样，`$tag` 命中的是它的成员；成员写在数据里时，它收的第一个参数是它所在的那个值。

## 9. 一次执行给出什么

`interpret` 给的答案有四支：

- `Ok(VibaNode)`：`__ret__` 的值。
- `$viba_program_err str`（Python 侧 `VibaProgramErr`）：**这份程序或环境不行**——编不过、文件不在、
  没有 `__ret__`、`$env` 没给。它不说"哪一步的实现坏了"，所以不带步名。
- `$underlying_viba_op_failed Failure`（`UnderlyingVibaOpFailed`）：**某一步的实现坏了**，或者它答了
  没有叶子的东西。`$msg` 给人读，`$step`、`$reason` 给程序读。
- `$not_my_duty_exception Duty`（`NotMyDutyException`）：**这一步不在这台机器上作答**。这不是失败，是
  递延——程序停在那儿，等有实现的一方接着做。`interpret` 不带库函数，所以"没有实现"是常态。

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

路径必须稳定才回放得上：`environ.sub_env << environ << "一个稳定的名字"` 稳定；
`tmp_sub_env` 每次都是新路径，挂在它底下的不纯步骤每次都重走——这是它给出的信号：这一步该显名保存了
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

## 12. 接下来读什么

- [`viba-style.md`](viba-style.md)：写设计时的规矩——tag 怎么起名、提示怎么写、什么时候用和、什么时候
  用积。
- [`viba-interpreter.md`](viba-interpreter.md)：执行这一层——环境、模块、闭包、按需的实参、一次执行会得到
  什么、幂等与快照。
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
```

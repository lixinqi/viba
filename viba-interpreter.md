# 执行 viba 模块

一个 viba 文件是一个 Module，而 Module 有两种读法：

1. **类型推导**：把它当类型读（`viba.is_sub_type`、`viba-reflect.md`）；
2. **求值执行**：把它跑起来（`viba.interpret`）。

同一份语法，两种模式。跑的时候，模块**就是函数**：输入是 `environ`，输出是 `__ret__`。

规则、呈证与度量也是这么写的：一次度量就是一次调用，证据是那次运行留下的实例，判定是程序
答出来的 `bool`——见 `viba-compliance.md`。

```python
from viba.interpret import interpret

interpret("add_demo.viba", environ)      # -> Result[VibaNode]
```

`interpret(viba_main_file, environ, viba_path=None, get_file=None)`：`viba_path` 相当于 PYTHONPATH（冒号分隔，
按顺序找 `<name>.viba`，dotted 名当路径走；空条目和不存在的目录跳过）；import 的那个文件所在的目录总是
先找——**被 import 进来、又在自己的文件里 import 的模块，也按它自己的文件找**（链多深都一样）。
写了 import 的文件里的名字，按 import 绑定的名字解析（`import a.b as c` 绑 `c`，`import a.b` 绑 `a.b`）。
`viba_path` 也可以直接给一个路径（`Path`）；给了别的类型是 `VibaProgramErr`，不是把 `AttributeError` 抛出来。

## 源从哪来：get_file

`get_file` 是 `Optional[$file_content str <- $file_path str]`：

```python
interpret("main.viba", environ, get_file=files.get)   # 一次运行全在内存里
```

- **留空（`None`）就读文件系统**，和以前一样（`Path.read_text()`）。
- **给了就一律走它**，不再碰文件系统：连主文件也从它那儿读。所以宿主可以把整次运行架在内存、
  数据库或者别的地方上，路径只是字符串。
- 它收到的路径**按字符串给**（`$file_path str`），就是这次要找的那个候选路径（绝对还是相对，
  取决于 `viba_path`/主文件是怎么写的）。
- **"这个路径上没有文件"：这两种都算**——返回 `None`、或者抛 `FileNotFoundError`，于是查找继续
  去下一个地方（先 import 旁边，再 `viba_path` 按顺序），全都说没有就是
  `module 'x' not found (...)`。主文件说没有就是 `no such file: ...`。
- **返回非字符串、或者抛别的异常，是 `VibaProgramErr`**（`get_file(...) raised ...` / `... not the file's text`），
  不是把异常扔给调用方；源编不过照旧是 `cannot parse ...`。
- **同一个文件只问一次**：模块按路径认，已经加载过的（哪怕换了别名）不会再问第二次。

## 一个可执行的模块

```viba
# file name: add_demo.viba
add =
	int
	<- $env Environment
	<- $a int
	<- $b int
	<- {
		add two integer
	}

print =
	void
	<- $env Environment
	<- $x Any
	<- {
		print to stdout
	}

__ret__ =
	add
	<< $env environ
	<< $a 999999
	<< $b 1
```

规则：

- **没有 `__ret__` 的文件是类型，不是程序**：跑它给 `VibaProgramErr`。
- **`environ` 是内建变量**：类型推导时它是 `Environment` 类型，计算时是真实的那个环境。
- **函数体里的 `{...}` 是说明**：它不是参数，`<<` 给完实参之后链就落到结果上——
  `(B <- $a A) << $a A` 就是 `B`。
- **`{...}` 只给提示，不给实现**：提示只说这一步要实现什么，主要逻辑得有人照着它写出来，再交到
  `get_func` 上。写这些函数的是 agent（见「宿主侧：Environment」），viba 一个都不带。
- **每个可执行函数都要依赖 environ**：签名里必须有 `$env Environment` 这个参数，调用时也必须给；
  否则 `VibaProgramErr`（不是猜一个默认值）。
- **接实参时逐个核类型**：设计已经写明每个参数要什么（`$a int`），而写出来的实参自己带着类型
  （可序列化数据、字面量、函数链、环境），所以判定层那套 `<:` 在这里就能用——给错了是**程序错**
  （`VibaProgramErr`），不是"某一步的实现坏了"。判不出类型的（宿主自己的值、判定 settle 不了的）
  照旧放过去，交给实现那一步的人；按需的实参不先算，所以那个参数等宿主叫它的时候才核，不叫就不核。
- `__ret__` 必须是值：可序列化数据、环境，或者一个**闭包**（见「环境就是执行」）。

## 环境就是执行

函数和模块是同一个东西：都是一条链，都要 `$env Environment` 这个参数（模块的链末了还多一个参数
`__args__`）。

```viba
add : int <- $env Environment <- $a int <- $b int     # 一个函数
lib : __ret__ <- $env Environment <- __args__         # 一个模块：末了那个参数是它的 __args__
```

**给环境就是执行**，也只有这一个动作是执行：给它的那一刻，`interpret` 去问宿主哪一步来实现
（`get_func(module_path, func_name)`），并按这次调用的 storage 路径把它认下来（调用的身份）。

**不给环境，链就是一个闭包**——一个值：

```viba
half = add << $a 40            # 类型 int <- $env Environment <- $b int：还欠 $b 和环境
g    = add << $a 1 << $b 2     # 类型 int <- $env Environment：实参给齐了，只欠环境
sg   = lib << 3 << 4           # 类型 __ret__ <- $env Environment：模块也一样，实参给齐了，只欠环境
__ret__ = g                    # 一次运行可以答一个闭包：给出去的是"还没执行的活"
```

- 闭包是**可序列化数据**：写下来的函数名加上已经算好的实参，所以它能存、能传、能当 `__ret__`、能序列化。
- 闭包里只能装**可序列化数据**，只读、能写下来——别的函数和模块的返回值本来就满足这个要求。
  环境从来不装在闭包里：它是执行那一刻才给的。
- 同一个闭包可以**反复执行**：换一个环境就是换一次执行。
  `sg << (environ.tmp_sub_env << environ)` 和 `sg << (environ.sub_env << environ << "again")` 是两次运行。
- **给了环境的链要么跑完，要么报错**：环境给了、实参还欠着，是"准备不完整"的程序错——这种
  半成品存不下来，也不该存。所以"给了一半"和"没给环境"是两件完全不同的事。
- 宿主那一侧永远只见得到可序列化数据（见「宿主侧：Environment」）：闭包到了宿主手里就是数据，可以拿着、存着、递回来。

## 模块的实参：`__args__`

一个模块要收参数，就把参数写成 `__args__`——一份**积类型**，成员就是这次调用的实参：

```viba
# file square_sum.viba
__args__ =
  Object
  * $a int
  * $b int

args = __args__

__ret__ = add
  << environ
  << (mul << environ << args.a << args.a)
  << (mul << environ << args.b << args.b)
```

```viba
# file main.viba
import square_sum

__ret__ = square_sum << (environ.tmp_sub_env << environ) << 3 << 4
```

- **给环境就是执行**：每个成员一个 `<<`，按位置给（`<< 3 << 4`）或按 tag 给（`<< $b 4 << $a 3`）
  都行；按 tag 给时顺序随意。**环境给了，成员就得齐**：少一个就是 `VibaProgramErr`，话里点名少了谁
  （`module 'square_sum' was given 1 of its 2 __args__: $b missing`）。多给、给重、给了没有的 tag，
  同样当场报错。
- **不给环境，它就是闭包**：`sg = square_sum << 3 << 4` 是一个值——写下来的模块名加上已经算好的
  实参，可以存、可以传、可以当 `__ret__`、可以序列化，也可以**换个环境再执行一次**
  （`sg << (environ.tmp_sub_env << environ)`）。"必须给全"只对执行成立，对闭包不成立。
- **`()` 只在显式写出空实参时写**：零实参的模块，执行就是 `lib << <环境>`；`sg = lib << ()` 是
  把"准备好的空实参"写出来存成闭包。
- **`__args__` 的成员上写 `CalledByNeed` 没有用**：模块的实参是它读到的可序列化数据（`args.a` 那种），
  不是一次"要的时候再来拿"的调用——按需说的是宿主那一侧的实参。
- **模块体里 `__args__` 就是那份实参积**：`args = __args__` 只是个别名，运行时 `args` 是这次调用
  的实参（`args.a` 是 `$a` 那个成员的值，按 tag 寻址——这里说的**地址**是数据存下来的那条路径，
  由一串步子接成（按 tag、按位置、按下标、按键），一段一段走下去就是寻址；`$a` 是它的一步，模块
  路径和 `cur_storage_path/<name>.viba` 也是地址。「幂等与快照」里的"快照地址"、「一次执行会得到什么」里的"这次调用的地址"都是这个
  意思，步子分哪几种见 [`viba-reflect.md`](viba-reflect.md) 第 4 节）；判定层读同一个名字读到的是
  成员**声明的类型**（`args.a` 就是 `int`）。一个名字，两层各读各的——和 `environ` 一样。
- 实参必须是可序列化数据：它是积的一部分，交出去的东西得是能写下来的值。
- **实参要装得下那个参数**：`$a int` 那个参数给 `"x"` 是**程序写错了**，当场 `VibaProgramErr`
  （`module 'square_sum': "x" does not fit $a int: "x" <: int does not hold`），宿主根本看不见这个
  实参。函数那边同理：`add` 的 `$a int` 给字符串也一样。

## 调用别的模块

```viba
import add_demo as demo

ret = demo << (environ.sub_env << environ << "add_demo") << ()

__ret__ = demo.print << environ << ret
```

- `demo` 是模块，当函数用：给它一个环境、再给它 `__args__`（这份没声明实参，那个参数写 `()`），
  它跑完给出它的 `__ret__`。
- `demo.print` 是这个模块里的函数。
- `environ.sub_env << environ << "add_demo"` 拿一个子环境：**它带着父级的 compute**，storage 路径是
  `父路径/add_demo`（见「幂等与快照：结果要能回放」）。同一个调用还可以写成
  `$sub_env << environ << "add_demo"`（见「如何调用方法：链头写 tag」）。

**每次模块调用都要有自己的 storage 路径**：那条路径是这次调用的身份——宿主拿到的
`get_func(module_path, func_name)` 里的 `module_path` 就是它，两次激活落在同一条路径上，宿主就
分不出谁是谁（`root` 下主文件自己的 `add` 与某个模块的 `add` 会看成一个）。所以一次运行里
**任何两次模块调用不许用同一条路径**，重复就是 `VibaProgramErr`，并且把正确写法写在错误里：

```
module 'lib' was handed the storage path 'root', which another module call already
used: give each module call a sub-environment of its own (environ.sub_env << environ << ...)
```

注意 `environ.sub_env << environ << "x"` 对同一个父环境是**同一个** storage（同名子环境按需造一次、之后
复用），所以同一个模块调两次要给两个名字，或者让宿主给出两份不同的 storage。主文件自己也算
一次激活，占着它那条路径——直接 `lib << environ` 就是撞车。

### 如何调用方法：链头写 tag

`$tag << X << a` 就是 `X.tag << X << a`：**tag 写在链头时，它命中的是第一个参数的成员，
而那个值也接着被交给成员当第一个实参**。`environ` 的成员因此有两种写法，一样长：

```viba
environ.sub_env << environ << "add_demo" # 点号，环境自己写出来
$sub_env << environ << "add_demo"        # tag 写链头，同一个调用，短一些
$sub_env << $env environ << $sub_env_name "add_demo"   # 都按 tag 给，也对
```

`viba/builtin.viba` 里 `Environment` 的成员就是这样声明的——环境是它的第一个参数：

```viba
  * $sub_env (Environment <- $env Environment <- $sub_env_name str)
  * $tmp_sub_env (Environment <- $env Environment)
```

成员也可以写在数据里：一个积的 `$f` 里放一个函数名时，`$f << box << environ << 1` 就是
`box.f << box << environ << 1` —— 成员收的第一个参数是它所在的那个值（`$box Box`），环境跟在后面。

tag 本身**不是值**：`method = $sub_env` 编不过，标签只有写在链头、后面跟着第一个参数时才成立。
第一个参数必须写出来——它是取成员的那一个，省略了就成了一次没有成员的调用。第一个参数是别的值
也一样：`$tag` 命中的是它的成员，不在就报错。

不想起名字就用 `tmp_sub_env`：

```viba
lib << (environ.tmp_sub_env << environ)
```

它像临时文件一样，**每次调用都给一个新的子环境**（路径是 `父路径/tmp_<随机>`），所以两次调用
天然各占一条路径。`viba/builtin.viba` 里 `Environment` 的成员因此写的是
`$tmp_sub_env (Environment <- $env Environment)`：它只收那个环境，名字不用给。

路径每次都不同，这是 `tmp_sub_env` 的语义：它给**纯函数调用**、或者**结果不留的调用**用。一个
需要快照、要靠回放才幂等的函数如果挂在它底下，回放自然命中不了——每次的路径都是新的，快照会
一次次写进不同的 `tmp_` 目录（`store_root/root/tmp_xxxx/…viba`），那条不纯的路于是每次都重走，
**幂等检查会失败**。这不是缺陷，而是它给出的信号：这个函数需要显名保存，应该换到
`environ.sub_env << environ << "一个稳定的名字"` 上去。`tests/test_interpreter_idempotence.py` 里两半都有用例：
显名路径第二次跑就回放，临时路径两次都重算、并且留下两份快照。

## 宿主侧：Environment

`Environment` 由宿主提供。三个名字里**只有 `Environment` 是 viba 里看得见的那一个**
（`viba/builtin.viba`）：模块的 `$env` 那个参数要的就是它，`environ` 也是它。

```viba
Environment =
    Object
  * $viba_path str
  * $sub_env (Environment <- $env Environment <- $sub_env_name str)
  * $tmp_sub_env (Environment <- $env Environment)
```

storage 与 compute 是**宿主的词汇**，viba 里没有一个名字指得到它们（它们不是类型，是宿主
对象），所以只在宿主侧出现（`viba.interpret`）：

```python
EnvironmentStorage(cur_storage_path, sub_storage=None, store_root_dir=None)
EnvironmentCompute(get_func)                             # get_func(module_path, func_name)
Environment(storage, compute, viba_path=None)            # sub_env / tmp_sub_env 给子环境
```

`viba_path` 是模块的搜索路径：一次运行里它跟着 environment 走，`sub_env`/`tmp_sub_env` 把父级的
那条原样交给孩子，于是"这个模块的 import 去哪里找"就是它被交给的那个 environment 说了算。

`EnvironmentStorage` 面向 viba 一侧暴露的概念：`cur_storage_path`（这条路径就是这次调用的
身份）、`sub(name)`（子 storage，`sub_env` 用它）、`store_root_dir`（快照放在哪个目录下，
不给就是默认的临时目录 `…/viba-store`）、`read_text(file_path)` / `write_text(file_path,
content)`（在 store root 底下的纯文本读写，读不到返回 `None`）。子 storage 带着父级的
`store_root_dir`。

`get_func(module_path, func_name)` 返回一个可调用对象；没有就返回 `None`，于是这次调用是递延
（`$not_my_duty_exception Duty`，见最后一节）——它也可以直接抛 `NotMyDutyException`，那是一个
路由在说"这条差事不归我"；它自己抛别的异常，则是这一步的 `$underlying_viba_op_failed`。
`module_path` 是**调用时那个 environment 的 storage 路径**——所以同一个 `add`，从
`root/add_demo` 进来和从 `root` 进来，宿主看到的是不同的路径，可以路由到不同的实现；也正是
这个路径，加上 `func_name`，构成了答案里那个 `$step`。

`HostLanguageFunc` 与 interpreter 匹配：Python interpreter 里就是一个 Python 函数，收到的参数是
**已经算好的实参**，按书写顺序给——实例是 `viba.reflect.VibaNode`，别的（environ 在内）是它本身。
返回值是 `VibaNode`，或者一个普通 Python 值（落到函数声明结果的一个叶子上）。

参数里出现 **viba 函数**（`$f (int <- $env Environment <- $x int)` 这种高阶签名）时，宿主拿到的是**可序列化数据**：
那个函数名本身（一个 `VibaNode`），或者一个闭包——写下来的函数名加上已经算好的实参。**宿主调不动它**：
viba 函数不是宿主侧的 Python 可调用对象，执行它们只有一条路，就是给环境（`<< environ`），那是 viba 那一侧
的事。所以宿主得到的是数据：可以拿着、存着、原样递回来，不能 `f(env, x)`。

宿主自己造一个 `VibaNode` 当答案也可以；但**列表、字典、可调用对象这类答不了**——它们没有对应的叶子，
只能答 `VibaNode`、标量或 `None`（`None` 就是 `nil`）。

**interpret 不认识任何具体函数**：viba 默认不带任何库函数，实现全部来自 `get_func`，
谁写、怎么生成，interpret 不感知——用法上这个"谁"就是 agent：它读的正是文件里 `{...}` 那句提示，
把它补成一个函数。同一份文件交给两个 agent，得到两份实现，文件本身不变。

## 幂等与快照：结果要能回放

一个 viba 程序的结果要可回放：同一份输入、同一批路径，跑多少次都该是同一个结果。可执行函数
里唯一不保证这一点的东西是**宿主函数**——它可能读时钟、掷骰子、调服务。所以不纯的那些由它
自己负责：**把答案快照到 `EnvironmentStorage` 里，下一次跑同一次调用就直接回放**。

```python
from viba.interpret import read_snapshot, write_snapshot, replayed

def roll(env, n):
    def compute():
        return random.randint(1, 10 ** 6)       # 不纯的那一步
    return replayed(env, compute, f"roll-{n.value}")
```

- `replayed(env, compute, name)`：**有快照就回放，没有就算出来、存下来**。这是最常用的写法。
  想分开写就是 `read_snapshot(env, name)`（没有返回 `None`）与 `write_snapshot(env, value, name)`。
- 快照地址由**这次调用的 storage 路径**决定：`snapshot_path(env, name)` 是
  `cur_storage_path/<name>.viba`，读写在 `store_root_dir` 底下。所以同一次调用（同一条路径）才
  会命中同一条快照；换了路径（另一个名字的子环境）就是另一次调用。
- **要回放就要一条稳定的路径**：跨两次运行能命中，靠的是两次运行里那条路径一样。用
  `sub_env << environ << "稳定的名字"` 就是稳定的；`tmp_sub_env` 每条路径都是新的，挂在它底下的调用不适合
  保存要回放的东西（上一节：这正是它给出的"该显名保存了"的信号）。
- **快照是序列化的 viba 数据**（`viba.serialize` 写出来的 `value = …`），不是 pickle：存下来
  的东西可以被人读、被人看、被人拿去喂类型推导。回放时解析回实例，叶子和原来一样。
- **存不了、回放不出来就是错**：`VibaProgramErr`（宿主抛出来，interpret 转成 `VibaProgramErr`），不会静默给个默认值。

于是"随机"也能回放：

```viba
roll =
	int
	<- $env Environment
	<- $n int
	<- { roll a die: not a pure function, so its answer is snapshotted }

__ret__ = roll << $env environ << $n 1
```

第一次跑算出 731204 并存进 `store_root/root/roll-1.viba`；第二次跑读到它，那条不纯的路一次
都不走，结果还是 731204。`tests/test_interpreter_idempotence.py` 里就是这么验的：同一个 store 跑两遍值
相同、不纯函数只被调用一次；换一个 store 才会重新算。

## 分支：用积选择，用和汇合

viba 没有专门的 `if` 语法。分支由积与和组合出来：**积决定某一支是否存在，和把仍然存在的分支汇合起来**。

解释器只需要归约三个代数恒等式：

```viba
nil * a = a
never * a = never
never | a = a
```

`nil` 是积的单位元，所以 `nil * a` 保留 `a`；`never` 是积的吸收元，所以
`never * a` 让整支消失；`never` 同时是和的单位元，所以汇合时会从和里被剥掉。

和里只剩一个非 never 分支时，结果就是该分支；所有分支都是 never 时，结果是 never；
多个非 never 分支同时存在时，结果保留为和值，不擅自选择其中一支。

`branch.viba` 与 `branch.py` 提供两个通用开关。它们接收分支值 `$v Any`，返回 `Any`，而那个参数写着
`CalledByNeed[Any]`（见「按需的实参」）——只有它会"要的时候才算"：

```viba
id_or_never =
    Any
  <- $env Environment
  <- $condition bool
  <- $v CalledByNeed[Any]

never_or_id =
    Any
  <- $env Environment
  <- $condition bool
  <- $v CalledByNeed[Any]
```

`branch.py` 是宿主实现，和其他可执行函数一样由 `EnvironmentCompute` 显式注册：

```python
import branch

def get_func(module_path, func_name):
    return branch.get_func(module_path, func_name)
```

被标记的函数拿到的是 **getter** 而不是值，所以开关叫哪个才算哪个：

```python
def id_or_never(get_env, get_condition, get_v):
    if get_condition().value:
        return get_v()
    return _never()
```

- `id_or_never`：condition 为真时按单位元 `nil * v = v` 返回 `get_v()`，否则按 `never * v = never`
  返回 never——**`get_v` 不会被叫**，那一支的实参表达式根本不算；名字里的 `id` 就是单位元
  （identity）：留下这个值的那个选择子；
- `never_or_id`：condition 为真时按 `never * v = never` 返回 never，否则按单位元 `nil * v = v`
  返回 `get_v()`。

```python
a = foo()
if a >= 0:
    return a
else:
    return 0
```

对应：

```viba
import branch

a = foo << $env environ
condition = ge << $env environ << $x a << $y 0

__ret__ =
    Oneof
    | (branch.id_or_never << $env environ << $condition condition << $v a)
    | (branch.never_or_id << $env environ << $condition condition << $v 0)
```

当 `a = 1` 时：第一支是 `nil * 1 = 1`，第二支是 `never * 0 = never`，最终 `1 | never = 1`。
当 `a = -1` 时：第一支是 `never * -1 = never`，第二支是 `nil * 0 = 0`，最终 `never | 0 = 0`。

开关把 `$v` 一并接收进来，对外统一返回 `Any`；nil/never 是它内部用来决定保留还是消去 `$v` 的代数机制。
解释器提供积与和的通用归约，库函数负责把 condition 映射成这次分支的结果。

**这两支都要算** 本来是这个设计的代价：积与和只决定"哪一支留下"，不代表"哪一支被算"。要让没走的那
一支不算，就得让那个参数的实参**要的时候才算**——这件事由 `CalledByNeed[T]` 管：写在参数上，
`interpret` 不先算它，而是把"要用的时候再来拿"的东西交给宿主，宿主叫它才算。

### 按需的实参：`CalledByNeed`

`viba/builtin.viba` 里定义了一个特殊标记——它和 `Environment` 一样对每个模块可见，**不需要
import**（`builtin.viba` 是最低优先级的内建库，见 [`viba-style.md`](viba-style.md) 第 12 节）。
标记本身是里面那个保留 tag（`=` 左边是名字）：

```viba
CalledByNeed[Arg] =
    Object
  * $__called_by_need_tag_yanatutt__ ()
  * $arg Arg
```

**标记标的是实参，不是函数**：写成 `$v CalledByNeed[Any]` 的那个参数，`interpret` 不先算它：

- 那个参数的实参被包成一个无参 lambda（getter），连同这次调用的环境一起交给宿主；
- 宿主叫它才算，不叫就一次都不求值；
- 同一个调用里**别的实参照旧给值**——环境是 `Environment` 自己，条件给的是它写下来的那份可序列化数据。所以宿主那一侧
  是"值 + 那个参数的 getter"，不是"全是 getter"。

```python
def id_or_never(env, condition, get_v):
    if condition.value:                # 可序列化数据是 VibaNode，bool 要 .value
        return get_v()                 # 只有这一支会算
    return _never()
```

getter 答出来的就是宿主平常会直接拿到的那份东西：可序列化数据是 `VibaNode`，环境是 `Environment` 自己，
viba 函数给的是它写下来的样子（名字或闭包）。getter 在求值时出的错**原样**回到 run：没有实现就是递延，实现坏了
就是 `$underlying_viba_op_failed`，程序写错就是 `$viba_program_err`——错的是那个实参，不是叫它的人。

**按需的那个参数存不下来**，别的都能：它的实参没算过，不是可序列化数据，所以装不进闭包。链走完时它还在、而环境
也没给，就是程序错（`the $v argument is computed only when it is wanted, so it cannot be stored`）。
所以偏应用在别的参数上照旧：只有按需的那个参数不能先给。

```viba
switch = branch.id_or_never << $condition condition   # 闭包：条件先定下来
__ret__ = switch << $v (tick << environ) << environ   # 给按需的那个参数 + 环境，才执行
```

getter **最多算一次**：第一次问出结果（值或停下），之后每一次问都拿同一个。所以宿主问两遍不会让
副作用发生两遍——和 eager 调用里那个实参只求值一次是同一件事。

一处代价要记住：递延时**那份 `$call` 里没有按需的那个参数**。工单要固定的可序列化数据来自已算出的实参，而它
根本没算过——递延里只有 `$step`、`$reason`，和其余算过的实参。

绑定和这两个开关一起用的时候，谁先算、谁不算，看
[`tests/test_interpreter_let_branch.py`](tests/test_interpreter_let_branch.py)：20 份可以打开的文件
（`tests/data/let_branch/*.viba`），每条只在表里写该跑出什么。

标记说的是那个参数怎么给，不是类型，所以**类型层也读穿它**：`CalledByNeed[T]` 在判定层和描述符层
就是 `T`。于是 `id_or_never` 的类型是 `Any <- $condition bool <- $v Any`，
`id_or_never << $env environ` 也归约得下去（少掉 `$env` 那个参数）。三层读的是同一份读法
（`viba/partial.py` 的 `by_need_type`），所以运行、判定、描述符不会各读各的。

名字自己不算数：本地定义盖过内建，所以只有定义体里带那个保留 tag 的名字才算标记——一个模块自己定义
个同名类型，那它就只是个类型。

用例：判定层 [`tests/test_is_sub_type.py`](tests/test_is_sub_type.py) 的 `run_lazy_marker_cases`，
描述符层 [`tests/data/type_descriptor/api/lazy_marker/case.viba`](tests/data/type_descriptor/api/lazy_marker/case.viba)。

## 类型层的模块

同一个文件在**类型推导**里也是"environ 进、实参进、`__ret__` 出"：名字绑到的是一个模块（import
的名字）时，它当函数读，类型就是它的 `__args__`（没声明就是空的 `()`）：

```viba
__ret__ <- $env Environment <- __args__
```

所以这几条都成立（`tests/test_is_sub_type.py` 里有用例）：

```viba
demo = import add_demo as demo         # 概念上
demo << $env environ << ()  <:  int     # 就是 __ret__ 的类型
int <: demo << $env environ << ()       # 反过来也成立：两者同型
demo.add <: int <- $env Environment <- $a int <- $b int
design.Only <: $x int                   # module.MyType 照旧，没有被顶掉
square_sum << $env environ << $a 3 <: int <- $b int   # 给了一半：剩下的是函数
```

- 只认 **import 绑定的那个名字**（`import a.b as c` 的 `c`，`import a.b` 的 `a.b`）。`module.Name`
  仍然是那个模块里的定义，和以前一样按最长的前缀解析。
- 没有 `__ret__` 的模块不是程序：`demo << $env environ << ()` 在类型层也是 `VibaProgramErr`。
- `environ` 在类型层是内建名字，类型为 `Environment`（`viba/builtin.viba`），所以 `<< $env environ`
  这个参数在类型上也对得上。
- 实参按 tag 给（`<< $a 3`）或按位置给（`<< 3`）都认；位置那一支要求给的实参装得下那个参数
  （`3 <: int`）。**给一半在类型上是一种类型**——剩下的那个函数；在值层给一半是程序错。
- `__args__` 的成员在模块体里按 tag 读：`args.a` 在类型层就是那个成员声明的类型。`__args__` 不是
  积类型的话，两层都当场报错。

## 表达式里的绑定

`A = (a := 7  a)` 右边那个括号就是一个**表达式**：绑定在前、结果最后，整块的值就是结果。
多条绑定不挤一行，每条一行（见 [`viba-style.md`](viba-style.md) 第 6 节）：

```viba
A = (
  a := 3
  b := 4
  add << $env environ << $a a << $b b
)
```

求值规矩：

- 绑定按书写顺序算，算完就放进这一块自己的作用域；结果在同一个作用域里求值，所以看得见它们；
- 名字**不出块**：`A = (a := 3  a)` 之后写 `__ret__ = a` 是 `no definition named 'a'`；
  遮蔽一个定义也一样，块一结束那个定义照旧；
- 绑定可以遮蔽 `environ`、import 的名字、甚至 `int` 这种内建名——查找顺序是"块内先看"；
- 绑定是**先算的**（ML 的 `let`，不是懒参数）：没用到的绑定照样算。懒的是"整块作为某个被标记
  函数的实参"这件事——那时整块变成 getter，宿主不叫，绑定和结果都不算；
- getter 记住它被写下时所在的作用域：块结束后被叫，仍然看得见当时的绑定。

值位置上的可序列化数据（tag、积、元组）不跑表达式，所以绑定不能写进可序列化数据里：`$point (p := …  p)` 报
`a binding belongs in a value, not inside material`。要让可序列化数据里出现一个算出来的值，就让一个函数
把它答出来（「宿主侧：Environment」里那些宿主函数就是这么干的）。

`:=` 是**计算的写法**。定义体平时由解释器当值算（`A = (a := 7  a)` 就是这么用的），但类型层
只判类型：它在定义体或别的类型位置上遇到绑定块，答的是 `a binding is computation, not a type: …`。
这一层没有绑定，也没有等着谁去做的代换——想在类型里给中间量起名，那就是另一条定义。

## 一份文件跑不出递归

函数递归执行要"再一次进到同一个定义"。一份文件自己的定义图是无环的——**文件里的定义不许绕回
自己**——所以单独给一份文件，怎么跑都跑不出递归：写出来的调用点就那么多，`<<` 一处一处写死。

```viba
A = B
B = A
# A -> B -> A: one file's definitions may not go round — recursion takes two files
```

- **绕回自己当场报错**，不是等栈崩：一份文件里 `A = A`、`A = B` 与 `B = A`、以及"算 A 的时候
  又去要 A"（`A = use << $env environ << $x A`）都是 `VibaProgramErr`，话里给出绕的路径。
- **没真跑起来的不算**：按需的实参没人叫就不算递归（`A = ignore << $env environ << $x A` 答 7，
  因为 `ignore` 不叫那个 getter）；`List[T] = Object * $tail List[T] | nil` 这种**类型**自引用也
  不算——类型不执行。
- **跨文件不设这条限制**：两份文件可以互相 import、互相调用，设计层不拦。真的绕回去（`a.x` 要
  `b.y`、`b.y` 又要 `a.x`）时，这次运行报
  `a.x -> b.y -> a.x: the run came back to where it started`；模块调用成环报
  `module 'a' is already running: a module call cycle`。两条都是"这次跑不完"，不是设计不合规。

## 一次执行会得到什么

`interpret` 返回的 `Result` 比别处多两支，而多出来的那两支都带着"是哪一步"：

```viba
Result[T] =
    Oneof
  | $ok T
  | $viba_program_err str                  # 关于程序或环境，不指某一步
  | $underlying_viba_op_failed Failure     # 某个宿主实现自己坏了
  | $not_my_duty_exception Duty            # 这一步不归这个宿主

Step =
    Object
  * $module_path str
  * $func_name str

Failure =
    Object
  * $msg str
  * $step Step
  * $reason str

Duty =
    Object
  * $step Step
  * $call ...
  * $reason str
```

- `Ok(VibaNode)` 是 `__ret__` 的值；
- `$viba_program_err str`（Python 侧是 `VibaProgramErr`）说的是这一份**程序或环境**不行：编不过、
  文件不在、没有 `__ret__`、`$env` 没给……它不说"哪一步的实现坏了"，所以不带步名；
- `$underlying_viba_op_failed Failure`（`UnderlyingVibaOpFailed`）：**某一步的实现坏了**，或者它
  答了没有叶子的东西。`$msg` 给人读，`$step` 与 `$reason` 给程序读——"哪一步"是个字段，不是嵌在
  句子里的；
- `$not_my_duty_exception Duty`（`NotMyDutyException`）：**这一步不在这台机器上作答**。这不是失败，是递延——程序停在
  那儿，等有实现的一方接着做（[`roadmap.md`](roadmap.md)）。`interpret` 不带库函数，所以"没有
  实现"是常态，不是错误。

`$step` 的两个字段就是 `get_func(module_path, func_name)` 收到的那两个：路径是这次调用的**地址**，
所以同一个定义、另一个案子，是另一步。`$call` 是这一步拿到的**实例**，按它写下来的样子——一份
Prepare 要固定的正是这份实例，所以拿着递延就能把工单写出来，不必再跑一次。宿主值（首先是
environment）不是实例，不随 `$call` 走：接手的那一侧自己造环境。`$reason` 只有这几种：

| `$reason` | 意思 |
|---|---|
| `no implementation` | `get_func` 答了 `None` |
| `refused` | `get_func` 抛了递延（一个路由说"这条差事不归我"） |
| `get_func raised` | `get_func` 自己坏了 |
| `raised` | 实现抛了 |
| `no leaf` | 实现答了没有叶子的东西 |

递延与失败都会一路穿回调用方，而且**原样上传、不被改写**：被调用的模块里那一步没实现，带回来的
那一步是**里面那一次调用**（`root/模块名` 下的那个定义），不是外面那一层；穿过运行、再交给宿主的可
调用对象时也一样。`get_func` 抛递延时，run 会把缺的补上：步名与 `$call` 用它知道的这次调用，
`$reason` 留着宿主自己说的（没说就是 `refused`）。

`$viba_program_err` 的那句话（`err_msg` 一栏）长这样：

```
no such file: ...                     文件不在
no definition named 'x' in module 'm' 名字解析不了
module 'x' not found (...)            import 找不到文件
module 'm' has no __ret__: ...        类型，不是程序
... takes no $env Environment ...     可执行函数没依赖 environ
... was not given the environment     调用时没给 environ
... was not given an Environment      给了，但不是 Environment
module 'x' is already running         模块调用成环
... storage path ... already used     两次模块调用用了同一条 storage 路径
... is a function still waiting ...   __ret__ 不是值
cannot read ...                       文件读不了
cannot parse ...                      编译不过（语法错误）
get_file(...) raised ...              get_file 自己抛了
get_file(...) answered bytes, ...     get_file 答的不是文件的文本
viba_path is a string ...             viba_path 给错了类型
get_file is a function ...            get_file 给错了类型
```

`$underlying_viba_op_failed` 的那句话（`$msg` 一栏，`$step` 与 `$reason` 是另外两个字段）：

```
get_func('root', 'add') raised ...    get_func 自己坏了
add raised ZeroDivisionError(...)     实现抛了
add answered list, which is no leaf   实现答了没有叶子的东西
environ.sub_env raised ...            环境上挂的宿主函数坏了
```

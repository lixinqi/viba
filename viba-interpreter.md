# 执行 viba 模块

一个 viba 文件是一个 Module，而 Module 有两种读法：

1. **类型推导**：把它当类型读（`viba.is_sub_type`、`viba-reflect.md`）；
2. **求值执行**：把它跑起来（`viba.interpret`）。

同一份语法，两种模式。跑的时候，模块**就是函数**：`__def__` 是它的签名，输入是环境与实参，
输出是 `__ret__`。

规则、呈证与度量也是这么写的：一次度量就是一次调用，证据是那次运行留下的实例，判定是程序
答出来的 `bool`——见 `viba-compliance.md`。

```python
from viba.interpret import interpret

interpret("add_demo.viba", environ)      # -> Result[VibaNode]
```

`interpret(viba_main_file, environ, viba_path=None, get_file=None, list_files=None)`：`viba_path` 相当于 PYTHONPATH（冒号分隔，
按顺序找 `<name>.viba`，dotted 名当路径走；空条目和不存在的目录跳过）；import 的那个文件所在的目录总是
先找——**被 import 进来、又在自己的文件里 import 的模块，也按它自己的文件找**（链多深都一样）。
一个名字也可能是一个**泛型目录**（`<name>/__generic__.viba`）：文件先找，目录随后，两者都按这里的
顺序（见 [`viba-pattern.md`](viba-pattern.md)）。
写了 import 的文件里的名字，按 import 绑定的名字解析（`import a.b as c` 绑 `c`，`import a.b` 绑 `a.b`）。
点分名字还有一层意思：`a.b = A` 定义的是 `a` 的 `$b` 成员（[`viba-style.md`](viba-style.md) 第 5 节），
所以 `a.b` 当类型读是它的声明类型，写在链头就是那一步的调用 —— 宿主拿到的 `func_name` 是写下来的整串。
同一个名字写两次，后写的覆盖先写的：读名字得到的是后写的那一份。
`viba_path` 也可以直接给一个路径（`Path`）；给了别的类型是 `VibaProgramErr`，不是把 `AttributeError` 抛出来。

## 源从哪来：get_file

`get_file` 是 `Optional[$file_content str <- $file_path str]`：

```python
interpret("main.viba", environ, get_file=files.get)   # 一次运行全在内存里
```

- **留空（`None`）就读文件系统**（`Path.read_text()`）。
- **给了就一律走它**，就不碰文件系统：连主文件也从它那儿读。所以宿主可以把整次运行架在内存、
  数据库或者别的地方上，路径只是字符串。
- 它收到的路径**按字符串给**（`$file_path str`），就是这次要找的那个候选路径（绝对还是相对，
  取决于 `viba_path`/主文件是怎么写的）。
- **"这个路径上没有文件"：这两种都算**——返回 `None`、或者抛 `FileNotFoundError`，于是查找继续
  去下一个地方（先 import 旁边，再 `viba_path` 按顺序，最后是**内建目录** `viba/` ——
  `builtin.viba` 与包自己的泛型 `Y/`、`y_helper/` 就在那里），全都说没有就是
  `module 'x' not found (...)`。主文件说没有就是 `no such file: ...`。
- **返回非字符串、或者抛别的异常，是 `VibaProgramErr`**（`get_file(...) raised ...` / `... not the file's text`），
  不是把异常扔给调用方；源编不过照旧是 `cannot parse ...`。
- **同一个文件只问一次**：模块按路径认，已经加载过的（哪怕换了别名）不会再问第二次。

## 目录里有什么：list_files

`list_files` 是 `Optional[list[str] <- $dir_path str]`，答的是这个目录**直接**装着的名字：

```python
interpret("main.viba", environ, get_file=files.get, list_files=names_of)
```

- **留空（`None`）就列文件系统**（`os.listdir`）。
- **自己供文件的宿主，也要自己供目录内容**：泛型是一个目录，决断的第一件事就是问这个目录里
  有哪些文件。只给了 `get_file`、又要用泛型，是 `VibaProgramErr`，说清楚要一起给 `list_files`。
- **"这个目录读不到"**：返回 `None`、抛 `FileNotFoundError`，或者返回的不是字符串表，都是
  `VibaProgramErr`。

## 一个可执行的模块

```viba
# file name: add_demo.viba
__def__ =
    void
  <- $env Env

args = __get_args__ << __def__

add =
	int
	<- $env Env
	<- $a int
	<- $b int
	<- {
		add two integer
	}

print =
	void
	<- $env Env
	<- $x Any
	<- {
		print to stdout
	}

__ret__ =
	add
	<< args.env
	<< $a 999999
	<< $b 1
```

规则：

- **没有 `__ret__` 的文件是类型，不是程序**：跑它给 `VibaProgramErr`。
- **`Env` 是内建类型名**（[`viba/builtin.viba`](viba/builtin.viba) 里 `Env = Environment`）：`__def__`
  里 `$env Env` 那个参数要的就是它；模块体里读这个环境用 `args.env`。
- **函数体里的 `{...}` 是说明**：它不是参数，`<<` 给完实参之后链就落到结果上——
  `(B <- $a A) << $a A` 就是 `B`。
- **`{...}` 只给提示，不给实现**：提示只说这一步要实现什么，主要逻辑得有人照着它写出来，再交到
  `get_func` 上。写这些函数的是 agent（见「宿主侧：Environment」），viba 一个都不带。
- **每个要执行的函数都要依赖环境**：签名里必须有 `$env Env` 这个参数，调用时也必须给；
  否则 `VibaProgramErr`（不是猜一个默认值）。
- **环境不是答案**：只有内建函数（[`viba/builtin.viba`](viba/builtin.viba) 里 `Environment` 的成员）
  能把 `Env` 声明成返回值。模块自己的 `__def__`、模块里写的函数、积成员里的链，结果那个位置写
  `Env` 或 `Environment` 都当场报错——环境是调用的规矩，不是能交出去的值。解释器读文件时拦，
  判定层与描述符层在把一个函数当函数读时拦。
- **接实参时逐个核类型**：设计已经写明每个参数要什么（`$a int`），而写出来的实参自己带着类型
  （可序列化数据、字面量、函数链、环境），所以判定层那套 `<:` 在这里就能用——给错了是**程序错**
  （`VibaProgramErr`），不是"某一步的实现坏了"。判不出类型的（宿主自己的值、判定 settle 不了的）
  就放过去，交给实现那一步的人；函数类型的那个参数不在这里算，所以它等宿主叫它的时候才核，
  不叫就不核。
- `__ret__` 必须是值：可序列化数据、环境，或者一个**闭包**（见「环境就是执行」）。

## 环境就是执行

函数和模块是同一个东西：都是一条链，都要 `$env Env` 这个参数；模块的那条链就是它的 `__def__`，
结果写在这条链的最前面。

```viba
add : int <- $env Env <- $a int <- $b int     # 一个函数
lib : int <- $env Env <- $a int <- $b int     # 一个模块：`__def__` 就是这条链
```

**给环境就是执行**，也只有这一个动作是执行：给它的那一刻，`interpret` 去问宿主哪一步来实现
（`get_func(module_path, func_name)`），并按这次调用的 storage 路径把它认下来（调用的身份）。

**不给环境，链就是一个闭包**——一个值：

```viba
half = add << $a 40            # 类型 int <- $env Env <- $b int：还欠 $b 和环境
g    = add << $a 1 << $b 2     # 类型 int <- $env Env：实参给齐了，只欠环境
sg   = lib << $a 3 << $b 4     # 类型 int <- $env Env：模块也一样，实参给齐了，只欠环境
__ret__ = g                    # 一次运行可以答一个闭包：给出去的是"还没执行的活"
```

- 闭包是**可序列化数据**：写下来的函数名加上已经算好的实参，所以它能存、能传、能当 `__ret__`、能序列化。
- 闭包里只能装**可序列化数据**，只读、能写下来——别的函数和模块的返回值本来就满足这个要求。
  环境从来不装在闭包里：它是执行那一刻才给的。
- 同一个闭包可以**反复执行**：换一个环境就是换一次执行。
  `sg << (args.env.tmp_env << args.env)` 和 `sg << (args.env.sub_env << args.env << "again")` 是两次运行。
- **给了环境的链要么跑完，要么报错**：环境给了、实参还欠着，是"准备不完整"的程序错——这种
  半成品存不下来，也不该存。所以"给了一半"和"没给环境"是两件完全不同的事。
- 宿主那一侧永远只见得到可序列化数据（见「宿主侧：Environment」）：闭包到了宿主手里就是数据，可以拿着、存着、递回来。
- **写给函数类型的那个参数是例外**：`$get_v (Any <- $env Env)` 这种参数交给宿主的，是一个**能带着环境调它自己**的东西 —— 宿主决定算不算、在哪个环境下算，还可以带上自己的实参（`f(env, 1, 2)`：写下来的那次调用按这些实参走完）。它同时也记得自己代表哪份值，所以只把活转手交出去的宿主把它递回来，拿到的还是那份值。`branch.py` 就是这么用的：它给 `get_v` 一个子环境（`sub_env(env, "echo_or_never")`），于是那一支在自己的路径下算出来。

## 模块的参数：`__def__`

一个模块要收参数，就把它们写进 `__def__`——模块当函数读时的整条链：结果在前，参数在后。要执行的
模块必须在里面声明**恰好一个** `$env Env`，环境就从它来：

```viba
# file square_sum.viba
__def__ =
    int
  <- $env Env
  <- $a int
  <- $b int

args = __get_args__ << __def__

__ret__ = add
  << args.env
  << (mul << args.env << args.a << args.a)
  << (mul << args.env << args.b << args.b)
```

```viba
# file main.viba
__def__ =
    int
  <- $env Env

args = __get_args__ << __def__

import square_sum

__ret__ = square_sum << (args.env.tmp_env << args.env) << 3 << 4
```

- **`__get_args__` 把这次调用的输入读回成一份积**：推导时它答的是 `__def__` 那些参数写成的积类型
  （`args.a` 是 `int`，`args.env` 是 `Env`），计算时它答的是上游真实传过来的数据（`args.a` 是
  当初给进来的那份值，`args.env` 是真实的环境）。一个模块要读自己的参数，就写
  `args = __get_args__ << __def__`，再按 tag 取。
- **结果不能是环境**：`__def__` 的第一个元素写 `Env` 或 `Environment` 当场报错，模块里别的函数
  也一样（只有内建函数答得了一个环境，见上面「环境不是答案」）。运行时的答案不受影响：
  `__ret__ = args.env` 交回去的仍然是那个环境，只是没人声明它的类型
  （[`tests/data/environment/the_env.viba`](tests/data/environment/the_env.viba)）。
- **给环境就是执行**：每个参数一个 `<<`，按位置给（`<< 3 << 4`）或按 tag 给（`<< $b 4 << $a 3`）
  都行；按 tag 给时顺序随意。环境那个参数按值认：给一个 `Environment`、或者按 `$env` 这个 tag 给，
  都落在那里；不带 tag 的实参跳过它，依次落到其余参数上——所以 `square_sum << 3 << 4` 是把 3 和 4
  给 `$a`、`$b`，不是给环境。**环境给了，参数就得齐**：少一个就是 `VibaProgramErr`，话里点名少了谁
  （`module 'square_sum' was given 1 of its 2 parameters: $b missing`）。多给、给重、给了没有的 tag，
  同样当场报错。
- **不给环境，它就是闭包**：`sg = square_sum << $a 3 << $b 4` 是一个值——写下来的模块名加上已经
  算好的实参，可以存、可以传、可以当 `__ret__`、可以序列化，也可以**换个环境再执行一次**
  （`sg << (args.env.tmp_env << args.env)`）。"必须给全"只对执行成立，对闭包不成立。
- **一份要别人给参数的模块，单独跑只会报错**：宿主跑一份主文件时只给环境，没人写下 `$f`、`$n`，
  所以 `args.f` 读不到东西（[`tests/data/y_combinator/`](tests/data/y_combinator/) 里那两条记录
  就是这个）。
- **参数是模块读到的可序列化数据**（`args.a` 那种），不是一次"要的时候再来拿"的调用：函数类型的
  参数说的是宿主那一侧的实参，模块的参数给的就是值。
- **一个片段在写它的那个模块里读**：名字、闭包、模块的参数，都带着"它是哪个模块写的"这一条，
  交出去、存下来、再读的时候还是回那儿解——`F` 在一份文件里是 import，在另一份里什么都不是。
  所以 `args.f` 交回去的是**当初给进来的那份值**，不是照着收它的那一方重新读一遍。
- **模块体里 `args` 就是那份实参积**：`args.a` 是 `$a` 那个成员的值，按 tag 寻址——这里说的
  **地址**是数据存下来的那条路径，由一串步子接成（按 tag、按位置、按下标、按键），一段一段走下去
  就是寻址；`$a` 是它的一步，模块路径和 `cur_storage_path/<name>.viba` 也是地址。「幂等与快照」里的
  "快照地址"、「一次执行会得到什么」里的"这次调用的地址"都是这个意思，步子分哪几种见
  [`viba-reflect.md`](viba-reflect.md) 第 4 节；判定层读同一个名字读到的是成员**声明的类型**
  （`args.a` 就是 `int`）。一个名字，两层各读各的。
- 实参必须是可序列化数据：它是积的一部分，交出去的东西得是能写下来的值。**环境不是它的数据成员**
  ——环境是宿主的对象，`args.env` 读得到，整份积交给宿主时不在里面；要交给宿主的是一个一个成员
  （`sum_of << args.env << args.a << args.b`）。
- **实参要装得下那个参数**：`$a int` 那个参数给 `"x"` 是**程序写错了**，当场 `VibaProgramErr`
  （`module 'square_sum': "x" does not fit $a int: "x" <: int does not hold`），宿主根本看不见这个
  实参。函数那边同理：`add` 的 `$a int` 给字符串也一样。

## 调用别的模块

```viba
import add_demo as demo

ret = demo << (args.env.sub_env << args.env << "add_demo")

__ret__ = demo.print << args.env << ret
```

- `demo` 是模块，当函数用：给它一个环境（它只收这一个参数），它跑完给出它的 `__ret__`。
- `demo.print` 是这个模块里的函数。
- `args.env.sub_env << args.env << "add_demo"` 拿一个子环境：**它带着父级的 compute**，storage 路径是
  `父路径/add_demo`（见「幂等与快照：结果要能回放」）。同一个调用还可以写成
  `$sub_env << args.env << "add_demo"`（见「如何调用方法：链头写 tag」）。

**一次模块调用的身份就是它的 storage 路径**：那条路径就是这次调用的地址——宿主拿到的
`get_func(module_path, func_name)` 里的 `module_path` 就是它。一条路径上只该有一次调用，三种
情形分得很清楚：

- **这条路径正在跑**（同一个地址上还没答完，又要进去一次）：真是环，`VibaProgramErr`，话里给出
  写法：

  ```
  the storage path 'root' is already running a call, so 'lib' cannot run there too:
  give each module call a sub-environment of its own (args.env.sub_env << args.env << ...)
  ```

  主文件自己也算一次激活，占着它那条路径——直接 `lib << args.env` 就是这一种。
- **这条路径答过了，而且是同一个模块**：同一个子计算被问了第二次，把它答过的那份**回放**回去
  （见「幂等与快照：结果要能回放」）。`args.env.sub_env << args.env << "x"` 对同一个父环境是
  **同一个** storage（同名子环境要到才造、造一次、之后复用），所以同一个模块两次用同一个名字，
  是**一次计算、两次取用**。
- **这条路径答过了，但换了一个模块**：两条不同的调用挤一个地址，宿主分不出谁是谁 → `VibaProgramErr`，
  话里给出正确写法（`... which another module call already used`）。

### 如何调用方法：链头写 tag

`$tag << X << a` 就是 `X.tag << X << a`：**tag 写在链头时，它命中的是第一个参数的成员，
而那个值也接着被交给成员当第一个实参**。环境的成员因此有三种写法，一样长：

```viba
args.env.sub_env << args.env << "add_demo" # 点号，环境自己写出来
$sub_env << args.env << "add_demo"         # tag 写链头，同一个调用，短一些
$sub_env << $env args.env << $sub_env_name "add_demo"  # 都按 tag 给，也对
```

`viba/builtin.viba` 里 `Environment` 的成员就是这样声明的——环境是它的第一个参数：

```viba
  * $sub_env (Environment <- $env Env <- $sub_env_name str)
  * $tmp_env (Environment <- $env Env)
```

成员也可以写在数据里：一个积的 `$f` 里放一个函数名时，`$f << box << args.env << 1` 就是
`box.f << box << args.env << 1` —— 成员收的第一个参数是它所在的那个值（`$box Box`），环境跟在后面。

tag 本身**不是值**：`method = $sub_env` 编不过，标签只有写在链头、后面跟着第一个参数时才成立。
第一个参数必须写出来——它是取成员的那一个，省略了就成了一次没有成员的调用。第一个参数是别的值
也一样：`$tag` 命中的是它的成员，不在就报错。

不想起名字就用 `tmp_env`：

```viba
lib << (args.env.tmp_env << args.env)
```

它像临时文件一样，**每次调用都给一个新的子环境**（路径是 `父路径/tmp_<随机>`），所以两次调用
天然各占一条路径。`viba/builtin.viba` 里 `Environment` 的成员因此写的是
`$tmp_env (Environment <- $env Env)`：它只收那个环境，名字不用给。

路径每次都不同，这是 `tmp_env` 的语义：它给**纯函数调用**、或者**结果不留的调用**用。一个
需要快照、要靠回放才幂等的函数如果挂在它底下，回放自然命中不了——每次的路径都是新的，快照会
一次次写进不同的 `tmp_` 目录（`store_root/root/tmp_xxxx/…viba`），那条不纯的路于是每次都重走，
**幂等检查会失败**。这不是缺陷，而是它给出的信号：这个函数需要显名保存，应该换到
`args.env.sub_env << args.env << "一个稳定的名字"` 上去。`tests/test_interpreter_idempotence.py` 里两半都有用例：
显名路径第二次跑就回放，临时路径两次都重算、并且留下两份快照。

## 宿主侧：Environment

`Environment` 由宿主提供。三个名字里**只有 `Environment` 是 viba 里看得见的那一个**
（[`viba/builtin.viba`](viba/builtin.viba)，短名字是 `Env`）：`__def__` 里 `$env` 那个参数要的就是
它，模块体里读它用 `args.env`。

```viba
Environment =
    Object
  * $viba_path str
  * $sub_env (Environment <- $env Env <- $sub_env_name str)
  * $tmp_env (Environment <- $env Env)
```

storage 与 compute 是**宿主的词汇**，viba 里没有一个名字指得到它们（它们不是类型，是宿主
对象），所以只在宿主侧出现（`viba.interpret`）：

```python
EnvironmentStorage(cur_storage_path, sub_storage=None, store_root_dir=None)
EnvironmentCompute(get_func)                             # get_func(module_path, func_name)
Environment(storage, compute, viba_path=None)            # sub_env / tmp_env 给子环境
```

`viba_path` 是模块的搜索路径：一次运行里它跟着 environment 走，`sub_env`/`tmp_env` 把父级的
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
**已经算好的实参**，按书写顺序给——实例是 `viba.reflect.VibaNode`，别的（环境在内）是它本身。
返回值是 `VibaNode`，或者一个普通 Python 值（落到函数声明结果的一个叶子上）。

参数里出现 **viba 函数**（`$f (int <- $env Env <- $x int)` 这种高阶签名）时，宿主拿到的是**可序列化数据**：
那个函数名本身（一个 `VibaNode`），或者一个闭包——写下来的函数名加上已经算好的实参。**宿主调不动它**：
viba 函数不是宿主侧的 Python 可调用对象，执行它们只有一条路，就是给环境（`<< args.env`），那是 viba 那一侧
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

**模块调用按同一条规矩办**：一次调用的身份就是它的 storage 路径，所以那条路径答过之后，同一个
模块再来问同一个地址，拿到的就是答过的那份（见「调用别的模块」里的三种情形）——一次运行里
省掉的是重复计算，跨运行回放靠的是上面那把快照。

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
  `args.env.sub_env << args.env << "稳定的名字"` 就是稳定的；`tmp_env` 每条路径都是新的，挂在它底下的调用不适合
  保存要回放的东西（上一节：这正是它给出的"该显名保存了"的信号）。
- **快照是序列化的 viba 数据**（`viba.serialize` 写出来的 `value = …`），不是 pickle：存下来
  的东西可以被人读、被人看、被人拿去喂类型推导。回放时解析回实例，叶子和存进去的时候一样。
- **存不了、回放不出来就是错**：`VibaProgramErr`（宿主抛出来，interpret 转成 `VibaProgramErr`），不会静默给个默认值。

于是"随机"也能回放：

```viba
__def__ =
    int
  <- $env Env

args = __get_args__ << __def__

roll =
	int
	<- $env Env
	<- $n int
	<- { roll a die: not a pure function, so its answer is snapshotted }

__ret__ = roll << $env args.env << $n 1
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

`branch.viba` 与 `branch.py` 提供两个开关。它们接收分支值，返回 `Any`；`$get_v` 那个参数写着函数类型
`(Any <- $env Env)`（见「环境就是执行」），所以那次实参不在这里算——宿主叫它的时候才算，
在宿主给的那个环境里算：

```viba
echo_or_never =
    Any
  <- $env Env
  <- $cond bool
  <- $get_v (Any <- $env Env)

never_or_echo =
    Any
  <- $env Env
  <- $cond bool
  <- $get_v (Any <- $env Env)
```

`branch.py` 是宿主实现，和其他可执行函数一样由 `EnvironmentCompute` 显式注册：

```python
import branch

def get_func(module_path, func_name):
    return branch.get_func(module_path, func_name)
```

`$get_v` 收到的是一个能带着环境调它自己的东西，所以开关叫哪个才算哪个：

```python
def echo_or_never(env, cond, get_v):
    if cond.value:
        return get_v(sub_env(env, "echo_or_never"))
    return _never()
```

- `echo_or_never`：condition 为真时按单位元 `nil * v = v` 返回 `get_v`，否则按 `never * v = never`
  返回 never——**`get_v` 不会被叫**，那一支写下来的实参根本不求值（非严格求值）；
- `never_or_echo`：condition 为真时按 `never * v = never` 返回 never，否则按单位元 `nil * v = v`
  返回 `get_v`。

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

a = foo << $env args.env
condition = ge << $env args.env << $x a << $y 0

__ret__ =
    Oneof
  | (branch.echo_or_never << $env args.env << $cond condition << $get_v (builtin.echo << $x a))
  | (branch.never_or_echo << $env args.env << $cond condition << $get_v (builtin.echo << $x 0))
```

那个实参**最多求值一次**（memoization，也叫 sharing）：第一次问出结果（值或停下），之后每一次问都拿
同一个。所以宿主问两遍不会让副作用发生两遍。

只有函数类型的那个参数是非严格的（non-strict）；别的实参按值求值（call-by-value），函数的 `$env`
也照旧按值给。想给一个求好的值，用 `builtin.echo` 把它包成"给它一个环境就答 V"的那个函数。

这跟一份定义什么时候求值是同一条策略，见下面「求值策略：按需求值（call-by-need）」一节。

条件、值和开关本身都可以写在别的文件里，用例只写一条链去调它们；一个调用先给一半、剩下的由用例
补齐，也是同一个值。`echo_or_never` / `never_or_echo` 本身也是这样一份可以 import 的文件。这条
路子的边角案例在
[`tests/test_interpreter_branch_switch.py`](tests/test_interpreter_branch_switch.py)：一份用例一个
文件（`tests/data/branch_switch/*.viba`），覆盖开关方向、条件的各种写法、每一种值的种类、部分计算、
别的文件里的开关、副作用次数、走不到那一支里的递延与失败，以及多个开关并起来的和。

## 类型层的模块

同一个文件在**类型推导**里也是"参数进、`__ret__` 出"：名字绑到的是一个模块（import 的名字）
时，它当函数读，类型就是它的 `__def__` 去掉环境那个参数：

```viba
int <- $a int <- $b int       # `__def__ = int <- $env Env <- $a int <- $b int` 读出来的类型
```

**环境不在类型里。** `$env Env` 是调用的规矩——给环境就是执行，一份函数定义没有那个参数就跑不起来
——不是设计的一个参数：没有人替它写一个值。所以运行、判定、描述符三层读出来的函数类型里都没有它，
`int <- $env Env <- $n int` 与 `int <- $n int` 是同一个类型；写成模块、写成函数定义，只要是同样的
设计参数，就是同一个类型。

所以这几条都成立（`tests/test_is_sub_type.py` 里有用例）：

```viba
demo = import add_demo as demo         # 概念上
demo << args.env  <:  int         # 就是 __ret__ 的类型；只收环境的模块，给环境就是执行
int <: demo << args.env           # 反过来也成立：两者同型
demo.add <: int <- $env Env <- $a int <- $b int
design.Only <: $x int                   # module.MyType 仍然是那个模块里的定义
square_sum << args.env << 3 <: int <- $b int   # 给了一半：剩下的是函数
```

- 只认 **import 绑定的那个名字**（`import a.b as c` 的 `c`，`import a.b` 的 `a.b`）。`module.Name`
  仍然是那个模块里的定义，按最长的前缀解析。
- 没有 `__ret__` 的模块不是程序：`demo << args.env` 在类型层也是 `VibaProgramErr`。
- 不带 tag 的实参跳过环境那个参数，和值层同一套规矩；环境那个参数给的就是环境本身，写 `args.env`
  或按 `$env` 这个 tag 给都认。
- 实参按 tag 给（`<< $a 3`）或按位置给（`<< 3`）都认；位置那一支要求给的实参装得下那个参数
  （`3 <: int`）。**给一半在类型上是一种类型**——剩下的那个函数；在值层给一半是程序错。
- **`__get_args__ << __def__` 在类型层答的是积类型**：成员就是 `__def__` 的那些参数——`args.env`
  是 `Env`、`args.a` 是 `int`。`__def__` 不是函数链的话，两层都当场报错。

## 模式：一个泛型是一个目录

`import demo.is_base_type as is_base_type` 也可能找到的是一个**目录**（目录里有 `__generic__.viba`
标记）：那是一个泛型，它的每个**数字文件名**是一个模式（[`viba-pattern.md`](viba-pattern.md)）。
它不是一个模块——单独写 `is_base_type` 不是类型，`is_base_type << $x 1` 不是调用。能写的只有应用：

```viba
Flag = is_base_type[bool]              # 决断选中 100.viba；它的 __def__ 是 true
```

跑起来的时候，决断**只看写下来的那几个类型**（静态动作），选中谁，就把谁的 `__def__` 拿到
**那个文件**里读：形参名（文件里没定义的那个名字，比如 `A`）绑定到实参里对应的那一部分，
这个绑定在那个文件的每一次求值里都算数，包括它自己定义里的那一层。

- 写下来是字面量，答的就是那个值（`__def__ = true` 答 `true`）；
- 写下来是萃取到的形参，答的就是那份类型（`__def__ = A`，答 `int`）；
- 写下来是数据，答的就是数据，形参名已经换成实参里写的那份（`__def__ = (A, B)` 答 `(int, str)`）；
- 写下来是**函数链**，这个应用代表的就是那次**调用**，跟"定义体是函数链"一样：可以接着给实参
  （`wrapper[add] << args.env << add << $a 1 << $b 2`），也可以当闭包递出去。宿主按**泛型的名字**找实现
  （写下来是 `wrapper[...]`，它拿到的是 `"wrapper"`），跟按定义名找实现同一种做法；
- 那个函数链所在的文件自己写了 `__ret__` 时，这个应用就是**它那一次模块调用**：环境与实参按它
  的 `__def__` 给（调用方给环境，欠着的实参接着给），活由它的 `__ret__` 自己算，宿主不必按泛型
  的名字另找一份实现。它跑在自己的**子环境**里，那一层的名字是这份文件的**数字**（`100.viba`
  跑在 `…/100` 下），所以一次调用的地址仍然写得下来、复现得了 —— 调用方不必自己再包一层。
  文件没写 `__ret__` 时是上一条：`__def__` 就是它答的类型，宿主按泛型的名字实现；
- 其余照平时的规矩：`int` 这个名字在值的位置上不是值。

`(A <- $env Env <- B)` 这种写着函数类型的参数，交给宿主的是它代表的那次调用：宿主带着环境叫它，
还可以带上自己的实参（`f(env, 1, 2)`），那些实参落进这次调用还欠的槽位。多喂的那一个由 viba 这边
说清楚（`takes no more arguments`），不是把宿主那边的 `TypeError` 报成实现坏了。

方括号里的实参本身写成一次 `<<` 时（`wrapper[add << $a 2]`），决断先把它**当类型读出来** ——
已经给过 `$a 2` 之后剩下的那条链条 —— 再照模式读开，所以"已经给过一部分实参的函数"也是一份
能收的实参。给实参要写出 tag（`$a 2`）：`$env Env` 是调用的规矩，不是实参的位置。

一个在选中的文件里建起来的、还欠着实参的调用（半成品）**记着决断绑下的那些名字**，跟它记得
自己写在哪个模块一样。这样的半成品递到别处再读时 —— 当实参交给另一个模块，或者存成闭包之后
再给实参 —— 文件里只写了名字的位置仍然指回**调用方写下的那一份**，包括"这个名字又出现在另一个
泛型应用的方括号里"那一层：决断读那种实参时回到写它的模块，而不是在写着这个名字的文件里读。
`tests/data/y_combinator/` 的两份泛型用的就是这一点（`Y/100.viba` 的 `__def__` 里写着
`y_helper[F]`，`F` 是它的形参）。

## 成员按名字读：`__tagged__` 与 `$__getattr__`

名字写在字符串里时，成员照样取得出来。`__tagged__["hello"] << X` 就是 `$hello << X` ——
一参数的 `__tagged__` 是一个成员，只写在链头（[`viba-pattern.md`](viba-pattern.md) 第 3 节）。

名字是**一份可以算出来的值**时（决断萃取出来的符号、从数据里读到的 str），用内建的成员
`$__getattr__`：

```viba
__ret__ = $__getattr__ << args << "name"          # 就是 args.name
__ret__ = $__getattr__ << box << name << args.env << 1   # name 是算出来的 str
```

它读一个值的成员：积里那条 tag 的那一份，环境上就是那个宿主属性。成员是函数时，那个值也照样当它
的第一个实参给出去（跟 `$tag << X` 一样）；成员就是一份值（`args.n`）时不给：它是值，值收不下
实参，`$__getattr__ << args << "n"` 就是 `args.n` 本身。名字不是字符串、或者这个值没有那个成员，
都是程序错。

判定层读同一条链时，名字是写下来的字符串（或解析得出字符串的名字）就给出**那个成员的类型**；
名字读不出来时给 `Any`：哪个成员是运行的时候才知道的，设计说不出它的类型。

宿主自己供文件时，`list_files` 也要一起给：决断要问目录里有什么（见上面「目录里有什么：list_files」一节）。

## 求值策略：按需求值（call-by-need）

viba 是**声明式**（declarative）的，不是命令式执行的：一份文件里的定义是**绑定**（binding，跟 let
绑定的地位一样），不是按写下来的次序一条一条跑的语句。求值由需求驱动（demand-driven）—— 一个绑定
在被**用到**的时候才求值，而且求值一次就把结果留下来（memoization，也叫 sharing），那次的结果就是
它以后的值。这就是**按需求值**（call-by-need），也就是惰性求值（lazy evaluation）。

- **没人用到的绑定不参与求值**：[`tests/data/values/nested.viba`](tests/data/values/nested.viba)
  里那份 `unused` 没人用，宿主一次都没被叫。
- **绑定可以前向引用**（forward reference）：`__ret__ = x` 与 `x = 7` 换个次序还是 7
  （[`defined_after.viba`](tests/data/values/defined_after.viba)）。写下来的次序不是求值的次序。
- **一个绑定只求值一次**：两处用同一个名字，宿主只被叫一次
  （[`memo.viba`](tests/data/values/memo.viba)）；同名写两份时，后写的**遮蔽**（shadow）前一份
  （[`twice_defined.viba`](tests/data/values/twice_defined.viba)）。

这条策略不是省一次求值：递归要"再一次进到同一个定义"，而一份文件里的绑定不许成环（下一节），所以
往下那一层只能先写成一个名字 —— 名字就是一份文件手里那个**还没求值的计算**（thunk）。写它的那一层
不用它，用它的那一层才把它求出来。`tests/data/y_functions/steps/` 那批用例就是这样写的：往下调一层
与取余各是一个绑定，基例那一侧不用它们，于是那一侧既不会去算 `a` 除以 0 的余数，也不会再往下调一层
（从基例那一侧进来的那一份是
[`main/gcd_zero.viba`](tests/data/y_functions/main/gcd_zero.viba)）。

一个调用的实参默认**按值求值**（call-by-value）；只有函数类型的那个参数**非严格**（non-strict）——
写在那个参数上的调用连同环境交给宿主，宿主用到它时才算，见上面「分支：用积选择，用和汇合」那一节。
求值到一半的绑定被自己用到（`A = B` 与 `B = A`，或者 `A = use << $env args.env << $x A`）是循环定义，
当场报 `one file's definitions may not go round`，而不是等栈崩（下一节）。

## 一份文件跑不出递归

递归要"再一次进到同一个定义"，而按需求值不会把一份文件里的绑定反复求出来：一个绑定就是它那一个
thunk，求一次就定了。一份文件自己的定义图因此是无环的——**文件里的定义不许绕回自己**——所以单独
给一份文件，怎么跑都跑不出递归：写出来的调用点就那么多，`<<` 一处一处写死。

```viba
A = B
B = A
# A -> B -> A: one file's definitions may not go round — recursion takes two files
```

- **绕回自己当场报错**，不是等栈崩：一份文件里 `A = A`、`A = B` 与 `B = A`、以及"求 A 的时候
  又要 A"（`A = use << $env args.env << $x A`）都是 `VibaProgramErr`，话里给出绕的路径。
- **没被求值的不算**：函数类型的实参没人叫就不算递归（`A = ignore << $env args.env << $x A` 答 7，
  因为 `ignore` 不叫那个实参）；`List[T] = Object * $tail List[T] | nil` 这种**类型**自引用也
  不算——类型不求值。
- **跨文件不设这条限制**：两份文件可以互相 import、互相调用，设计层不拦。真的绕回去（`a.x` 要
  `b.y`、`b.y` 又要 `a.x`）时，这次运行报
  `a.x -> b.y -> a.x: the run came back to where it started`；模块调用再进同一条路径报
  `the storage path '...' is already running a call`；同一个模块带着**同一份实参**又在跑报
  `module 'a' is already running with the same arguments`——同样输入的一次计算还在里面进行，
  它停不下来。这些都是"这次跑不完"，不是设计不合规。
- **同名、不同实参、各自一条新路径的调用是两次调用**，不是环：函数递归执行要"再一次进到同一个
  定义"，而固定的实参换一次就是另一层。不动点正是这样展开的：
  [`tests/data/y_combinator/`](tests/data/y_combinator/) 里的 `Y F 10` 答 55
  （Z 组合子的 viba 版：欠着实参的调用就是值，把自应用停在半成品上）。
- **两份文件互调的边角**：这条路子本身有一份用例集，
  [`tests/test_interpreter_mutual_recursion.py`](tests/test_interpreter_mutual_recursion.py)：100 条，
  每条是一对互相 import 的文件（`tests/data/mutual_recursion/left_*.viba` 与 `right_*.viba`）。
  分支与部分计算决定那条路走不走：写在函数类型那个参数里的递归调用，条件不成立时一次都不发作；
  一个调用先给一半（`right.f << $a 1`）是一个值，没进入另一份文件，也不发作；而条件那个参数照旧
  先算，所以写在条件里的递归调用一定发作。每条用例跑出值、跨文件绕回去、模块调用成环、闭包、
  `never`、和值、积或写下来的名字之一。

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
  实现"很正常，不是错误。

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
... takes no $env Env parameter ...   要执行的函数没声明环境参数
... needs an Environment ...          一次模块调用没给环境（它只收环境）
... was not given an Environment      给了，但不是 Environment
... storage path ... already running  同一条 storage 路径上已经有一次调用在跑
module 'x' ... same arguments         同一个模块带着同一份实参又在跑（没有进展）
... storage path ... already used     两个模块挤同一条 storage 路径
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
the environment's sub_env raised ... 环境上挂的宿主函数坏了
```

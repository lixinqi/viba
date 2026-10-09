# 执行 viba 模块

一个 viba 文件是一个 Module，而 Module 有两种取法：

1. **类型推导**：把它当类型判（`viba.is_sub_type`、`viba-reflect.md`）；
2. **求值执行**：把它跑起来（`viba.interpret`）。

同一份语法，两种模式。跑的时候，模块**就是函数**：`__decl__` 是它的签名，输入是环境与实参，
输出是 `__impl__`。

它回答的就是声明里那个 `InterpretResult`，**一份 viba 数据**（一个节点）：`$ok` 那一支带这次执行
算出来的值，`$err` 那一支带它为什么停下。声明放在 [`viba/interpret_result.viba`](viba/interpret_result.viba)
里 —— 那份文件就是这个类型的定义，`interpret` 与 `exec` 的回答按它取、按它判
（`viba/interpret.py` 的 `is_interpret_result` 用 `is_sub_type` 判它，所以"回答到底符不符合这个定义"
本身是可查的）。它用到的名字里，`Environment` 与 `VObject` 不在定义里（前者在「宿主侧：Environment」，
后者在 [`viba-reflect.md`](viba-reflect.md)）：

它的签名（viba 一型）：

```viba
interpret =
    InterpretResult
  <- $viba_main_file str
  <- $environ Environment
```

Python 侧的入口是 `interpret(viba_main_file, environ)`，回来的是一个 `VObject`：`$ok` 是这次执行算出来的值，
`$err` 是这次执行是怎么停下的。声明本身（`InterpretResult`、`InterpretError`、四种停法各自的那些字段）
放在 [`viba/interpret_result.viba`](viba/interpret_result.viba)，取它就是取那份文件。

要跑的模块**就在手里**、不从文件取时，用 `exec`：参数形式和 `interpret` 一样（环境、`get_file`、`list_files`），
只是第一份不是文件路径，而是那份模块本身 —— 它的文本，或者已经落成 viba 数据的那一份。

```viba
exec =
    InterpretResult
  <- $viba_code (str | VObject)
  <- $environ Environment
```

- `viba_code` 是**模块文本**时，它给出的模块就是这次的主模块：`__decl__` 是它的签名、`__impl__` 是它的结果，
  和 `interpret` 同一套规矩。
- `viba_code` 是**一个节点**时，不用落成文本再反序列化：那份数据给出模块，`Module` 就是那个模块（`__decl__`、
  `__impl__` 照旧）；不是 `Module` 的一份数据就是这次主模块的 `__impl__`，而**这次运行给它环境** ——
  `$call` 是一份给出这次调用的数据（名是数据，判定层取这种链就判成 `Any`），给它一个环境就是执行它，
  所以把 `$call` 那一格交给 `exec` 就是「把这次调用再做一遍」，中间没有文本
  （`tests/test_interpreter_roundtrip.py` 里那条 `as_node` 的路）。
- **主模块没有文件、也没有名字**：`import` 只按环境的搜索路径找（它旁边没有目录可以找），`$stack` 最外那一帧
  的 `$file_path` 是空串，解析不了的名字报的是 `in module ''`，文本编不过时说 `<viba_code>`（那份文本编译时
  用的就是这个名字）。它也不绑名字：给出的模块不是谁 import 得到的文件。
- 别的都一样：环境不是 `Environment`、`viba_path` 给了别的类型、`get_file` / `list_files` 不是函数、第一份
  既不是文本也不是节点，都是 `VibaProgramErr`，话里点名是哪个入口。

模块在哪找是**环境**的事：`Environment.viba_path` 相当于 PYTHONPATH，冒号分隔，按顺序找 `<name>.viba`
（dotted 名当路径走；空条目和不存在的目录跳过）。import 的那个文件所在的目录总是
先找——**被 import 进来、又在自己的文件里 import 的模块，也按它自己的文件找**（链多深都一样）。
一个名字也可能是一个**泛型目录**（`<name>/__generic__.viba`）：文件先找，目录随后，两者都按这里的
顺序（见 [`viba-pattern.md`](viba-pattern.md)）。
带 import 的文件里的名字，按 import 绑定的名字解析（`import a.b as c` 绑 `c`，`import a.b` 绑 `a.b`）。
点号取的是成员：`a.b` 是在 `a` 上取成员 `b`（[`viba-style.md`](viba-style.md) 第 5 节），
所以 `a.b` 当类型判是那个成员的声明类型，出现在链头就是那一步的调用 —— 这一步问的是它**在哪个模块、
在那儿叫什么**（下一节的 `get_func`）；定义左边只有一个名字，`a.b = A` 编不过。用别名包一层也一样：
`b_scale = foo.bar.b_scale` 之后 `b_scale << …` 问的还是 `foo.bar` 那个模块里的 `b_scale` ——
一步是按它**代表的那次调用**问的，不是按它在那儿叫什么名字问的。
同一个名字不许在同一个模块里定义两次：定义两次是程序错误，取一个名字不必猜它指哪一份。
`Environment.viba_path` 也可以直接给一个路径（`Path`）；给了别的类型是 `VibaProgramErr`，不是把
`AttributeError` 抛出来。**同一个文件只问一次**：模块按路径认，已经加载过的（哪怕换了别名）不会再问
第二次。

## 一个可执行的模块

```viba
# file name: add_demo.viba
__decl__ =
    void
  <- $env Env

args = __get_args__ << __decl__

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

__impl__ =
	add
	<< args.env
	<< $a 999999
	<< $b 1
```

规则：

- **没有 `__impl__` 的文件是类型，不是程序**：跑它给 `VibaProgramErr`。
- **`Env` 是内建类型名**（[`viba/builtin.viba`](viba/builtin.viba) 里 `Env = Environment`）：`__decl__`
  里 `$env Env` 那个参数要的就是它；模块体里要用环境，用 `args.env`。
- **函数体里的 `{...}` 是说明**：它不是参数，`<<` 给完实参之后链就落到结果上——
  `(B <- $a A) << $a A` 就是 `B`。
- **`{...}` 只给提示，不给实现**：提示只说这一步要实现什么，逻辑要另外补上，落成函数再交到
  `get_func` 上（见「宿主侧：Environment」）。viba 一个函数都不带。
- **内建算子就是这样的几个成员**：`viba/builtin.viba` 里 `builtin` 的成员（`$add`、`$lt`、
  `$concat`……），每个一张签名，实现在宿主手里。`builtin.add << …` 与 `add << …` 是同一次调用，
  宿主拿到的名字都是 `builtin.add`；自己模块里同名定义优先，所以它排在任何别的名字之后
  （[`viba-style.md`](viba-style.md) 第 11 节）。
- **内建库里的模块与泛型，也照样按名字取**：`viba/` 里 `builtin.viba` 旁边那几个模块（`Y.viba`、
  `apply.viba`、`sub_env_run.viba`、`sequential.viba`），以及 `viba/builtin/` 下的那几个泛型
  （`is_closure/`、`unclosure/`、`sequential_impl` 用的 `sequential_step/`、`sequential_arg/`）
  都不需要 import，带的那个前缀（`builtin.sub_env_run`、`builtin.is_closure`）
  叫的是同一个。`sub_env_run` 是其中的一个：给它的环境取那个名字的
  孩子，链上剩下的实参交给那次调用（[`viba/sub_env_run.viba`](viba/sub_env_run.viba)）。
  `is_closure` 与 `unclosure` 是那两个泛型：前者问一份源码里的东西是不是闭包（有实参、没给环境
  的调用），后者把闭包拆成 `f`（那条 api）与 `captured`（收下的那份积）。调用模式一个 `<<` 对一个
  实参，所以两份各有 1..16 段的 16 个文件（[`viba-pattern.md`](viba-pattern.md) 第 2 节）；
  文件名前面带着份数，决断因此只取份数对得上的那一份（同第 4.1 节）。
- **`sequential` 把一串步骤按源码里的次序跑完**，整个链的结果就是最后那一步的结果
  （[`viba/sequential.viba`](viba/sequential.viba)）：`sequential << $x (…) << $y (…) << (…)` 是一个
  闭包，环境最后给。每一步都是一次调用；除最后那一步之外，每一步前面带一个 tag（`$x (…)`）—— 跑那次
  调用，结果记在那个名字下；**最后那个参数不带 tag**，它没有名字，它的结果就是整条链的结果。想返回
  前面某一步的值，就把那个名字取出来交给内建的 `echo`（`echo << $x V` 原样给出 V）：
  `<< (echo << $x ($var "x"))`。步骤的实参里也可以是变量引用（`$a ($var "x")`），它们在调用跑起来
  之前换成值。步骤数决定 `sequential_impl` 选哪一份文件（2..64 步各一份；一步没有 tag 钉住个数，
  按那次调用的实参个数各一份），一次调用的实参个数决定 `sequential_step` 选哪一份，
  一个实参的源码形式决定 `sequential_arg` 选哪一份。这几份目录里的文件名前面带着份数（`2_200.viba`、
  `1_101.viba`），所以一条链只取它自己那几份，79 份模式文件不必翻一遍
  （[`viba-pattern.md`](viba-pattern.md) 第 4.1 节）。每一步、每个实参的调用都跑在自己的数据路径上
  （`step0`、`arg0` 那样）：一条数据路径只处理一次调用。
  变量引用在源码里的样子像一个 tag（`$var "x"`），设计时看不出它将来是什么值，所以它落在的槽位
  要收得下这个源码形式（`$a Any` 那样）；落成 `$a int` 是程序错。一步的闭包可以是这个文件里的定义、
  import 进来的模块，或者内建算子都可以；链头是内建**模块**（`sub_env_run`、`apply`、`Y`）时，
  类型侧还看不出这个片段（判定与实参核对那一层报 PartialError），要它现在就能用，就把它包进一个
  本文件里的模块再当步骤。
- **要执行的函数都得拿到环境**：`__decl__` 里可以有 `$env Env` 这个参数，也可以没有。调用时给环境的方式
  一样：一个 `<<` 给一个环境值（`square_sum << (args.env.tmp_env << args.env) << 3 << 4`）。用了，环境
  就给这个参数，模块体用 `args = __get_args__ << __decl__` 和 `args.env` 取到它。没用，**环境不进门**：
  它不是这次调用收到的一员，`args.env` 取不到（用它是程序错），它只用来跑这次调用 —— 这次调用跑在
  自己的子环境里，名字就是模块在源码里的那个名字 —— 再缀在这次调用返回的那次调用末尾（见「环境就是
  执行」）。环境没给，这次调用就还是个闭包，不会拿一个默认环境顶上。
- **环境不能当结果**：只有内建函数（[`viba/builtin.viba`](viba/builtin.viba) 里 `Environment` 的成员）
  能把 `Env` 声明成返回值。模块自己的 `__decl__`、模块里定义的函数、积成员里的链，结果那个位置是
  `Env` 或 `Environment` 都当场报错——环境是调用的规矩，不是能交出去的值。解释器取文件时拦，
  判定层与描述符层在把一个函数当函数判时拦。
- **接实参时不核类型**：设计说 `$a int`，那是给**判定层**看的（把一个函数链判成某个类型、给一份
  设计取实参时才用得上）；"这一次调用给了这一个值"这一步不走判定 —— 给 `"x"` 就给 `"x"`，它交到
  那一步的实现手里，谁拿它做整数运算谁在那里撞上（报的是 `$underlying_viba_op_err`，不是程序错）。
  函数类型的那个实参也一样：不在这里算，宿主叫它的时候才算。
- `__impl__` 必须是值：可序列化数据、环境，或者一个**闭包**（见「环境就是执行」）。

## 环境就是执行

函数和模块是同一个东西：都是一条链，环境是这次调用的规矩 —— 放在 `$env Env` 这个参数上，或者由
调用方单独给；模块的那条链就是它的 `__decl__`，结果在这条链的最前面。

```viba
add        : int <- $env Env <- $a int <- $b int   # 一个函数
add_module : int <- $env Env <- $a int <- $b int   # 一个模块：`__decl__` 就是这条链
```

**给环境就是执行**，也只有这一个动作是执行：给它的那一刻，`interpret` 去问宿主哪一步来实现
（`get_func(module_path, func_name)`），而这一步跑在给它的那个环境的数据路径上。

**不给环境，链就是一个闭包**——一个值：

```viba
half = add << $a 40                  # 类型 int <- $env Env <- $b int：还欠 $b 和环境
g    = add << $a 1 << $b 2           # 类型 int <- $env Env：实参给齐了，只欠环境
sg   = add_module << $a 3 << $b 4    # 类型 int <- $env Env：模块也一样，实参给齐了，只欠环境
__impl__ = g                         # 一次运行可以给出一个闭包：给出去的是"还没执行的活"
```

- 闭包是**可序列化数据**：源码里的函数名加上已经算好的实参，所以它能存、能传、能当 `__impl__`、能序列化。
- 闭包里只能装**可序列化数据**，不能改、能序列化成源码——别的函数和模块的返回值本来就满足这个要求。
  环境从来不装在闭包里：它是执行那一刻才给的。
- 同一个闭包可以**反复执行**：换一个环境就是换一次执行。
  `sg << (args.env.tmp_env << args.env)` 和 `sg << (args.env.sub_env << args.env << "again")` 是两次运行。
- **给了环境的链要么跑完，要么报错**：环境给了、实参还欠着，是"准备不完整"的程序错——这种
  半成品存不下来，也不该存。所以"给了一半"和"没给环境"是两件完全不同的事。
- 宿主那一侧永远只见得到可序列化数据（见「宿主侧：Environment」）：闭包到了宿主手里就是数据，可以拿着、存着、递回来。
- **函数类型的那个参数是例外**：`$get_v (Any <- $env Env)` 这种参数交给宿主的，是一个**能带着环境调它自己**的东西 —— 宿主决定算不算、在哪个环境下算，还可以带上自己的实参（`f(env, 1, 2)`：源码里的那次调用按这些实参走完）。它同时也记得自己代表哪份值，所以只把活转手交出去的宿主把它递回来，拿到的还是那份值。`branch.py` 就是这么用的：它给 `get_v` 一个子环境（`sub_env(env, "echo_or_never")`），于是那一支在自己的路径下算出来。

## 两种语义：模块与函数

一份文件就是**一个模块**，它只有两种意思。**模块语义**是"一份定义的映射"：顶层每份定义都是它的一个
成员，按名字取 —— `foo_module.Bar`，泛型应用选中那份文件时同样按名字取（`g[T].value`、`g[T].type`）。
**函数语义**要一份签名加一份结果：`__decl__` 是签名（当函数判时的整条链，结果在前、参数在后），
`__impl__` 是结果。**没有 `__decl__` 就没有函数语义** —— 它只是模块，调它是程序错误；没有 `__impl__`
是设计、不是程序。同一个模块不许把同一个名字定义两次。

## 模块的参数：`__decl__`

一个模块要收参数，就把它们放进 `__decl__`——模块当函数判时的整条链：结果在前，参数在后。
`$env Env` 至多一个。用不用它，调用时给环境的方式一样：一个 `<<` 给一个环境值；用了，环境就给这个
参数，模块体用 `args.env` 取；没用，环境不进门，模块体取不到它，它只用来跑这次调用，再缀在这次调用
返回的那次调用末尾：

```viba
# file square_sum.viba
__decl__ =
    int
  <- $env Env
  <- $a int
  <- $b int

args = __get_args__ << __decl__

__impl__ = add
  << args.env
  << (mul << args.env << args.a << args.a)
  << (mul << args.env << args.b << args.b)
```

```viba
# file main.viba
__decl__ =
    int
  <- $env Env

args = __get_args__ << __decl__

import square_sum

__impl__ = square_sum << (args.env.tmp_env << args.env) << 3 << 4
```

- **`__get_args__` 把这次调用的输入起成一份积**：推导时它给出的是 `__decl__` 那些参数落成的积类型
  （`args.a` 是 `int`，`args.env` 是 `Env`），计算时它给出的是上游真实传过来的数据（`args.a` 是
  当初给进来的那份值，`args.env` 是真实的环境）。一个模块要取自己的参数，就用
  `args = __get_args__ << __decl__`，再按 tag 取。
- **结果不能是环境**：`__decl__` 的第一个元素是 `Env` 或 `Environment` 当场报错，模块里别的函数
  也一样（只有内建函数给得了一个环境，见上面「环境不能当结果」）。运行时的结果不受影响：
  `__impl__ = args.env` 交回去的仍然是那个环境，只是没人声明它的类型
  （[`tests/data/environment/the_env.viba`](tests/data/environment/the_env.viba)）。
- **给环境就是执行**：每个参数一个 `<<`，按位置给（`<< 3 << 4`）或按 tag 给（`<< $b 4 << $a 3`）
  都行；按 tag 给时顺序随意。环境按值认：给一个 `Environment`，或者按 `$env` 这个 tag 给，都算给它，
  没有 `$env Env` 的模块也一样认。不带 tag 的实参跳过环境，依次落到其余参数上——所以
  `square_sum << 3 << 4` 是把 3 和 4 给 `$a`、`$b`，不是给环境。**环境给了，参数就得齐**：少一个就是
  `VibaProgramErr`，话里点名少了谁（`module 'square_sum' was given 1 of its 2 parameters: $b missing`）。
  多给、给重、给了没有的 tag，同样当场报错。
- **带 `...` 的那个参数收"剩下的实参"**：链条上接着给的那些都归它，在那里打包成一份积 ——
  `Y << step << ($sub_env << args.env << "Y") << $a 7 << $b 0` 里 `Y` 的 `$args ...` 收到的是
  `$a 7 * $b 0`（`args.args` 取回它）。用 `Any` 的参数不打包：它要的是一份已经给出的积，`apply
  << f << ($a 1 * $b 2)` 就是；落成两条 `<<`（`apply << f << $a 1 << $b 2`）是"多给"，当场报错。
  `...` 那个参数按惯例放在最后，它前面的那些参数照旧按位置或按 tag 给。
- **环境不进门时它随结果往后走**：没有 `$env Env` 的模块，环境只用来跑这次调用；这次调用返回的那次
  调用还欠着环境，它就被缀在那里。`apply << f << args << env` 是现成的例子：`apply` 收下函数与积、跑出
  `apply_impl[...] << args.f << args.args`，环境缀上去；`apply_impl` 那一支再跑出 `f << $a 1 << $b 2`，
  环境又缀上去，凑成 `f << $a 1 << $b 2 << env`，那次调用才跑。链条多长都这样一层层往后递。
  这次调用返回的那次调用**自己有 `$env Env`** 时，它拿到的就是它跑起来的那一份数据路径：解释器按它
  源码里的名字（模式文件按它的文件号）给它造一份子环境，免得两个模块挤同一条路径。
- **不给环境，它就是闭包**：`sg = square_sum << $a 3 << $b 4` 是一个值——源码里的模块名加上已经
  算好的实参，可以存、可以传、可以当 `__impl__`、可以序列化，也可以**换个环境再执行一次**
  （`sg << (args.env.tmp_env << args.env)`）。"必须给全"只对执行成立，对闭包不成立。
- **一份要别人给参数的模块，单独跑只会报错**：宿主跑一份主文件时只给环境，没人给 `$f`、`$n` 值，
  所以 `args.f` 取不到东西（[`tests/data/y_combinator/`](tests/data/y_combinator/) 里那两条记录
  就是这个）。
- **参数是模块取到的可序列化数据**（`args.a` 那种），不是一次"要的时候再来拿"的调用：函数类型的
  参数说的是宿主那一侧的实参，模块的参数给的就是值。**一份积交进来时，积里的成员在这里就算好** ——
  给出这份积的那次调用里，`$n below` 的成员值是 `below` 的值；`below` 里取的是这一层的实参，所以
  只有给出它的那次调用算得出来（`viba-pattern.md` 的 `apply` 就靠这一点：那份积就在调用方那个模块里）。
  另一个模块里的积不在这里算：那是它自己的事。
- **一个片段在给出它的那个模块里取**：名字、闭包、模块的参数，都带着"它是哪个模块给出的"这一条，
  交出去、存下来、再取的时候还是回那儿解——`F` 在一份文件里是 import，在另一份里什么都不是。
  所以 `args.f` 交回去的是**当初给进来的那份值**，不是照着收它的那一方重新取一遍。
- **模块体里 `args` 就是那份实参积**：`args.a` 是 `$a` 那个成员的值，按 tag 寻址——这里说的
  **数据路径**由一串步子接成（按 tag、按位置、按下标、按键），一段一段走下去就是寻址；`$a` 是它
  的一步，模块路径和 `cur_storage_path/<name>.viba` 也是数据路径。「幂等与快照」里的"快照数据
  路径"、「一次执行会得到什么」里的"这次调用的数据路径"都是这个意思，步子分哪几种见
  [`viba-reflect.md`](viba-reflect.md) 第 4 节；判定层取同一个名字取到的是成员**声明的类型**
  （`args.a` 就是 `int`）。一个名字，两层各取各的。
- 实参必须是可序列化数据：它是积的一部分，交出去的东西得是能序列化的值。**环境不是它的数据成员**
  ——环境是宿主的对象，`args.env` 取得到，整份积交给宿主时不在里面；要交给宿主的是一个一个成员
  （`sum_of << args.env << args.a << args.b`）。
- **实参装不下那个参数时，这里不拦**：`$a int` 那个参数给 `"x"`，这个字符串就交给那一步的实现，
  它怎么对待是它的事（拿它做整数运算，就崩在它那里）。要在**设计**上拦住这种调用，判的是那份设计
  —— 判定层把一个函数链判成某个类型时要求 `3 <: int`（见「类型层的模块」），一次运行不走那道判定。

## 调用别的模块

```viba
import add_demo as demo

ret = demo << (args.env.sub_env << args.env << "add_demo")

__impl__ = demo.print << args.env << ret
```

- `demo` 是模块，当函数用：给它一个环境（它只收这一个参数），它跑完给出它的 `__impl__`。
- `demo.print` 是这个模块里的函数。
- `args.env.sub_env << args.env << "add_demo"` 拿一个子环境：**它带着父级的 compute**，数据路径是
  `父路径/add_demo`（见「幂等与快照：结果要能回放」）。同一个调用还可以用
  `$sub_env << args.env << "add_demo"`（见「如何调用方法：tag 放在链头」）。

**一条数据路径上只该有一次调用**：一次运行里，每个模块调用跑在自己的数据路径上（那是环境的
`cur_storage_path`，不是 `get_func` 收到的 `module_path`）。三种情形分得很清楚：

- **这条路径正在跑**（同一个数据路径上还没处理完，又要进去一次）：真是环，`VibaProgramErr`，话里给出
  源码形式：

  ```
  the storage path 'root' is already running a call, so 'main' cannot run there too:
  give each module call a sub-environment of its own (args.env.sub_env << args.env << ...)
  ```

  主文件自己也算一次激活，占着它那条路径——直接 `main << args.env` 就是这一种。没有 `$env Env` 的模块
  由解释器给它造这一份子环境；一次调用返回的那次调用自己有 `$env Env` 时也一样造一份（否则它拿到的
  就是跑着那次调用的数据路径），名字都是模块在源码里的那个名字（选中的模式文件用它的文件号）。所以同一个
  模块在同一个父环境下被用两次，是同一个数据路径、一次计算；要把两次分开，调用方自己给一份子环境
  （`$sub_env << args.env << "one"` / `"two"`）。
- **这条路径处理过了，而且是同一个模块**：同一个子计算被问了第二次，把它处理过的那份**回放**回去
  （见「幂等与快照：结果要能回放」）。`args.env.sub_env << args.env << "x"` 对同一个父环境是
  **同一个** storage（同名子环境要到才造、造一次、之后复用），所以同一个模块两次用同一个名字，
  是**一次计算、两次取用**。
- **这条路径处理过了，但换了一个模块**：两条不同的调用挤一个数据路径，宿主分不出谁是谁 → `VibaProgramErr`，
  话里给出正确的源码形式（`... which another module call already used`）。

### 如何调用方法：tag 放在链头

`$tag << X << a` 就是 `X.tag << X << a`：**tag 出现在链头时，它命中的是第一个参数的成员，
而那个值也接着被交给成员当第一个实参**。环境的成员因此有三种源码形式，一样长：

```viba
args.env.sub_env << args.env << "add_demo" # 点号，环境自己给出
$sub_env << args.env << "add_demo"         # tag 放在链头，同一个调用，短一些
$sub_env << $env args.env << $sub_env_name "add_demo"  # 都按 tag 给，也对
```

`viba/builtin.viba` 里 `Environment` 就是这样声明的——环境是它的第一个参数：

```viba
Environment =
    Object
  * $viba_path str
  * $sub_env (Env <- $env Env <- $sub_env_name str)
  * $tmp_env (Env <- $env Env)
  * $get_parent ((Env | nil) <- $env Env)
  * $get_root ((Env | nil) <- $current (Env | nil))
  * $get_relative_path (str <- $current Env <- $root (Env | nil))
  * $find_by_relative_path (Env <- $relative_path str <- $root (Env | nil))
  * $compress_env_path (Env <- $sup Env <- $sub Env)
  * $try_compact (Env <- $env Env <- $env_chain_length_limit int)
  * $uncompress_relative_path (str | nil)
```

成员也可以放在数据里：一个积的 `$f` 里放一个函数名时，`$f << box << args.env << 1` 就是
`box.f << box << args.env << 1` —— 成员收的第一个参数是它所在的那个值（`$box Box`），环境跟在后面。

tag 本身**不是值**：`method = $sub_env` 编不过，标签只有出现在链头、后面跟着第一个参数时才成立。
第一个参数必须给出来——它是取成员的那一个，省略了就成了一次没有成员的调用。第一个参数是别的值
也一样：`$tag` 命中的是它的成员，不在就报错。

### 链上的成员：父级、根、相对路径、找回来、压数据路径

子环境是由 `sub_env`/`tmp_env` 从父级造出来的，它记着那个父级，所以一条链能从任意一层往上走。
这些成员把这条链取出来（见「调用别的模块」）：

- **`$get_parent << args.env`**：造这一层出来的那一层（`sub_env`/`tmp_env` 给的那一个）。链顶没有
  父级，交出来就是 `nil`。**这条链就是解释器自己的栈**：一次还在跑的模块调用占一个环，跟着它往上走
  能走到这次运行最外面那一次调用（见「栈太长时：把系统栈变成工作清单」）。
- **`$get_root << args.env`**：这一层那条链的根（没有父级的那个环境）。`nil` 进 `nil` 出。
- **`$get_relative_path << args.env << <root>`**：这一层的数据路径从那个根往下取出来的那一段 ——
  `root/a/b` 从 `root/a` 看是 `"b"`，根看自己是 `""`。`$root` 给 `nil` 就用这条链自己的根；
  给的根不是它的祖先就当场报错。
- **`$find_by_relative_path`**：反过来，从一份根按相对路径走下来。路径里每一段都是一个目录名
  （`.`、`..` 不是），`""` 就是那份根本身。它的源码形式是点号那一种，`root` 给 `nil` 时从**取到这个
  成员的那个环境**往下找：

  ```viba
  args.env.find_by_relative_path << "a/b" << nil              # 从 args.env 往下找
  args.env.find_by_relative_path << "a/b" << ($get_root << args.env)  # 从根往下找
  ```

  落成 `$find_by_relative_path << args.env << "a/b"` 会把那个环境放到相对路径的位置上：报错，
  话里给出正确的源码形式。**走出来的是"一层"，不是一条链**：那些段说的是它的数据路径，不是调用，
  它的父级就是你从哪一层往下找的那一层（`root` 给 `nil` 时就是取到这个成员的那个环境）。三十段的
  路径给出来也是一层；`get_relative_path` 照样报出那三十段，因为它走的是数据路径。
- **`$compress_env_path << <sup> << <sub>`**：把 `sub` 的数据路径压成 `sup` 旁边的一个名字，
  形式是 `{sup 的路径}_{sha1(sub 的路径)}`（后面那 40 位是十六进制）。`sup` 的路径必须是 `sub`
  的路径的前缀，而且落在名字边界上（`<sup>/…` 或者已经压过的 `<sup>_…`）；压出来的数据路径长度只跟
  `sup` 的名字有关，跟 `sub` 有多深无关 —— 递归里每层的数据路径因此不
  会越接越长。它不是 `sub_env`：不往下一层走，而是在 `sup` 所在的那个目录里放一个扁平的名字。
  压掉的那个原数据路径记在返回值的 `uncompress_relative_path` 上；同一个 `sup` 与 `sub` 给同一个
  数据路径。`sup` 自己的名字所在的目录里没有地方可放（`sup` 是链根）时，当场报错。**压的只是
  数据路径**：名字放在 `sup` 自己的名字所在的目录里（`sup.storage.parent`，数据路径自己那条往上
  的路），而走出来那个环境的父级是 `sub`（它站的那一层）—— 链是 `sub_env` 那样的调用形成的，
  压缩既不从链上拿走一个环，也不看链。
- **`$uncompress_relative_path << args.env`**（也落成 `args.env.uncompress_relative_path`）：
  压过数据路径的那一层记着它压掉的原数据路径；别的环境上是 `nil`。它是一个**值成员**，不带参数。

`tests/test_interpreter_env_paths.py` 把每个成员自己的规矩、几种给错的情形，以及它们合起来的往返
（根 → 相对路径 → 按那条路径找回来、压出来的数据路径还能按路径走）都钉住了。

不想起名字就用 `tmp_env`：

```viba
add_demo << (args.env.tmp_env << args.env)
```

它像临时文件一样，**每次调用都给一个新的子环境**（路径是 `父路径/tmp_<随机>`），所以两次调用
天然各占一条路径。`viba/builtin.viba` 里 `Environment` 的成员因此是
`$tmp_env (Env <- $env Env)`：它只收那个环境，名字不用给。

路径每次都不同，这是 `tmp_env` 的语义：它给**纯函数调用**、或者**结果不留的调用**用。一个
需要快照、要靠回放才幂等的函数如果挂在它底下，回放自然命中不了——每次的路径都是新的，快照会
一次次落进不同的 `tmp_` 目录（`store_root/root/tmp_xxxx/…viba`），那条不纯的路于是每次都重走，
**幂等检查会失败**。这不是缺陷，而是它给出的信号：这个函数需要显名保存，应该换到
`args.env.sub_env << args.env << "一个稳定的名字"` 上去。`tests/test_interpreter_idempotence.py` 里两半都有用例：
显名路径第二次跑就回放，临时路径两次都重算、并且留下两份快照。

## 宿主侧：Environment

`Environment` 由宿主提供。三个名字里**只有 `Environment` 是 viba 里看得见的那一个**
（[`viba/builtin.viba`](viba/builtin.viba)，短名字是 `Env`）：`__decl__` 里 `$env` 那个参数要的就是
它，模块体里取它用 `args.env`。

它的定义（八个成员）见上面「如何调用方法：tag 放在链头」那一节。

storage 与 compute 是**宿主的词汇**，viba 里没有一个名字指得到它们（它们不是类型，是宿主
对象），所以只在宿主侧出现（`viba.interpret`）：

```python
EnvironmentStorage(cur_storage_path, sub_storage=None, store_root_dir=None,
                  parent=None)                           # 它的名字坐在哪个目录里
EnvironmentCompute(get_func)                             # get_func(module_path, func_name)
Environment(storage, compute, viba_path=None, parent=None,
            uncompress_relative_path=None)                  # sub_env / tmp_env 给子环境
```

两处 `parent` 是两件事。**`Environment` 的 `parent` 是造它出来的那个环境**（`sub_env`/`tmp_env`
会填上，宿主自己 `Environment(...)` 造出来的就是 `None`，那它自己就是一条链的根）—— 链上那些成员
取的就是它。**`EnvironmentStorage` 的 `parent` 是它自己的名字坐在哪个目录里**（`sub` 填上），是
数据路径自己那条往上的路：`compress_env_path` 把一个名字放到 `sup` 旁边时用的是它，不碰链。

链上那些成员在宿主侧就是 `viba.interpret` 里几个普通函数：

```python
get_parent(current)                         # 造它出来的那一层；链顶是 None（`nil`）
get_root(current)                           # 链顶那个环境；None（`nil`）进 None 出
get_relative_path(current, root)            # 从 root 往下的那一段数据路径
find_by_relative_path(relative_path, root)  # 从 root 按那一段数据路径走下来（一层，不是一条链）
compress_env_path(sup, sub)                 # 把 sub 的数据路径压成 sup 旁边 名字_sha1，放在
                                            # sup.storage.parent 里
env_chain_length(current)                   # 这条链有几个环：`$try_compact` 比的就是它
```


`uncompress_relative_path` 不是函数，是环境上的一个值：`compress_env_path` 把压掉的原数据路径
存进去，别的环境上是 `None`（viba 那边取出来就是 `nil`）。

它们同时也是 `Environment` 的成员，所以 viba 那边两种源码形式都行（见「如何调用方法：tag 放在链头」）。
`find_by_relative_path` 挂在环境上时多一件事：`root` 给 `nil` 就用取到它的那个环境，所以每个环境
自己那一份记着它是谁。

`viba_path` 是模块的搜索路径：一次运行里它跟着 environment 走，`sub_env`/`tmp_env` 把父级的
那条原样交给孩子，于是"这个模块的 import 去哪里找"就是它被交给的那个 environment 说了算。

`EnvironmentStorage` 面向 viba 一侧暴露的概念：`cur_storage_path`（就是这次调用的数据路径）、`sub(name)`（子 storage，`sub_env` 用它）、`store_root_dir`（快照放在哪个目录下，
不给就是默认的临时目录 `…/viba-store`）、`get_text(file_path)` / `put_text(file_path,
content)`（在 store root 底下的纯文本存取，取不到返回 `None`）。子 storage 带着父级的
`store_root_dir`。

`get_func(module_path, func_name)` 返回一个可调用对象；没有就返回 `None`，于是这一次调用没有实现
（结果里 `$msg` 是 `no implementation`，见最后一节）——它也可以**交回那份"没有实现"的数据**
（`viba.interpret.not_implemented()`，见最后一节），这一步的实现自己交回它也一样。它抛异常，run 也照这一支
作答，`$msg` 开头是 `get_func raised`。
`module_path` 是**这一步声明在哪个模块**，也就是它在 viba 里被 import 的那个名字：`import foo.bar`
（或者 `import foo.bar as foo_bar`）里的 `foo.bar`；就地声明的步骤说的是 `interpret` 跑的那份模块
（一份文件按它的名字，`exec` 那份代码按 `<viba_code>`）。`func_name` 是这一步**在那个模块里的名字**：
`foo.bar.b_scale << …` 问的是 `get_func("foo.bar", "b_scale")`，`foo_bar.b_scale << …` 问的是同一次。

内建词汇是一个模块（`builtin`），它的成员是那儿的名字：`builtin.add << …` 与不带模块名的 `add << …`
问的都是 `get_func("builtin", "add")` —— 两种源码形式是一次调用。宿主按那两个名字找实现，
两个服务各有一份 `b_scale` 时，`foo.bar` 说的是哪一份。

结果里 `UnderlyingOpErr` 的头两个字段是这一步的**整名**与它那个模块的**路径**：`$module_path` 是
`module_path` 换成路径（`foo.bar` 是 `/foo/bar`，内建词汇是 `/builtin`），
`$full_qualified_func_name` 是整名
（`foo.bar.b_scale`）—— 名字带模块，所以拿到它的人不看别的也知道是哪一步。

**这次调用跑在哪儿**不在这里：那是给它的那个 environment 的数据路径，宿主从环境上取
（`env.storage.cur_storage_path`）。所以同一个模块的两条数据路径是同一批步骤、两次计算。

`func_name` 是这一步**在 `module_path` 那个模块里的名字**：`demo.print << …` 问的是
`get_func("demo", "print")` —— 模块说哪一份，名字说那儿的哪一个；`demo` 这个模块自己叫它什么就是
什么。别名不改变这件事：`b_scale = foo.bar.b_scale` 之后问的还是 `("foo.bar", "b_scale")`（上一节）。
一个只服务一个模块的宿主，直接拿 `func_name` 就行了。

宿主语言里的函数与 interpreter 对得上：Python interpreter 里就是一个 Python 函数，收到的参数是
**已经算好的实参**，按源码里的次序给——实例是 `viba.reflect.VObject`，别的（环境在内）是它本身。
返回值是 `VObject`，或者一个普通 Python 值（落到函数声明结果的一个叶子上）。

参数里出现 **viba 函数**（`$f (int <- $env Env <- $x int)` 这种高阶签名）时，宿主拿到的是**可序列化数据**：
那个函数名本身（一个 `VObject`），或者一个闭包——源码里的函数名加上已经算好的实参。**宿主调不动它**：
viba 函数不是宿主侧的 Python 可调用对象，执行它们只有一条路，就是给环境（`<< args.env`），那是 viba 那一侧
的事。所以宿主得到的是数据：可以拿着、存着、原样递回来，不能 `f(env, x)`。

宿主自己造一个 `VObject` 当结果也可以；但**列表、字典、可调用对象这类给不了**——它们没有对应的叶子，
只能给出 `VObject`、标量或 `None`（`None` 就是 `nil`）。

**interpret 不认识任何具体函数**：viba 默认不带任何库函数，实现全部来自 `get_func`，
谁补上、怎么生成，interpret 不感知——文件里 `{...}` 那句提示只说这一步要实现什么，实现从外面交进来；
同一份文件配上两份 `get_func`，就是两份实现，文件本身不变。

## 幂等与快照：结果要能回放

一个 viba 程序的结果要可回放：同一份输入、同一批路径，跑多少次都该是同一个结果。可执行函数
里唯一不保证这一点的东西是**宿主函数**——它可能看时钟、掷骰子、调外部系统。所以不纯的那些由它
自己负责：**把结果快照到 `EnvironmentStorage` 里，下一次跑同一次调用就直接回放**。

**模块调用按同一条规矩办**：一条数据路径上只该有一次调用，所以那条路径处理过之后，同一个
模块再来问同一条路径，拿到的就是处理过的那份（见「调用别的模块」里的三种情形）——一次运行里
省掉的是重复计算，跨运行回放靠的是上面那把快照。

```python
from viba.interpret import get_snapshot, put_snapshot, replayed

def roll(env, n):
    def compute():
        return random.randint(1, 10 ** 6)       # 不纯的那一步
    return replayed(env, compute, f"roll-{n.value}")
```

- `replayed(env, compute, name)`：**有快照就回放，没有就算出来、存下来**。这是最常用的源码形式。
  想分开用就是 `get_snapshot(env, name)`（没有返回 `None`）与 `put_snapshot(env, value, name)`。
- 快照数据路径由**这次调用的数据路径**决定：`snapshot_path(env, name)` 是
  `cur_storage_path/<name>.viba`，存取都在 `store_root_dir` 底下。所以同一次调用（同一条路径）才
  会命中同一条快照；换了路径（另一个名字的子环境）就是另一次调用。
- **要回放就要一条稳定的路径**：跨两次运行能命中，靠的是两次运行里那条路径一样。用
  `args.env.sub_env << args.env << "稳定的名字"` 就是稳定的；`tmp_env` 每条路径都是新的，挂在它底下的调用不适合
  保存要回放的东西（上一节：这正是它给出的"该显名保存了"的信号）。
- **快照是序列化的 viba 数据**（`viba.serialize` 序列化出来的 `value = …`），不是 pickle：存下来
  的东西可以被人看、被人拿去喂类型推导。回放时解析回实例，叶子和存进去的时候一样。
- **一次落盘就是完整的一份**：`put_text` 把内容放在目标旁边再改名过去，所以取的那一方取到的要么是还没有，
  要么是完整的一份 —— 另一个进程可能正在取同一条路径，谁都不想取到半份快照。
- **存不了、回放不出来就是错**：`VibaProgramErr`（宿主抛出来，interpret 转成 `VibaProgramErr`），不会静默给个默认值。

于是"随机"也能回放：

```viba
__decl__ =
    int
  <- $env Env

args = __get_args__ << __decl__

roll =
	int
	<- $env Env
	<- $n int
	<- { roll a die: not a pure function, so its answer is snapshotted }

__impl__ = roll << $env args.env << $n 1
```

第一次跑算出 731204 并存进 `store_root/root/roll-1.viba`；第二次跑取到它，那条不纯的路一次
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

积是**扁的**：一个成员自己是一份积时，它贡献的是自己的成员。`A * (B * C)` 与 `(A * B) * C`
是同一份积；一个名字代表一份积时也一样 —— 在 `inner = tagged["b", 1] * tagged["c", 2]` 之下，
`tagged["a", 0] * inner` 就是三个成员的那份积。判定层摊开积用的是同一条规则（`is_sub_type`），
所以一份积的**值**与它的**类型**说的是同一件事：按 tag 取成员时，隔着这层名字也取得到
（`sequential_impl` 的 `vars` 就是一层层这样堆起来的）。带 tag 的成员不摊开：`$x (A * B)` 是
`$x` 那一个成员，它的值才是那份积。

和里只剩一个非 never 分支时，结果就是该分支；所有分支都是 never 时，结果是 never；
多个非 never 分支同时存在时，结果保留为和值，不擅自选择其中一支。

`branch.viba` 与 `branch.py` 提供两个开关。它们接收分支值，返回 `Any`；`$get_v` 那个参数的类型是函数类型
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
  返回 never——**`get_v` 不会被叫**，那一支源码里的实参根本不求值（惰性求值）；
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

__impl__ =
    Oneof
  | (branch.echo_or_never << $env args.env << $cond condition << $get_v (builtin.echo << $x a))
  | (branch.never_or_echo << $env args.env << $cond condition << $get_v (builtin.echo << $x 0))
```

那个实参**最多求值一次**（memoization，也叫 sharing）：第一次问出结果（值或停下），之后每一次问都拿
同一个。所以宿主问两遍不会让副作用发生两遍。

只有函数类型的那个参数是惰性的（lazy，non-strict）；别的实参按值求值（call-by-value），函数的 `$env`
也照旧按值给。想给一个求好的值，用 `builtin.echo` 把它包成"给它一个环境就给出 V"的那个函数。

这跟一份定义什么时候求值是同一条策略，见下面「求值策略：按需求值（call-by-need）」一节。

条件、值和开关本身都可以放在别的文件里，用例只给一条链去调它们；一个调用先给一半、剩下的由用例
补齐，也是同一个值。`echo_or_never` / `never_or_echo` 本身也是这样一份可以 import 的文件。这条
路子的边角案例在
[`tests/test_interpreter_branch_switch.py`](tests/test_interpreter_branch_switch.py)：一份用例一个
文件（`tests/data/branch_switch/*.viba`），覆盖开关方向、条件的各种源码形式、每一种值的种类、部分计算、
别的文件里的开关、副作用次数、走不到那一支里的「没有实现」与失败，以及多个开关并起来的和。

## 类型层的模块

同一个文件在**类型推导**里也是"参数进、`__impl__` 出"：名字绑到的是一个模块（import 的名字）
时，它当函数判，类型就是它的 `__decl__` 去掉环境那个参数：

```viba
int <- $a int <- $b int       # `__decl__ = int <- $env Env <- $a int <- $b int` 判出来的类型
```

**环境不在类型里。** `$env Env` 是调用的规矩 —— 给环境就是执行，用不用这个参数都一样 —— 它不是设计
里的一个参数：没有人替它给一个值。所以运行、判定、描述符三层取出来的函数类型里都没有它，
`int <- $env Env <- $n int` 与 `int <- $n int` 是同一个类型；落成模块、落成函数定义，只要是同样的
设计参数，就是同一个类型。

所以这几条都成立（`tests/test_is_sub_type.py` 里有用例）：

```viba
demo = import add_demo as demo         # 概念上
demo << args.env  <:  int         # 就是 __impl__ 的类型；只收环境的模块，给环境就是执行
int <: demo << args.env           # 反过来也成立：两者同型
demo.add <: int <- $env Env <- $a int <- $b int
design.Only <: $x int                   # module.MyType 仍然是那个模块里的定义
square_sum << args.env << 3 <: int <- $b int   # 给了一半：剩下的是函数
```

- 只认 **import 绑定的那个名字**（`import a.b as c` 的 `c`，`import a.b` 的 `a.b`）。`module.Name`
  仍然是那个模块里的定义，按最长的前缀解析。
- 没有 `__impl__` 的模块不是程序：`demo << args.env` 在类型层也是 `VibaProgramErr`。
- 不带 tag 的实参跳过环境那个参数，和值层同一套规矩；环境那个参数给的就是环境本身，用 `args.env`
  或按 `$env` 这个 tag 给都认。
- 实参按 tag 给（`<< $a 3`）或按位置给（`<< 3`）都认；位置那一支要求给的实参装得下那个参数
  （`3 <: int`）。**给一半在类型上是一种类型**——剩下的那个函数；在值层给一半是程序错。
- **`__get_args__ << __decl__` 在类型层给出的是积类型**：成员就是 `__decl__` 的那些参数——`args.env`
  是 `Env`、`args.a` 是 `int`。`__decl__` 不是函数链的话，两层都当场报错。

## 模式：一个泛型是一个目录

`import demo.is_base_type as is_base_type` 也可能找到的是一个**目录**（目录里有 `__generic__.viba`
标记）：那是一个泛型，它的每个文件是一个模式：文件名是决断顺序（一个数字），前面还可以带上它取几份
（[`viba-pattern.md`](viba-pattern.md)）。
它不是一个模块——单独用 `is_base_type` 不是类型，`is_base_type << $x 1` 不是调用。能用的只有应用：

```viba
Flag = is_base_type[bool]              # 决断选中 100.viba；它的 __decl__ 是 true
```

跑起来的时候，决断**只看源码里的那几个类型**（静态动作），选中谁，就把谁的 `__decl__` 拿到
**那个文件**里取：形参名（文件里没定义的那个名字，比如 `A`）绑定到实参里对应的那一部分，
这个绑定在那个文件的每一次求值里都算数，包括它自己定义里的那一层。

- 源码里是字面量，给出的就是那个值（`__decl__ = true` 给出 `true`）；
- 源码里是萃取到的形参，给出的就是那份类型（`__decl__ = A`，给出 `int`）；
- 源码里是数据，给出的就是数据，形参名已经换成实参里给的那份（`__decl__ = (A, B)` 给出 `(int, str)`）；
- 源码里是**函数链**，这个应用代表的就是那次**调用**，跟"定义体是函数链"一样：可以接着给实参
  （`call_it[list[int]].type << args.env << 1`），也可以当闭包递出去。宿主按**泛型的名字**找实现
  （源码里是 `call_it[...]`，它拿到的是 `"call_it"`），跟按定义名找实现同一种做法；
- 那个函数链所在的文件自己有 `__impl__` 时，这个应用就是**它那一次模块调用**：环境与实参按它
  的 `__decl__` 给（调用方给环境，欠着的实参接着给），活由它的 `__impl__` 自己算，宿主不必按泛型
  的名字另找一份实现。它跑在自己的**子环境**里，那一层的名字是这份文件的**决断顺序**（`2_200.viba`
  跑在 `…/200` 下），所以一次调用的数据路径仍然存得下来、复现得了 —— 调用方不必自己再包一层。
  文件没有 `__impl__` 时是上一条：`__decl__` 就是它给出的类型，宿主按泛型的名字实现；
- 其余照平时的规矩：`int` 这个名字在值的位置上不是值。

`(A <- $env Env <- B)` 这种类型是函数类型的参数，交给宿主的是它代表的那次调用：宿主带着环境叫它，
还可以带上自己的实参（`f(env, 1, 2)`），那些实参落进这次调用还欠的槽位。多喂的那一个由 viba 这边
说清楚（`takes no more arguments`），不是把宿主那边的 `TypeError` 报成实现坏了。

方括号里的实参本身是一次 `<<` 时（`itself[add << $a 2].type`），决断先把它**当类型判出来** ——
已经给过 `$a 2` 之后剩下的那条链条 —— 再照模式展开，所以"已经给过一部分实参的函数"也是一份
能收的实参。给实参要带上 tag（`$a 2`）：`$env Env` 是调用的规矩，不是实参的位置。

一个在选中的文件里建起来的、还欠着实参的调用（半成品）**记着特化时换上的那些实参**，跟它记得
自己在哪个模块一样。这样的半成品交给另一个模块当实参、或者存成闭包之后再给实参时，只给出形参
的位置仍然指回**调用方给的那一份**。`apply` 用的就是这一点：`apply_impl[args.args]` 里的 `args.args`
换成的是调用方给的那份积，数成员、认 tag 都在这份积上做（`tests/data/apply/`）。

## 成员按名字取：`tagged` 与 `$__getattr__`

名字在字符串里时，成员照样取得出来。`tagged["hello"] << X` 就是 `$hello << X` ——
一参数的 `tagged` 是一个成员，只出现在链头（[`viba-pattern.md`](viba-pattern.md) 第 3 节）。

名字是**一份可以算出来的值**时（决断萃取出来的符号、从数据里取到的 str），用内建的成员
`$__getattr__`：

```viba
__impl__ = $__getattr__ << args << "name"          # 就是 args.name
__impl__ = $__getattr__ << box << name << args.env << 1   # name 是算出来的 str
```

它取一个值的成员：积里那条 tag 的那一份，环境上就是那个宿主属性。名字后面**还接着实参**时，成员是
被当方法调的，X 就照 `$tag << X` 那样当它的第一个实参给出去（`$__getattr__ << box << name << args.env
<< $x 1` 就是 `box.f << box << args.env << $x 1`）；名字后面**没有实参**时，取出来的就是那个成员
本身 —— 一份值（`$__getattr__ << args << "n"` 就是 `args.n`），或者它代表的那次调用，原样交出去。
名字不是字符串、或者这个值没有那个成员，都是程序错。

判定层取同一条链时，名字是源码里的字符串（或解析得出字符串的名字）就给出**那个成员的类型**；
名字取不出来时给 `Any`：哪个成员是运行的时候才知道的，设计说不出它的类型。

## 按地址取元素：`$__getitem__`

容器里的元素按**地址**取，地址也是一份可以算出来的值：位置给 int、键给 str，正是访问协议里那两条
地址步子（`$at_index int`、`$at_key str`，[`viba-reflect.md`](viba-reflect.md)）。

```viba
__impl__ = $__getitem__ << xs << 1                            # xs[1]，xs 是 list
__impl__ = $__getitem__ << table << "k"                       # table["k"]，table 是 dict
__impl__ = $__getitem__ << ($__getitem__ << xs << 1) << 0     # xs[1][0]
```

名字后面的方括号是这条链的**简式**：`xs[1]` 就是 `$__getitem__ << xs << 1`，
`table["k"]` 就是 `$__getitem__ << table << "k"`（容器也可以是一条名字路径：`box.xs[1]`）。
一个名字没有哪个泛型回答它时，`A[b]` 就是这一步取用；`A` 指不着任何东西时，报的就是那个名字
（`no definition named ...`）。方括号只跟在名字后面、只一层：`xs[0][1]` 要落成
`$__getitem__ << xs[0] << 1`。放在**值**的位置上它才是这一步取用；放在积的成员里时它跟别的成员一样
是源码里的数据，不在这里取。

它先拿容器、再拿地址：地址还没给时它是一个值（`pick = $__getitem__ << xs`，再 `pick << 1` 才是取）；
地址后面**还接着实参**时，取出来的那个元素被接着当调用给（`$__getitem__ << xs << 0 << $x 1` 就是
`xs[0] << $x 1`）。它自己**不收环境**：取元素是数据那一步，环境只在取出来的元素是一次调用、你还要它跑
的时候才放在链上（`$__getitem__ << xs << 0 << $env args.env << $x 41`）。这里跟 `$__getattr__` 有一处
不同：成员被当方法调时，X 会额外当它的第一个实参给出去；元素不是容器的成员方法，所以取出来的那次调用
**不会**多收到容器这一个实参。地址不是 int 也不是 str、那个位置没有元素、那个键不在、拿到的不是容器
（或者给了 dict 一个位置、给了 list 一个键），都是程序错。按名字取成员是另一个内建成员：
`$__getattr__`（上一节）。

## 问在不在：`$__in__`

容器里有没有某一份，也是内建成员：`$__in__` 先拿容器、再拿要找的那一份，答一个 `bool`。

```viba
__impl__ = $__in__ << xs << 3                 # 3 在不在这个 list 里
__impl__ = $__in__ << table << "k"            # "k" 在不在这个 dict 的键里
```

list / set / tuple 装的是元素，所以问的是有没有落成一个样子的元素；dict 装的是键，所以问的是那个
字符串在不在键里（值不在其中）。跟 `$__getitem__` 一样，它**不收环境**：这是数据那一步。要找的那一份
还没给时它是一个值（`pick = $__in__ << xs`，再 `pick << 3` 才是问）；被问的不是容器、给 dict 的不是
一个字符串、自己没拿到要找的那一份，都是程序错。

## 容器字面量

`ListLiteral[...]` / `SetLiteral[...]` / `DictLiteral[("k", v), ...]` 放在值的位置上就是那个容器本身：
一次运行把它算成一个值，能数、能按地址取（`$__getitem__`）、能当实参交给宿主、也能原样序列化回去。
它的成员是源码里的可序列化数据，所以嵌套（`ListLiteral[ListLiteral[1]]`）、混合
（`ListLiteral[1, "a"]`）、空的（`ListLiteral[]`、`DictLiteral[]`）都照这样给。同一段文本放在**类型表达式**
里代表的是另一件事：那时它是 `list[a | b | c]` 之类的居民（[`viba/type.viba`](viba/type.viba)）。

## 求值策略：按需求值（call-by-need）

viba 是**声明式**（declarative）的，不是命令式执行的：一份文件里的定义是**绑定**（binding，跟 let
绑定的地位一样），不是按源码里的次序一条一条跑的语句。求值由需求驱动（demand-driven）—— 一个绑定
在被**用到**的时候才求值，而且求值一次就把结果留下来（memoization，也叫 sharing），那次的结果就是
它以后的值。这就是**按需求值**（call-by-need），也就是惰性求值（lazy evaluation）。

- **没人用到的绑定不参与求值**：[`tests/data/values/nested.viba`](tests/data/values/nested.viba)
  里那份 `unused` 没人用，宿主一次都没被叫。
- **绑定可以前向引用**（forward reference）：`__impl__ = x` 与 `x = 7` 换个次序还是 7
  （[`defined_after.viba`](tests/data/values/defined_after.viba)）。源码里的次序不是求值的次序。
- **`__impl__` 倒着放也行**：它放在所有定义之前，用到的每一步都放在后面，算出来还是同一个结果
  （[`impl_given_first.viba`](tests/data/values/impl_given_first.viba)）。
- **一个绑定只求值一次**：两处用同一个名字，宿主只被叫一次
  （[`memo.viba`](tests/data/values/memo.viba)）；同一个名字在一个模块里定义两份是程序错误
  （[`twice_defined.viba`](tests/data/values/twice_defined.viba)）。

这条策略不是省一次求值：递归要"再一次进到同一个定义"，而一份文件里的绑定不许成环（下一节），所以
往下那一层只能先落成一个名字 —— 名字就是一份文件手里那个**还没求值的计算**（thunk）。定义它的那一层
不用它，用它的那一层才把它求出来。`tests/data/y_functions/steps/` 那批用例就是这样：往下调一层
与取余各是一个绑定，基例那一侧不用它们，于是那一侧既不会去算 `a` 除以 0 的余数，也不会再往下调一层
（从基例那一侧进来的那一份是
[`main/gcd_zero.viba`](tests/data/y_functions/main/gcd_zero.viba)）。

一个调用的实参默认**按值求值**（call-by-value）；只有函数类型的那个参数**惰性**（lazy，non-strict）——
那个参数上的调用连同环境交给宿主，宿主用到它时才算，见上面「分支：用积选择，用和汇合」那一节。
求值到一半的绑定被自己用到（`A = B` 与 `B = A`，或者 `A = use << $env args.env << $x A`）是循环定义，
当场报 `one file's definitions may not go round`，而不是等栈崩（下一节）。

## 一份文件跑不出递归

递归要"再一次进到同一个定义"，而按需求值不会把一份文件里的绑定反复求出来：一个绑定就是它那一个
thunk，求一次就定了。一份文件自己的定义图因此是无环的——**文件里的定义不许绕回自己**——所以单独
给一份文件，怎么跑都跑不出递归：给出的调用点就那么多，`<<` 一处一处定死。

```viba
A = B
B = A
# A -> B -> A: one file's definitions may not go round — recursion takes two files
```

- **绕回自己当场报错**，不是等栈崩：一份文件里 `A = A`、`A = B` 与 `B = A`、以及"求 A 的时候
  又要 A"（`A = use << $env args.env << $x A`）都是 `VibaProgramErr`，话里给出绕的路径。
- **没被求值的不算**：函数类型的实参没人叫就不算递归（`A = ignore << $env args.env << $x A` 得 7，
  因为 `ignore` 不叫那个实参）；`List[T] = Object * $tail List[T] | nil` 这种**类型**自引用也
  不算——类型不求值。
- **跨文件不设这条限制**：两份文件可以互相 import、互相调用，设计层不拦。真的绕回去（`a.x` 要
  `b.y`、`b.y` 又要 `a.x`）时，这次运行报
  `a.x -> b.y -> a.x: the run came back to where it started`；模块调用再进同一条路径报
  `the storage path '...' is already running a call`；同一个模块带着**同一份实参**又在跑报
  `module 'a' is already running with the same arguments`——同样输入的一次计算还在里面进行，
  它停不下来。这些都是"这次跑不完"，不是设计有错。
- **同名、不同实参、各自一条新路径的调用是两次调用**，不是环：函数递归执行要"再一次进到同一个
  定义"，而固定的实参换一次就是另一层。不动点正是这样展开的：
  [`tests/data/y_combinator/`](tests/data/y_combinator/) 里的 `Y F 10` 得 55
  （Z 组合子的 viba 版：欠着实参的调用就是值，把自应用停在半成品上）。
- **两份文件互调的边角**：这条路子本身有一份用例集，
  [`tests/test_interpreter_mutual_recursion.py`](tests/test_interpreter_mutual_recursion.py)：100 条，
  每条是一对互相 import 的文件（`tests/data/mutual_recursion/left_*.viba` 与 `right_*.viba`）。
  分支与部分计算决定那条路走不走：函数类型那个参数里的递归调用，条件不成立时一次都不发作；
  一个调用先给一半（`right.f << $a 1`）是一个值，没进入另一份文件，也不发作；而条件那个参数照旧
  先算，所以条件里的递归调用一定发作。每条用例跑出值、跨文件绕回去、模块调用成环、闭包、
  `never`、和值、积或源码里的名字之一。

## 栈太长时：把系统栈变成工作清单

一次模块调用拿到一层自己的环境（`$sub_env`、`$tmp_env`），环境的父子链（`$get_parent`）就多一个环。
**这条链就是解释器自己的栈**：一个环对应一次还在跑的模块调用，所以它随着递归一层层变长。

Y 的每一层都往下调一层，一个够深的递归因此会把解释器自己的 Python 栈压爆：`fib_acc` 跑到两千层时，
链上一万多个环（一层五六个），解释器自己的栈跟着长同样多，`RecursionError` 先来。所以 `y_helper`
在自己的模块体之前给出一个**压缩点**：

```viba
# viba/y_helper.viba
env = $try_compact << args.env << 32
```

- **`$try_compact << <环境> << <上限>`** 是 `Environment` 的成员（[`viba/builtin.viba`](viba/builtin.viba)），
  答的是给它的那个环境：链上环数不超过上限就什么都不做，超过上限就把**此刻正在跑的调用**交出来，
  由这次运行最外面那一次调用一个一个地再做（`global_resumable_stack`，这次运行自己的工作清单），
  做完把这个环境给回来。
- **压的是此刻正在跑的那条链**，所以给它的环境必须是此刻正在跑的某一次调用的环境（同一份，
  不是同一条路径上另外一份）。给一个解释器不在其中的环境 —— 一次查找走出来的那一层、早先某次
  调用手里留着的那一份、没有调用跑过的子环境 —— 名字说的链不在系统栈上，压不出东西来，环境原样
  答回来（`tests/data/compaction/main/elsewhere.viba`）。
- **答一个环境，是为了把这一步排进顺序**：定义按需求值，用它的那一处算的时候它才算。`y_helper` 下面
  那几个定义用的就是这个 `env`（`env.find_by_relative_path << …`、`$compress_env_path << … << env`），
  而它们又是这次调用的环境 —— 要跑这一层，就得先把这个名字算出来，所以压缩点一定排在
  "这一层跑起来"之前。
- **每个待做的调用是两样东西**：它跑的那条数据路径（`$current_env_path`）和它本身（`$call`，类型是
  `Any <- Env`，见 [`viba/resumable.viba`](viba/resumable.viba)）。做它的时候，环境按那条数据路径从
  链顶取回来（`$find_by_relative_path`）：同一条路径、同一份 storage，而它是**一层** —— 路径里那些
  段是数据路径，不是调用，所以取回来的这一层挂在链顶下面，链短了，递归在里头接着长到上限，再压
  一次。
- **已经答过的调用不会再做**：一次模块调用的答案按它跑的那条数据路径记着（`_Runner.done`），重做一份
  模块体时，走到那条已答的调用就停在那儿，把记着的答案交出来。所以一次重做的代价是有界的：只重走
  这份模块体自己那几步和它欠的那次调用，不会把下面整棵递归树重走一遍（会被掀掉几次，见「重做的
  次数不是零，也不总是一个小常数」）。
- **每一段从哪儿起算，就从哪儿量**：解释器记着这一段的起点链长
  （`_Runner.compacted_chain_length`：运行自己那一次调用是它拿到环境的地方，重做的那几次是取回来
  那一层的链长），链不超过它就不再压。所以数据路径压不短的那些也有上限：重做一遍只多长一条链，
  不会把同一批调用一遍遍交回去。
- **同时压在系统栈上的调用因此是有限的**：`tests/test_interpreter_compaction.py` 钉着这件事 —— 101 层
  的 `Y` 递归一共做了 1201 次模块调用，同一时刻在栈上的从不超过 28 个，最长的一条链 37 环（上限 32）。

**机制在解释器，政策在 viba。** 阈值不是解释器定的：`$try_compact` 收的那一个数就是政策，眼下由
[`viba/y_helper.viba`](viba/y_helper.viba) 给 32，解释器自己一个阈值都没有 —— 没有模块调用它，就
没有压缩这件事。它答的还是给它的那个环境，所以它下面那些定义照旧拿这个环境用：压缩只换"这些调用
在哪里做"，不换求值语义。

**它依赖的是一条比"幂等"更窄的前提：一次调用的答案只由它跑的那条数据路径决定。** 压过的数据路径是
输入的函数，所以重做一份模块体时，它下面那些调用都能在 `_Runner.done` 里找到答案，走到那儿就停住 ——
"多做一遍"的代价是这份模块体自己那几步，不是它下面整棵子树。这条前提（见「幂等与快照」）本来就要，
压缩没有再加一条；它不成立的那两处因此只是多一个后果：`tmp_env` 每次给一条新路径，重做时命中不了
已答的调用，它下面那棵子树就一遍遍地重算；没走 `replayed` 的不纯步骤会被多问几次。

**重做的次数不是零，也不总是一个小常数**：一次调用下面有多少个压缩点，它就可能被掀掉多少次。一条
只有一个递归调用的链上量到的是两遍左右 —— 101 层那条链：1201 次模块调用、`note` 198 笔、答案 100；
分叉多、压得多的那种模块体会被做得更多，而每次多做的代价仍然只是它自己那几步。

一个模块自己也可以给压缩点（`env = $try_compact << args.env << 0` 那样），上限是那次调用算出来的值
（可以是一条定义算出来的）。在不是一次运行的地方调用它（宿主自己拿 `environ.try_compact(…)`），
链没过线就把那个环境答回来，过线了当场报 `no run is here`：压的是这次运行的工作清单，没有运行就
没有可压的。

**不纯的宿主步骤会被重做**：重做一份模块体时，它自己那几步重走一遍。可回放的那几步走 `replayed`
（见「幂等与快照」），所以重做给的是同一个答案；没走快照、又不纯的那几步两次跑出来的东西可能不同 ——
这是它自己该处理的事，不是压缩带进来的新规矩。

**一个没有基例的递归不再由解释器喊停**：以前它会在解释器自己的栈上撞到上限（`RecursionError`，报成
某一步失败），现在链可以一直长下去，调用也一直做下去。同一个模块带着同一份实参又在跑那种环还是当场报
（见「一份文件跑不出递归」）。

**按数据路径取回来的那一层不记着压掉的原路径**：`$uncompress_relative_path` 是 `nil`
（它是 `compress_env_path` 放在那一个环境上的值，取回来的这一层是新造的）。压过的
路径本身认不出来是哪一条原路径 —— 名字里只有一个 sha1。

**重做的那一层看到的链是取回来的那一条**：它的 `get_parent` 是链顶（它是链顶下面一层），不是被
掀掉那条长链上的一段。链就是此刻的系统栈，这一点压缩前后都按事实说：那一段调用已经不在栈上，
它在工作清单里。

**为什么不是全手工栈，也不是 continuation。** 两条路都能把栈深解掉，只是各要付一笔这里没付的代价。

- **全手工栈**：要动的是解释器自己的每一个求值边界 —— 每一处 `<<`、每一次给实参、每一次取成员，都得
  能停在半路，把"接下来做什么"留成数据，整层求值因此是一台状态机；而这份代价每次调用都要付，不管
  递归深不深。现在只有过线之后的那些调用付。
- **continuation（无栈）**：要动的是宿主那一侧 —— 一步要么交回"接着做什么"，要么只能由外面再进来
  一次，于是 `get_func` 交回的那个可调用对象不再只是"收实参、答一个值"，宿主那一侧的函数跟着变；每
  一步也多一份 continuation 的成本。

**这个设计选的是第三条路**：结果本来就要能回放（见「幂等与快照」），于是系统栈照旧用，只在链太长
的时候把它变成工作清单 —— 没过上限的运行照旧只有系统栈，过了上限的那些才付"多做几遍模块体"这份
代价。

## 把一次调用落成可执行的：`__dyn_call__` 与 `__dyn_method__`

一步是按**名字**实现的：`get_func(module_path, func_name)` 收到的那一个。名字出现在链头时只在给出它的那个模块里成立
——`a.b.c` 在那里指得着，交出去、存下来、换个模块取就指不着了，因为它是一条名字路径，不是一个自己站得住的调用。
所以把一次调用落成**可执行的数据**时，名字跟着数据走。下面这两个名字由 `interpret` 自己回答：它们不向 `get_func`
要实现，一个模块自己的定义也盖不住它们。

```viba
__dyn_call__ =
    Any
  <- $env Env
  <- $name str
  <- $args ...

__dyn_method__ =
    Any
  <- $env Env
  <- $name str
  <- $value Any
  <- $args ...
```

- **`__dyn_call__ << env << "a.b.c" << 实参…` 就是 `a.b.c << env << 实参…`**：名字当数据交给宿主
  （`get_func(<给的那个环境的数据路径>, "a.b.c")`），后面给的实参按源码里的次序接着交给它。它问的是**宿主**，
  不是照这个名字去找一份 `.viba` 文件：要跑一个模块，给的是它的名字（`demo << env`），不是这里。
  名字不是字符串、这次调用没拿到环境，都是**程序错**。
- **成员是某份值的成员时走 `__dyn_method__`**：`$foo << bar << 实参…` 取的是 `bar` 的成员 `foo`，而 `bar`
  是一份值、不是名字路径，所以这里的名字只能是数据：`__dyn_method__ << env << "foo" << bar << 实参…`
  就是 `$foo << bar << 实参…` —— 那份值当成员的第一个实参，环境是这次调用的规矩。成员自己是一份值时
  （`uncompress_relative_path` 那种）取出来就是这个值；值就是环境时（`args.env.tmp_env` 那种），
  取的是那个环境自己的那一个。
- **不带环境就是闭包**：`__dyn_call__ << "add" << $a 1` 是一个值 —— 判定层取它就判成 `Any`，因为名是数据、
  没有名字可展开；给它一个环境就执行，和别的没给环境的调用一样。
- **`$call` 用的就是这两种源码形式**（见「一次执行会得到什么」）：名字是字符串数据，环境剔掉 —— 链上有环境的那一处也剔掉，
  剩下的就是那个闭包。**一份数据里凡是调用，都按这两种源码形式给**：给宿主的数据里，那一处调用不给名字，给的就是
  那个调用本身。成员那一支正是这样：`$f << box << args.env << 1` 落成
  `__dyn_method__ << "f" << ($f (__dyn_call__ << "inc") * $y 2) << 1` —— 成员那一层保留，而那份值里
  `$f` 这一格是调用 `inc`，就落成 `__dyn_call__ << "inc"`。反序列化时不必解析 `inc` 这个名字，也就不必让
  给出它的那个模块在场。函数类型的实参也一样：`$f inc` 落成 `$f (__dyn_call__ << "inc")`。
- **设计层判同一段源码形式，说的是 `Any`**：名字是数据，设计说不出这次调用答什么，所以它既不核后面那几个实参，
  也不多说。

环境上的成员收不下给它的东西时，报的是第四种停法（`$environment_api_invalid_argument_err`），
不是「某一步失败了」——见下一节。

## 一次执行会得到什么

`interpret` 与 `exec` 回答的是**一份 viba 数据**：声明里那个 `InterpretResult` 的一个节点
（定义在 [`viba/interpret_result.viba`](viba/interpret_result.viba)）。它只有两支：`$ok` 是这次执行算出来的
值，`$err` 是它停下的方式。按数据取它就行 —— 两支各有自己的 tag，停下来之后那一支再按 tag 往下取一层：

```viba
answer = interpret("demo.viba", environ)
answer.by_tag("$ok")                      # 值那一支（就是 __impl__ 的答案）
answer.by_tag("$err")                     # 停下那一支
answer.by_tag("$err").by_tag("$not_implemented_err")
                                          # 哪一种停法，再往下就是它的字段
```

`viba/interpret.py` 的 `is_interpret_result` 用 `is_sub_type` 拿同一个定义判这份数据，所以"这个回答是不是
一个 `InterpretResult`"是能查的；声明改了而回答没跟着改，当场就是 `False`。

停下的方式有四种：

- `$viba_program_err ProgramErr`：**这一份程序或环境不行** —— 编不过、文件不在、没有 `__impl__`、
  `$env` 没给……它不说"哪一步"，所以不带步名；`$msg` 一句话说是什么事，`$stack` 是这次执行走过的调用链（见下）；
- `$underlying_viba_op_err UnderlyingOpErr`：**这一步坏了** —— 它的实现抛了、给出了没有叶子的东西，
  或者 `get_func` 自己坏了；
- `$not_implemented_err UnderlyingOpErr`：**这一步没有实现** —— `get_func` 那里没有它（`get_func` 返回 `None`，
  或者交回了这一支）。这不是失败：`interpret` 不带库函数，所以"没有实现"很正常。
- `$environment_api_invalid_argument_err EnvironmentApiInvalidArgumentErr`：**环境上的一个 api 收不下给它的
  东西** —— `Environment` 的成员（`sub_env`、`tmp_env`、`get_relative_path`……），或者宿主挂在环境上的那些，
  把这次调用退回来了：给的根不是它的祖先、路径里带 `.` 或 `..`、那个环境底下没有 storage……它是 viba 自己
  这一侧的活：`get_func` 从来没有被问过它们，所以这不是"某一步失败了"，也不带步名。

后两支的字段是**同一个 `UnderlyingOpErr`**，靠 tag 分开 —— 调用方按 tag 分，不必看那句话。`$msg` 一句话说清是
什么事，**开头就是原因**：`no implementation`（`get_func` 给了 `None`）、`get_func raised`（`get_func` 自己坏了）、
`raised`（实现抛了，它自己没说为什么时就是这个词）、`no leaf`（实现给出了没有叶子的东西）。

`EnvironmentApiInvalidArgumentErr` 的 `$api_name` 是**哪一个 api**，就是它定义里那个名字：
`Environment.sub_env`（[`viba/builtin.viba`](viba/builtin.viba) 里 `Environment` 的成员就是这几个）；
`$args` 是**给它的实参**，能落成数据的那几份，按源码里的次序，环境不在里面（它是那个 api 自己的值）——
所以 `sub_env << "kid"` 记的就是 `"kid"`，只收环境的 `tmp_env` 记的是 `nil`。它的 `$msg` 也开头就是原因：
`raised`（那个 api 收不下，抛了）。

一次执行回答的必须是数据，所以**环境不是答案**：环境是这次调用的规则，不是值，一个模块的 `__impl__` 落成
`args.env` 时，这次执行停下，报的就是上面第一种（`the run answered the environment`），与文件里的一条链被判定层拦下时
是同一个道理（[`viba/partial.py`](viba/partial.py)）。

`$stack` 是这次执行停下的**调用链**，最外一帧在前：一帧就是一次调用点 —— 调用出现在哪个 `.viba` 文件里
（`$file_path`）、在哪一行（`$lineno`）。最外那一帧是主文件本身，它不是被谁调用的，行号是 0。

`$module_path` 是这一步声明在哪个模块（换成路径，`/foo/bar`），
`$full_qualified_func_name` 是它的整名
（`foo.bar.b_scale`，也就是 `get_func` 收到的那两个合起来：模块和它在那儿叫的名字）。这次调用跑在
哪儿不在里面 —— 那是给它的环境的数据路径。拿 `$call` 再跑一遍时，环境由你自己给：给在哪儿它就跑在
哪儿；名字里带着模块，所以问的是哪个模块的哪一步从名字就看得出。

`$call` 是**这次调用本身，落成能再跑一遍的样子**：一步坏了的时候，取它的那份运行不一定是给出它的那个
模块，所以名在这里当数据走，而且是**整名** —— `__dyn_call__ << "foo.bar.b_scale" << $x 2`（就是
`foo.bar.b_scale << $x 2`）；反序列化时按最后一个点切开，前面是模块、后面是名字（上一节）。成员是某份
值的成员时保留成员那一层：`$f << box << args.env << 1` 落成
`__dyn_method__ << "f" << ($f (__dyn_call__ << "inc") * $y 2) << 1` —— 环境那一处剔掉了，那份值里
`$f` 这一格是调用 `inc`，就按能跑的形式给出（上一节）。同一个规矩也在函数类型的实参上：`$f inc` 落成
`$f (__dyn_call__ << "inc")`。
环境不在里面：给出环境就是执行这次调用，它不是这次调用的数据，另一次运行自己造。声明里这一格是
`Any`：名是数据，判定层取这种链就判成 `Any`（没有名字可展开，[`viba/partial.py`](viba/partial.py) 的
`reduce_partial`），所以它就是这个类型。把它当数据交给一次运行、让那次运行给它一个环境
（`held = hold << $env args.env`，再 `held << args.env`），这次调用就跑起来了；也可以把它用
`serialize` 序列化成 viba 源代码（`call = __dyn_call__ << "add" << $a 1`）再跑，或者把它**直接当节点**
交给 `exec`（把停下来那一支里的 `$call` 那一格交给它：它本来就是节点，不用序列化成源码再反序列化）。实参里是函数、
是闭包也一样，它们本来就是可序列化的 —— 照着它就能把这次调用重新做一遍，不必重跑一次运行，也不必让给出它的
那个模块在场（上一节）。这三条路都钉在用例里
（[`tests/test_interpreter_roundtrip.py`](tests/test_interpreter_roundtrip.py)：二十七份，先缺一步、
补上、把 `$call` 跑一遍，含同一个名字在两条数据路径上、以及一轮一轮补下去的那种）。

这四种停法都会一路穿回调用方，而且**带回来的那一步是里面那一次调用、不是外面那一层**：被调用的模块里
那一步没实现，带回来的就是那个定义（`root/模块名` 下那个），穿过运行、再交给宿主的可调用对象时也一样。

宿主自己要说这句话时，**交回那份数据**，不抛异常：`get_func` 在本来该回答 `None` 的地方交回它，
这一步的实现在本来该回答一个值的地方交回它。`viba.interpret.not_implemented(...)` 给的就是这份数据
（`$not_implemented_err (...)`，字段都在，留空也行）：

```python
from viba.interpret import not_implemented

def get_func(module_path, func_name):
    if (module_path, func_name) == ("service_b", "b_scale"):
        return not_implemented()               # 这一步没有实现，别的不用给
    if (module_path, func_name) == ("service_b", "b_scale2"):
        return lambda env, x: not_implemented("no b_scale2 here")
    return None                                # 这个模块别的步骤也没有实现
```

它已经填好的东西（步名、`$call`、`$msg`、tag）一律留着，缺的由 run 用它知道的这次调用补上
（留空不是错：步名与 `$call` 照补，`$msg` 按 tag 给 —— `$not_implemented_err` 时是 `no implementation`，
`$underlying_viba_op_err` 时是 `raised`）。所以 `not_implemented()` 一个字段都不填也一样，`$call` 不会是空的。
实现抛 Python 异常仍然算"实现坏了"（`$msg` 开头 `raised:`），`get_func` 抛算"`get_func` 坏了"
（`get_func raised:`）—— 那两句话说的是这层代码自己出了事，不是"这一步没有实现"。
环境上的 api 那次也一样：坏的是里面那一个 api，带回来的就是它，不是外面那一层。

`$viba_program_err` 的 `$msg` 长这样：

```
no such file: ...                     文件不在
no definition named 'x' in module 'm' 名字解析不了
module 'x' not found (...)            import 找不到文件
module 'm' has no __impl__: ...        类型，不是程序
... needs an Environment ...          一次模块调用没给环境（它只收环境）
... was not given an Environment      给了，但不是 Environment
... storage path ... already running  同一条数据路径上已经有一次调用在跑
module 'x' ... same arguments         同一个模块带着同一份实参又在跑（没有进展）
... storage path ... already used     两个模块挤同一条数据路径
... is a function still waiting ...   __impl__ 不是值
the run answered the environment ...   __impl__ 回答的是环境，而环境不是值
... was given no Environment to run in  __dyn_call__ 拿到环境才能跑
... was given no name to call         __dyn_call__ 没拿到名字
... was given no value ...            __dyn_method__ 没拿到取值的那份值
cannot take ...                       文件取不了
cannot parse ...                      编译不过（语法错误）
viba_path is a string ...             viba_path 给错了类型
```

`$underlying_viba_op_err` 的 `$msg` 一行（`$module_path`、`$full_qualified_func_name`、`$call` 是另外三个字段），
每种原因都在开头那一个词上：

```
no implementation                      `get_func` 给了 `None`，或者宿主交回这一支时没说一句话
get_func raised: get_func('root', 'add') raised ...    `get_func` 自己坏了
raised: add raised ZeroDivisionError(...)   实现抛了
raised                                实现自己交回这一支时没说为什么
no leaf: add answered list, ...        实现给出了没有叶子的东西
```

`$environment_api_invalid_argument_err` 的 `$msg` 一种，`$api_name` 与 `$args` 是另外两个字段：

```
raised: Environment.sub_env raised AttributeError(...)   那个 api 收不下给它的东西
```

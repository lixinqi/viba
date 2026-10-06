# 模式：一个泛型是一个目录，某个文件给出结果

泛型不是一份带形参的定义，而是一组**文件**：每一个文件说"实参长这样的时候，结果是这个"。
调用方把实参写在方括号里（`is_base_type[bool]`），**决断**是一个静态动作 ——
它只看写下来的那几个实参，不看程序跑到哪儿了。

**特化是一次编译时的调用**：`g[实参]` 就是调用 `g`，把选中的那份文件 `pattern` 里的形参换成
这次给的实参。所以形参叫什么名字不要紧，它代表的是调用方给的那一份；形参换成实参之后，实参里
带着的名字按调用方那份文件算（第 4 节）。

```viba
import demo.is_base_type as is_base_type

Flag = is_base_type[bool]          # true
```

一个泛型由三件事定下来：

- 它放在一个目录里，**目录名就是泛型名**；
- 目录下必须有标记文件 `__generic__.viba`；
- 其余 `.viba` 文件每一个都是一个模式，**文件名是它的决断顺序**（一个数字），数字前面还可以写上
  这一份**读几份**（第 4.1 节）。

```
demo/is_base_type/__generic__.viba      标记：这个目录是一个泛型
demo/is_base_type/100.viba              pattern bool | int | float | str
demo/is_base_type/200.viba              pattern A
demo/is_base_type/2_250.viba            同一份东西的另一种写法：读两份，决断顺序 250
```

`import demo.is_base_type as is_base_type` 找到的是那个**目录**：找模块的地方按顺序试
`demo/is_base_type.viba`，然后才是 `demo/is_base_type/__generic__.viba`（[`viba-interpreter.md`](viba-interpreter.md)
第 1 节）。目录里有标记文件，它就是一个泛型；没有，这个目录与 viba 无关。

## 1. 标记文件 `__generic__.viba`

标记文件只需要存在，并且能编译。它写了什么，决断不看；习惯上里面写一行注释：

```viba
# __generic__.viba
```

它不能写 `pattern`：`pattern` 只写在模式文件里，而模式文件的名字是一个数字
（第 2 节）。它不是模块，也不是模式文件 —— 它是一个目录的标记。

## 2. 一个模式文件：`pattern` 与它给出的那个成员

一个模式文件按顺序写若干行 `pattern`，**每个形参一行**：

```viba
# demo/is_base_type/100.viba
pattern bool | int | float | str

value = true
```

一行 `pattern` 写一个模式，模式写的是**这个形参收什么**，两种写法：

- **写下来的类型**：实参必须能落进它，落不落得进由判定层自己的子类型（`is_sub_type`）判。
  `bool | int | float | str` 收 `bool`（`true` 也收得下），不收 `list[int]`。
- **文件里没定义的名字**：那是一个**形参**，表示"这里的东西萃取出来"。`pattern A` 里的 `A`
  就是这个文件里没有的（自己没定义、不是内建名、也没有从它的 import 里来），于是实参在这个位置
  是什么，`A` 就是什么。

同名形参写两次是**相等**：两处都收得下彼此才算命中。`is_compatable` 的第三个文件就是这样：

```viba
# demo/is_compatable/200.viba
pattern A
pattern A

value = true
```

`pattern` 可以带结构，形参就在结构里面：

```viba
# demo/element_type_of/100.viba
pattern list[A]

type = A
```

```viba
# demo/ret_type_of/300.viba
pattern A <- B <- C

type = A
```

模式写成结构时，一份一份对着读 —— 模式那一份写成什么，就认实参里的什么：

| 模式那一份写成 | 认什么 | 形参拿到什么 |
|----------------|--------|--------------|
| 一个名字，文件里没定义的（`A`） | 任何一份 | 那一份整块，tag 也在里面 |
| 一个和（`A \| B`） | 实参的每个分支都落在某个分支上（实参不是和时，把它自己当一个分支） | 落在哪个分支上，那一支里的形参 |
| 一个积（`A * B`） | 实参也是积，成员个数一样 | 逐位，顺序照书写 |
| 一条指数链（`A <- B <- C`） | 实参也是链，位置个数一样（结果在前、实参按书写顺序） | 逐位，顺序照书写 |
| 一个元组（`(A, B)`） | 实参也是元组，个数一样 | 逐位，顺序照书写 |
| 一个应用（`list[A]`） | 构造器指同一个东西，参数个数一样 | 逐个参数 |
| 一个 tag（`$tag T`） | 实参的 tag 就是 `$tag` | `T` 拿 tag 里面那一层 |
| `tagged[x, T]` | 实参带任何 tag | `x` 拿 tag 的符号（一个字符串），`T` 拿里面那一层 |
| 一次调用（`F << A`） | 实参是**写下来的一次调用**，给的实参个数和模式里的 `<<` 一样多，而且还没给环境 | `F` 拿链头（这条 api），每个 `<<` 按书写顺序拿一个实参 |
| 哪一段里没有形参 | 那一段整体交给 `is_sub_type` 判 | 不萃取 —— `A <- (() \| nil)` 里的 `(() \| nil)` 就是这样 |

积、指数链、元组、应用这四行，实参对应的那一层不是积 / 链 / 元组 / 应用时就是没命中：
这不是错，换下一个文件接着试。

**实参写成调用时怎么读。** 读它的那一层先把这个调用化成它代表的类型（那条链，这个实参已经坐在
自己的位置上），所以**链模式**（`A <- B <- C`）看的是链上**剩下**的那条类型。**调用模式**
（`F << A`）不看剩下的类型，看的是实参**写下来**的那次调用：**一个 `<<` 对一个实参**，
`pattern F << A` 只认给了一个实参的调用、`pattern F << A << B` 只认给了两个的。所以长度不同
就得各写一个文件（`viba/apply_impl/` 就是这么按长度分的，1..16 各一份）。另外，那次调用还得
没给过环境，才叫闭包 —— 给环境就是执行，所以：

- 给过环境的调用不认：环境那个 `$env` tag 也好，某一份的类型就是 `Env`（模块里的 `args.env`
  就是）也好，都算给过；
- 光一个名字、一个叶子不是调用，不认；
- 一份外面的 tag 是这一份的一部分：`F << A` 认不出 `$x (add << $a 1)`，要写 `pattern $x (F << A)`，
  同一个意思也可以写 `tagged[x, F << A]`。

`is_closure` 与 `unclosure` 就是两个这样的泛型
（[`viba/builtin/is_closure/`](viba/builtin/is_closure/1_100.viba)、
[`viba/builtin/unclosure/`](viba/builtin/unclosure/1_100.viba)）：各写 1..16 段的 16 个文件，
前者给出 `value`（`true` / `false`，16 段以上落到最后那份兜底），后者给出 `f`（链头那条 api）与
`captured`（收下的那些实参合成的一份积）。内建的 `sequential` 也是这么分的：它的
[`sequential_step/`](viba/builtin/sequential_step/1_200.viba) 认一步那次调用（带 1..16 个 tag 实参
各一份），[`sequential_arg/`](viba/builtin/sequential_arg/100.viba)
认一个实参（一次变量引用，或者照写的任何一份），
[`viba/sequential_impl/`](viba/sequential_impl/2_200.viba) 按步骤数各一份（2..64；一步没有 tag 钉住
个数，按写下来的调用实参个数各一份）
（[`viba-interpreter.md`](viba-interpreter.md)）。这几份目录里的文件名前面都写着份数
（第 4 节），所以决断不必翻完整个目录。

形参拿的是那一位**写下来的整块** —— tag 也是那一块的一部分。`pattern A <- $env Env <- B <- C`
对着 `int <- $env Env <- $a int <- $b int` 时，`B` 是 `$a int`、`C` 是 `$b int`；这个成员
把它们写回哪里，tag 就跟着回到哪里。只想要 tag 里面那一层，就把 tag 写进模式：

| 模式那一位 | 收什么 | 形参拿到什么 |
|------------|--------|--------------|
| `B` | 任何 tag | 整块（`$a int`） |
| `$a B` | tag 必须是 `$a` | 里面那一层（`int`） |
| `tagged[n, B]` | 任何 tag | `n` 是符号（`"a"`），`B` 是里面那一层 |

**哪个形参带 tag、哪个不带，由模式自己说了算**：要原样传下去就写裸名字（`wrapper` 就是靠这个
把调用方的 tag 一路带过去的），要把 tag 认下来 —— 限定它、改名、或者拿它跟别处比 —— 就在模式里
写明。也正因为裸名字带着 tag，拿两个形参去拼同一个积时同一个 tag 撞上第二次会当场报错
（`the tag $a is written twice in one product`）：那本来就是写错的设计，要改名就用第三行那种写法。

成员的名字由这份文件自己起：**给出的是一份类型就叫 `type`，是一份值就叫 `value`**（跟 C++ 模板的
`::type` / `::value` 一回事），调用方照名字取 —— `ret_type_of[int <- int].type` 是 `int`，
`is_base_type[bool].value` 是 `true`。泛型特化文件**不是函数**，所以它们不写 `__decl__`：
`__decl__` 只属于"这个模块要当函数"的那种文件。

`pattern A` 这种没有任何限定的写法**直接收下对象自身**：`A` 拿到的是实参写下来的那一整份
（`pattern A` 加 `type = A` 就是恒等；[`demo/itself/100.viba`](tests/data/pattern/demo/itself/100.viba)）。
有限定的写法（`pattern bool | int`、`pattern list[A]`）才是"先看合不合，再按位置抽"。

**被匹配的对象是一个模块时，读的是它的 `__decl__`**：模块当函数读出来的那条链（结果在前、参数在后）
就是拿来按位置拆的那一份 —— `pattern A <- B` 匹配一个 `__decl__ = int <- $env Env` 的模块，
`A` 是 `int`、`B` 是 `$env Env`（[`ret_of_a_module.viba`](tests/data/pattern/ret_of_a_module.viba)、
[`decl_pair_of_a_module.viba`](tests/data/pattern/decl_pair_of_a_module.viba)）。模块没有 `__decl__`
就是没有函数签名，这种模式没有东西可拆。

这个文件给出的是**它自己那个模块**（模块语义）：决断选中谁，调用方就在谁身上按名字取一个成员，形参名换成实参里对应的
那一部分。它可以是任何类型：

```viba
pattern list[A]
type = A                        # 给出萃取到的那一个类型
```

```viba
pattern A <- B <- C
type = (A, B)                   # 给出一个元组类型
```

```viba
pattern A
type = (int <- $env Env <- int)   # 给出一个函数类型（带环境的那个写法也照写）
```

函数类型照平时的两种读法：

- **在类型层读**：`$env Env` 那个参数是调用的规矩、不进类型，所以上面那条就是 `int <- int`；
- **在值那一层读**：这个应用代表的就是那次**调用**，跟"一个定义体是函数链"完全一样 ——
  `X = gen[T]` 是一份写下来的调用，可以继续给实参（`gen[T] << args.env << …`），也可以
  当闭包递出去。宿主按**泛型的名字**找这一步的实现（写下来是 `wrap[...]`，它拿到的是
  `"wrap"`），跟按定义名找实现同一种做法。

### 摊开实参积：`apply`

"拿一个函数换一个调用"写成一份普通的模块：[`demo/wrapper.viba`](tests/data/pattern/demo/wrapper.viba)
收下一个函数和它那一份实参积，把活交给 `apply`（[`viba/apply.viba`](viba/apply.viba)）。`apply`
自己也不知道实参有几个 —— 它把那份积交给泛型 `apply_impl[args.args]`，由那份决定数出来：

```viba
# viba/apply_impl/1_100.viba：名字上写着读一份，积里一个成员就选这一份
pattern tagged[arg0_name, Arg0]

__decl__ =
    Any
  <- $f Any
  <- $args Any

__impl__ =
    args.f
  << args.env
  << ($__getattr__ << args.args << arg0_name)
```

`pattern tagged[arg0_name, Arg0]` 每一位收下积里一个成员的 tag（`arg0_name` 是符号 `"x"`），
`$__getattr__ << args.args << arg0_name` 再按那个名字把成员取回来 —— 成员原来带什么 tag，取回来
还是那个 tag，于是它落进函数对应的那个参数。积里有几个成员就选哪一份文件，1 到 16 各一份
（[`viba/apply_impl/`](viba/apply_impl)），所以调用方不用写实参个数。

**环境不必是一个参数**：`$env Env` 可以不写，环境由调用方单独给一个 `<<`。这时环境不进门：模块体
取不到 `args.env`（写它是程序错），环境只用来跑这次调用，再缀在这次调用返回的那次调用末尾。
`apply << f << args` 只是把这两件收下（这时它还是个闭包），给了环境才跑：

```viba
add2 = add << $a 2

__impl__ = wrapper << add2 << $b 1 << args.env     # 3
```

这四步都有用例，都在 [`tests/data/apply/`](tests/data/apply)：`direct`（不套 `apply`）、
`apply_once`、`apply_twice`、`apply_thrice` —— 一个函数被 `apply` 套 0 到 3 层，每套一层，环境就往后
缀一次；外面那份积里的成员用 `$f`、`$args` 两个 tag 写，因为决断的 `pattern tagged[...]` 是按 tag 认
它们的。

`add2` 是给过 `$a 2` 的那次调用，还欠 `$b` 和环境；`wrapper` 把函数与积交给 `apply`，`apply` 跑出
`add2 << $b 1`，环境再缀到这次调用的末尾，于是它凑齐了。每返回一层就缀一次：`wrapper` 返回的那次调用、
`apply` 返回的那次调用、`apply_impl` 返回的那次调用，一层层往后递，直到落进函数自己的 `$env Env`
那一位。

给实参要写出它落在哪个 tag 上（`$a 2`）：不写 tag 的实参按位置落到函数自己的参数上，环境不占位置
（`viba-interpreter.md` 的「模块的参数」一节）。

一个文件里也可以有自己的定义，成员读到自己定义的名字就在这个文件里读：

```viba
# demo/wrapped_item/100.viba
pattern list[A]

Wrapped = $item A

type = Wrapped                  # 结果是 $item 实参里的元素
```

### 选中的文件自己会算：`__impl__`

决断只把选中的那份文件读成它自己那个模块；`__impl__` 是**运行时**的事。一份模式文件同时也可以是一个能跑的模块：它的
`__decl__` 里带 `$env Env`、文件里写了 `__impl__`（有签名才算函数），那么这次应用**就是它这一次模块调用** —— 环境与
实参按它的 `__decl__` 给（调用方给环境，欠着的实参接着给），活由它自己的 `__impl__` 算，宿主不必
按泛型的名字另找一份实现。它跑在自己的**子环境**里，那一层的名字就是这份文件的**数字**
（`100.viba` 跑在 `…/100` 下），所以一次调用的数据路径仍然写得下来、复现得了，调用方不必自己再包
一层。成员是个函数链、又写在链头时，它就是那一次调用，跑起来由宿主按**泛型自己的名字**实现。

模式文件里的其它定义跟别的文件一样，是按需求值的**绑定**（binding）：被用到的时候才求值，求一次
就把结果留下来（call-by-need 与 memoization，见 [`viba-interpreter.md`](viba-interpreter.md) 的
「求值策略：按需求值（call-by-need）」一节）。写的次序不是求值的次序，没人用到的那一份不参与求值 ——
`tests/data/y_functions/steps/` 那批用例就靠这一点：往下调一层是一个绑定，基例那一侧不用它。

一份模式文件的成员里也可以拿自己的形参再特化一次：那次特化同样是编译时的调用，换成的是
**调用方给的那个实参**，不是这个形参名本身。这样建起来的半成品记着这次替换，交给另一个模块当
实参、或者存成闭包之后再给实参时，仍然是那次替换。

`Y.viba`、`y_helper.viba` 与 `apply.viba`、`apply_impl/` 就住在包的内建目录里（`viba/`，
`builtin.viba` 旁边），那里是搜索路径的最后一站，所以写 `Y << …` 就拿到 `Y`（`import Y` 也行），谁也不必把包的
目录写进 `viba_path`（`viba-interpreter.md`）。Y 与 y_helper 都是**普通模块**，都不写 `pattern`，
都写 `$env Env`：给环境就是执行，所以每一层的数据路径由调用方写下来 —— 主文件写
`args.env.sub_env << args.env << "Y"`，`Y.viba` 给 helper 写 `"y_helper"`，步文件往下调时写
`"low"` 那样的名字。步自己那一层只是名字的出处：`y_helper` 用 `convert_sub_to_sibling` 把它的
数据路径压成 workspace 旁边的一个名字（`{workspace 的路径}_{sha1(那一层数据路径)}`），于是往下多少层，
数据路径都只有一个哈希那么长；`caller_workspace_relative_path` 每层原样往下传，压掉的原数据路径记在
压出来那一层的 `uncompress_relative_path` 上（`viba-interpreter.md`）。
`Y << f` 就是那一步的不动点（`y_helper << $f f << $y_helper y_helper`），
`y_helper` 是自应用那一步 —— 它给这一步的是"上一步拿到的自己"，也就是下一次 `y_helper` 调用。
这一层的实参接着 `<<` 写在后面（`Y << F << ($sub_env << args.env << "Y") << $n 3 << $m 4`），
`Y` 那个写 `$args ...` 的参数把它们收成一份积；`apply` 与 `apply_impl` 的 `$args` 写的是 `Any`，
要的是一份写好的积，`apply` 按积里有几个成员选 `apply_impl` 那一支，所以这几个模块都不必按参数
个数分文件。`tests/data/y_functions/` 拿它跑 20 个递归函数（阶乘、斐波那契、阿克曼……）。

## 3. 符号写在字符串里：`tagged`

tag 是数据路径（`$a`），设计里写的就是它，所以一个只作为**字符串**存在的名字没有地方放。`tagged`
就是那个地方 —— 它是内建的，收一个符号，或者一个符号加一个类型：

```viba
Tagged = tagged["a", T]                   # 就是 Tagged = $a T

__impl__ = tagged["hello"] << persion      # 就是 $hello << persion
```

符号写在字符串里（`"a"`，或者带 `$` 的 `"$a"`），而且必须是一个合法符号：字母、数字、`_`，不以数字
开头。写错了当场报错（`'a b' is no symbol for a tag`），参数个数不是 1 或 2 也当场报错。
`tagged` 是内建名，谁也不能拿它定义（`tagged = int` 编不过）。

一参数的写法是一个**成员**，跟 `$hello` 一样只写在链头：

```viba
__impl__ = tagged["hello"] << persion        # 就是 $hello << persion
```

两参数的写法是一个类型，各层读到的就是那个 tag。

**符号的位置上可以是一个形参。** `pattern` 行里那样写时，形参拿到的是实参那个 tag 的符号
（不带 `$` 的字符串）：

```viba
# demo/arg_name_of/100.viba
pattern tagged[arg_name, T]

value = arg_name                # 给出 "a"
```

于是 `arg_name_of[$a int]` 给出字符串 `"a"`：**一个 tag 的名字可以当值用**。反过来，决断把这个字符串
交给这个成员，它就用同一个 tag 建回来：

```viba
# demo/tagged_again/100.viba
pattern tagged[arg_name, T]

type = tagged[arg_name, T]           # 又建回 `$a int`
```

读各层的时候，符号要么是写下来的字符串，要么是一个解析得出字符串的名字（决断绑给它的那个）；
两者都不是时这不是一个 tag，各层照它原本的写法报错。

## 4. 决断：数字从小到大，第一个命中的赢

文件的数字就是顺序，**不要求连续**：`100.viba`、`150.viba`、`300.viba` 照数字排。
命中一个就停，后面的不再看。

- 一个文件的 `pattern` 行数与这次应用写的实参个数不同，它不可能是这一支，跳过。所以
  **「这个泛型收几个实参」也是每个文件自己的事**：几行 `pattern` 就收几个，不同个数各写一个
  文件（`num_variadic_args` 里就是 0 / 1 / 2 / 3 各一个），一行不写就是收零个；一个模式是一个
  类型，里面没有 `...` 这种写法；
- 所有文件的行数都与实参个数不同，直接报错（`generic 'x' takes 2 parameters, not 1`）；
- 没有任何文件命中，直接报错（`no pattern of 'x' matches [bool]: the decision failed`）。

**决断失败是程序错误**，不是 `never`，也不是 `false`：一个决断没有结果的泛型应用是写错的设计。

### 4.1 名字上写的份数：事先分好的桶

一个泛型的文件多了（`sequential_impl` 有 79 份，`apply_impl`、`unclosure`、`sequential_step` 各
16 份，`is_closure` 17 份 —— 它多一份兜底 `1700.viba`），逐个编过来才认出哪一份接得住，代价就
落在每次应用上。所以**文件名
在决断顺序前面可以再写一个数：这一份读几份**（`2_200.viba` 读两份，决断顺序是 200）。决断先数
一遍这次应用摆出来几份，只读那个份数的文件 —— 份数是事先分好的**桶**，跳到桶里就不必翻整份目录。

数的是每个实参摆出来几份，再相加（一个文件几行 `pattern` 就读几个实参，所以按实参分别数）。
数得出来的是写法里已经把部分写死的那几种，一共三类 —— **积**（`A * B`、元组 `(A, B)`）、
**函数链**（`A <- B`）、**应用**（`list[int]`、`$a int`、写下来的一次调用 `f << a`）：

| 实参写成 | 摆出几份 |
|----------|----------|
| 一次调用（`f << a << b`） | 它给的那几个实参；另外，它照 `<<` 化开之后剩下的那条链摆几份，也算 |
| 一个积 | 成员个数 |
| 一条链 | 位置个数 |
| 一个元组 | 元素个数 |
| 一个应用（`list[int]`） | 实参个数 |
| 一个 tag（`$a int`） | 一份 |
| 一个和 | 数不出来（见下） |
| 别的（一个叶子、一个名字） | 数不出来 |

**和类型不参与分桶。** 一个和实参被 `pattern` 行读开时，读开的是这个实参自己写下的分支：`pattern
A | B | C` 在 `int | str` 上读两份，在 `int | str | bool` 上读三份，同一份文件两种应用都命中。所以
在和的这一侧，份数不是文件的性质，而是实参的性质；写进名字里，换一次应用就对不上，那一份反而被
跳过去。因此：

- 实参里有和时，这次应用**什么都不跳**：目录里的文件都读一遍，跟没有写份数时一样；
- `pattern` 行写成和时，名字上不许写份数，写了就在读到那一份时报错（`…and a \`pattern\` line in
  it reads no count`）。

**数不出来就什么都不跳**：这次应用把目录里的文件都读一遍，跟没有写份数时一样。所以份数只是
省下不必读的文件，不是必要条件 —— 名字上写了份数的文件，命中与否仍然由 `pattern` 行说了算。

名字上写的份数**必须与这个文件的 `pattern` 行读的份数一致**：一个文件几行 `pattern`，各读几份
就加几份（`pattern A * B` 读两份，`pattern F << A << B` 也读两份，`pattern A` 读不出固定份数）。
不一致时，决断读到那一份就报错：

- `…/3_100.viba: the name says the file reads 3 parts, and its `pattern` lines read 2`
- `…/2_100.viba: the name says the file reads 2 parts, and a `pattern` line in it reads no count`

份数对不上的文件这一轮不编：那一份里写错了什么（连编不过）都留着下次再说。份数对得上、或者根本
没写份数的文件照样编，编不过当场报。目录里一份都没命中时，整个目录都会被读一遍 —— 那时份数不再
跳过任何文件 —— 好把话说完：实参个数没有文件接就报 `takes N parameters, not M`，有文件接但都
不命中就报 `no pattern of … matches`。

```viba
# viba/apply_impl/，名字上的份数是积的成员个数
# 1_100.viba    pattern tagged[arg0_name, Arg0]
# 2_200.viba    pattern tagged[arg0_name, Arg0] * tagged[arg1_name, Arg1]

# viba/sequential_impl/，名字上的份数是步骤数（一步没有 tag 钉住个数，按那次调用的实参个数）
# 2_200.viba    两步：pattern tagged[step0_name, Step0] * Last
# 1_101.viba    一步，这次调用给了一个实参：pattern F << tagged[arg0_name, Arg0]
```

一个泛型里可以两种文件都有：写了份数的照份数跳，没写份数的每次都读（`is_closure/1700.viba`
就是那份兜底：`pattern A` 接任何实参，所以它读不出份数，也就不写）。

形参换成实参之后，实参里的名字按**调用方那份文件**算：`element_type_of[list[Local]]` 里
`A` 换成 `Local`，`Local` 是调用方文件里的名字（那份文件里 `Local = int` 的话，`A` 就是 `int`）。

方括号里写的是**这个文件的形参**时，替换的落点由调用方定：`apply.viba` 的成员里写着
`apply_impl[args.args]`，而 `args.args` 是这次调用收到的一个成员 —— 换成的是**调用方给的那份积**
（调用方写 `apply << add << ($a 1 * $b 2)`，积就是它给的）。数成员、认 tag 都在这份积上做，积里
带着调用方文件里的名字（`$n below` 里的 `below`）也按那份文件算。

## 5. 同一个应用，三层各读一次

- **判定层**（`is_sub_type`、描述符池）：`gen[T1, T2]` 展开成选中的那个文件
  （那个成员）在**它自己的模块**里的读法；形参名通过 `AstNodeType.env_get` 这条自由名字通道
  绑定到实参里的对应部分。所以 `element_type_of[dict[int, str]]` 是 `(int, str)`，
  `ret_type_of[int <- int]` 是 `int`。
- **计算层**（`viba.interpret`）：选中文件的这一次激活带着这些绑定，那个成员就在那个激活里
  求值 —— 写下来是 `true` 就给出 `true`，写下来是 `A` 就把萃取到的那份交回去。求值按平时
  的规矩走：`int` 这个名字在值的位置上仍然不是值，`(int, str)` 是元组数据。选中的文件写了
  `__impl__` 时，这一层读的就是**它那一次模块调用**（见第 2 节末）。
- **完整性**（`viba.is_complete`）：看完整性的那一遍按同样一次决断往下走；决断失败就是
  这个设计不完整。

**泛型不是模块。** 它没有定义，名字单独写出来不是一个类型（`X = g` 解不出来），它也不能当
函数调用（`g << $x 1` 报 `is a generic: it answers an application (g[T, ...]), not a call`）。
能写的只有应用：`g[T, ...]`。

## 6. 报错

| 写错的地方 | 报的错 |
|------------|--------|
| 目录里没有 `__generic__.viba` | 找不到模块（它就不是泛型） |
| 模式文件的文件名不是数字（也不是「份数_数字」） | `is named by its order, a number` |
| 名字上写的份数与 `pattern` 行读的份数不符 | `the name says the file reads N parts, and its \`pattern\` lines read M` |
| 名字上写了份数，`pattern` 行却读不出固定份数 | `the name says the file reads N parts, and a \`pattern\` line in it reads no count` |
| 模式文件写了 `__decl__` 却不是函数链 | `__decl__ is not a function type: …` |
| `__generic__.viba` 里写了 `pattern` | `one of the numbered files of its generic's directory` |
| 别处的文件写了 `pattern` | 同上 |
| 行数与实参个数不同 | `takes N parameters, not M` |
| 没有文件命中 | `no pattern of 'x' matches [...]: the decision failed` |

## 7. 自己写一份

1. 建目录 `demo/is_positive/`，放一个 `__generic__.viba`，里面写 `# __generic__.viba`。
2. 写 `100.viba`：

   ```viba
   pattern int | float

   value = true
   ```

3. 写 `200.viba`：

   ```viba
   pattern A

   value = false
   ```

4. 在能跑的文件里用：

   ```viba
   import demo.is_positive as is_positive

   type = Any <- $env Env

   __impl__ = $a is_positive[int] * $b is_positive[str]
   ```

`is_positive[int]` 命中 `100.viba`，给出 `true`；`is_positive[str]` 落到 `200.viba`，给出 `false`。

再看一份：数写下来的那个应用带几个实参。结构写在模式里，顺序从具体到宽松，数字也不要求连续：

```viba
# demo/num_generic_args/100.viba
pattern dict[A, B]

value = 2
```

```viba
# demo/num_generic_args/210.viba
pattern set[A]

value = 1
```

```viba
# demo/num_generic_args/300.viba
pattern A

value = 0
```

```viba
import demo.num_generic_args as num_generic_args

Args = num_generic_args[dict[int, str]]      # 2
One = num_generic_args[list[int]]            # 1
NoArgs = num_generic_args[bool]              # 0
```

再一份：数**自己收到几个实参**。行数就是个数，所以每个个数一个文件；一行不写就是零个。

```viba
# demo/num_variadic_args/50.viba
value = 0
```

```viba
# demo/num_variadic_args/200.viba
pattern A
pattern B

value = 2
```

```viba
import demo.num_variadic_args as num_variadic_args

None_ = num_variadic_args[]                  # 0
One = num_variadic_args[bool]                # 1
Two = num_variadic_args[bool, str]           # 2
```

## 8. 实现放在哪

| 位置 | 做什么 |
|------|--------|
| `viba/pattern.py` | 目录的读法、`structural_pattern_match` 结构匹配与萃取、`decide` 决断、`reduce_application` 给各层用 |
| `viba/type.py` | 名字解析遇到泛型时说明白它等的是应用 |
| `viba/is_sub_type.py` | 应用在判定里展开（`env_get` 绑定形参） |
| `viba/viba_type_descriptor.py` | 描述符池把 `名字.数字` 那几个文件合成一个泛型 |
| `viba/reflect.py` | 按类型读实例时，泛型应用先决断再展开 |
| `viba/interpret.py` | 目录当模块导入、应用求值（函数链的结果就是那次调用） |
| `viba/is_complete.py` | 完整性按决断往下走 |
| `viba/viba_ast/tagged.py` | `tagged`：符号写在字符串里的读法，写下来的字符串在这里折成 tag |
| `tests/test_pattern.py` + `tests/data/pattern/` | 本文的例子与全部报错 |

泛型的决断第一件事就是问这个目录里有哪些文件。

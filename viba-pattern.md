# 模式：一个泛型是一个目录，某个文件给出答案

泛型不是一份带形参的定义，而是一组**文件**：每一个文件说"实参长这样的时候，答案是这个"。
调用方把实参写在方括号里（`is_base_type[bool]`），**决断**是一个静态动作 ——
它只看写下来的那几个实参，不看程序跑到哪儿了。

```viba
import demo.is_base_type as is_base_type

Flag = is_base_type[bool]          # true
```

一个泛型由三件事定下来：

- 它放在一个目录里，**目录名就是泛型名**；
- 目录下必须有标记文件 `__generic__.viba`；
- 其余 `.viba` 文件每一个都是一个模式，**文件名是数字**，那个数字就是它的决断顺序。

```
demo/is_base_type/__generic__.viba      标记：这个目录是一个泛型
demo/is_base_type/100.viba              pattern bool | int | float | str
demo/is_base_type/200.viba              pattern A
```

`import demo.is_base_type as is_base_type` 找到的是那个**目录**：找模块的地方按顺序试
`demo/is_base_type.viba`，然后才是 `demo/is_base_type/__generic__.viba`（[`viba-interpreter.md`](viba-interpreter.md)
第 1 节）。目录里有标记文件，它就是一个泛型；没有，这个目录与 viba 无关。

## 1. 标记文件 `__generic__.viba`

标记文件只需要存在，并且能编译。它写了什么，决断不看；习惯上里面写一行注释：

```viba
# __generic__.viba
```

它不能写 `pattern`：`pattern` 只写在模式文件里，而模式文件的名字是数字
（第 2 节）。它不是模块，也不是模式文件 —— 它是一个目录的标记。

## 2. 一个模式文件：`pattern` 与 `__def__`

一个模式文件按顺序写若干行 `pattern`，**每个形参一行**：

```viba
# demo/is_base_type/100.viba
pattern bool | int | float | str

__def__ = true
```

一行 `pattern` 写一个模式，模式写的是**这个形参收什么**，两种写法：

- **写下来的类型**：实参必须能落进它，判据就是判定层自己的子类型（`is_sub_type`）。
  `bool | int | float | str` 收 `bool`（`true` 也收得下），不收 `list[int]`。
- **文件里没定义的名字**：那是一个**形参**，表示"这里的东西萃取出来"。`pattern A` 里的 `A`
  就是这个文件里没有的（自己没定义、不是内建名、也没有从它的 import 里来），于是实参在这个位置
  是什么，`A` 就是什么。

同名形参写两次是**相等**：两处都收得下彼此才算命中。`is_compatable` 的第三个文件就是这样：

```viba
# demo/is_compatable/200.viba
pattern A
pattern A

__def__ = true
```

`pattern` 可以带结构，形参就在结构里面：

```viba
# demo/element_type_of/100.viba
pattern list[A]

__def__ = A
```

```viba
# demo/ret_type_of/300.viba
pattern A <- B <- C

__def__ = A
```

结构对结构的匹配是按位置读的：积对积、和对和、指数链对指数链（结果在前、实参按书写顺序）、
元组对元组、`list[A]` 对 `list[实参]`（构造器要指同一个东西，参数个数要一样）、`$tag T` 对
`$tag 实参`。哪一段里没有形参，那一段就整体交给 `is_sub_type` 判：`A <- (() | nil)` 里的
`(() | nil)` 就是这么判的。

形参拿的是那一位**写下来的整块** —— tag 也是那一块的一部分。`pattern A <- $env Env <- B <- C`
对着 `int <- $env Env <- $a int <- $b int` 时，`B` 是 `$a int`、`C` 是 `$b int`；`__def__`
把它们写回哪里，tag 就跟着回到哪里。只想要 tag 里面那一层，就把 tag 写进模式：

| 模式那一位 | 收什么 | 形参拿到什么 |
|------------|--------|--------------|
| `B` | 任何 tag | 整块（`$a int`） |
| `$a B` | tag 必须是 `$a` | 里面那一层（`int`） |
| `__tagged__[n, B]` | 任何 tag | `n` 是符号（`"a"`），`B` 是里面那一层 |

**哪个形参带 tag、哪个不带，由模式自己说了算**：要原样传下去就写裸名字（`wrapper` 就是靠这个
把调用方的 tag 一路带过去的），要把 tag 认下来 —— 限定它、改名、或者拿它跟别处比 —— 就在模式里
写明。也正因为裸名字带着 tag，拿两个形参去拼同一个积时同一个 tag 撞上第二次会当场报错
（`the tag $a is written twice in one product`）：那本来就是写错的设计，要改名就用第三行那种写法。

`__def__` 是**这个文件答的类型**：决断选中谁，谁就被读成它的 `__def__`，形参名换成实参里对应的
那一部分。它可以是任何类型：

```viba
pattern list[A]
__def__ = A                       # 答萃取到的那一个类型
```

```viba
pattern A <- B <- C
__def__ = (A, B)                  # 答一个元组类型
```

```viba
pattern A
__def__ = (int <- $env Env <- int)   # 答一个函数类型（带环境的那个写法也照写）
```

函数类型照平时的两种读法：

- **在类型层读**：`$env Env` 那个参数是调用的规矩、不进类型，所以上面那条就是 `int <- int`；
- **在值那一层读**：这个应用代表的就是那次**调用**，跟"一个定义体是函数链"完全一样 ——
  `X = gen[T]` 是一份写下来的调用，可以继续给实参（`gen[T] << args.env << …`），也可以
  当闭包递出去。宿主按**泛型的名字**找这一步的实现（写下来是 `wrap[...]`，它拿到的是
  `"wrap"`），跟按定义名找实现同一种做法。

这一点让"拿一个函数换一个调用"的泛型写得出来。模式读的是**写下来的链条**，所以函数自己的
`$env Env` 那一位也写在模式里；`__def__` 按同一条链条答回去，只是在函数本身前面多要一份
那份函数：

```viba
# demo/wrapper/100.viba
pattern A <- $env Env <- B

__def__ = A <- $env Env <- (A <- $env Env <- B) <- B
```

```viba
# 调用方：add 是两元的，inc 是一元的
add =
    int
  <- $env Env
  <- $a int
  <- $b int
  <- { add the two }

__ret__ = wrapper[add] << args.env << add << $a 1 << $b 2     # 3
```

给环境是执行，给 `add` 的是函数本身，`$a 1`、`$b 2` 是它的实参（按它们落的那一位的 tag 写）。
`(A <- $env Env <- B)` 那一位写着函数类型，所以交给宿主的是**它代表的那次调用**：宿主带着
环境叫它，还可以带上自己的实参（`f(env, 1, 2)`），那些实参落进这次调用还欠的槽位 ——
`wrapper` 的实现就是 `f(env, *rest)`，把收到的实参原样转给那个函数。

方括号里的实参本身也可以是一次 `<<`。`add << $a 2` 是"已经给过 `$a 2` 的那次调用"，所以它
**作为一个类型读出来就是剩下的那条链条**（`int <- $env Env <- $b int`），决断照这条链条读开。
上面那份泛型的 100.viba 收一元函数、200.viba 收二元函数，所以一个已经给过一部分实参的函数
落在一元那一支上：

```viba
add2 = add << $a 2

__ret__ = wrapper[add2] << args.env << add2 << $b 1     # 3
```

给实参要写出它落在哪个 tag 上（`$a 2`）：`$env Env` 是调用的规矩，不是实参的位置，不写 tag 的
实参先去撞这个参数，报 `2 does not fit $env Env`。

一个文件里也可以有自己的定义，`__def__` 读到自己定义的名字就在这个文件里读：

```viba
# demo/wrapped_item/100.viba
pattern list[A]

Wrapped = $item A

__def__ = Wrapped                 # 结果是 $item 实参里的元素
```

`__ret__` 决断不看：那是**运行时**的事。模式文件同时也可以是一个能跑的模块
（`__def__` 带 `$env Env`、`__ret__` 写自己要跑的东西），那与它当模式被选中没有关系。

## 3. 符号写在字符串里：`__tagged__`

tag 是地址（`$a`），设计里写的就是它，所以一个只作为**字符串**存在的名字没有地方放。`__tagged__`
就是那个地方 —— 它是内建的，收一个符号，或者一个符号加一个类型：

```viba
Tagged = __tagged__["a", T]                   # 就是 Tagged = $a T

__ret__ = __tagged__["hello"] << persion      # 就是 $hello << persion
```

符号写在字符串里（`"a"`，或者带 `$` 的 `"$a"`），而且必须是一个合法符号：字母、数字、`_`，不以数字
开头。写错了当场报错（`'a b' is no symbol for a tag`），参数个数不是 1 或 2 也当场报错。
`__tagged__` 是内建名，谁也不能拿它定义（`__tagged__ = int` 编不过）。

一参数的写法是一个**成员**，跟 `$hello` 一样只写在链头：

```viba
__ret__ = __tagged__["hello"] << persion        # 就是 $hello << persion
```

两参数的写法是一个类型，各层读到的就是那个 tag。

**符号的位置上可以是一个形参。** `pattern` 行里那样写时，形参拿到的是实参那个 tag 的符号
（不带 `$` 的字符串）：

```viba
# demo/arg_name_of/100.viba
pattern __tagged__[arg_name, T]

__def__ = arg_name                 # 答 "a"
```

于是 `arg_name_of[$a int]` 答字符串 `"a"`：**一个 tag 的名字可以当值用**。反过来，决断把这个字符串
交给 `__def__`，`__def__` 就用它把同一个 tag 建回来：

```viba
# demo/tagged_again/100.viba
pattern __tagged__[arg_name, T]

__def__ = __tagged__[arg_name, T]          # 又建回 `$a int`
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

**决断失败是程序错误**，不是 `never`，也不是 `false`：一个没有答案的泛型应用是写错的设计。

实参写在哪一层，萃取到的类型就归哪一层：`element_type_of[list[Local]]` 里 `A` 是 `Local`，
它在**写这个实参的模块**里解析（`Local = int` 的话它就是 `int`）。

## 5. 同一个应用，三层各读一次

- **判定层**（`is_sub_type`、描述符池）：`gen[T1, T2]` 展开成选中的那个文件
  （`__def__`）在**它自己的模块**里的读法；形参名通过 `AstNodeType.env_get` 这条自由名字通道
  绑定到实参里的对应部分。所以 `element_type_of[dict[int, str]]` 是 `(int, str)`，
  `ret_type_of[int <- int]` 是 `int`。
- **计算层**（`viba.interpret`）：选中文件的这一次激活带着这些绑定，`__def__` 就在那个激活里
  求值 —— 写下来是 `true` 就答 `true`，写下来是 `A` 就把萃取到的那份交回去。求值按平时
  的规矩走：`int` 这个名字在值的位置上仍然不是值，`(int, str)` 是元组数据。
- **完整性**（`viba.is_complete`）：看完整性的那一遍按同样一次决断往下走；决断失败就是
  这个设计不完整。

**泛型不是模块。** 它没有定义，名字单独写出来不是一个类型（`X = g` 解不出来），它也不能当
函数调用（`g << $x 1` 报 `is a generic: it answers an application (g[T, ...]), not a call`）。
能写的只有应用：`g[T, ...]`。

## 6. 报错

| 写错的地方 | 报的错 |
|------------|--------|
| 目录里没有 `__generic__.viba` | 找不到模块（它就不是泛型） |
| 模式文件的文件名不是数字 | `is named by its order, a number` |
| 模式文件没有 `__def__` | `a pattern file writes __def__, the type it answers` |
| `__generic__.viba` 里写了 `pattern` | `one of the numbered files of its generic's directory` |
| 别处的文件写了 `pattern` | 同上 |
| 行数与实参个数不同 | `takes N parameters, not M` |
| 没有文件命中 | `no pattern of 'x' matches [...]: the decision failed` |

## 7. 自己写一份

1. 建目录 `demo/is_positive/`，放一个 `__generic__.viba`，里面写 `# __generic__.viba`。
2. 写 `100.viba`：

   ```viba
   pattern int | float

   __def__ = true
   ```

3. 写 `200.viba`：

   ```viba
   pattern A

   __def__ = false
   ```

4. 在能跑的文件里用：

   ```viba
   import demo.is_positive as is_positive

   __def__ = Any <- $env Env

   __ret__ = $a is_positive[int] * $b is_positive[str]
   ```

`is_positive[int]` 命中 `100.viba`，答 `true`；`is_positive[str]` 落到 `200.viba`，答 `false`。

再看一份：数写下来的那个应用带几个实参。结构写在模式里，顺序从具体到宽松，数字也不要求连续：

```viba
# demo/num_generic_args/100.viba
pattern dict[A, B]

__def__ = 2
```

```viba
# demo/num_generic_args/210.viba
pattern set[A]

__def__ = 1
```

```viba
# demo/num_generic_args/300.viba
pattern A

__def__ = 0
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
__def__ = 0
```

```viba
# demo/num_variadic_args/200.viba
pattern A
pattern B

__def__ = 2
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
| `viba/interpret.py` | 目录当模块导入、应用求值（函数链的答案就是那次调用）、`list_files` 这个取目录内容的钩子 |
| `viba/is_complete.py` | 完整性按决断往下走 |
| `viba/viba_ast/tagged.py` | `__tagged__`：符号写在字符串里的读法，写下来的字符串在这里折成 tag |
| `tests/test_pattern.py` + `tests/data/pattern/` | 本文的例子与全部报错 |

宿主要自己供文件时（`interpret(..., get_file=...)`），目录里有什么也要一起供
（`list_files`）：决断的第一件事就是问这个目录里有哪些文件。

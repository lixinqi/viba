# 特化：一个泛型是一个目录，某个文件给出答案

泛型不是一份带形参的定义，而是一组**文件**：每一个文件说"实参长这样的时候，答案是这个"。
调用方把类型写在方括号里（`is_base_type[bool]`），特化的**决断**是一个静态动作 ——
它只看写下来的那几个类型，不看程序跑到哪儿了。

```viba
import demo.is_base_type as is_base_type

Flag = is_base_type[bool]          # true
```

一个泛型由三件事定下来：

- 它放在一个目录里，**目录名就是泛型名**；
- 目录下必须有标记文件 `__generic__.viba`；
- 其余 `.viba` 文件的**文件名是数字**，那个数字就是它的决断顺序。

```
demo/is_base_type/__generic__.viba      标记：这个目录是一个泛型
demo/is_base_type/100.viba              specialize bool | int | float | str
demo/is_base_type/200.viba              specialize A
```

`import demo.is_base_type as is_base_type` 找到的是那个**目录**：找模块的地方按顺序试
`demo/is_base_type.viba`，然后才是 `demo/is_base_type/__generic__.viba`（[`viba-interpreter.md`](viba-interpreter.md)
第 1 节）。目录里有标记文件，它就是一个泛型；没有，这个目录与 viba 无关。

## 1. 标记文件 `__generic__.viba`

标记文件只需要存在，并且能编译。它写了什么，决断不看；习惯上里面写一行注释：

```viba
# __generic__.viba
```

它不能写 `specialize`：`specialize` 只写在特化文件里，而特化文件的名字是数字
（第 2 节）。它不是模块，也不是特化 —— 它是一个目录的标记。

## 2. 一个特化文件：`specialize` 与 `__def__`

一个特化文件按顺序写若干行 `specialize`，**每个形参一行**：

```viba
# demo/is_base_type/100.viba
specialize bool | int | float | str

__def__ = true
```

`specialize` 后面写的是**这个形参收什么**，两种写法：

- **写下来的类型**：实参必须能落进它，判据就是判定层自己的子类型（`is_sub_type`）。
  `bool | int | float | str` 收 `bool`（`true` 也收得下），不收 `list[int]`。
- **文件里没定义的名字**：那是一个**形参**，表示"这里的东西萃取出来"。`specialize A` 里的 `A`
  就是这个文件里没有的（自己没定义、不是内建名、也没有从它的 import 里来），于是实参在这个位置
  是什么，`A` 就是什么。

同名形参写两次是**相等**：两处都收得下彼此才算命中。`is_compatable` 的第三个文件就是这样：

```viba
# demo/is_compatable/200.viba
specialize A
specialize A

__def__ = true
```

`specialize` 可以带结构，形参就在结构里面：

```viba
# demo/element_type_of/100.viba
specialize list[A]

__def__ = A
```

```viba
# demo/ret_type_of/300.viba
specialize A <- B <- C

__def__ = A
```

结构对结构的匹配是按位置读的：积对积、和对和、指数链对指数链（结果在前、实参按书写顺序）、
元组对元组、`list[A]` 对 `list[实参]`（构造器要指同一个东西，参数个数要一样）、`$tag T` 对
`$tag 实参`。哪一段里没有形参，那一段就整体交给 `is_sub_type` 判：`A <- (() | nil)` 里的
`(() | nil)` 就是这么判的。

`__def__` 是**这个文件答的类型**：决断选中谁，谁就被读成它的 `__def__`，形参名换成实参里对应的
那一部分。它可以是任何类型：

```viba
specialize list[A]
__def__ = A                       # 答萃取到的那一个类型
```

```viba
specialize A <- B <- C
__def__ = (A, B)                  # 答一个元组类型
```

```viba
specialize A
__def__ = (int <- $env Env <- int)   # 答一个函数类型（带环境的那个写法也照写）
```

函数类型照平时的两种读法：在类型层读，`$env Env` 那个参数是调用的规矩、不进类型，所以它就是
`int <- int`；在值的位置上，函数类型与别处一样不是值（值那一层读的是写在那儿的类型本身）。

一个文件里也可以有自己的定义，`__def__` 读到自己定义的名字就在这个文件里读：

```viba
# demo/wrapper/100.viba
specialize list[A]

Wrapped = $item A

__def__ = Wrapped                 # 结果是 $item 实参里的元素
```

`__ret__` 决断不看：那是**运行时**的事。特化文件同时也可以是一个能跑的模块
（`__def__` 带 `$env Env`、`__ret__` 写自己要跑的东西），那与它当特化被选中没有关系。

## 3. 决断：数字从小到大，第一个命中的赢

文件的数字就是顺序，**不要求连续**：`100.viba`、`150.viba`、`300.viba` 照数字排。
命中一个就停，后面的不再看。

- 一个文件的 `specialize` 行数与这次应用写的实参个数不同，它不可能是这一支，跳过。所以
  **「这个泛型收几个实参」也是每个文件自己的事**：几行 `specialize` 就收几个，不同个数各写一个
  文件（`num_variadic_args` 里就是 0 / 1 / 2 / 3 各一个），一行不写就是收零个；一个模式是一个
  类型，模式里没有 `...` 这种写法；
- 所有文件的行数都与实参个数不同，直接报错（`generic 'x' takes 2 parameters, not 1`）；
- 没有任何文件命中，直接报错（`no specialization of 'x' matches [bool]: the decision failed`）。

**决断失败是程序错误**，不是 `never`，也不是 `false`：一个没有答案的泛型应用是写错的设计。

实参写在哪一层，萃取到的类型就归哪一层：`element_type_of[list[Local]]` 里 `A` 是 `Local`，
它在**写这个实参的模块**里解析（`Local = int` 的话它就是 `int`）。

## 4. 同一个应用，三层各读一次

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

## 5. 报错

| 写错的地方 | 报的错 |
|------------|--------|
| 目录里没有 `__generic__.viba` | 找不到模块（它就不是泛型） |
| 特化文件的文件名不是数字 | `is named by its order, a number` |
| 特化文件没有 `__def__` | `a specialization writes __def__, the type it answers` |
| `__generic__.viba` 里写了 `specialize` | `one of the numbered files of its generic's directory` |
| 别处的文件写了 `specialize` | 同上 |
| 行数与实参个数不同 | `takes N parameters, not M` |
| 没有文件命中 | `no specialization of 'x' matches [...]: the decision failed` |

## 6. 自己写一份

1. 建目录 `demo/is_positive/`，放一个 `__generic__.viba`，里面写 `# __generic__.viba`。
2. 写 `100.viba`：

   ```viba
   specialize int | float

   __def__ = true
   ```

3. 写 `200.viba`：

   ```viba
   specialize A

   __def__ = false
   ```

4. 在能跑的文件里用：

   ```viba
   import demo.is_positive as is_positive

   __def__ = Any <- $env Env

   __ret__ = $a is_positive[int] * $b is_positive[str]
   ```

`is_positive[int]` 命中 `100.viba`，答 `true`；`is_positive[str]` 落到 `200.viba`，答 `false`。

再看一份：数一个类型带几个类型实参。结构写在模式里，顺序从具体到宽松，数字也不要求连续：

```viba
# demo/num_generic_args/100.viba
specialize dict[A, B]

__def__ = 2
```

```viba
# demo/num_generic_args/210.viba
specialize set[A]

__def__ = 1
```

```viba
# demo/num_generic_args/300.viba
specialize A

__def__ = 0
```

```viba
import demo.num_generic_args as num_generic_args

Args = num_generic_args[dict[int, str]]      # 2
One = num_generic_args[list[int]]            # 1
NoArgs = num_generic_args[bool]              # 0
```

再一份：数**自己收到几个模板实参**。行数就是个数，所以每个个数一个文件；一行不写就是零个。

```viba
# demo/num_variadic_args/50.viba
__def__ = 0
```

```viba
# demo/num_variadic_args/200.viba
specialize A
specialize B

__def__ = 2
```

```viba
import demo.num_variadic_args as num_variadic_args

None_ = num_variadic_args[]                  # 0
One = num_variadic_args[bool]                # 1
Two = num_variadic_args[bool, str]           # 2
```

## 7. 实现放在哪

| 位置 | 做什么 |
|------|--------|
| `viba/specialize.py` | 目录的读法、`structural_pattern_match` 结构匹配与萃取、`decide` 决断、`reduce_application` 给各层用 |
| `viba/type.py` | 名字解析遇到泛型时说明白它等的是应用 |
| `viba/is_sub_type.py` | 应用在判定里展开（`env_get` 绑定形参） |
| `viba/viba_type_descriptor.py` | 描述符池把 `名字.数字` 那几个文件合成一个泛型 |
| `viba/reflect.py` | 按类型读实例时，泛型应用先决断再展开 |
| `viba/interpret.py` | 目录当模块导入、应用求值、`list_files` 这个取目录内容的钩子 |
| `viba/is_complete.py` | 完整性按决断往下走 |
| `tests/test_specialize.py` + `tests/data/specialize/` | 本文的例子与全部报错 |

宿主要自己供文件时（`interpret(..., get_file=...)`），目录里有什么也要一起供
（`list_files`）：决断的第一件事就是问这个目录里有哪些文件。

# Viba 内建 API

这份文档列出 viba 程序里不用 `import` 就能用的每一个名字：每个算子怎么调、答案是什么类型、
`Environment` 的成员、按位置或按键取元素的那几个成员、内建的容器、按名字取得的内建模块，
以及一次运行停下来时给出什么。每一节只讲这一节的名字，别处要用的规矩在那一节点名。

签名本身在 [`viba/builtin.viba`](viba/builtin.viba)；一次调用怎么跑、环境是什么、结果是什么，
在 [`viba-interpreter.md`](viba-interpreter.md)；容器的源码形式与用法在
[`viba-style.md`](viba-style.md) 第 8 节。算子的含义不在本层：本层给签名与实参，
实现由 `get_func` 那一方给出（[`viba-interpreter.md`](viba-interpreter.md)，
「宿主侧：Environment」）。

## 1. 这些名字从哪来

内建的名字有三层。

| 层 | 名字 | 在哪 |
|---|---|---|
| 算子与环境的类型 | `builtin` 这个概念的 69 个成员，`Environment`，以及同一个类型的另一个名字 `Env` | `viba/builtin.viba` |
| 解释器自己实现 | `$__getattr__`、`$__getitem__`、`$__in__`、`$__len__`、`$__keys__`、`__dyn_call__`、`__dyn_method__`、`__get_args__` | `viba/interpret.py` |
| 内建模块与泛型 | `apply`、`if`、`sequential`、`sub_env_run`、`Y`、`y_helper`、`resumable`、`interpret_result`、`type`、`viba_type_descriptor`、`is_closure`、`unclosure` | `viba/` 与 `viba/builtin/` |

同一个成员有两种源码形式：`add` 与 `builtin.add` 是同一次调用，宿主两次拿到的名字都是
`builtin.add`。谁赢：模块自己给的定义先，然后是 `import` 进来的名字，内建是最后一站 ——
一个模块自己给出 `add`，裸名 `add` 就是它自己的那个；`builtin.add` 仍然拿到内建那个。

算子的签名都按同一种排法给出：**结果在前，然后 `$env Env`，再是其余参数**。

```viba
add : int <- $env Env <- $x int <- $y int
```

调用时一个 `<<` 给一个参数，按 tag 给（顺序随意）或按位置给。环境就是 `$env Env` 那个参数：
环境按值认（给一个 `Environment`），按 `$env` 这个 tag 给也算；不带 tag 的实参跳过环境，
依次落到其余参数上。给它就是执行这次调用，不给就还是闭包。

```viba
add << $env args.env << $x 3 << $y 4     # 跑了，答案 7
add << args.env << $x 3 << $y 4          # 同样跑了：第一个实参就是环境
add << $x 3 << $y 4                      # 没给环境：这条链是一个闭包
```

两条要记住的规矩：

- **接实参时不核类型。** 签名里的类型是给判定层看的；把 `"two"` 给 `$y int` 不会在调用处被拦，
  那个值交到实现手里，实现拿它做整数运算时在那里撞上，报的是 `$underlying_viba_op_err`，
  不是程序错（[`viba-interpreter.md`](viba-interpreter.md)，「模块的参数：`__decl__`」）。
- **环境给了，参数就得齐。** 少一个、多给、同一个 tag 给两次、给一个签名里没有的 tag，
  都在调用处报 `$viba_program_err`，话里点名是哪一个参数。

## 2. 算子

下面每张表都省去了 `$env Env` 这个参数：每个算子都先收它。实参一列给出 tag 与类型，
「答案」一列给出结果的类型。

### 2.1 `int` 的算术

| 名字 | 实参 | 答案 | 含义 |
|---|---|---|---|
| `add` | `$x int` `$y int` | `int` | x 与 y 的和 |
| `sub` | `$x int` `$y int` | `int` | x 减 y |
| `mul` | `$x int` `$y int` | `int` | x 与 y 的积 |
| `div` | `$x int` `$y int` | `int` | x 除以 y，向下取整 |
| `rem` | `$x int` `$y int` | `int` | x 除以 y 的余数 |
| `neg` | `$x int` | `int` | x 换成相反的符号 |
| `abs` | `$x int` | `int` | x 去掉符号 |
| `min` | `$x int` `$y int` | `int` | 两者中小的那个 |
| `max` | `$x int` `$y int` | `int` | 两者中大的那个 |
| `pow` | `$x int` `$y int` | `int` | x 的 y 次幂 |

### 2.2 `float` 的算术

名字都在 `int` 那套后面加 `_f`，实参与答案的类型换成 `float`，含义一一对应。

| 名字 | 实参 | 答案 | 含义 |
|---|---|---|---|
| `add_f` | `$x float` `$y float` | `float` | 和 |
| `sub_f` | `$x float` `$y float` | `float` | 差 |
| `mul_f` | `$x float` `$y float` | `float` | 积 |
| `div_f` | `$x float` `$y float` | `float` | 商 |
| `rem_f` | `$x float` `$y float` | `float` | 余数 |
| `neg_f` | `$x float` | `float` | 相反符号 |
| `abs_f` | `$x float` | `float` | 去掉符号 |
| `min_f` | `$x float` `$y float` | `float` | 小的那个 |
| `max_f` | `$x float` `$y float` | `float` | 大的那个 |
| `pow_f` | `$x float` `$y float` | `float` | x 的 y 次幂 |

### 2.3 比较

三个类型各有一套，答案都是 `bool`。`eq` / `ne` 是相等与不等，其余四个按名字的意思对照：
`lt` 小于、`le` 小于或等于、`gt` 大于、`ge` 大于或等于。

| 类型 | 名字 | 实参 | 答案 |
|---|---|---|---|
| `int` | `lt` `le` `gt` `ge` `eq` `ne` | `$x int` `$y int` | `bool` |
| `float` | `lt_f` `le_f` `gt_f` `ge_f` `eq_f` `ne_f` | `$x float` `$y float` | `bool` |
| `str` | `lt_str` `le_str` `gt_str` `ge_str` `eq_str` `ne_str` | `$x str` `$y str` | `bool` |
| `bool` | `eq_bool` `ne_bool` | `$x bool` `$y bool` | `bool` |

`str` 的四个序关系按排序先后比。`eq_f` 是精确相等，没有容差。

### 2.4 `str`

| 名字 | 实参 | 答案 | 含义 |
|---|---|---|---|
| `concat` | `$x str` `$y str` | `str` | 两段首尾接起来 |
| `len_str` | `$x str` | `int` | 这段字符串有多少个字符 |
| `upper_str` | `$x str` | `str` | 全大写 |
| `lower_str` | `$x str` | `str` | 全小写 |
| `substr` | `$x str` `$start int` `$end int` | `str` | 从第 `$start` 个字符到第 `$end` 个字符，含头不含尾 |
| `char_at` | `$x str` `$at int` | `str` | 第 `$at` 个位置上的那一个字符 |
| `find` | `$x str` `$needle str` | `int \| nil` | `$needle` 第一次出现的位置；没有就是 `nil` |
| `contains` | `$x str` `$needle str` | `bool` | `$needle` 在不在这段里 |
| `starts_with` | `$x str` `$prefix str` | `bool` | 以 `$prefix` 开头 |
| `ends_with` | `$x str` `$suffix str` | `bool` | 以 `$suffix` 结尾 |
| `split` | `$x str` `$separator str` | `list[str]` | 按 `$separator` 切成的几段 |
| `join` | `$parts list[str]` `$separator str` | `str` | 几段用 `$separator` 接起来 |
| `replace` | `$x str` `$old str` `$new str` | `str` | 每一处 `$old` 换成 `$new` |
| `trim` | `$x str` | `str` | 去掉首尾的空白 |
| `repeat` | `$x str` `$times int` | `str` | `$times` 个 `$x` 接起来 |
| `eq_str` `ne_str` | `$x str` `$y str` | `bool` | 两段是否相同 |
| `lt_str` `le_str` `gt_str` `ge_str` | `$x str` `$y str` | `bool` | 排序先后 |

几处要记住的：

- **切片含头不含尾**：`substr << $env args.env << $x "viba" << $start 1 << $end 3` 是 `"ib"`。
  `$start` 与 `$end` 都要求在字符串里（`0` 到 `len_str` 之间，而且 `$start` 不大于 `$end`），
  `char_at` 的位置也要存在；越界不是给一段截短的，是这一步停下。
- **五种给出的东西没有答案**：`substr` 与 `char_at` 越界、`split` 的分隔符是空字符串、
  `replace` 的 `$old` 是空字符串、`repeat` 的次数是负的。这些都由实现停下，不是程序错 ——
  这些算子的含义在实现那一侧，本层只给签名（第 1 节）。实现抛出时，停法是
  `$underlying_viba_op_err`（第 9 节）。
- **找不到是 `nil`，不是 -1，也不是停**：`find` 的答案类型是 `int | nil`。
- **`$__in__` 不认 `str`**（第 4.3 节）：问一段字符串在不在用 `contains`。
- **`split` 的答案是一个容器**：`parts = split << …` 之后按位置取段（`$__getitem__ << parts << 1`
  是第二段），也能直接交给 `join`。宿主给出这个答案时，交的是一个 `VObject` 或者一块
  `ListLiteral` 的语法树，不是一个 Python 列表（宿主能答的只有 `VObject`、语法树、标量、`None`）。

### 2.5 `bool`

| 名字 | 实参 | 答案 | 含义 |
|---|---|---|---|
| `and` | `$x bool` `$y bool` | `bool` | 两个都真才真 |
| `or` | `$x bool` `$y bool` | `bool` | 有一个真就真 |
| `not` | `$x bool` | `bool` | 另一个真值 |
| `xor` | `$x bool` `$y bool` | `bool` | 恰好一个真 |
| `eq_bool` `ne_bool` | `$x bool` `$y bool` | `bool` | 两个真值是否相同 |

`and` 与 `or` 都是算子，两个实参都会算出来再交给实现 —— 这里没有短路。

### 2.6 换类型

`x_to_y` 是从类型 x 换到类型 y。

| 名字 | 实参 | 答案 | 含义 |
|---|---|---|---|
| `int_to_float` | `$x int` | `float` | x 当成 float |
| `float_to_int` | `$x float` | `int` | 去掉小数部分 |
| `int_to_str` | `$x int` | `str` | x 用十进制表示出来的字符串 |
| `float_to_str` | `$x float` | `str` | x 用数字表示出来的字符串 |
| `bool_to_str` | `$x bool` | `str` | `"true"` 或 `"false"` |
| `str_to_int` | `$x str` | `int` | 这段字符串里的那个整数 |
| `str_to_float` | `$x str` | `float` | 这段字符串里的那个数 |

`str_to_int` 与 `str_to_float` 拿到一个不是数的字符串时，
是实现在那里停下，报 `$underlying_viba_op_err`。

### 2.7 把值交回去，与 `if` 用的两个开关

| 名字 | 签名 | 含义 |
|---|---|---|
| `echo` | `Any <- $env Env <- $x Any` | 原样把 `$x` 交回去 |
| `echo_or_never` | `Any <- $env Env <- $cond bool <- $get_v (Any <- $env Env)` | 条件真就给那一支的值，否则给 `never` |
| `never_or_echo` | `Any <- $env Env <- $cond bool <- $get_v (Any <- $env Env)` | 条件假就给那一支的值，否则给 `never` |

`echo` 的用处是把一个已经算好的值变成"欠着环境的一次调用"：`echo << $x V` 对任何环境都答 V，
所以它能放在函数类型的参数位置上、能当 `sequential` 的最后一步、能当 `if` 的一支。

`echo_or_never` 与 `never_or_echo` 的 `$get_v` 是**函数型的参数**：源码里那个实参不在原地算，
交给实现，由实现决定算不算、在哪个环境下算。`if` 就是这两个开关的和
（[`viba/if.viba`](viba/if.viba)，第 6.3 节）。

## 3. `Environment`

`Environment` 是环境这个类型，`Env` 是同一个类型的另一个名字。它是内建里唯一能把环境声明成结果的东西：
模块自己的 `__decl__`、模块里定义的函数、积成员里的链，结果那个位置给的是 `Env` 或 `Environment`
都当场报错（[`viba-interpreter.md`](viba-interpreter.md)，「环境不能当结果」）。

成员有两种调用形式，同一次调用：挂在环境值上（`args.env.sub_env << args.env << "child"`），
或者把 tag 放在链头（`$sub_env << args.env << "child"`）。下面一列「实参」里的 `$env Env`
就是被取成员的那个环境本身；它同时也是交给成员的第一个实参。

| 成员 | 答案 | 实参 | 含义 |
|---|---|---|---|
| `$viba_path` | `str` | 无 | 这个环境查 `import` 时用的路径；这是一个值，不是函数 |
| `$sub_env` | `Env` | `$env Env` `$sub_env_name str` | 这个环境下叫这个名字的子环境 |
| `$tmp_env` | `Env` | `$env Env` | 名字没人挑过的子环境，每次调用都是新的，任意两次调用不共用一条数据路径 |
| `$get_parent` | `Env \| nil` | `$env Env` | 造出这个环境的那个环境；链的根给 `nil` |
| `$get_root` | `Env \| nil` | `$current Env \| nil` | 这条链最上面的环境；`nil` 进 `nil` 出 |
| `$get_relative_path` | `str` | `$current Env` `$root Env \| nil` | `$current` 的数据路径从 `$root` 看是什么；根用 `nil` 表示这条链自己的根 |
| `$find_by_relative_path` | `Env` | `$relative_path str` `$root Env \| nil` | 上面那件事反过来：从 `$root` 沿这条路径走到的那个环境；`nil` 表示从自己来的那个环境起 |
| `$compress_env_path` | `Env` | `$sup Env` `$sub Env` | 把 `$sub` 的数据路径压成一个名字，挂在 `$sup` 旁边 |
| `$uncompress_relative_path` | `str \| nil` | 无 | 被压过的那条路径；别的环境上是 `nil`。这是一个值，不是函数 |
| `$try_compact` | `Env` | `$env Env` `$env_chain_length_limit int` | 链太长时把现在跑着的调用变成工作清单；答案就是给它的那个环境 |

### 3.1 `$sub_env` 的名字

名字是字符串，也可以是**源码里那次调用由哪几段组成**的一份积，各段用 `/` 接起来，
这样一条数据路径能对着代码看回去：

```viba
("step0" * step0_name * Step0)      # 子环境叫 `step0/a/add`
```

三段的含义是：这次调用站在哪一位、它带哪个 tag、它的链头是哪个函数。每一段是什么，就占什么：

| 那一段是什么 | 占什么 |
|---|---|
| 字符串 | 它自己 |
| 一个名字（tag） | 那个名字，不带 `$` |
| 一次调用 | 它的链头所指的那个函数的名字 |
| 别的（叶子、`$var "x"` 这样的变量引用、`void`） | 什么都不占，这一段跳过 |

整个名字就是一个叶子时，那个叶子就是名字。几段都占不到东西（`1 * 2`）时当场报错，
话的开头是 `a storage name is a string or pieces of the source that name something, and this names nothing`，
冒号后面是那一段源码。

解释器自己造的子环境用的是另一套名字：模块在源码里的名字，或者它选中的那个模式文件的整名
（`sequential_step.300`）—— 父环境的数据路径已经以这个名字结尾时只用文件号那一段，不再重复一遍。

### 3.2 数据路径与环境的链

两件事要分开：

- **数据路径**（英文 storage path）是一次调用跑在哪条路径上，成员的名字就是它。压缩
  （`$compress_env_path`）、按路径找回来（`$find_by_relative_path`）都只跟它有关。
- **环境的链**是 `$get_parent` 走的那条链，每给一次调用一个自己的环境就多一环。压缩不改这条链：
  被压出来的环境站的位置不变，它的 `$get_parent` 还是它替身的那个环境。

所以 `$find_by_relative_path` 的答案只有**一个**环境，不是一串：路径的每一段只说明它站在哪，
不是一次调用。三十段的一条路径给一个环境，而 `$get_relative_path` 照样报得出那三十段。

### 3.3 `$compress_env_path`

给它的 `$sup` 的数据路径必须是 `$sub` 的数据路径的前缀，而且在名字的边界上：`$sub` 在 `$sup` 下面
（`<sup>/…`），或者它本来就是从 `$sup` 压出来的一条路径（`<sup>_…`）。压法是把 `$sub` 的整条
数据路径做一次 SHA-1，接在 `$sup` 自己那条路径后面当成一个名字：`<sup 的路径>_<40 位十六进制>`。
被压掉的那条路径放进答案的 `$uncompress_relative_path`，工具靠它取回原来的路径；别的环境上这个
成员是 `nil`。递归按这个办法压，数据路径的长度就不随递归深度增长。

### 3.4 `$try_compact`

给它的环境必须是现在跑着的一次调用之一。`$get_parent` 那条链有几环，超过 `$env_chain_length_limit`
给的限度时，现在跑着的这些调用不再挂在这条链上继续做，而是变成一份工作清单，
由这次运行自己的调用一次一个地做回来，解释器那侧的栈就不再随递归变深。答案就是给它的那个环境，
所以用法是：先起一个名字，再用那个名字：

```viba
env = $try_compact << args.env << 32
```

限度是一个整数（`true` 不算整数）。给的如果是一个解释器并不在内的环境 —— 某次查表的答案、
更早的调用收到过的环境、没有调用跑在上面的子环境 —— 那它说的链根本不在系统栈上，
这时压不动任何东西，环境原样交回。宿主在一次运行之外调它则是环境的 api 拒绝：报
`$environment_api_invalid_argument_err`。

## 4. 解释器自己实现的五个算子

这五个是解释器自己实现的，不算 `builtin` 的成员，也不用 `import`：前三个从一个值里取东西（按名字取成员、按位置或按键取元素、问一份在不在），后两个量一个容器（数一数有多少份、取字典的键）。

### 4.1 `$__getattr__`

按名字取一个成员，名字当数据给出。第一个实参是被取成员的那份值，第二个是名字（一个字符串），
后面跟着的实参交给取出来的成员。

```viba
$__getattr__ << box << "f" << args.env << $x 1
# 就是 box.f << box << args.env << $x 1
```

取出来的是一次调用时，它当成员被调用：**被取的那个值本身排在实参最前面**，所以上面那条的
`box` 落进 `f` 的第一个参数。名字后面不给实参时，取出来的就是那个成员本身（一个值，
或者一个还欠着实参的调用），可以直接往下传。

名字不是字符串、或者不是一个合法符号（字母数字下划线、不以数字开头）时，报 `$viba_program_err`；
那一份值上没有这个成员，同样报程序错。

### 4.2 `$__getitem__`

按位置或按键取一个元素。第一个实参是容器，第二个是位置或键。

```viba
$__getitem__ << xs << 1          # 列表、集合、元组的第 1 个元素
$__getitem__ << table << "k"     # 字典里键 "k" 的值
```

位置给 `int`（`true` 不算 `int`），键给 `str`，两种各对应一条取元素的步子。答案就是那个元素，
所以后面可以接着给实参（`$__getitem__ << xs << 0 << $x 1` 是 `xs[0] << $x 1`）。
`xs[1]` 与 `table["k"]` 是它的简式；方括号只跟在名字后面，一层，`xs[0][1]` 要落成
`$__getitem__ << xs[0] << 1`。

拒绝的情形都报 `$viba_program_err`：没有那个位置（越界）、没有那个键、给字典一个位置、
给列表或元组一个键、给的位置或键不是 `int` / `str`、根本没给位置或键、被取的那份值不是容器。
越界与缺键的话都是 `no such address`。

### 4.3 `$__in__`

问一个东西在不在容器里，答案 `bool`。第一个实参是容器，第二个是要找的那个东西。

```viba
$__in__ << xs << 3               # 3 在不在这个列表 / 集合 / 元组里
$__in__ << table << "k"          # 键 "k" 在不在这个字典里
```

列表、集合、元组按元素本身比；字典只比键，不比值，而且给的必须是一个字符串键。
被问的不是这三种容器（也没有给字典一个字符串键）时报 `$viba_program_err`。

### 4.4 `$__len__`

数一个容器有多少份：`list` / `set` / tuple 数元素，`dict` 数键。答案是一个 `int`。

```viba
$__len__ << xs                   # 这个列表有几个元素
$__len__ << table                # 这个字典有几个键
$__len__ << (10, 20, 30)         # 元组按位置数，所以也数得出：3
```

跟 `$__getitem__`、`$__in__` 一样**不收环境**：这是数据那一步。数不出个数的都报程序错：
一个叶子、一个带 tag 的积（`Object * $a 1 * $b 2`）、一段字符串 —— 字符串用 `len_str`（第 2.4 节）。

容器有了个数，走一遍才有可能：`$__len__` 给几段，`$__getitem__` 给第几段，两者配起来。
语言里没有循环，走一遍要自己用 `Y` 递归（第 6.5 节）。

### 4.5 `$__keys__`

取一个字典的键，答案是一份 `list[str]`，顺序由实现定。

```viba
$__keys__ << table                                   # 键组成的列表
$__getitem__ << ($__keys__ << table) << 0            # 第一个键
$__len__ << ($__keys__ << table)                     # 有几个键
```

答案是一份能接着用的列表：能取（`$__getitem__`）、能数（`$__len__`）、能交给 `join`
（第 2.4 节）接成一段。不是字典报程序错；键不是字面量的字典也报程序错 —— 这一支走的是源码形式，
一个算出来的键没有名字可以交出去。

取字典的值就是 `$__keys__` 加 `$__getitem__` 走一遍（[`viba-reflect.md`](viba-reflect.md) 第 5.4 节）。

## 5. 容器

三个容器是内建的，用法就这几件事：源码形式、字面量、取元素、问在不在、数一数、取字典的键
（[`viba-style.md`](viba-style.md) 第 8 节）。

**源码形式**：`list[T]` 按位置取，`set[T]` 没有固定顺序（按位置取时，从头到尾的顺序由实现定），
`dict[K, V]` 按键取；键用 `str`。可以任意嵌套：`list[dict[str, int]]`。

```viba
Items = $items list[int]
Tags = $tags set[str]
Table = $table dict[str, int]
```

**字面量**有它自己的源码形式，站在值的这一侧：放在值的位置上就是那个容器本身。

```viba
ListLiteral[1, "x"]          # 一个列表
SetLiteral[1, 2]             # 一个集合
DictLiteral[("k", 1)]        # 一个字典
ListLiteral[]                # 空列表
```

字面量是 `list[...]` 之类的居民，所以能当实参交给宿主、能序列化回源码。
元组不是这三个之一，它是按位置排的积，源码形式是 `(10, 20)`：括号里按位置摆，不带 tag。

**取元素**用 `$__getitem__`（第 4.2 节），简式是 `xs[1]` / `table["k"]`。
**问在不在**用 `$__in__`（第 4.3 节）。**数一数**用 `$__len__`（第 4.4 节），
**取字典的键**用 `$__keys__`（第 4.5 节）。

这三个名字（连 `ListLiteral` / `SetLiteral` / `DictLiteral`）不能拿来定义：定义名和泛型形参里
出现它们，解析器当场拒。

## 6. 不用 import 的内建模块

这些模块与泛型按名字就能取，在模块自己的定义与 `import` 之后、`builtin` 之前；
`builtin.<名字>` 也是同一个。它们都是模块，所以调用它们就是给环境（第 1 节）。

### 6.1 `apply` —— 把一份积摊回一次调用

```viba
apply : Any <- $f Any <- $args Any
```

```viba
apply << f << ($a 1 * $b 2)          # 一个闭包：还欠环境
apply << f << ($a 1 * $b 2) << env   # 就是 f << 1 << 2 << env
```

`$args` 要的是一份**已经给出的积**，所以积按 `($a 1 * $b 2)` 一个实参给出，不是两条 `<<`；
后面再给环境，这次调用才跑。`apply` 自己不声明 `$env Env`，所以环境是缀在它返回的那次调用后面
往前递的（[`viba-interpreter.md`](viba-interpreter.md)，「环境不进门时它随结果往后走」）。

### 6.2 `sub_env_run` —— 在指定名字的子环境里跑一次调用

```viba
sub_env_run : Any <- $sub_env_name str <- $env Env <- $f Any <- $args ...
```

```viba
sub_env_run << $sub_env_name "low" << env << f << $a 1 << $b 2
# 就是 f << ($sub_env << env << "low") << $a 1 << $b 2
```

`$env` 是这次调用自己的环境，调用方按模块调用的规矩给它一个（`args.env.sub_env << args.env << "run"`，
或者 `args.env.tmp_env << args.env`）；子环境是那一层的孩子（`<那一层>/low`）。
`$args ...` 是"剩下的实参"：链上接着给的都归它，在那里打包成一份积。一个实参都不收的调用
不是它管的 —— 那种调用直接把环境给它（`f << (args.env.sub_env << args.env << "low")`）。
名字那一段也可以是第 3.1 节那种积。

### 6.3 `if` —— 条件挑的那一支，只有那一支

```viba
if : Any <- $env Env <- $cond bool <- $t (Any <- $env Env) <- $f (Any <- $env Env)
```

```viba
if
  << $env (args.env.sub_env << args.env << "if")
  << $cond condition
  << $t (counters.tick)
  << $f (builtin.echo << $x 7)
```

两支都是**函数类型的参数**，所以源码里的那次调用不在原地算：它交给被选中的那个开关，
由开关在它自己的子环境里跑。没被选中的那一支答 `never`，和 `never | a = a` 一起被丢掉 ——
那一支的调用从头到尾没算过，连它的实参也没算。条件是一个 `bool`，
所以"能不能问出一个 `bool`"决定 `if` 用得上用不上（第 10 节）。

一支要好几步时，用 `sequential` 包起来（第 6.4 节）：它的步骤是还欠着环境的调用，
开关给它们环境时才跑。

### 6.4 `sequential` —— 按顺序跑一条链，答最后一步的答案

```viba
sequential : Any <- $steps ...
```

```viba
sequential
  << $x (add << $a 1 << $b 2)              # 带 tag：跑了，答案记在 $x 下
  << (echo << $x ($var "x"))               # 最后一步不带 tag：它的答案就是整条链的答案
```

每一步都是一个调用，跟在 `<<` 后面，按源码里的顺序，一步不落。除最后一步外每一步都带 tag：
这一步跑了，答案记在那个 tag 下。最后那一个实参不带 tag，它的答案就是整条链的答案 ——
要把前面记住的名字交回去，就用 `echo` 把它交出去（`echo << $x ($var "x")`）。
一次调用的实参里可以给出变量引用（`$a ($var "x")`），这一步跑之前，那些名字由前面记住的值答出来，
所以它们落在的槽位要收得下这个源码形式（给成 `$a Any` 那样）。

这条链本身是闭包：环境最后给，给了才跑。步骤可以是本文件里的定义、`import` 进来的模块，
或者内建算子。链头是内建**模块**（`sub_env_run`、`apply`、`Y`）时，判定那一侧还看不出这一段，
要这一步现在就能用，就把它包进本文件里的一个模块再当步骤
（[`viba-interpreter.md`](viba-interpreter.md)，「一个可执行的模块」）。

每一步跑在自己的一条数据路径上，名字由它站的位置、带的 tag 和调的函数拼出来
（`step0/a/add`，最后一步是 `last/echo`），一个实参一个孩子（`arg0/a/mul`），
所以一条数据路径只处理一次调用，也能对着代码看回去。

### 6.5 `Y` —— 一个步骤的不动点

```viba
Y : Any <- $env Env <- $f Any <- $args ...
```

```viba
Y << ($sub_env << args.env << "Y") << f << $a 7 << $b 0
```

这就是把 `f` 交给它自己的下一层，带上这一层的实参：`$args ...` 收下 `$a 7 * $b 0`。
入口与递归是同一件事 —— 交给 `f` 的下一层就是这个调用再往下一层。环境是它的参数：
给环境的那一方同时也给出这一层的存储位置，所以已经跑在某条路径上的模块要给 `Y` 一个自己的子环境
（`args.env.sub_env << args.env << "Y"`）。

### 6.6 按名字取得到、但不是拿来当步骤调的

- `y_helper` 是 `Y` 自己用的下一层（第 6.5 节），`Y` 调用它；程序里不用直接调它。
- `resumable` 给出工作清单那两条定义（`ResumableVibaTaskStack`、`ResumableVibaTask`），
  `interpret_result` 给出一次运行的结果与四种停法的声明
  （[`viba/interpret_result.viba`](viba/interpret_result.viba)），
  `type` 给出类型那一套（`Type`、`ModuleType`、`AstNodeType`、`ModuleGetType`、`IsSubType`），
  `viba_type_descriptor` 给出描述符那一套。这四个是定义的名字，取来当类型用。

## 7. 判断源码里的一次调用是不是闭包

`is_closure` 与 `unclosure` 是**判定层**的泛型：它们看的是源码里的那一段是什么，
所以把 `is_closure[X].value` 放在类型的位置上，取到的就是判定给出的东西；
把它放进 `__impl__`，跑出来的就是那个值。

```viba
is_closure[add << $a 1].value            # true：给了实参、没给环境
is_closure[add].value                    # false：光一个名字，还没有实参
is_closure[add << args.env << $a 1].value  # false：环境给了，它跑了
is_closure[0].value                      # false：一个叶子

unclosure[add << $a 1].f                 # 链头那条 api 的类型
unclosure[add << $a 1].captured          # 收下的那份积：`$a 1`
```

一次调用按 `<<` 一个一个数实参：有几段就认给了几个实参的调用。两份泛型各给出 1..16 段的
16 个文件。环境不算实参，两种给法都不算：按 `$env` 这个 tag 给的，或者那一段的类型就是 `Env` ——
环境给了就是跑了，那条链不认作闭包。外面的 tag 是这一段的一部分，`pattern F << A` 认不出
`$x (add << $a 1)`。

超过 16 段的调用没有文件接：`is_closure` 落到最后那份兜底给出 `false`，
`unclosure` 当场说没有模式。

## 8. 名当数据的一次调用

`__dyn_call__` 与 `__dyn_method__` 这两个名字由解释器自己回答，不向 `get_func` 要实现，
一个模块自己的定义也盖不住它们；`__get_args__` 也是解释器自己回答的（第 8.3 节）。

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

### 8.1 `__dyn_call__`

链头是一条名字路径（`a.b.c`）的那种调用：名字当数据给出，交给宿主，后面给的实参按源码里的次序接着交给它。

```viba
__dyn_call__ << env << "a.b.c" << $x 2        # 就是 a.b.c << env << $x 2
__dyn_call__ << "add" << $a 1                 # 没给环境：一个闭包
```

名字是整名（`"foo.bar.b_scale"` 那样）。它问的是宿主，不是照这个名字去找一份 `.viba` 文件；
要跑一个模块给的是它的名字（`demo << env`），不是这一个。不给环境时它就是一个闭包 ——
判定层把这种链判成 `Any`，因为名字是数据，没有东西可以展开。一次运行里"某一步没有实现"
或"某一步坏了"那一支，`$call` 里记的就是这种形式，所以它不需要原来那个模块在场就能再跑一遍。

### 8.2 `__dyn_method__`

成员是某份值的成员时走这一个：名字是数据，第二份是这个值。

```viba
__dyn_method__ << env << "foo" << bar << $x 1     # 就是 $foo << bar << env << $x 1
```

成员那一层保留：一份值里 `$f` 那个成员是调用 `inc` 时，落成
`__dyn_call__ << "inc"`（[`viba-interpreter.md`](viba-interpreter.md)，
「把一次调用落成可执行的：`__dyn_call__` 与 `__dyn_method__`」）。

### 8.3 `__get_args__`

把这次调用收到的输入起成一份积，成员就是 `__decl__` 那些参数，按 tag 取：

```viba
__decl__ =
    int
  <- $env Env
  <- $a int
  <- $b int

args = __get_args__ << __decl__
# args.env 是环境，args.a 是给 $a 的那份值，args.b 是给 $b 的那份值
```

判定时它给出的是那些参数落成的积类型，计算时给出的是上游真正传过来的数据。
每个模块要取自己的参数都用这一句。

## 9. 一次运行停下来时给出什么

一次运行答一支：`$ok` 拿到答案，`$err` 拿到停下来的原因
（[`viba/interpret_result.viba`](viba/interpret_result.viba)）。停法有四种。

| tag | 什么时候 | 带什么 |
|---|---|---|
| `$viba_program_err` | 程序或环境跑不起来：参数没给齐、名字找不到、容器上没有那个位置或键、被取成员的那份值上没有这个成员 | `msg`（一句话）、`stack`（这次运行经过的调用：文件名与行号） |
| `$environment_api_invalid_argument_err` | 环境自己的成员（本层自己跑的那些 api，`get_func` 从来收不到它们）拒绝了程序给它的实参 | `msg`、`api_name`（`Environment.sub_env` 那样）、`args`（给它的东西，环境不含在内） |
| `$underlying_viba_op_err` | 这一步坏了：`get_func` 抛了异常、实现自己抛了异常，或者实现答了一个没有叶子的东西 | `msg`（`get_func raised` / `raised` / `no leaf` 开头）、`module_path`、`full_qualified_func_name`、`call` |
| `$not_implemented_err` | `get_func` 那里没有这一步的实现 | 同上四个成员，`msg` 是 `no implementation` |

`$call` 就是这次调用本身，落成能再跑一遍的样子（`__dyn_call__` 加上名字和源码里的实参，
环境不在里面，第 8.1 节）。

## 10. 还没有提供的

上面列的是现在有的。下面这一节列出没有的，免得照着一份签名里找不到的东西动手。
这些都不是"以后再补"的清单，只是把现状说清楚：要它们就得由 `get_func` 那一方按自己的方式给出。

**按类型看**：

| 类型 | 没有的 |
|---|---|
| `int` | 位运算（与、或、异或、移位）；`int_to_bool`（用 `ne << $y 0` 顶） |
| `float` | 取整、开方、四舍五入；`eq_f` 是精确相等，没有容差 |
| `str` | 字符与码点互转、反转、数某个子串出现几次、左右补齐、按空白切词；`str` 不算容器，所以 `$__getitem__` 与 `$__in__` 都不认它（取一个字符用 `char_at`，问在不在用 `contains`，都在第 2.4 节） |
| `bool` | `bool_to_int`、`str_to_bool` |
| `nil` | 什么都没有：没有判空，没有和 `nil` 比相等 |
| `list` / `set` / `dict` / 元组 | 枚举（只有 `$__len__` 加 `$__getitem__` 自己数着走）、增加与删除元素、`values` / `items`、首尾相接、切片、集合的并与交、按结构比相等 |

**几处要绕路的地方**：

- **容器数得出，但走一遍要自己搭。** `$__len__` 给几段（第 4.4 节），`$__getitem__` 给第几段，
  两者配起来才能从头到尾走一遍，而语言里没有循环，得自己用 `Y` 递归。字典的键有 `$__keys__`，
  字典的值没有对应的东西（`values` / `items` 都没有）。
- **`bool` 只有几条来源。** 三类比较（`int` / `float` / `str`）、`bool` 的逻辑算子、
  还有 `$__in__`。所以以下几件事在语言里问不出来，只能交给宿主：一份和类型数据是哪一个分支、
  一个积里有没有某个成员、一个容器空不空、一份数据是不是 `nil`。
- **参数都是定长的。** `min` / `max` 只收两个，`concat` 只接两段，没有可变参数；
  三个数取最小要套两层。
- **换类型的失败是实现的停法，不是程序错。** `str_to_int` 拿到一个不是数的字符串报的是
  `$underlying_viba_op_err`（第 2.6 节），语言里没有给出 `Result` 的版本。

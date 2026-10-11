# The interpreter, as a design

这一份说的是解释器**本身**有哪些东西：一次运行是什么、宿主交进来的是什么、一次调用走到哪儿、
入口答什么。每一节的主体是一段 viba；说明只说三件事——这一节在整份设计的哪一层、名字打哪儿来、
哪些还没形式化。

这一份不是程序：不跑，也不参与 `interpret` 的求值。它是一份**读得进去的设计**，跟
[`type.viba`](viba/type.viba)、[`viba_type_descriptor.viba`](viba/viba_type_descriptor.viba)、
[`interpret_result.viba`](viba/interpret_result.viba) 同一个写法，只是把那个写法铺到解释器的本体上。

怎么核这一份：每一段都能 `parse`；把各段按次序接起来是一个模块，里面每个引用到的名字都能在包里
解析出来（定义在本模块、`viba/builtin.viba`、或者上面那三个 import 指过去的模块）。它过不了
`is_complete`——那一位不认 `Any`（[`interpret_result.viba`](viba/interpret_result.viba) 自己也过不了
它那一关），而一份"答什么"的设计里 `Any` 是躲不开的。

## 0. 这一份的模块，与它借用的名字

```viba
# interpreter_design — the interpreter's own design, as viba.
#
# Names this design takes from elsewhere, and does not write again:
#   Environment / Env          the environment's own type, and its eleven members
#   InterpretResult, …         what `interpret` and `exec` answer
#   ResumableVibaTaskStack, …  the work list a compacted run is turned into
#
# Everything else these definitions mention is defined below.

import builtin as vocabulary
import interpret_result as answer
import resumable as work

Optional[T] = T | nil                      # absence is nil; a sum, not a tag

# 附在类型上的话：值是单位，不占字段，只带一段说明。
# `viba_type_descriptor.viba` 用 `Comments`，`type.viba` 用 `Hint`，而这两个名字在包里没有定义。
# 这一份要用，所以自己定义——`RemoteBuddy-core/MEETING_ADT.md` 也是这么做的。

Comments[Note] = Object
Appendix[Note] = Object
Hint[Note] = Object
```

## 1. 顶层：一次运行

一次运行是这几样凑在一起的：一个入口、一个宿主机给的环境、已经答下的那些、以及（压缩过以后）
还没做的那些。往下每一节都是它的一部分。

```viba
Run =
    Object
  * $entry Entry
  * $environ HostEnvironment
  * $answers Answers
  * $running list[Activation]
  * $work work.ResumableVibaTaskStack
  * Comments[{
      一次运行。入口是一份文件或者一段源码；环境是宿主造的（见第 2 节）；`$answers` 是这条运行
      已经答下的那些（按需求值：一个绑定求一次就留下）；`$running` 是此刻跑着的那些调用
      （第 4 节的激活）；`$work` 是压缩之后还没做的调用（`$try_compact` 干的事，见
      viba-interpreter.md「栈太长时」）。
      Python 那边对应 `_Runner`：它是这几个东西加上一路上的账（`running_calls`、`done`、
      `held_paths`……）。
    }]

Entry =
    Oneof
  | $a_file str                            # `interpret`：要跑的那份 .viba 文件
  | $a_text str                            # `exec`：整份模块的源码
  | $some_data Any                         # `exec`：已经站着模块那份数据的可序列化数据
  * Comments[{ 入口的三种给法：一份文件、一段源码、一段已经是模块的数据。 }]

Answers =
    list[AnswerAt]
  * Appendix[{
      # 一条数据路径一个答案。位置不是身份，路径才是：同一条路径问第二次拿到的就是同一个答案。
      # Python 那边是 `runner.done`（按路径字符串查）。
    }]

AnswerAt =
    Object
  * $at Address
  * $value Any
```

## 2. 宿主交进来的三件

这三样是**宿主的词汇**：viba 那一侧一个名字都指不到它们（`Environment` 例外，见下）。
把它们写在这儿，是为了让"宿主该交什么"有个能读的定义，而不是散在文档的句子里。

```viba
EnvironmentStorage =
    Object
  * $cur_storage_path str                  # 这次调用跑在哪条路径上（成员的名字就是它）
  * $store_root_dir str                    # 快照放在哪个目录下
  * $sub (EnvironmentStorage <- $name str) # 子 storage：同一个 store root，路径更长一段
  * $get_text (Optional[str] <- $file_path str)   # store root 底下的纯文本；取不到是 nil
  * $put_text (str <- $file_path str <- $content str)
  * Comments[{
      文件、子模块、快照放在哪儿。注意它自己的 `$parent` 是**数据路径**往上的那一层，跟环境的
      `$get_parent`（环境链）是两件事——`$compress_env_path` 用的是前者。
    }]

EnvironmentCompute =
    Object
  * $get_func GetFunc
  * Comments[{ 实现从哪儿来：只有这一条路。解释器一个具体函数都不认识。 }]

HostEnvironment =
    vocabulary.Environment
  * $storage EnvironmentStorage
  * $compute EnvironmentCompute
  * $viba_path str                          # import 去哪个目录找
  * $parent Optional[HostEnvironment]        # 造它出来的那一层；宿主自己造的就是链的根
  * Comments[{
      宿主手上那个环境。viba 那一侧看得见的是 `vocabulary.Environment`（`viba/builtin.viba` 里
      那十一条成员），它里面**没有** storage 与 compute——那两个是宿主的词汇，viba 里一个名字都
      指不到。这一份把宿主那半边也写出来，好让"环境是什么"有一个完整的定义。
      两处 `$parent` 是两件事：这里的 `$parent` 是**环境链**（造它出来的那一层，`$get_parent` 走
      的那条），`EnvironmentStorage` 那边的 `$parent` 是**数据路径**往上的那一层。
      八个成员函数在宿主侧就是 `viba.interpret` 里几个普通函数：get_parent、get_root、
      get_relative_path、find_by_relative_path、compress_env_path、env_chain_length……
    }]

GetFunc =
    HostImplementation <- $module_path str <- $func_name str
  * Hint[{
      `$module_path` 是这一步**声明在哪个模块**（import 的那个名字；就地声明的是这次跑的模块）；
      `$func_name` 是它**在那个模块里的名字**。问的是"这一步谁来答"，不是"它在哪儿跑"。
    }]

HostImplementation =
    Oneof
  | $a_step HostStep                       # 宿主那边一个可调用
  | $nothing nil                           # 没人实现这一步
  | $said_so NotImplemented                # 实现自己说"没有人做这一步"

HostStep =
    Hint[{
      宿主交进来的一个可调用。它收到的是**已经算好的实参**，按源码里的次序；实例是 VObject，别的
      原样（环境在内）。它返回 VObject、一个普通值，或者 nil。viba 那一侧**调不动它**：它不是 viba
      函数，执行 viba 函数只有一条路——给环境。所以高阶参数上，宿主拿到的是数据（函数名，或者一个
      闭包），可以拿着、存着、递回来。
    }]
```

## 3. 一次调用

调用是这一份设计里最小的活动单位：一个链头、一串已经给掉的实参、以及有没有环境。
**给环境才算跑**；没给的就是一个值（闭包），可以存着，晚点再给。

```viba
Call =
    Object
  * $names Name
  * $head WhatIsCalled
  * $given list[Given]
  * $environ Optional[vocabulary.Environment]
  * Comments[{
      一次调用。`$head` 是链头（名字、成员、或者一次应用），`$given` 是已经给掉的实参，`$environ`
      是这次调用拿到的环境：nil 的时候这次调用还是一个值（闭包），不是一次跑。
      Python 那边是 `_Pending`（还没拿到环境）与它落地后的样子。
    }]

Name =
    Object
  * $file_path str                         # 写在哪份文件里
  * $lineno int                            # 第几行
  * Comments[{ 一次调用写在哪儿：出错的答复要把这个带回去（`interpret_result.viba` 里那两条）。 }]

WhatIsCalled =
    Oneof
  | $a_definition Name                     # 本文件里的一条定义
  | $a_module_member str                   # 别的模块里的一个名字（import 进来的）
  | $a_member_taken str                    # 从一个值上取成员：`box.f`、`$get_attr << box << "f"`
  | $the_builtin str                       # viba 自带的那一套（`builtin.add`，或者不带模块名的 `add`）

Given =
    Object
  * $tag Optional[str]                     # `$env`、`$x`……按 tag 给；nil 是按位置给的
  * $value Any
  * Comments[{
      一个已经给掉的实参。函数类型的那个参数是**惰性**的：它不算，直到有人要它——那时候连同环境
      一起交给宿主。别的参数按值算（call-by-value）。
    }]
```

## 4. 运行里的一个模块：激活与它的绑定

一个模块配一个环境，就有一次激活；它那份文件里的每条定义是一条绑定，绑定是**还没求值的计算**，
求一次就把答案留下（按需求值）。

```viba
Activation =
    Object
  * $call Call
  * $at Address
  * $bindings list[Binding]
  * $defined list[str]
  * Comments[{
      一个模块在一个环境里跑起来。`$at` 是它的数据路径（身份），`$bindings` 是这份文件里的那些
      绑定，`$defined` 是已经求出来的那些名字。Python 那边是 `_Activation`。
    }]

Binding =
    Object
  * $name str
  * $at Address
  * $answer Optional[Any]
  * Comments[{
      一条绑定：还没求值的计算（thunk）加一个"求出来了没有"。一份文件自己的定义图不许成环——
      绕回自己当场报错，不是等栈崩。求过一次就不再算（memo），同一条路径问第二次拿到的是同一个答案。
    }]

Address =
    str
  * Appendix[{
      # 数据路径：一次调用跑在哪条路径上（存下来的那条路径）。位置的每一段说明它站在哪，
      # 不是一次调用——所以从一条路径找回来的是一个环境，不是一串。
      # 太长的时候 `$compress_env_path` 把一段路径压成一个名字挂在旁边，`$try_compact` 把此刻
      # 跑着的调用换成一份工作清单。
    }]
```

## 5. 解释器自己回答的五个成员

这五个不是宿主实现的：解释器自己有。它们跟环境那十一个成员一样，都是**成员**——环境排在第一位，
两种写法（挂在值上、tag 放链头）是同一次调用。它们现在只在 Python 和文档里，这一节把它们的签名
写成 viba。

```viba
InterpreterMembers =
    Object
  * $get_attr (Any <- $env Env <- $box Any <- $name str)
  * $get_item (Any <- $env Env <- $container Any <- $key Any)
  * $in (bool <- $env Env <- $container Any <- $element Any)
  * $len (int <- $env Env <- $container Any)
  * $keys (list[str] <- $env Env <- $table Any)
  * Comments[{
      解释器自己回答的五个成员。它们不走宿主的 `get_func`，所以一份程序调到它们不需要谁去声明——
      这跟"角色 api 声明了什么，宿主才有什么"是两回事。
      `$get_attr` 后面还能接着给实参（取出来的成员本身是一次调用）；`$get_item` 按位置或者按键取；
      `$in` 问在不在（字典只比键）；`$len` 数一个容器有几份；`$keys` 取字典的键。
    }]
```

## 6. 关键 api

入口两个，各自跟一个宿主的两件事，加上一步写完的取放。

```viba
Interpret =
    answer.InterpretResult <- $viba_main_file str <- $environ Env

Exec =
    answer.InterpretResult <- $viba_code Any <- $environ Env

GetSnapshot =
    Any <- $env Env <- $name str

PutSnapshot =
    Any <- $env Env <- $value Any <- $name str
  * Appendix[{
      # 入口两个：`interpret` 跑一份文件，`exec` 跑一段源码（或者一段已经站着模块的数据）。
      # 两者都答 `answer.InterpretResult`——`$ok` 带这次跑出来的值，`$err` 带为什么停。
      # 取放两个是宿主自己用的：一个算子的答案落在它那条数据路径下
      # （`<那条路径>/<算子名>.viba`），回放的时候就从那儿拿。幂等说的就是这件事：同一条路径问
      # 第二次不该再算一遍。
    }]

NotImplemented =
    Hint[{ 实现自己交回"没有人做这一步"时用的那份数据（`viba.interpret.not_implemented`）。 }]
```

## 7. 还没形式化的

这些在 Python 里、在 [`viba-interpreter.md`](viba-interpreter.md) 里都有，但这一份还没给它们定义：

- **惰性的两条路**：函数类型的实参走 `_HostArgument`，别的惰性槽走 `_Getter`；两者的分工只写在
  代码里。
- **一次运行的账**：`done` 按路径字符串查、`_is_running_there` 按对象 `is` 比——两套并存，为什么
  还没写成定义（上面 `Answers` 里只写了按路径那一套）。
- **在途实参**：`_Given`（给到一半的实参）与 `_Raised`（跨宿主边界带回来的结果）只在 Python 里。
- **宿主函数包成环境成员**：`_HostFunction`、`_GetArgs`、`_DynCall` 三个包法，`__get_args__` /
  `__dyn_call__` / `__dyn_method__` 三个语法位置。
- **压缩的边界**：`_Compact` 信号、`_hold`、`_driven`、`_ResumableCall` 与 `work.ResumableVibaTask`
  的字段是不是逐一对上（阈值那条策略在 viba 里，机制在这边）。

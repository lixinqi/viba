# viba/exec：运行 viba 代码

一个模块怎么变成一次调用、一次调用要什么、跑完得到什么——这里放这些说明。

一次运行是 `__decl__` 进、`__impl__` 出；逻辑不在这里写，由宿主通过 `get_func` 交进来。

想从头学、一路学到跑起来：[`viba_tutorial.md`](viba_tutorial.md)。

## 一个模块什么时候是函数

同一个文件既是**模块**，也是——当它自己这么说时——**函数**：

- 当模块看，它是一张定义表：每个顶层定义是它的一个成员，按名字取（`foo_module.Bar`，泛型应用落到
  的文件也一样：`g[T].value`、`g[T].type`）。
- 当函数看，它是"一个声明加一个 body"：`__decl__` 说它返回什么、收哪些参数（环境在其中），
  `__impl__` 是它返回的东西。没有 `__decl__` 的文件不是函数，调用它是程序错误；没有 `__impl__`
  的文件是类型，不是程序，跑它给 `VibaProgramErr`。一个模块不能把同一个名字定义两次。

要参数的模块在 `__decl__` 里声明，再用 `args = __get_args__ << __decl__` 取回来：当类型看，那条链
的参数构成一个积类型；当计算看，`args` 是这次调用交进来的数据。**给掉环境才跑**，没给环境的调用是
一个闭包——一个值，可以留着、传走，或者序列化之后再跑。跑一个模块就是求 `__impl__`，以及按需地求它
用到的绑定（[`../core/viba-reference.md`](../core/viba-reference.md#求值)）。

`{ add two integers }` 这样的提示块是文件对某一步实现的全部说法：每一步要实现的地方都只有一条提示
——说这一步是干什么的——没有逻辑。所以一个文件自己跑不了；逻辑那一半由调用方经 `get_func` 交进来。

## `interpret` 与 `Environment`

`interpret` 吃一个文件，把它跑起来。签名写成 viba 类型；`Environment`（宿主的）和 `VObject`（一格
viba data，带读它的访问器）是它用到、但不在这里定义的两个名字，两者都在
[`viba-interpreter.md`](viba-interpreter.md)：

```viba
interpret =
    InterpretResult
  <- $viba_main_file str
  <- $environ Environment
```

答的是声明的那份结果，**作为一个 viba data 的节点**：`$ok` 装着 `__impl__` 返回的东西，`$err` 装着
为什么停下。声明——`InterpretResult`、`InterpretError`，以及每种停下的字段——在
[`../viba/interpret_result.viba`](../interpret_result.viba)，结果按它来取、按它来判
（`is_interpret_result`，建在 `is_sub_type` 上）。

宿主提供环境：快照放在哪儿（`EnvironmentStorage`，默认一个临时目录），以及那些实现
（`EnvironmentCompute`，一个 `get_func(module_path, func_name)`）。

```python
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            exec, interpret)
from viba.reflect import access

def get_func(module_path, func_name):
    # module_path 是那一步声明在哪儿，和它被 import 时的写法一样（`demo`、`foo.bar`）；
    # func_name 是那一步在那儿叫什么（`add`、`print`）。
    if module_path == "demo" and func_name == "add":
        return lambda env, a, b: a.value + b.value
    return None

environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))

result = interpret("add_demo.viba", environ)
print(result)                                     # $ok 1000000
print(access.leaf(result.by_tag("$ok")).ok_value) # 1000000——宿主返回的那个数

ADD_DEMO = open("add_demo.viba").read()   # 同一个入口，模块以文本给出
print(exec(ADD_DEMO, environ))            # 同一次运行，同一格数据

# `exec` 收模块的文本，或者已经代表那棵模块树的那格 viba data：不是模块树的数据就是这次运行的
# `__impl__`，环境由这次运行给——所以一次停下带的 `$call` 能再跑一次，不用先序列化出去。
print(exec(access.leaf(result.by_tag("$ok")).ok_value, environ))
```

可执行的函数可以带一个 `$env Env` 参数，也可以一个环境参数都不带。没有那个参数，环境就在这次调用
之外——它不是成员，所以 `args.env` 只在参数在的时候才有。

## 章节

| 文档 | 讲什么 |
|------|---------|
| [`viba_tutorial.md`](viba_tutorial.md) | 从头学：和、积、指数、`<<`、泛型，直到一个模块真跑起来 |
| [`viba-interpreter.md`](viba-interpreter.md) | 跑一个模块：`__decl__` 进、`__impl__` 出，按需求值的策略 |
| [`viba-primitives.md`](viba-primitives.md) | 内建 API：算子、`Environment`、成员算子、容器、不要 import 就看得见的内建模块 |
| [`viba-interpreter-design.md`](viba-interpreter-design.md) | 解释器自己，用 viba 写：一次运行、宿主的三件东西、一次调用、一个活动、解释器自己处理的五个成员、两个入口 |

## 这里的代码

| 模块 | 干什么 |
|------|--------|
| `interpret.py` | 跑一个模块：`__decl__` 进、`__impl__` 出——见 `viba-interpreter.md` |
| `interpret_result.viba` | 一次运行的答案声明（`InterpretResult`） |
| `builtin.viba` | 每个模块都看得见的内建词表——`Environment` / `Env`、`builtin.echo`、内建算子（`builtin.add`、`builtin.lt_f`……），以及 `builtin.echo_or_never` / `builtin.never_or_echo` 两个开关 |
| `if.viba` | 内建 if/else：条件挑一支，只算那一支的调用——两个开关合成的和 |
| `sub_env_run.viba` | 在一个具名子环境里跑一次调用——`Y.viba`、`apply.viba` 在它旁边 |
| `resumable.viba` | 太深的调用链拆成的工作清单：一次一个任务，带它的数据路径和那次调用 |
| `sequential.viba`、`sequential_impl/` | 按顺序跑一串步骤的内建：每个步骤数一个文件，每一步在自己的子环境里跑，名字由站的位置、带的 tag 和调用的函数给（`step0/a/add`、`last/echo`），最后一步的结果就是这一串的结果 |
| `builtin/is_closure/`、`builtin/unclosure/` | 闭包上的两个内建泛型：源码里的一次调用是不是闭包，以及把它拆开——每个实参个数（1..16）一个文件 |
| `builtin/sequential_step/` | `sequential` 怎么跑一步：一次带 1..16 个 tag 实参的调用，每个实参在自己的子环境里算，名字同样给法（`arg0/a/mul`） |
| `builtin/tree_apply_closure_only_without_env/` | 一次调用的一个实参怎么当成它自己的树算出来：`$var "x"` 变成那个名字持有的值，一次调用同样往下算一层再在这个环境的子环境里跑，别的值原样——每个实参个数一个文件 |



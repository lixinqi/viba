# 执行 viba 模块

一个 viba 文件是一个 Module，而 Module 有两种读法：

1. **类型推导**：把它当类型读（`viba.is_sub_type`、`viba-reflect.md`）；
2. **数值计算**：把它跑起来（`viba.interpreter`）。

同一份语法，两种模式。跑的时候，模块**就是函数**：输入是 `environ`，输出是 `__ret__`。

```python
from viba.interpreter import interpret

interpret("add_demo.viba", environ)      # -> Result[VibaNode]
```

`interpret(viba_main_file, environ, viba_path=None)`：`viba_path` 相当于 PYTHONPATH（冒号分隔，
按顺序找 `<name>.viba`，dotted 名当路径走；空条目和不存在的目录跳过）；import 的那个文件所在的目录总是
先找——**被 import 进来、又在自己的文件里 import 的模块，也按它自己的文件找**（链多深都一样）。
写了 import 的文件里的名字，按 import 绑定的名字解析（`import a.b as c` 绑 `c`，`import a.b` 绑 `a.b`）。

## 一个可执行的模块

```viba
# file name: add_demo.viba
add :=
	int
	<- $env Environment
	<- $a int
	<- $b int
	<- {
		add two integer
	}

print :=
	void
	<- $env Environment
	<- $x Any
	<- {
		print to stdout
	}

__ret__ :=
	add
	<< $env environ
	<< $a 999999
	<< $b 1
```

规则：

- **没有 `__ret__` 的文件是设计，不是程序**：跑它给 `Err`。
- **`environ` 是内建变量**：类型推导时它是 `Environment` 类型，计算时是真实的那个环境。
- **函数体里的 `{...}` 是说明**：它不是参数，`<<` 给完真参数之后链就落到结果上（见
  `contract-enough.md` 的 `T << X`）。
- **每个可执行函数都要依赖 environ**：签名里必须有 `$env Environment` 这一格，调用时也必须给；
  否则 `Err`（不是猜一个默认值）。
- `__ret__` 必须是值。还差参数没给全的函数不是值，`Err`。

## 调用别的模块

```viba
import add_demo as demo

ret := demo << (environ.sub_env << "add_demo")

__ret__ := demo.print << environ << ret
```

- `demo` 是模块，当函数用：给它一个 environment，它跑完给出它的 `__ret__`。
- `demo.print` 是这个模块里的函数。
- `environ.sub_env << "add_demo"` 拿一个子环境：**它带着父级的 compute**，storage 路径是
  `父路径/add_demo`（见下）。

## 宿主侧：Environment

`Environment` 由宿主提供，至少两个概念：

```viba
Environment :=
    Object
  * $sub_env (Environment <- $sub_env_name str)

EnvironmentStorage :=
    Object
  * $cur_storage_path str
  * $sub_storage (EnvironmentStorage <- $sub_storage_name str)
  * {
      other storage information
    }

EnvironmentCompute :=
    Object
  * $get_func (HostLanguageFunc <- $module_path str <- $func_name str)
  * {
      HostLanguageFunc matches the interpreter: with the python interpreter it
      is a python function. get_func is called by interpret, and a module's
      sub-environment holds the parent's compute.
    }
```

Python 侧就是这三个类（`viba.interpreter`）：

```python
EnvironmentStorage(cur_storage_path, sub_storage=None)   # sub(name) 给子 storage
EnvironmentCompute(get_func)                             # get_func(module_path, func_name)
Environment(storage, compute)                            # sub_env(name) 给子环境
```

`get_func(module_path, func_name)` 返回一个可调用对象，没有就返回 `None`（于是 `Err`）。
`module_path` 是**调用时那个 environment 的 storage 路径**——所以同一个 `add`，从
`root/add_demo` 进来和从 `root` 进来，宿主看到的是不同的路径，可以路由到不同的实现。

`HostLanguageFunc` 与 interpreter 匹配：Python interpreter 里就是一个 Python 函数，收到的参数是
**已经算好的实参**，按书写顺序给——材料是 `viba.reflect.VibaNode`，别的（environ 在内）是它本身。
返回值是 `VibaNode`，或者一个普通 Python 值（落到函数声明结果的一个叶子上）。

参数里出现 **viba 函数**（`$f (int <- $env Environment <- $x int)` 这种高阶签名）时，宿主拿到的是一个
Python 可调用对象：它照那个函数自己的顺序给参数（可执行函数的 `$env` 也要给），拿回答案，答案同样是
`VibaNode` 或普通 Python 值。所以高阶函数在宿主侧就是普通的高阶 Python 函数：

```python
def twice(env, f, x):
    return f(env, x).value + f(env, x).value
```

宿主自己造一个 `VibaNode` 当答案也可以；但**列表、字典、可调用对象这类答不了**——它们没有对应的叶子，
只能答 `VibaNode`、标量或 `None`（`None` 就是 `nil`）。

**interpret 不认识任何具体函数**：viba 默认不带任何库函数，实现全部来自 `get_func`，
谁写、怎么生成，interpret 不感知。

## 类型层的模块

同一个文件在**类型推导**里也是"environ 进、`__ret__` 出"：名字绑到的是一个模块（import 的名字）时，
它当函数读，类型就是

```viba
__ret__ <- $env Environment
```

所以下面这几条都成立（`tests/test_is_sub_type.py` 里有用例）：

```viba
demo := import add_demo as demo         # 概念上
demo << $env environ  <:  int           # 就是 __ret__ 的类型
int <: demo << $env environ             # 反过来也成立：两者同型
demo.add <: int <- $env Environment <- $a int <- $b int
design.Only <: $x int                   # module.MyType 照旧，没有被顶掉
```

- 只认 **import 绑定的那个名字**（`import a.b as c` 的 `c`，`import a.b` 的 `a.b`）。`module.Name`
  仍然是那个模块里的定义，和以前一样按最长的前缀解析。
- 没有 `__ret__` 的模块不是程序：`demo << $env environ` 在类型层也是 `Err`。
- `environ` 在类型层是内建名字，类型为 `Environment`（`viba/builtin.viba`），所以 `<< $env environ`
  这一格在类型上也对得上。

## 错误

`interpret` 返回 `Result`：`Ok(VibaNode)` 是 `__ret__` 的值，`Err(str)` 说明哪一步不行：

```
no such file: ...                     文件不在
no definition named 'x' in module 'm' 名字解析不了
module 'x' not found (...)            import 找不到文件
module 'm' has no __ret__: ...        设计，不是程序
no implementation for 'add' in ...    get_func 没给
get_func(...) raised ...              get_func 自己抛了
add raised ZeroDivisionError(...)     宿主函数抛了
... takes no $env Environment ...     可执行函数没依赖 environ
... was not given the environment     调用时没给 environ
... was not given an Environment      给了，但不是 Environment
module 'x' is already running         模块调用成环
... is a function still waiting ...   __ret__ 不是值
... answered list, which is no leaf   宿主答了没有叶子的东西
cannot read ...                       文件读不了
cannot parse ...                      编译不过（语法错误）
```

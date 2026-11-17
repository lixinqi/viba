# 重演：记录入口调用，按参照重跑

A 调 B、C、Compare(B, C)。C 带状态，重演时不能再跑一遍，只能用录制时存下的那一份返回值；B 是被考核的
函数，每次重演都要重算；Compare 拿新的 B 和那份固定的 C 重新算。重演按一个 tag 找到之前记录的所有
入口调用，逐个重跑。

## 1. 录制与重演

**录制**：入口 A 跑一次。`reenact_entry` 把这次调用的入口与实参写进本地的 store，然后在这次调用
自己的数据路径下跑入口；入口里经过 `reenact_with_reference[C, tag]` 的调用，跑 C 一次，把它的
返回值存进 store，位置是那次调用的数据路径加 tag。

**重演**：`reenact` 按 tag 读出所有调用，逐个用记录里的实参再调入口。重演在另一个 store 里跑：
B、Compare 照常算；`reenact_with_reference` 不调 C，按同一条数据路径去录制时的 store 里取那份
返回值，读出来就用，不复制。

录制用的 store 往后只读；每次重演用一个新 store。参照从旧的取，重算的写进新的。

## 2. 存储

`<base>` 是入口被调用时那个环境的数据路径。

```
<base>/reenact/<tag>/count.viba              已经记录了几次
<base>/reenact/<tag>/<n>/entry.viba          第 n 次的入口名（一个字符串）
<base>/reenact/<tag>/<n>/args.viba           第 n 次的实参积
<base>/reenact/<tag>/<n>/…/<tag>.viba        参照，放在那次调用的数据路径下
<base>/reenact/<tag>/<n>/answer.viba         重演时第 n 次算出来的返回值
```

`entry.viba`、`args.viba`、`answer.viba` 都用快照那套序列化（`value = …`）；`count.viba` 是一个
整数。`$record_reenact_call` 读到 n，写回 n+1，这次调用跑在 `<base>/reenact/<tag>/<n>` 上。同一个
tag 的录制不要并发——这是读-改-写。

参照不另起命名空间：`reenact_with_reference` 录制时用 `put_snapshot(env, answer, tag)` 存到
`<store>/<数据路径>/<tag>.viba`；重演时按同一条数据路径去 `reference_store_root_dir` 里读。路径是
参照的一部分，所以入口两次要跑在同一条路径上（第 4 节）。

## 3. Python 侧

`ReferencedEnvironment` 比 `Environment` 多一个字段：

```python
class ReferencedEnvironment(Environment):
    def __init__(self, storage, compute, viba_path=None, parent=None,
                 uncompress_relative_path=None, reference_store_root_dir=None):
        super().__init__(storage, compute, viba_path, parent,
                         uncompress_relative_path)
        self.reference_store_root_dir = reference_store_root_dir
```

`sub_env`/`tmp_env` 从它造出来的仍是 `ReferencedEnvironment`，带着同一个
`reference_store_root_dir`；少了这一步，入口往下走一层就丢了参照。

环境上多五个**运行时成员**（解释器实现，不进 `get_func`：它们要碰 `EnvironmentStorage`）：

| 名字 | 返回什么 |
|---|---|
| `$make_referenced_env` | 同一条数据路径、同一个 storage 的 `ReferencedEnvironment`，只多了参照 store |
| `$record_reenact_call` | 写下这次调用，返回它跑的名字 `"reenact/<tag>/<n>"` |
| `$reenact_reference` | `$found bool * $value Any`：重演环境里读参照，录制环境里 `$found false` |
| `$store_reenact_reference` | 录制时存下返回值；重演时原样返回，不写 |
| `$reenact_calls` | 逐个重跑记录下来的调用，返回所有返回值组成的列表 |

它们**不进 `Environment` 那个积**：积里放 `Any` 会让 `is_complete` 判它不完整，而 `Any` 正是本层
要补的那一格。运行时成员和 `$get_attr`、`$get_item` 那几个一样，靠解释器实现，不靠积声明。

`$reenact_calls` 是一个宿主侧的循环：它从参照 store 读第 n 次的实参，把入口按录制时那个调用
（`apply << $f 入口 << $args 实参积`）跑一遍，返回值存成 `answer.viba`。入口是 viba 数据（名字或
调用，带着它的名字所在的模块），所以宿主把它交给解释器自己的调用机器跑，入口可以是 viba 写的。
一个名字交给宿主时，模块跟着它走：`_argument_value` 把 `data_module` 放在那份 `VObject` 上。

## 4. viba 侧

两个泛型放在 `viba/builtin/` 下（`__generic__.viba` 加文件，和 `is_closure` 一样），名字对每个
模块可见，不用 import。

```viba
reenacted_func_a = reenact_entry[reenacted_func_a_impl, "reenacted_func_a"]

# reenacted_func_a_impl 里不直接调 reference_c：
cached_reference_c =
    reenact_with_reference[reference_c, "cached_reference_c"]
  << $env args.env
  << $x args.x

# 重演，在 $make_referenced_env 造出来的环境上：
__impl__ =
    reenact
  << $env (args.env.sub_env << args.env << "replay")
  << $f reenacted_func_a_impl
  << $tag "reenacted_func_a"
```

`reenact_entry[F, S]` 的形参收入口与 tag，实参收在 `$args ...` 里；`__decl__` 是
`Any <- $env Env <- $args ...`。`__impl__` 先把这次调用记在**入口被调用时那个环境**下（包装模块
自己那一层的上一级），拿回调用名，再在它的子环境上 `apply << $f F << $args args.args`。入口的
签名不从名字上再抄一遍（名字的指数链不在源码里，模式看不到它），所以 `reenacted_func_a` 收任意实参，原样转给入口。
要把签名保住，就把第一个实参给成调用形式（`reenact_entry[A_impl << $x T, "tag"]`），模式按位置
收签名，每个实参个数一份文件——本设计按名字给。

`reenact_with_reference[F, S]` 同样收实参、同样转发。先问 `$reenact_reference << $env args.env << $tag S`：
`$get_attr` 取 `found` 与 `value`；`found` 为真就用 `if` 的那一支把 `value` 交回（重演环境里缺这份
参照是程序错）；为假就 `apply` F，再 `$store_reenact_reference` 存下它。

`reenact` 的声明是 `list[Any] <- $env Env <- $f Any <- $tag str`。调用方给它一个子环境（模块调用
要一层自己的），它的上一级就是录制时入口被调用的那个 `<base>`；`$reenact_calls` 在那里读记录、
跑入口、收返回值。重演调的是记录里的入口，不是 A，所以不会记第二遍。

## 5. 边界

- **重演要在同一个 `<base>` 上跑。** `reenact` 的上一层就是录制时入口的调用者；调用方给它的子环境
  名字随意（`"replay"`），路径对不上，参照就找不到。
- **重演 store 每次新建。** 复用同一个重演 store，B 的 `replayed` 会在同一条调用路径上命中，B 就
  不重算了；B 的步骤照常走 `replayed` 也行，只要 store 是新的。
- **参照只认稳定路径。** 入口里给 C 的环境路径在两次运行里要一样；`tmp_env` 的名字每次都变，
  参照找不到。
- **录制 store 只读。** 重演不往里写；同一 tag 的录制串行做。
- **缺参照是错。** 重演时参照只从录制里读；读不到就是那次录制没走到这个 tag，当场报错，不退回重算。

## 6. 文件

- [`viba/interpret.py`](../interpret.py)：`ReferencedEnvironment`、`$make_referenced_env`、四个
  重演成员、`sub_env`/`tmp_env` 的字段传递、`_argument_value` 带上 `data_module`。
- [`viba/builtin.viba`](../builtin.viba)：`Environment` 这个积（运行时成员不在这里）。
- `viba/builtin/reenact_entry/`、`viba/builtin/reenact_with_reference/`：两个泛型，各一份文件。
- `viba/reenact.viba`：`reenact` 模块。
- [`tests/test_reenact.py`](../../tests/test_reenact.py)、`tests/data/reenact/`：录制两遍、重演一遍、
  缺参照、普通环境上重演。

# 合规：规则与呈证

一份规则（rule）说"什么算合规"，一份呈证（witness）说"这一次是什么情况"，两者合起来给一个
判定。这一章讲这套东西怎么用**可执行模块**来写：规则是程序，呈证是材料，判定就是跑那个程序。

## 1. 一个规则是一个程序

规则就是一个 viba 程序：`environ` 进、`bool` 出。判定不是另一套推理，就是把文件跑起来。

```viba
# viba/compliance/demo/rule_distance.viba
# 那一刻他们至少隔了 5 吗？
import case_points as witness

Point := $x int * $y int
Case := $victim Point * $suspect Point * $at str

measure_distance :=
	int
	<- $env Environment
	<- $case Case
	<- {
		两人当时相距多远：不是纯函数，答案会被存档（见第 4 节）
	}

distance_at_least_5 :=
	bool
	<- $env Environment
	<- $d int
	<- {
		至少 5 吗？
	}

__ret__ :=
	distance_at_least_5
	<< $env environ
	<< $d (
		measure_distance
		<< $env environ
		<< $case (witness << (environ.sub_env << "witness"))
	)
```

读法：

- **`__ret__` 是判定**。它是 `bool`，跑完就是答案：真合规、假不合规。
- **条件是它自己的函数**。`distance_at_least_5` 就是这条规则的条件；规则想有几个条件、想怎
  么组合（`<<` 给参数、定义复用、模块拆开），都是写程序的事。规则能表达任何算得出来的条件。
- **函数体里的 `{...}` 是说明**：实现来自宿主的 `get_func(module_path, func_name)`（见
  `viba-interpreter.md`），interpret 不带任何库函数。
- **每个可执行函数都要 `$env Environment`**：这是 interpreter 的规矩，规则也不例外。

## 2. 呈证是材料

呈证是规则要判的那份材料，它自己也是一个程序：`environ` 进、材料出。写法和它的类型一模一
样——tag 与积写在值位置上就是材料。

```viba
# viba/compliance/demo/case_points.viba
__ret__ :=
    $victim ($x 0 * $y 0)
  * $suspect ($x 3 * $y 4)
  * $at "12:30"
```

规则那边把呈证当模块调一次，拿到材料，再交给自己的函数：

```viba
<< $case (witness << (environ.sub_env << "witness"))
```

材料里有什么，规则就按反射协议读什么（`$victim`、`$suspect`、`$at`，以及各自的 `$x`/`$y`）；
读的工具在宿主侧，是 `viba.reflect`（`viba-reflect.md`）。呈证不必长得像规则：它就是事实。

## 3. 判定

```python
from viba.compliance import is_compliant

verdict = is_compliant("rule_distance.viba", environ)   # -> Result[bool]
```

- 规则答 `bool`，判定就是它：`Ok(True)` / `Ok(False)`。
- 答的不是 `bool`，是 `Err`（"a verdict is a bool"）。
- 规则编不过、没有 `__ret__`、`$env` 没给、宿主函数抛了……都是 `Err`，说明哪一步不行。
- 一次运行的环境（`Environment`、storage、`get_func`）怎么给，见 `viba-interpreter.md`：
  `is_compliant` 就是 `interpret` 加"读出那个 bool"。

## 4. 不纯的那一步：Prepare

程序里唯一不能保证"跑多少次都一样"的东西是宿主函数：它可能读时钟、掷骰子、调服务。
`measure_distance` 就是这样一个函数。它不能自己答一个数就算了——那样结果不可回放。它走
`measure`：

```python
# viba/compliance/demo/host.py
def measure_distance(self, env, case):
    def compute(prepared):
        victim = _point(prepared, "victim")
        suspect = _point(prepared, "suspect")
        return int(round(((victim[0] - suspect[0]) ** 2
                          + (victim[1] - suspect[1]) ** 2) ** 0.5))
    return measure(env, "distance", case, compute)
```

`measure(environ, name, call, compute)` 做的事：

1. 读这次调用的 **Prepare**（见下）。里面已经有量出来的值 → **直接回放，`compute` 一次不调**；
2. 没有 → 调 `compute(call)`（不纯的那一步），把值写进本轮 store 的 Prepare 里，再答它。

**Prepare 是"这一次测量"的存档**：调用（参数定了、结果声明了）与量出来的值，一个文件：

```viba
# <store>/root/prepare/distance.viba
value :=
    $call (
        $victim ($x 0 * $y 0)
      * $suspect ($x 3 * $y 4)
      * $at "12:30"
    )
  * $measured 5
```

- 它是**序列化的 viba 数据**，不是 pickle：人读得懂，也能解析回材料。
- 路径是 `<storage 路径>/prepare/<name>.viba`；`name` 由调用方取（同一个调用用同一个名字）。
- Prepare 里的 `$call` 压过现场写的 `call`：**被固定的那次调用才算数**——这正是"准备"的含义。

## 5. 备份与回放

一次运行读 Prepare 的规矩在 `viba/compliance/storage.py` 的 `PreparedStorage`：

- **读：先看本轮 store，再看备份**（`prepare_root_dir`）。备份是运行之前就存好的证据。
- **写：只进本轮 store，永不写备份**。运行不许改证据。

于是"一次准备、多次判定"：

| 运行 | 量距离 | 条件看到 | 结果 |
|---|---|---|---|
| 第一次（没有备份） | 走了一次 (0,0)—(3,4) | 5 | `Ok(true)` |
| `prepare_run` | 回放刚量过的那份 | 5 | `Ok(true)`，并把 Prepare 记进备份 |
| 之后每一次（有备份） | **一次都不走**，读备份里的 5 | 5 | `Ok(true)` |

把备份留下来的是**一次专门的运行**：

```python
from viba.compliance import prepare_run

prepare_run("rule_distance.viba", environ)   # 跑一遍，把它量过的 Prepare 记进备份
```

`tests/data/compliance/backup/` 里放着一份事先备份好的 Prepare：拿它当 `prepare_root_dir`
运行，那条不纯的路一次都不走，判定仍是同一个。

## 6. 句柄一览

判定与准备（`viba/compliance/judge.py`）：

```python
is_compliant(rule_file, environ) -> Result[bool]   # 跑规则，读它的判定
prepare_run(rule_file, environ) -> Result[bool]    # 跑一遍，并把量过的 Prepare 记进备份
measure(environ, name, call, compute)              # 量一次：有 Prepare 就回放，没有就量并存
```

直接读写 Prepare：

```python
prepare_path(name)                     # 它在 store 里的名字：prepare/<name>
read_prepare(environ, name)            # 存着的那份材料，或 None
record_prepare(environ, name, call, measured=None)   # 写进本轮 store
measured_of(prepare)                   # (值, True)；没有量过是 (None, False)
call_of(prepare)                       # 被固定的那次调用，或 None
```

storage：

```python
PreparedStorage(cur_storage_path, sub_storage=None,
                store_root_dir=None, prepare_root_dir=None)
read_text(file_path)      # 本轮 store，然后备份；都没有是 None
write_text(file_path, content)           # 只写本轮 store
record_text(file_path, content)          # 写进备份本身（prepare_run 用它）
```

## 7. 自己写一份

1. **写呈证**：一个模块，`__ret__` 是材料（tag、积、字面量、元组都行）。
2. **写规则**：一个模块，`__ret__` 是 `bool`；把条件写成它自己的函数，每个函数带
   `$env Environment`。
3. **实现宿主那侧**：`get_func(module_path, func_name)` 给出这些函数的实现；纯的照常写，
   不纯的用 `measure` 包住，名字自取。
4. **跑判定**：给一个 `Environment`（storage + compute）。要留证据就 `prepare_run` 一次，
   之后每次 `is_compliant` 都读备份。
5. **要检查材料**：用 `viba.reflect` 的地址与叶子读（`viba-reflect.md`），或 `read_prepare`
   直接读存档。

`tests/test_compliance.py` 是一份可以照抄的完整例子；`viba/compliance/demo/` 是上面这套的
最小可运行版本。

---

`viba-rule.md` 是更早的一套规则做法（用子类型判定、另立 `Metric`/`Predicate` 等词），已废弃；
现在的做法就是这一章。

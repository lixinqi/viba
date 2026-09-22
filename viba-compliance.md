# 合规：规则与呈证都用 interpret 来跑

规则不再另立一层词汇。**一个规则就是一个 viba 程序**：`environ` 进、`bool` 出；判定就是
跑它——`is_compliant(rule_file, environ)`。

```python
from viba.compliance import is_compliant

is_compliant("rule_distance.viba", environ)     # -> Result[bool]
```

这么做的结果：

- **没有判定层**。witness <: rule 的那套子类型判定不在了；规则里的谓词是它自己的函数，
  判定结果是程序跑出来的 `bool`。
- **witness 也是程序**：`environ` 进、要判的材料出。材料就按它的类型写——tag 与积写在值
  位置上就是材料（`$victim ($x 0 * $y 0) * $at "12:30"`），不需要 `RuleObject`、`Metric`、
  `$value` 这些词。
- **不纯的一步留在程序里**：量距离要读时钟/服务/掷骰子，这件事仍然是宿主函数——但它现在
  必须把答案落成 **Prepare**，于是整次运行可以回放。

## 一个规则长什么样

```viba
# viba/compliance/demo/rule_distance.viba
import case_points as witness

Point := $x int * $y int
Case := $victim Point * $suspect Point * $at str

measure_distance :=
	int
	<- $env Environment
	<- $case Case
	<- { how far apart they were: not a pure function }

distance_at_least_5 :=
	bool
	<- $env Environment
	<- $d int
	<- { is that at least 5? }

__ret__ :=
	distance_at_least_5
	<< $env environ
	<< $d (
		measure_distance
		<< $env environ
		<< $case (witness << (environ.sub_env << "witness"))
	)
```

witness 那一份（`case_points.viba`）就是材料：

```viba
__ret__ :=
    $victim ($x 0 * $y 0)
  * $suspect ($x 3 * $y 4)
  * $at "12:30"
```

`measure_distance` 的实现（`viba/compliance/demo/host.py`）是宿主侧的一个普通函数，
它按 tag 读出 witness 里的坐标，算距离——**并且通过 `measure` 交出去**。

## Prepare：备份好的那次调用

`Prepare` 是「一次测量」的存档：**调用**（参数定了、结果声明了）与**量出来的值**。它在
storage 里就是一个文件：

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

它是**序列化的 viba 数据**，不是 pickle：人读得懂，也能解析回来当材料。

一次运行读 Prepare 的规矩（`viba/compliance/storage.py` 的 `PreparedStorage`）：

- **先看本轮 store，再看备份**：运行之前就备份好的那份（`prepare_root_dir`）是证据，
  本轮自己量出来的那份也在，两者都有时本轮那份优先；
- **写永远进本轮 store，不进备份**：备份是证据，运行不许改它；
- 备份是由一次**专门的运行**产生的：`prepare_run(rule_file, environ)` 跑一遍规则，把它量
  过的 Prepare 记进备份。之后任何一次 `is_compliant` 都只读它。

于是：

| | 量距离 | 谓词 | 结果 |
|---|---|---|---|
| 第一次跑（无备份） | 走了一次（0,0)-(3,4) | 看到 5 | `Ok(true)` |
| `prepare_run` | 回放（刚量过的那份） | 看到 5 | `Ok(true)`，并写下备份 |
| 之后每次（有备份） | **一次都不走**（回放备份里的 5） | 看到 5 | `Ok(true)` |

`tests/test_compliance.py` 就是这三行；`tests/data/compliance/backup/` 里放着一份**事先
备份好**的 Prepare，运行只读它——那条不纯的路一次都不走。

## 宿主侧看到的东西

- `measure(environ, name, call, compute)`：量 `name` 这次调用。有 Prepare 就回放，没有再
  算（`compute(call)` 是不纯那一步），然后把值写进本轮 store。**Prepare 里的调用压过这里
  写的调用**——被固定的那次调用才算数。
- `is_compliant(rule_file, environ) -> Result[bool]`：跑规则，读它的 `bool`；答的不是 bool
  就是 `Err`（"a verdict is a bool"）。
- `prepare_run(rule_file, environ) -> Result[bool]`：跑一遍，并把量过的 Prepare 记进备份。
- `read_prepare` / `record_prepare` / `measured_of` / `call_of` / `prepare_path`：直接读写
  Prepare 的时候用。

## 与旧规则层的关系

`viba-rule.md` 与 `viba/rule/*`（`RuleObject`、`Metric`、`Predicate`、`PredicationFailed`、
`not`、`generate_witness`、`is_compliant`、`check_rule_coding_style`、规则层的 reflect）是
**旧做法**，已废弃：判定要另立一套词，值靠 witness 自己写下来。现在的做法是上面的程序 +
Prepare，旧的那层不再演进。

旧套件的去向（迁移进行中，一代一代搬）：

| 旧套件 | 新家 |
|---|---|
| `test_rule_demo.py` | `test_compliance.py`（demo 与 Prepare 回放） |
| `test_rule_refutation.py` | 谓词答 false 的规则（同一套跑法，不再需要 `not[oneof]` 的判据） |
| `test_rule_reflect_api.py`、`test_rule_reflect_corpus.py` | witness 的材料按 `viba.reflect` 读，也就是核心协议（`test_reflect_edges.py`、`test_serialize.py`）那两套的写法 |
| `test_rule_coding_style_check.py`、`test_check_rule_coding_style.py` | 规则是程序这件事由 interpret 本身保证（`$env` 必须给、`__ret__` 必须是值），检查项收成 `test_compliance.py` 里的拒绝用例 |

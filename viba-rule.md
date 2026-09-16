# Viba.Rule 使用

## 1. 引言

Rule 不是 Viba 的新机制，只是 Viba 类型的一个应用。

一条规则（Rule）是以 `Metric` 与字面量为叶子的 Viba 类型描述，声明材料必须满足的结构与断言；一份呈证材料（Witness）也是一个 Viba 类型。判定就是子类型判定 `witness <: rule`。规则层用到的 `RuleObject` / `OneofRule` 标记与 `Predicate` / `Metric` / `PredicationFailed` / `not` 都只是 Viba 类型写法。

Rule 里可以带谓词（Predicate）与禁止性字段（`not`），书写形式见第 4 节与第 7 节。

## 2. 规则标记

规则以标记开头：`RuleObject` 或 `OneofRule`。

- `RuleObject` 与 `Object` 完全同义：积类型单位元，基数为 1。规则体为积类型时使用。
- `OneofRule` 与 `Oneof` 完全同义：和类型单位元，基数为 0。规则体为和类型时使用；其分支必须是其他规则的引用。

标记只是名字层面的声明：判定语义上 `RuleObject` 就是 `Object`，`OneofRule` 就是 `Oneof`。它唯一的额外含义是：一个 module 中，只有带标记的定义才会被认为是规则；其余定义（Metric 名字、辅助类型）不参与规则枚举。

## 3. 字段顺序纪律

积类型中，不带标签的字段一律放在前面，带标签的字段放在其后。

带标签的字段按名字对位，与书写顺序无关；不带标签的字段按位置对位。将不带标签的字段集中在前面，保证按位置对位的段落有稳定的顺序，不受带标签字段增删的影响。

## 4. Predicate 的书写形式

`Predicate` 是内建泛型（定义见 `viba/builtin.viba`），节点必须携带两个参数：

- 第一个参数 `{...}`：条件的自然语言描述（CodeBlock），不参与执行；
- 第二个参数 `$python_code {...}`：可执行的判定器代码。

`$python_code` 代码块内的花括号必须成对出现（CodeBlock 词法约束）。

判定器代码必须定义名为 `predicate` 的函数作为入口。`predicate` 恰好接受一个参数，参数名约定为 `self`：`self` 的属性与 Rule 中带标签的字段按名字对应（标签去掉 `$` 前缀），度量字段的实测值取其 `.value` 属性，嵌套的积字段按属性继续展开，元组字段按下标访问。`predicate` 返回布尔值：返回真值表示断言成立，返回假值表示断言不成立。

## 5. Metric 的书写形式

度量写作 `Metric[Name]`。`Metric[Name]` 是内建透传泛型（定义见 `viba/builtin.viba`）：度量的语义完全由 `Name` 承载。在判定中，`Metric` 只考虑纯数据类型：基类型（`bool` / `int` / `float` / `str`）及其容器（`list[T]`、`set[T]`、`dict[K, V]`）、单个带标签字段（如 `$tag T`）与积类型。函数、`Predicate` 等其他语义在判定中完全被忽视。

## 6. 断言与合取的书写形式

断言只允许出现在积类型中，作为带标签的字段。

多个约束同时成立时，必须使用积类型书写。若使用和类型，由于和的语义是"任一分支成立即成立"，断言字段可以被其他分支满足，导致断言失效不被发现。

## 7. 禁止性字段：`not`

禁止性约束写成带标签字段的 `not[...]`：

```viba
NoDeathPenaltyRule :=
  RuleObject
  * $not_crimes
      not[
        $homicide Homicide
        | $arson Arson
        | $robbery Robbery
      ]
```

`not[A]` 就是 `never <- A`（定义见 `viba/builtin.viba`）。上面每个分支的 `Homicide` / `Arson` / `Robbery` 是 Predicate 字段类型。

判定层对应三条：外壳（同 tag、全 `PredicationFailed`）成立；写成 `never <- B` 的按指数类型比（`never <- B <: never <- A` 当且仅当 `A <: B`）；写成带 tag 的积的按 `never <- (A | B) = (never <- A) * (never <- B)` 逐分支要求否证。

`PredicationFailed` 只在禁止性分支被要求；正向 Predicate 字段写它则表示断言不成立。

## 8. Witness（呈证材料）

Witness 与 Rule 结构平行：带 tag 的字段按 tag 一一对位，只有叶子按规则要求换掉。

- **标记**：Rule 用 `RuleObject` / `OneofRule` 起头；Witness 用同义的 `Object` / `Oneof` 起头，不带标记（带了会被当成规则枚举）。
- **Metric 字段**：Rule 写 `$len Metric[Len]`，Witness 填实测值，例如 `$len 42`；容器与带 tag 的积同理。
- **Predicate 字段**：断言成立就照写 `Predicate[...]`；断言不成立就写 `PredicationFailed[...]`。
- **禁止性字段**：保持 `not[oneof]` 外壳，只把每个分支的叶子换成 `PredicationFailed[...]`（见第 7 节）。
- **判定**：`is_compliant(witness, rule)`，True 是材料满足规则，False 是不满足。

一句话：Rule 说“要什么”，Witness 说“实际是什么”，两者之间只有子类型关系。

```text
Rule        := RuleObject * $len Metric[Len] * $check Predicate[{...}, $python_code {...}]
Witness     := Object     * $len 42          * $check Predicate[{...}, $python_code {...}]   # 断言成立
FailWitness := Object     * $len 42          * $check PredicationFailed[nil, str]            # 断言不成立
```

`PredicationFailed` 是毒剂：它在正向位置永不入席，所以一个字段从 `Predicate[...]` 变成 `PredicationFailed[...]`，整份材料就不满足这条规则。

禁止性规则对应的 Witness 保持 `not[oneof]` 形状：

```viba
LawAbidingWitness :=
  Object
  * $not_crimes
      not[
        $homicide PredicationFailed[nil, str]
        | $arson PredicationFailed[nil, str]
        | $robbery PredicationFailed[nil, str]
      ]
```

少一个分支、或某个分支留成正向的 `Homicide`，都判 `False`。

## 9. 书写示例

以下代码块是合法的 Viba 程序：

```viba
CodeLength := int
CyclomaticComplexity := int
IndentWidth := $spaces int
DocCoverage := $documented_lines int * $total_lines int
Keywords := list[str]

DemoRule :=
  RuleObject
  * $code_length Metric[CodeLength]
  * $assert_code_len_le_24
      Predicate[{code length <= 24}, $python_code {
def predicate(self):
    return self.code_length.value <= 24
}]
```

这个代码块由 `tests/test_check_rule_coding_style.py` 提取并执行验证：解析、反解析回环、子类型判定与书写规范检查。

## 10. 怎么用

规则层的 API 都在 `viba.rule`：

```python
from viba.rule import (
    generate_witnesses,
    reset_predication_by_python_code,
    is_compliant,
    check_rule_coding_style,
    check_determinate,
)
```

- `generate_witnesses(rule, count, seed=None, fail_prob=0.1)` — 按规则生成 `count` 份随机 Witness。`fail_prob` 是每个 `Predicate` 字段（以及每个 `not` 分支）被翻成违规形态的概率。
- `reset_predication_by_python_code(witness)` — 执行 Witness 里每个 `Predicate` 的 `$python_code`，返回假的换成 `PredicationFailed[...]`。配合 `fail_prob=0` 使用，就是让代码而不是随机数决定哪些断言不成立。
- `is_compliant(witness, rule)` — 判定 `witness <: rule`，返回 `Ok(True)` 或 `Ok(False)`。
- `check_rule_coding_style(rule)` — 检查规则是否符合本文档的书写规范，合规返回 `Ok(None)`，否则 `Err(第一条违规)`。
- `check_determinate(rule, count, seed=None)` — 先做书写规范检查，再以 `fail_prob=0` 生成 Witness、逐份执行 predicate 代码并判定；全部跑通返回 `Ok(None)`，任一份报错或 predicate 抛异常则返回 `Err`。

一个最小流程：

```python
from pathlib import Path

from viba.rule import (
    check_determinate,
    generate_witnesses,
    is_compliant,
    reset_predication_by_python_code,
)
from viba.type import AstNodeType, custom_module

module = custom_module(Path("rule.viba").read_text())
defs = {d.name: d for d in module.module.body}
rule = AstNodeType(defs["DemoRule"].body, module)

check_determinate(rule, 50)   # Result[None]：规范 + 代码 + 每份 witness 都判得出来

for witness in generate_witnesses(rule, 20, seed=1, fail_prob=0.0):
    witness = reset_predication_by_python_code(witness)   # 跑 predicate 代码
    print(is_compliant(witness, rule))                    # Ok(True) / Ok(False)
```

语料与更完整的回归在 `tests/test_rule_coding_style_check.py`：40 条 Metric/Predicate 规则、20 条 `not` 规则、坏 predicate 样例，以及生成 → 执行 → 判定的比例断言。

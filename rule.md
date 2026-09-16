# Rule 书写规范

## 1. 引言

本规范定义 Rule 的书写形式。Rule 是以 Metric 与字面量为叶子的 Viba 类型描述，声明材料必须满足的结构与断言。

Rule 中可以携带断言（Assert）。携带断言时，断言必须满足第 4 节规定的书写形式。

## 2. 规则标记

规则以标记开头：`RuleObject` 或 `OneofRule`。

- `RuleObject` 与 `Object` 完全同义：积类型单位元，基数为 1。规则体为积类型时使用。
- `OneofRule` 与 `Oneof` 完全同义：和类型单位元，基数为 0。规则体为和类型时使用；其分支必须是其他规则的引用。

标记只是名字层面的声明：判定语义上 `RuleObject` 就是 `Object`，`OneofRule` 就是 `Oneof`。它唯一的额外含义是：一个 module 中，只有带标记的定义才会被认为是规则；其余定义（Metric 名字、辅助类型）不参与规则枚举。

## 3. 字段顺序纪律

积类型中，不带标签的字段一律放在前面，带标签的字段放在其后。

带标签的字段按名字对位，与书写顺序无关；不带标签的字段按位置对位。将不带标签的字段集中在前面，保证按位置对位的段落有稳定的顺序，不受带标签字段增删的影响。

## 4. Assert 的书写形式

`Assert` 是内建泛型（定义见 `viba/builtin.viba`），节点必须携带两个参数：

- 第一个参数 `{...}`：条件的自然语言描述（CodeBlock），不参与执行；
- 第二个参数 `$python_code {...}`：可执行的判定器代码。

`$python_code` 代码块内的花括号必须成对出现（CodeBlock 词法约束）。

判定器代码必须定义名为 `handler` 的函数作为入口。`handler` 恰好接受一个参数，参数名约定为 `self`：`self` 的属性与 Rule 中带标签的字段按名字对应（标签去掉 `$` 前缀），度量字段的实测值取其 `.value` 属性，嵌套的积字段按属性继续展开，元组字段按下标访问。`handler` 返回布尔值：返回真值表示断言成立，返回假值表示断言不成立。

## 5. Metric 的书写形式

度量写作 `Metric[Name]`。`Metric[Name]` 是内建透传泛型（定义见 `viba/builtin.viba`）：度量的语义完全由 `Name` 承载。在判定中，`Metric` 只考虑纯数据类型：基类型（`bool` / `int` / `float` / `str`）及其容器（`list[T]`、`set[T]`、`dict[K, V]`）、单个带标签字段（如 `$tag T`）与积类型。函数、`Assert` 等其他语义在判定中完全被忽视。

## 6. 断言与合取的书写形式

断言只允许出现在积类型中，作为带标签的字段。

多个约束同时成立时，必须使用积类型书写。若使用和类型，由于和的语义是"任一分支成立即成立"，断言字段可以被其他分支满足，导致断言失效不被发现。

## 7. 书写示例

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
      Assert[{code length <= 24}, $python_code {
def handler(self):
    return self.code_length.value <= 24
}]
```

## 8. 验证

第 7 节的代码块由 `tests/test_rule_spec.py` 提取并执行验证，验证内容包括解析、反解析回环与子类型判定。

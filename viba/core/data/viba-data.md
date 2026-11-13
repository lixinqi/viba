# viba data：类型描述的那组值

viba 的类型不只是在说"该有哪些字段"：一个类型描述的是一组**值**，而字面量让一个值本身也能写成类型——
`$x 1 * $y 2` 是一个类型，它只有一个值。所以 viba 既能写数据，也能说数据该长什么样，用的是同一套源码。

这一层讲的是这个：一个值是什么、怎么从类型里取出来、什么时候一份设计真的在描述数据，以及它和 JSON 的关系。

## 1. 一个值是什么

一个值是有限的一棵树：

- **叶子**是字面量：`1`、`2.5`、`"a"`、`true`，以及语言自己的标量（`int` 与 `int8_t` …
  `uint64_t` 这些整数、`float`、`str`、`bool`）。
  单位元（`nil`、`never`、`Object`、`Oneof`）也算走到了头——它们没有内容。
- **每一层由 tag 或位置命名**：积的成员各有 tag（`$x`、`$y`），和的分支也各有 tag（或按值的种类分开），
  元组与容器按位置。
- **容器**是内建字面量：`ListLiteral[1, 2]`、`SetLiteral["a"]`、`DictLiteral[("k", 1)]`，还有元组。
- **提示块不是值**：`{...}` 是留给实现的那一步，它本身没有内容，要有人把它换成实际的值。

同一个类型下可以有很多值（`$x int * $y int` 的每个整数对都是），也可以只有一个（`$x 1 * $y 2`）。

## 2. 值也能写成类型

字面量是终结符，所以一个具体的值可以整份写成类型：

```viba
Point = $x 1 * $y 2          # 只有一个值：x 是 1、y 是 2
Mode  = "fast" | "slow"      # 两个值，按字面量分开
Empty = nil                  # 一个值：什么都没有
```

反过来，`Point = $x int * $y int` 描述的是所有整数对。两种写法在源码里没有区别——定义是类型，类型
描述值；一个定义说的是一个值还是很多个值，只看它的体。

## 3. 怎么取出来

取一个值要两样东西：类型侧的一段描述（描述符），和实现侧的一份数据；合起来是一格节点（`VObject`）。
从节点出发一次走一步（`VibaStep`）——一步是一个 tag、一个字段序号或一个容器位置——走到叶子取到字面量。

- 协议本身：[`viba-reflect.md`](viba-reflect.md)（`VibaRoot` / `VibaHas` / `VibaGet` / `VibaLeaf` /
  `VibaLength` / `VibaKeys`，以及实现方要交付的那一份）。
- 反过来把一个值写回源码：[`viba-serialize.md`](viba-serialize.md)。

## 4. 什么时候一份设计是在描述数据

一条数据路径是类型侧列得出来的一条地址：从某个定义出发，每一步都有成员可走，最后落在叶子上。一份设计
能反射到底，就是每个定义、每条数据路径都落得到叶子；这一问不碰任何实例，只答 true / false
（[`viba-is-complete.md`](viba-is-complete.md)）。

走不通的几处：

- `Any`：没有成员，也没有叶子。
- 提示块（`Hint[...]`、裸的 `{...}`、`...`）：内容等着实现，不在类型里。
- 泛型决断失败，或者形参还没绑上实参：这一遍说不出那一段是什么。
- 名字解析不到定义。

## 5. 和 JSON 的关系

JSON 的生态位是值：object、array、数字、字符串、布尔、null。viba 的值族比它宽：

| JSON | viba |
|------|------|
| object（键是字符串） | 积（成员是 tag），或 `dict[K, V]` |
| array | `list[T]`、`set[T]`、元组 |
| 数字 / 字符串 / 布尔 | 字面量与标量 |
| null | 单位元 `nil` |

多出来的是：和的标签、字面量类型（值本身就是类型）、单位元、元组、`set`。两处硬差别：JSON 的键只能
是字符串，viba 的 tag 是标识符；JSON 没有和，viba 的分支靠 tag 或值的种类分开。

所以一个 viba 类型能比一份 JSON Schema 说得更细；反过来能不能落成 JSON、落成什么样，是另外两章的事：
[`viba-check-convertable-to-json-schema.md`](../../wiki/viba-check-convertable-to-json-schema.md)、
[`viba-convert-to-json-schema.md`](../../wiki/viba-convert-to-json-schema.md)。

## 6. 名字

这一层叫 **viba data**，中文写**可序列化数据**——代码里是 `VibaData`、`viba_data()`、`as_viba_data()`。
它说的是"类型描述的那组值"，不是某个具体的实例，也不是某个 Python 对象。

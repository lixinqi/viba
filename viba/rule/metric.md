# Metric：度量、取证、追溯

`Metric` 把"一次度量"写成一条调用：函数是 `CoreFunc`，证据是这次调用的参数与结果。

## 1. 目标

1. **判断逻辑可追溯**：一个度量字段为什么成立、为什么不成立，从类型上看得出来。
2. **参数与结果可追溯**：度量不是"一个数"，是"一次调用"——参数和结果都写下来，
   之后用 reflect 逐地址取。
3. **约束明确**：什么能当度量，有一条写下来的接口。

## 2. 接口

```viba
Result[T] :=
    Oneof
  | $ok ($ok_value T)
  | $err ($err_msg str)

JsonLike :=
    Oneof
  | nil
  | bool
  | int
  | float
  | str
  | list[JsonLike]
  | set[JsonLike]
  | dict[str, JsonLike]

MetricFuncInterface := Result[JsonLike] <- never
```

- `Result` 用 reflect 协议里那一份：`$ok` 带值，`$err` 带说明。
- `JsonLike` 是 reflect 读得出叶子的那一族形状：叶子是 `bool` / `int` / `float` /
  `str` / `nil`，外面套容器或和。
- `MetricFuncInterface` 要求结果是 `Result[JsonLike]`；参数位写 `never`，谁都收得下
  它，所以参数的个数与形状都不构成障碍。例如 `GetDistance <: MetricFuncInterface`。

## 3. 取证：一次调用

```viba
# demo
DemoPoint := ($x int * $y int)

# definition
GetDistance := Result[int] <- DemoPoint <- DemoPoint

# before call：取证准备，先把参数定下来
Result[int] <- ($x 0 * $y 0) <- ($x 3 * $y 3)

# after call：固定证据
Result[9] <- ($x 0 * $y 0) <- ($x 3 * $y 3)
```

取证的支点：**同一段语法既是类型又是数据**。往右当类型，跟 `GetDistance` 比；往左当
数据，被 reflect 逐地址取。证据不需要第二种数据结构——after call 就是最终证据，
输入与输出都在里面。留 `before` 是因为取证分两步：先把参数定下来，再把结果填上固定。

## 4. 度量对象

```viba
# 度量基类
MetricObject := Object * $__metric_object_you_are_not_allowed_to_use_this_tag_name__ nil

Metric[CoreFunc] :=
    MetricObject
  * $func CoreFunc
  * Assert[{
      CoreFunc <: (Result[JsonLike] <- never)
      当取证的时候，CoreFunc的输入输出参数必须填上真实的值。
    }]
```

- 那个 tag 是**名义锚**：它不属于任何业务字段，保留给度量对象自己。`MetricObject`
  是不带标签的成员，按内联规则摊进 `Metric[CoreFunc]`，于是度量的成员就是：保留 tag
  那一位（`nil`）+ `$func`。
- 呈证材料里的度量字段因此是一份"带标记的证据"：

```viba
* $distance (
    $__metric_object_you_are_not_allowed_to_use_this_tag_name__ nil
    * $func (Result[9] <- ($x 0 * $y 0) <- ($x 3 * $y 3))
  )
```

## 5. 判定

一个度量字段成立，要同时满足：

1. **形状**：标记那一位（`nil`）+ `$func` 底下的一份证据——标记保证"这是一个度量"，
   不会被别的同形数据顶替。
2. **接口**：`CoreFunc <: MetricFuncInterface`，也就是 `Assert` 里写的第一句。
3. **说明块**：`Assert[{...}]` 是写下来的说明，不是类型，判定不看它。
4. **证据**：每位协变——每个参数的值属于声明的参数类型，结果的值属于声明的结果类型，
   例如 `($x 0 * $y 0) <: DemoPoint`、`Result[9] <: Result[int]`。
5. **值**：证据里每个叶子必须是字面量，不能是类型名。

第 4 条是"这次调用合不合法"，与"函数之间能不能顶替"是两回事，由单独一条读法/检查
承担（见第 8 节）。第 5 条判定做不到（`9 <: int` 与 `int <: int` 同形），同样是单独
一条检查：走过证据的每个地址，落到叶子时必须是字面量。有第 5 条，`JsonLike` 的意义
才落到实处——叶子落在它的原子里，reflect 才保证取得到东西。

## 6. 追溯

第 4 节那份证据的地址与叶子（`by_field_index(0)` 是结果位；参数的书写顺序就是地址
顺序）：

| 地址 | 叶子 |
|---|---|
| `$marker` | `nil` |
| `$func . by_field_index(0) . by_tag($ok) . by_tag($ok_value)` | `9` |
| `$func . by_field_index(1) . by_tag($x)` / `. by_tag($y)` | `0` / `0` |
| `$func . by_field_index(2) . by_tag($x)` / `. by_tag($y)` | `3` / `3` |

参数起了 tag 时（`<- $p0 DemoPoint`），地址从 `by_field_index(i)` 换成 `by_tag($p0)`。
谓词读测量值：走 `$func` 的结果位，再走 `$ok` / `$ok_value`。

## 7. 度量函数的写法

一个 `CoreFunc` 要当度量函数，需要满足：

1. 结果位是 `Result[JsonLike]`（第 5 节第 2 条）；
2. 参数与结果都写成可读的数据形状：叶子落在 `JsonLike` 的原子上（第 5 节第 5 条）；
3. 取证时参数与结果都填真实的值，不留类型名。

参数的写法自由：带 tag（`$p0 DemoPoint`）按名字取，不带 tag 按位置取；参数量不限。

## 8. 尚未确定

1. **证据的合法性怎么判**：判定层单开一条读法（每位协变），还是单独一条检查
   （第 5 节第 4 条）。
2. **"每个叶子必须是字面量"怎么保证**：一条单独检查的入口（第 5 节第 5 条）。
3. **`Metric[T] := $value T` 的去留**：换掉、并存，还是把 `$value` 留成糖。
4. **证据要不要写成定义**：每个度量字段都抄一遍保留 tag 太吵，可以让证据先写成一个
   定义，`$func` 那里引这个名字。
5. **度量函数的实现与枚举**：核心函数与实现怎么对应、谁发起调用并写下证据、"哪些
   定义是度量函数"要不要一条标记。

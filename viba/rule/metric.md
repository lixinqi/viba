# Metric：度量、取证、追溯

Rule 里的 `Metric` 目前是"一个值"：`Metric[T] := $value T`（见 `viba/builtin.viba`），
呈证时写 `$len ($value 42)`，谓词读 `.value`。这份文档把它换成"**一次调用**"：度量是
一个函数，证据是这次调用的参数与结果，全部写在类型里。

## 1. 目标

1. **判断逻辑可追溯**：一个度量字段为什么成立、为什么不成立，从类型上看得出来，
   不依赖判定之外的约定。
2. **参数与结果可追溯**：度量不是"一个数"，是"一次调用"——参数和结果都写下来，
   之后用 reflect 逐地址取。
3. **约束明确**：什么能当度量，有一条写下来的接口，不靠口头约定。

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

- `Result` 沿用 `viba-reflect.md` 第 5.1 节那一份（Python 侧是 `viba.type` 的 `Ok` /
  `Err`），不另立。
- `JsonLike` 是"reflect 读得出叶子"的那一族形状：叶子是 `bool` / `int` / `float` /
  `str` / `nil`，外面套容器或和。
- `MetricFuncInterface := Result[JsonLike] <- never` 读作"**结果**必须是
  `Result[JsonLike]`，参数怎么写都行"：`never` 在参数位是底，谁都收得下它；指数链
  比较时链长按"短了补 `never`、长了截掉"处理，所以参数的个数与形状都不构成障碍。
  于是"结果为 `Result[JsonLike]`"的函数都满足这条接口，例如
  `GetDistance <: MetricFuncInterface`。

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
输入与输出都在里面。

留 `before` 是因为取证分两步：先把参数定下来，再把结果填上固定。

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

一个度量字段成立，要同时满足下面几条。

### 5.1 形状

`Metric[CoreFunc]` 的居民 = 标记那一位（`nil`）+ `$func` 底下的一份证据。标记保证
"这是一个度量"，不会被别的同形数据顶替。

### 5.2 接口

`CoreFunc <: MetricFuncInterface`，也就是 `Assert` 里写的第一句：结果必须是
`Result[JsonLike]`（例如 `Result[int] <: Result[JsonLike]`）。

### 5.3 说明块要旁路

`Assert[{...}]` 这类块是**写下来的说明**，不是类型。判定不能去解析它，否则一个未定义
的名字会让整条判定变成 `Err`。

判定因此收一个 `VibaReflectConfig`（就是地址层读设计用的那个值）：`nil_eqv` /
`never_eqv` 里的名字算单位元，**裸名与应用同名同权**——`Assert` 在 `nil_eqv` 里时，
`Assert[{...}]` 就当 `nil` 读，判定不去解析它；不带标签的 `Assert[{...}]` 成员因此
不算成员。什么名字都不给时，判定保持原样。

### 5.4 证据的合法性

一次调用合不合法，逐位协变地看：每个参数的值属于声明的参数类型，结果的值属于声明的
结果类型。例如 `($x 0 * $y 0) <: DemoPoint`、`Result[9] <: Result[int]`。

这一条**不是**指数判定那条规则：指数规则比的是"函数之间能不能顶替"，参数位逆变，
`证据 <: CoreFunc` 会因为 `DemoPoint <: ($x 0 * $y 0)` 里的 `int <: 0` 而不成立。两者
不可互推，所以证据的合法性由单独一条读法/检查承担（落在哪一层见第 9 节）。

### 5.5 值必须是写下来的

证据里的每个叶子必须是字面量（`bool` / `int` / `float` / `str` / `nil`），不能是类型
名。判定做不到这一条——`9 <: int` 与 `int <: int` 都成立，结构与值在判定里同形——所以
它是一条单独的检查：走过证据的每个地址，落到叶子时必须是字面量（见第 9 节）。

有这一条，第 2 节 `JsonLike` 的意义才落到实处：叶子限定在它的原子里，reflect 才保证
取得到东西。

## 6. 追溯

第 4 节那份证据的地址与叶子（`by_field_index(0)` 是结果位；参数的书写顺序就是地址
顺序）：

| 地址 | 叶子 |
|---|---|
| `$marker` | `nil` |
| `$func . by_field_index(0) . by_tag($ok) . by_tag($ok_value)` | `9` |
| `$func . by_field_index(1) . by_tag($x)` / `. by_tag($y)` | `0` / `0` |
| `$func . by_field_index(2) . by_tag($x)` / `. by_tag($y)` | `3` / `3` |

参数起了 tag 时（`<- $p0 DemoPoint`），地址从 `by_field_index(i)` 换成
`by_tag($p0)`；判定不受影响，因为接口的参数位是 `never`。

谓词读测量值：老路径是 `Metric[T] := $value T` 那一层的 `.value`；新路径是 `$func`
的结果位再走 `$ok` / `$ok_value`。

## 7. 度量函数的写法

一个 `CoreFunc` 要当度量函数，需要满足：

1. 结果位是 `Result[JsonLike]`（接口，第 5.2 节）；
2. 参数与结果都写成可读的数据形状：叶子落在 `JsonLike` 的原子上（第 5.5 节）；
3. 取证时参数与结果都填真实的值，不留类型名。

参数的写法自由：带 tag（`$p0 DemoPoint`）按名字取，不带 tag 按位置取；参数量不限，
接口那边由 `never` 与链长规则兜住。

## 8. 与现状的关系

| 现状 | 本文 |
|---|---|
| `Metric[T] := $value T`（`viba/builtin.viba`） | `Metric[CoreFunc] := MetricObject * $func CoreFunc * Assert[{...}]` |
| 度量是"一个值"，谓词读 `.value` | 度量是"一次调用"，读 `$func` 的结果位 `$ok` / `$ok_value` |
| 证据只有一个值（`$len ($value 42)`） | 证据是完整的调用：每个参数 + 结果 |
| 数据形状由 `T` 决定，没有接口 | 数据形状由 `CoreFunc` 决定，接口是 `MetricFuncInterface` |
| 值与参数不可追溯 | 每个参数与结果都有地址（第 6 节） |

## 9. 尚未确定

1. **证据的合法性落在哪一层**：判定层新开一条读法（逐位协变），还是校验层一条检查
   （第 5.4 节）。
2. **"值必须是字面量"这条检查放哪**：与 `check_tag_and_inline` 同路的校验层入口
   （第 5.5 节）。
3. **判定拿到的单位表含哪些名字**：`Predicate` / `PredicationFailed` / `RuleObject`
   这些**有定义**的名字当单位元会改变现有判定语义（它们现在是积）。整份 config，还是
   只含说明块（`Assert` / `Hint` / `Appendix`）的一份。
4. **`Metric[T] := $value T` 的去留**：换掉、并存，还是把 `$value` 留成糖（第 8 节）。
5. **证据要不要起名字**：每个度量字段都抄一遍保留 tag 太吵；可以让证据先写成一个
   定义，`$func` 那里引这个名字。
6. **度量函数的实现与枚举**：核心函数与实现（Python）怎么对应、谁发起调用并写下
   证据；"哪些定义是度量函数"要不要一条标记或一条 `check_metric`。

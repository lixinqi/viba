# Metric：度量、取证、追溯

Rule 里的 `Metric` 现在只是"一个值"（`Metric[T] := $value T`，见 `viba/builtin.viba` 与
`viba-rule.md` 第 4、5 节）。这份文档要把它换成"**一次调用**"：度量是一个函数，
证据是这次调用的参数与结果，全部写在类型里。

## 1. 要什么

1. **判断逻辑可追溯**：一个度量字段为什么成立、为什么不成立，从类型上就该看出来，
   不藏在 Python 的约定里。
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

- `Result` 用现成的那份（`viba-reflect.md` 第 5.1 节就有；Python 侧是 `viba.type` 的
  `Ok` / `Err`），不另立一份。
- `JsonLike` 是"reflect 读得出叶子"的那一族形状：叶子是 `bool` / `int` / `float` /
  `str` / `nil`，外面套容器或和。
- `MetricFuncInterface := Result[JsonLike] <- never` 读作"**结果**必须是
  `Result[JsonLike]`，参数随便写"：`never` 在参数位是底（谁都收得下它），链长又按
  "sub 短了补 `never`、长了截掉"这条规则比，所以具体函数写几个参数都不影响。
  实测：`GetDistance <: MetricFuncInterface` 为真。

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

支点是这一条：**同一段语法既是类型又是数据**。往右当类型，跟 `GetDistance` 比；
往左当数据，被 reflect 逐地址取。所以取证不需要第二种数据结构——after call 就是最终
的证据，输入和输出都在里面。

`before` 也留着：取证从"参数先定"开始，`after` 才是固定下来的那一份。

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

- 那个长 tag 是**名义锚**：谁都别拿它当自己的字段名。`MetricObject` 是不带标签的
  成员，按内联规则摊进来，于是 `Metric[CoreFunc]` 的成员就是：保留 tag 那一位
  （`nil`）+ `$func`。
- 呈证材料里的度量字段因此长这样（实测通过，成员与地址见第 6 节）：

```viba
* $distance (
    $__metric_object_you_are_not_allowed_to_use_this_tag_name__ nil
    * $func (Result[9] <- ($x 0 * $y 0) <- ($x 3 * $y 3))
  )
```

- 每次写证据都要把那个长 tag 抄一遍（见第 7 节第 5 条）。

## 5. 判定

命中两件事：

1. **形状**：`Metric[CoreFunc]` 的居民 = 标记 + `$func` 的证据。
2. **接口**：`CoreFunc <: MetricFuncInterface`（就是 `Assert` 里写的那句话）。

接口那条实测是好的：`GetDistance <: MetricFuncInterface` 为真，`Result[int] <:
Result[JsonLike]` 为真。

**"证据算不算这个函数的一次调用"这一句，现在没有对应的判定**：

- 证据链写在 `$func` 底下，而 `$func` 的**类型**是 `CoreFunc`，所以判定会在那一格拿
  证据链跟 `CoreFunc` 比，走的是指数那条规则（参数逆变）：好调用过不了——每个位置
  单独看都是对的（实测 `($x 0 * $y 0) <: DemoPoint` 为真、`Result[9] <: Result[int]`
  为真），但整条链 `证据 <: CoreFunc` 会因为 `DemoPoint <: ($x 0 * $y 0)` 里的
  `int <: 0` 而 False。
- "一次调用是这个函数的合法调用"要的是每一位**协变**：每个参数的值属于声明的参数
  类型，结果的值属于声明的结果类型。这跟"函数之间能不能顶替"是两件事，所以要单开
  一条读法/检查（见第 7 节第 2 条）。

## 6. 追溯

拿第 4 节那份证据走 reflect，地址与叶子是（`by_field_index(0)` 是结果位，参数的书写
顺序就是地址顺序）：

```text
$marker                                                  -> nil
$func . by_field_index(0) . by_tag($ok) . by_tag($ok_value) -> 9
$func . by_field_index(1) . by_tag($x)                   -> 0
$func . by_field_index(1) . by_tag($y)                   -> 0
$func . by_field_index(2) . by_tag($x)                   -> 3
$func . by_field_index(2) . by_tag($y)                   -> 3
```

参数要是起了 tag（`<- $p0 DemoPoint`），地址就从 `by_field_index(1)` 换成
`by_tag($p0)`；判定那边不受影响（接口的参数位是 `never`，谁都收得下）。

谓词读测量值的老路径是 `.value`（`Metric[T] := $value T` 那一层）；新路径是
`$func` 的结果位再走 `$ok` / `$ok_value`。

## 7. 要讨论的

1. **`Result` / `JsonLike` 放哪**：`Result` 沿用现成的；`JsonLike` 是新的，我倾向
   放规则层的 `metric.viba`（度量概念属于规则层，core 不该认识它）。
2. **一次调用怎么判**（第 5 节）：证据链写在 `$func` 底下、类型是 `CoreFunc`，而
   "合法调用"要逐位协变，跟指数判定那条（参数逆变）不是一回事。要不要在判定层给它
   单开一条读法，还是把"证据合不合法"交给校验层（第 7 条那条检查）——你定。
3. **`Assert[{...}]` 这类说明块：已按"旁路"做掉**。`is_sub_type` 现在收
   `VibaReflectConfig`（就是地址层那个值），`nil_eqv` / `never_eqv` 里的名字算单位元，
   **裸名与应用同名同权**，于是 `Assert[{...}]` 就当 `nil` 读、判定不去解析它。实测：
   带上 config，`Metric[GetDistance] <: Metric[GetDistance]` 为真、`Assert[{x}] <: nil`
   为真、不带标签的 `Assert[{x}]` 成员直接不算成员；不带 config 仍是
   `Err("unresolvable constructor 'Assert'")`。
   剩下要定的是：规则层那份 config 里还列着 `Predicate` / `PredicationFailed` /
   `RuleObject` —— 那些**有定义**的名字一旦当单位元，判定语义会变（它们现在是积）。
   要不要给判定单开一份只含说明块的单位表（`Assert` / `Hint` / `Appendix`）？
4. **`JsonLike` 管到结果就够**：接口是 `Result[JsonLike] <- never`，参数位是 `never`
   （通配），所以 `DemoPoint` 这样的积当参数没问题，实测 `Result[int] <:
   Result[JsonLike]` 为真。真正要保证的是**叶子**：参数与结果里每个叶子都得是
   `JsonLike` 的原子（`bool` / `int` / `float` / `str` / `nil`）——这才是 reflect 取
   得到东西的那条保证，落到第 7 条那条检查里。
5. **证据要不要起个名字**：每个度量字段都抄一遍那个长 tag 太吵。可以让证据先写成
   一个定义（名字），`$func` 那里引这个名字；叶子还是字面量，"固定证据"这条不破。
6. **和 `Metric[T] := $value T` 的关系**：老写法（`$len ($value 42)`、谓词读 `.value`、
   语料里那批规则）怎么办——换掉、并存、还是把 `$value` 留成糖？这条定了才能动代码。
7. **"值必须是真的"谁来查**：判定做不到（`int` 和 `9` 在结构判定里同形：`9 <: int`
   和 `int <: int` 都成立），所以要一条单独的检查（每个叶子必须是字面量）。
   位置我倾向校验层，跟 `check_tag_and_inline` 一路。
8. **度量要不要可枚举**：像 `RuleObject` 那样，"哪些定义是度量函数"要不要有一条
   标记/查法，还是一条 `check_metric(rule)` 把接口那条 `Assert` 逐条验掉。

## 8. 现状对照

| 现在 | 这份设计 |
|---|---|
| `Metric[T] := $value T`（`viba/builtin.viba`） | `Metric[CoreFunc] := MetricObject * $func CoreFunc * Assert[{...}]` |
| 度量是"一个值"，谓词读 `.value` | 度量是"一次调用"，读 `$func` 的结果位 `$ok` / `$ok_value` |
| 证据只有一个值（`$len ($value 42)`） | 证据是完整的调用：每个参数 + 结果 |
| 数据形状由 `T` 决定，没有接口 | 数据形状由 `CoreFunc` 决定，接口是 `MetricFuncInterface` |
| 值与参数无法追溯 | 每个参数与结果都有地址（第 6 节） |

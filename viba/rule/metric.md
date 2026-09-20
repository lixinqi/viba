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

情景：一起案子，受害人与嫌疑人各有一个当时的坐标。要量的是两人在某一时刻相距多远。

```viba
# demo：人物坐标与时刻
PersonPoint := ($x int * $y int)
Moment := str

# definition：受害人、嫌疑人、时刻 → 两人当时的直线距离
GetDistance :=
    Result[int]
  <- $victim PersonPoint
  <- $suspect PersonPoint
  <- $at Moment

# prepare：取证准备——参数填真实的值，结果位留声明的类型
Prepare :=
    Result[int]
  <- $victim ($x 0 * $y 0)
  <- $suspect ($x 3 * $y 4)
  <- $at "12:30"

# after call：固定证据——12:30 受害人在 (0,0)、嫌疑人在 (3,4)，量出来相距 5
Result[5]
  <- (
      Result[int]
        <- $victim ($x 0 * $y 0)
        <- $suspect ($x 3 * $y 4)
        <- $at "12:30"
    )
```

取证的支点：**同一段语法既是类型又是数据**。after call 是数据：参数位放着 prepare
（被测调用的准备——参数已定，结果位留声明），结果位是度量量出来的值。取证的类型写作
`(Result[int] <- Prepare)`，判定见第 5 节；数据由 reflect 逐地址取，见第 6 节。

## 4. 度量对象

```viba
# 度量基类
# yanatuttn = you_are_not_allowed_to_use_this_tag_name
MetricObject := Object * $__metric_object_yanatuttn__ nil

Metric[CoreFunc] :=
    MetricObject
  * $func CoreFunc
  * $call_instance (Result[JsonLike] <- CoreFunc)
  * Appendix[{
      CoreFunc <: (Result[JsonLike] <- never)
      当取证的时候，CoreFunc的输入输出参数必须填上真实的值。
    }]
```

- 那个 tag 是**名义锚**：它不属于任何业务字段，保留给度量对象自己。`MetricObject`
  是不带标签的成员，按内联规则摊进 `Metric[CoreFunc]`，于是度量的成员就是：保留 tag
  那一位（`nil`）+ `$func` + `$call_instance`。
- `$func` 是设计侧的函数，`$call_instance` 是这次调用的实例。固定证据时不动 `$func`，
  只动 `$call_instance`。
- 呈证材料里的度量字段因此是一份"带标记的证据"：

```viba
* $distance (
    $__metric_object_yanatuttn__ nil
    * $func GetDistance
    * $call_instance (
        Result[5]
          <- (
              Result[int]
                <- $victim ($x 0 * $y 0)
                <- $suspect ($x 3 * $y 4)
                <- $at "12:30"
            )
      )
  )
```

## 5. 判定

一个度量字段成立，要同时满足：

1. **形状**：标记那一位（`nil`）+ `$func` + `$call_instance`——标记保证"这是一个度量"，
   不会被别的同形数据顶替。
2. **接口**：`CoreFunc <: MetricFuncInterface`，也就是 `Appendix` 里写的第一句。
3. **说明块**：`Appendix[{...}]` 是写下来的说明，不是类型，判定不看它。
4. **证据**：`$call_instance` 底下是一次调用——参数位是 `Prepare`，结果位是量出来的值：

   ```viba
   Result[5] <- Prepare

   Prepare :=
       Result[int]
     <- $victim ($x 0 * $y 0)
     <- $suspect ($x 3 * $y 4)
     <- $at "12:30"
   ```

5. **约束**：`(Result[int] <- Prepare) <: (Result[JsonLike] <- GetDistance)`——就是
   `$call_instance` 那一位的类型检查（第 4 节 `Metric[CoreFunc]` 里那一位声明为
   `Result[JsonLike] <- CoreFunc`）。

第 5 条就是普通的一次 `is_sub_type`，没有第二遍读法。值不进判定：它在数据里
（after call），由 reflect 逐地址取，见第 6 节。

## 6. 追溯

第 4 节那份证据的地址与叶子（`by_field_index(0)` 是结果位，`by_field_index(1)` 是被测
调用那一支；参数按 tag 寻址）：

| 地址 | 叶子 |
|---|---|
| `$__metric_object_yanatuttn__` | `nil` |
| `$call_instance . by_field_index(0) . by_tag($ok) . by_tag($ok_value)` | `5` |
| `$call_instance . by_field_index(1) . by_tag($victim) . by_tag($x)` / `. by_tag($y)` | `0` / `0` |
| `$call_instance . by_field_index(1) . by_tag($suspect) . by_tag($x)` / `. by_tag($y)` | `3` / `4` |
| `$call_instance . by_field_index(1) . by_tag($at)` | `"12:30"` |

`$func` 那一位不进这张表：它一直是设计函数的名字，没有叶子。

被测调用自己的结果位（`$call_instance . by_field_index(1) . by_field_index(0)`）停在
声明的 `Result[int]` 上，没有叶子——那是 `Prepare` 留下的。

参数不带 tag 时按位置取（`by_field_index(i)`）。谓词读测量值：走 `$call_instance` 的
结果位，再走 `$ok` / `$ok_value`。

## 7. 度量函数的写法

一个 `CoreFunc` 要当度量函数，需要满足：

1. 结果位落在 `Result[JsonLike]` 里（第 5 节第 2 条）；
2. 参数与量的值都写成可读的数据形状：叶子落在 `JsonLike` 的原子上（第 6 节）；
3. 固定证据时只动 `$call_instance`：`$func` 留设计函数的名字，`$call_instance` 的参数
   填成真实的值（`Prepare`），量出来的值放在结果位，被测调用自己的结果位留声明。

参数的写法自由：带 tag（本例 `$victim` / `$suspect` / `$at`）按名字取，不带 tag 按
位置取；参数量不限。

## 8. 边界

判定是定义之间的关系，不是对世界的断言。它保证证据与设计合得上、读得到，不保证量出来
的数是真的——后者归数据与取证，判定不假装拥有它没有的谓词。

每个名字都是它那个体的别名，没有独立身份，所以两个工具、两条规则引用同一个定义，读出来
的就是同一个东西：互读靠定义，不靠两份约定对不对得上。

由此有一条分界线要守：**被判定的部分不许是自然语言**。形状、类型、调用合法性进判定；
意图、理由、经过留在 `Appendix`——写下来可以，但不承担正确性。第 5 节第 3 条"说明块
不看"是这个意思：不是把说明省掉，而是把要求挪到能执行的地方。

编造只能落在设计侧：设计可以自由发明，证据必须被钉死（具体值、具体调用）。`$func` 与
`$call_instance` 分开就是这个落点——自由留给设计，落地留给证据。证据侧一松，判定就成
了自证。

这一层不关心执行：代价是它证明不了世界上的事，收益是判定不被执行细节污染。

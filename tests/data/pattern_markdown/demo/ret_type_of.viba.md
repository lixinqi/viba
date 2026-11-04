一条函数链的返回类型。链的写法是 `返回 <- 参数`，所以答案总在第一个位置上。

## `__generic__.viba`

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## `100.viba`

最里层只有一个单位元参数（`()` 或 `nil`）：答 `A`。

```viba
pattern A <- (() | nil)

type = A
```

## `200.viba`

一个参数：参数是什么不影响答案。

```viba
pattern A <- B

type = A
```

## `300.viba`

两个参数：照样只看第一个位置上的 `A`。

```viba
pattern A <- B <- C

type = A
```

`ret_type_of[(int, str) <- bool <- str]` 答 `(int, str)`。

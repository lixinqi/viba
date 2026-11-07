问一个问题：传进来的是不是常量 `"red"`。两个文件按编号从小到大试，第一个模式对得上的答。

## `__generic__.viba`

标记：这份文档是一个泛型。

## `100.viba`

常量 `"red"` 本身是一个类型——一个字面量类型，只有一个值。模式写成这个类型，实参是同一个字面量时
落得进去（由 `is_sub_type` 判）。

```viba 100.viba
pattern "red"

value = true
```

## `200.viba`

别的都落到这一支，答 `false`。

```viba 200.viba
pattern T

value = false
```

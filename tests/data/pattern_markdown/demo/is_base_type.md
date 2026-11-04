# is_base_type

问一个问题：传进来的类型落不落在叶子上——`bool`、`int`、`float`、`str`。两个文件按编号从小到大试，第一个模式对得上的答。

## __generic__.viba

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## 100.viba

叶子：四种标量都收，答 `true`。

```viba
pattern bool | int | float | str

value = true
```

`is_base_type[bool]`、`is_base_type[str]` 都停在这一支。

## 200.viba

除此之外的类型都落到这一支，答 `false`。

```viba
pattern A

value = false
```

`is_base_type[list[int]]` 答 `false`。

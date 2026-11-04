容器里装的是什么：`list`、`set`、`dict` 各一支，答元素类型。

## `__generic__.viba`

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## `100.viba`

列表：答它的元素。

```viba
pattern list[A]

type = A
```

## `200.viba`

集合：同样答元素。

```viba
pattern set[A]

type = A
```

## `300.viba`

字典：键和值都要，答一个元组。

```viba
pattern dict[A, B]

type = (A, B)
```

`element_type_of[dict[int, str]]` 答 `(int, str)`。

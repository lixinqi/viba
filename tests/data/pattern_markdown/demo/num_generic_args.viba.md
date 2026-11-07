一个类型应用带几个实参：按源码里写的数，`dict[A, B]` 是两个，`A` 是零个。

## `__generic__.viba`

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## `100.viba`

字典：两个实参。

```viba 100.viba
pattern dict[A, B]

value = 2
```

## `200.viba`

列表：一个。

```viba 200.viba
pattern list[A]

value = 1
```

## `210.viba`

集合：也是一个。

```viba 210.viba
pattern set[A]

value = 1
```

## `300.viba`

别的类型：零个。

```viba 300.viba
pattern A

value = 0
```

`num_generic_args[int <- int]` 也走这一支。

应用里给了几个实参——按个数分文件，从 `[]` 到一个、两个、三个。

## `__generic__.viba`

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## `50.viba`

一个都没给：`num_variadic_args[]`。

```viba 50.viba
value = 0
```

## `100.viba`

一个。

```viba 100.viba
pattern A

value = 1
```

## `200.viba`

两个。

```viba 200.viba
pattern A
pattern B

value = 2
```

## `300.viba`

三个。

```viba 300.viba
pattern A
pattern B
pattern C

value = 3
```

再多就没有文件了，应用会报决断失败。

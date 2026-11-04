把列表应用成一条函数类型：`A <- $env Env <- A`。

## `__generic__.viba`

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## 100.viba

接住列表，把元素放进返回位和参数位。

```viba
pattern list[A]

type = A <- $env Env <- A
```

`$env Env` 那个参数是给环境留的位置。

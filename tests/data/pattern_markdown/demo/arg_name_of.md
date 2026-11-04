参数带的 tag 叫什么名字：用 `tagged[arg_name, T]` 把它抓出来。

## `__generic__.viba`

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## 100.viba

`tagged` 模式认下 tag，`arg_name` 就是那个 tag 的符号；`value = arg_name` 答它。

```viba
# The tag an argument carries, answered as the symbol it spells: a
# `tagged` pattern names the tag it claims, so the name is a parameter.
pattern tagged[arg_name, T]

value = arg_name
```

`arg_name_of[$a int]` 答 `"a"`。

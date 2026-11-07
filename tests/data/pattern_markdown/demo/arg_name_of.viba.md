参数带的 tag 叫什么名字：用 `tagged[arg_name, T]` 把它抓出来。

## `__generic__.viba`

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## `100.viba`

`tagged` 模式认下 tag，`arg_name` 就是这个 tag 拼出来的符号——它在模式里是个参数；
`value = arg_name` 把它答回去。

```viba 100.viba
pattern tagged[arg_name, T]

value = arg_name
```

`arg_name_of[$a int]` 答 `"a"`。

# wrapped_item

把元素包进一个带 tag 的成员里，答那个成员。

## __generic__.viba

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## 100.viba

先接住列表，再把元素放进 `$item`；答的是 `Wrapped` 这个名字。

```viba
pattern list[A]

Wrapped = $item A

type = Wrapped
```

答案是定义名，所以取出来的是这个定义的成员结构。

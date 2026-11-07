问一个问题：传进来的是不是 `colors.Red` 这个类型——它定义在 `types.viba.md` 的 `colors.viba` 那一章里。
模式写的是自己定义的类型：它在模式文件里解析得到（那份文件 import 了 `types.colors`），判定照 `is_sub_type` 来。

## `__generic__.viba`

标记：这份文档是一个泛型。

## `100.viba`

```viba 100.viba
import types.colors as colors

pattern colors.Red

value = true
```

## `200.viba`

```viba 200.viba
pattern T

value = false
```

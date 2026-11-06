同一个比较，但 `units.Red` 和 `units.Blue` 都是单位元——结构一样，就是一个类型。所以问 `units.Blue`
也答 `true`：按结构判定，分不出这两个名字。

## `__generic__.viba`

标记：这份文档是一个泛型。

## `100.viba`

```viba
import types.units as units

pattern units.Red

value = true
```

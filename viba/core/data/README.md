# viba/core/data：类型描述的那组值

viba 的类型描述的是一组值，而字面量让一个值本身也能写成类型（`$x 1 * $y 2` 是一个只有一个值的类型）。
这里放这一侧的说明：

- **概念**：一个值是什么、怎么取、什么时候一份设计真的在描述数据、和 JSON 的关系——
  [`viba-data.md`](viba-data.md)。
- **取值**：反射协议——[`viba-reflect.md`](viba-reflect.md)。
- **写回源码**：序列化——[`viba-serialize.md`](viba-serialize.md)。
- **完整性**：这份设计反射得到底吗——[`viba-is-complete.md`](viba-is-complete.md)。

## 这里的代码

| 模块 | 干什么 |
|------|--------|
| `reflect.py` | 反射协议：给类型寻址、从实例里取值——见 `viba-reflect.md` |
| `serialize.py` | 把实例序列化成源码——见 `viba-serialize.md` |
| `is_complete.py` | 一份设计反射得到底吗——见 `viba-is-complete.md` |

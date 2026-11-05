# viba/core：语言、类型模型与子类型判定

viba 用纯类型描述设计：一份设计是一组类型与定义，类型之间按结构比，没有名义类型。这里放三样说明：

- **语言**：语法、一个定义怎么写、泛型——[`viba-reference.md`](viba-reference.md)、[`viba-style.md`](viba-style.md)、[`viba-pattern.md`](viba-pattern.md)。
- **类型模型**：类型怎么被寻址、描述符是什么、怎么从实例里取值、怎么把数据写回源码——[`viba-reflect.md`](viba-reflect.md)、[`viba-serialize.md`](viba-serialize.md)。
- **判定**：两个类型怎么比——[`viba-is-sub-type.md`](viba-is-sub-type.md)。

## 从哪儿开始

- 语言的一页参考：[`viba-reference.md`](viba-reference.md)——语法、终结符、运算符、求值、优先级。
- 定义怎么写：[`viba-style.md`](viba-style.md)，带一份刚写完时的自检清单。

## 章节

| 文档 | 讲什么 |
|------|---------|
| [`viba-reference.md`](viba-reference.md) | 语言参考：语法、终结符、运算符、求值、优先级 |
| [`viba-style.md`](viba-style.md) | 定义怎么写：tag、块头、容器，以及刚写完时的自检清单 |
| [`viba-pattern.md`](viba-pattern.md) | 泛型是一个目录：`pattern`、决断顺序，以及各层怎么取它 |
| [`viba-reflect.md`](viba-reflect.md) | 类型模型：描述符、给类型寻址、从实例里取值 |
| [`viba-serialize.md`](viba-serialize.md) | `serialize`：把数据写回源码，反射的逆 |
| [`viba-is-sub-type.md`](viba-is-sub-type.md) | 子类型判定：按结构、共归纳、没有名义类型 |

## 例：取一个类型的语法树

```python
from viba import viba_ast

tree = viba_ast.parse("Option[T] = $some T | nil")
print(viba_ast.dump(tree))
print(viba_ast.unparse(tree))   # canonical chain-style source
```

## 这里的代码

| 模块 | 干什么 |
|------|--------|
| `viba_ast/`、`parser.py` | 语法层：节点类、链式规范化、unparse、访问器、`dump`；`viba_ast.parse` 背后的 PLY 语法（自测核对 `viba-reference.md` 拼的就是那份语法） |
| `type.py`、`type.viba` | Type 模型、内建名字、`module_get_type` |
| `is_sub_type.py` | 子类型判定（`<<`、单位元、共归纳、`Any`）——见 `viba-is-sub-type.md` |
| `pattern.py` | 泛型与它的模式目录：模式匹配、决断——见 `viba-pattern.md` |
| `viba_type_descriptor.py`、`viba_type_descriptor.viba` | 描述符层：文件、定义、成员、类型表达式 |
| `reflect.py` | 反射协议：给类型寻址、从实例里取值——见 `viba-reflect.md` |
| `partial.py` | `<<` 走的约简，判定用 |


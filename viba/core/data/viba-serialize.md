# serialize：把 viba 数据序列化成源码

反射的逆：反射从设计出发、顺着数据路径取值；`serialize` 把手里的一格（`VObject`）写回 viba 源码。
出来的是规范形式（经 `viba.builder` 生），交出去之前用 `builder.check` 再解析一遍，所以它一定编得过。

## 1. 类型

```viba
Serialize =
    Result[str]
  <- $name str
  <- $value VObject[Any]
  <- Hint[{
      $name 是写出来的那个定义的名字。
      $value 是协议里的一格（viba-reflect.md 第 5 节的 VObject），带着它自己的访问器；
      序列化顺着这一格问下去，不问别的访问器。
      答 Ok(源码)：`name =` 加上那一格的写法；$err 说的是这份设计没有这一格的形式。
    }]
```

## 2. 怎么用

```python
from viba.reflect import access, VibaData
from viba import serialize

root = access.root(definition, VibaData(viba_data))
serialize.serialize("entry", root.ok_value)
# Ok('entry =\n  Object\n  * $left 1\n  * $right 2\n')
```

## 3. 顺着协议走

每一格都经协议的取法问出来——和取值用的是同一套——所以它不懂设计，也不懂数据是怎么绑的。写出来的
源码经 [`viba.builder`](../viba_builder.md) 生，和手工拼源码是同一个出口。

## 4. 出来的是什么

- **积以 `Object` 开头**，不管源码里的积头是不是单位元：多行块的第一行就说清这是哪一层
  （[`viba-style.md`](../viba-style.md) 第 6 节）。
- **和只给值所在的那一支**：值不带着别的分支，一支也不成和（[`viba-style.md`](../viba-style.md) 第 7 节）。
- **类型什么都没说的一格**（`Any`，顶类型）按它自己的源码形式写：类型没有成员可走，能说它是什么的
  只有那一格自己。槽位里带着 `Any` 的声明（`InterpretResult` 的 `$ok Any`）这样也能转回来。

## 5. 错误

答 `$err` 而不是写错一个形式：这一类是设计装不下的数据，由调用方决定怎么办。

- `a chain headed by a member has no form in the source here`
- `this piece is … itself, not a value`
- `nothing resides in never`
- `the inline chain comes back to 'A'`
- `no value here: $a`

## 6. 这一版不做的

- **不是原始源码**：出来的是规范形式，重排版、丢注释，不和输入逐字相同。
- **不管数据怎么来**：`VObject` 是调用方给的；序列化不查它背后是谁的数据。
- **不落盘**：这一层只把数据写成源码，不负责把源码存成文件或快照。

## 7. 测试

[`tests/test_serialize.py`](../../../tests/test_serialize.py)：每一层的写法、`Any` 的槽位、按位置与按 tag 的
成员、容器字面量、以及第 5 节那些 `$err`。

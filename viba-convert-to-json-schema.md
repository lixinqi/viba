# convert_to_json_schema：把类型写成 JSON Schema

类型要交给只认 JSON 的工具时——校验器、模型的 tool schema、审计脚本——先得有一份 JSON Schema。这个
函数把类型走一遍，写出这份 schema 文档；入口和 [`check_convertable_to_json_schema`](viba-check-convertable-to-json-schema.md)
一样，判定先跑：互转不了的类型没有 schema 可写，返回值就是判定的那条错。

```viba
Point = Object * $x float * $y float
```

```python
from viba.convert_to_json_schema import convert_to_json_schema

answer = convert_to_json_schema(["kb"], ("type_names", ["objects.Point"]))
```

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$ref": "#/$defs/objects.Point",
  "$defs": {
    "objects.Point": {
      "type": "object",
      "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"}
      },
      "required": ["x", "y"],
      "additionalProperties": false
    }
  }
}
```

## 1. 类型

```viba
ConvertToJsonSchema =
    Result[JsonStr]
  <- $viba_path list[str]
  <- $entry ($module_name ImportableTypeName | $type_names list[ImportableTypeName])
  <- $strict ((bool | nil) * { 默认为 False，表示允许给定 module_name 内部可以包含 <- 和 <<，函数内部会选择性忽略他们 })
  <- Hint[{
      $entry 给一个模块名：文档的 $defs 里是这个模块的每个定义；给一组 type 名：是这几个定义。
      先跑 check_convertable_to_json_schema：它答错就把它的错答回去。
      出来的是一份 JSON Schema（2020-12）：每个走到的定义按全名收进 $defs，引用它们的用 $ref，
      $ref 的根是第一个（模块的话是它的第一个定义）。
      积是对象（不带 tag 的成员摊平，按位置的不行）；和是 oneOf；带 tag 的一项是一个对象，键是 tag 去掉 $；
      list / set 是数组（set 加 uniqueItems）；dict 是对象（键是字符串）；元组是定长数组；标量按 JSON 的种类。
      带实参的定义按 `名字@摘要` 收，换个实参就是另一份。
      $strict 给 True：函数（`<-`）和调用（`<<`）不跳过，按判定拒掉；给 nil 或 False（默认）：
      这两种定义跳过，模块里的数据类型照常写。
    }]

JsonStr =
    str * Assert[{ json content }]

ImportableTypeName =
    str * Assert[{ 形如 xxx.yyy 等可被 import 的字符串 }]
```

## 2. 怎么调用

入口和 `check_convertable_to_json_schema` 是同两支：给模块名，或给一组 type 名。Python 里：

```python
convert_to_json_schema(["kb"], ("module_name", "objects"))
convert_to_json_schema(["kb"], ("type_names", ["objects.Point", "objects.Node"]))
```

答 `Ok(json_text)` 或 `Err(...)`（一句话，说的是第一处不成立的地方）。

第三个实参 `strict` 说函数（`A <- B`）和调用（`f << ...`）怎么处理：

- `None` 或 `False`（默认）：这两种定义跳过——程序模块的 `__decl__`、`__impl__` 不进文档，模块里的数据类型
  照常写；
- `True`：不跳过，按判定拒掉（`a function is no JSON value`）。

```python
convert_to_json_schema(["kb"], ("module_name", "demo"), strict=True)
```

被写进去的类型如果自己还走到函数，两种模式都答错：那个字段没有 schema 可写。

## 3. 文档长什么样

- **`$schema`**：`https://json-schema.org/draft/2020-12/schema`，固定这个方言。
- **`$defs`**：类型走到的每个定义收在这里，键是它的全名（`objects.Point`）。带实参的定义——泛型应用落到的定义、决断选中的文件里的定义——键是 `名字@摘要`（`EitherImpl.200.type@65f930a3`）：换个实参就是
  另一份 schema，摘要把它们分开。
- **`$ref`**：根是第一个入口定义；类型之间互相引用也都用 `$ref`，所以递归类型是有限的一份文档。
- 一份文档只写 `$schema`、`$ref`、`$defs` 三个键。

## 4. 类型写成什么

| viba | JSON Schema |
|------|-------------|
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `str` | `{"type": "string"}` |
| `bool` | `{"type": "boolean"}` |
| `nil` / `Object` | `{"type": "null"}` |
| `7` / `"a"` | `{"const": 7}` / `{"const": "a"}` |
| `never` | `false` |
| `list[T]` | `{"type": "array", "items": …}` |
| `set[T]` | 同上，加 `"uniqueItems": true` |
| `dict[str, T]` | `{"type": "object", "additionalProperties": …}` |
| 积 `Object * $a A * $b B` | 对象：`properties` / `required` / `additionalProperties: false` |
| 积里的内联位 | 摊进当前对象；按位置的成员没有键，答错 |
| 积里的单位元成员（`Object`、`nil`、`Hint[{...}]`、`Assert[{...}]`） | 不算成员；一个不剩就是 `{"type": "null"}`，只剩一个不带 tag 的成员时就是那一个的 schema |
| 和 | `{"oneOf": […]}`，一支一项；嵌套的和摊平；`Oneof` 头和 `never` 不算一项 |
| 带 tag 的一项 `$a A` | `{"type": "object", "properties": {"a": …}, "required": ["a"], "additionalProperties": false}` |
| 元组 `(A, B)` | `{"type": "array", "prefixItems": […], "items": false, "minItems": 2}` |
| 名字 | `{"$ref": "#/$defs/<全名>"}` |

`set` 和 `list` 在 JSON 里都是数组，`set` 多的只是 `uniqueItems`；tag 写成键时去掉前面的 `$`。

## 5. 错误

- 判定不过（[`viba-check-convertable-to-json-schema.md`](viba-check-convertable-to-json-schema.md) 第 4、6 节那些）——返回值就是
  判定的那条错；
- 入口本身不成立：模块找不到、type 名解析不到、`$entry` 不是那两支、`strict` 不是 bool 或 nil；
- 没有可写的类型：模块里全是函数和调用（`$strict` 没开），或者给的名字都是函数。

## 6. 这一版不做的

- **按位置的积成员**：没有 tag 就没有键，答错。
- **非字符串的 dict 键**：JSON 对象的键只能是字符串；`dict[int, T]` 答错（判定也查这一条）。
- **说明文字**：代码块里的说明（`Hint[{...}]`、`Assert[{...}]`）是单位元，不进 schema，也不写
  `title` / `description`。
- **方言**：只写 2020-12；要别的方言得自己转。

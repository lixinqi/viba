# 语言参考

语法在 [`viba/parser.py`](../parser.py) 里，下面就是它：一行一条产生式，解析器的自测每次都会把这份
和它自己的对一遍。对不上这份语法的源码编不过。`(empty)` 表示什么都不匹配——空程序、没有参数的定义、
没有实参的类型。

```ebnf
program : statement_list | (empty)
statement_list : statement | statement statement_list
statement : definition | import_stmt | pattern_stmt
definition : type_definition | generic_definition
type_definition : CLASS_NAME ASSIGN partial_expr
generic_definition : CLASS_NAME LBRACKET CLASS_NAME type_param_list RBRACKET ASSIGN partial_expr
type_param_list : COMMA CLASS_NAME type_param_list | (empty)
import_stmt : IMPORT class_path optional_alias
pattern_stmt : PATTERN partial_expr
optional_alias : AS CLASS_NAME | (empty)
class_path : CLASS_NAME | class_path DOT CLASS_NAME
partial_expr : partial_expr APPLY_OP adt_expr | member_head APPLY_OP adt_expr | adt_expr
member_head : TAGGED_CLASS_NAME
adt_expr : adt_expr SUM_OP product_expr | product_expr
product_expr : product_expr PROD_OP exponent_expr | exponent_expr
exponent_expr : exponent_expr EXP_OP unary_expr | unary_expr
unary_expr : tag_path member_expr | member_expr
tag_path : TAGGED_CLASS_NAME | tag_path DOT CLASS_NAME
name_tail : DOT CLASS_NAME name_tail | (empty)
member_expr : member_expr DOT CLASS_NAME | type_app_expr
type_app_expr : CLASS_NAME name_tail optional_type_args | primary_expr
optional_type_args : LBRACKET partial_expr adt_arg_list RBRACKET | LBRACKET RBRACKET | (empty)
adt_arg_list : COMMA partial_expr adt_arg_list | (empty)
primary_expr : CLASS_NAME | literal | NIL | NEVER | ANY | ELLIPSIS | LPAREN partial_expr RPAREN | LPAREN adt_expr_list RPAREN | HINT_BLOCK
adt_expr_list : partial_expr COMMA partial_expr | partial_expr COMMA adt_expr_list | (empty)
literal : FLOAT | INT | STRING | SINGLE_STRING | TRIPLE_STRING | BOOLEAN
```

## 终结符

| 终结符 | 源码里的样子 |
|--------|-------------|
| `CLASS_NAME` | `Option`、`GraphModule`——`\w+`（一段；`.` 是自己的终结符） |
| `TAGGED_CLASS_NAME` | `$x`、`$meta`——名字前一个 `$`（一段） |
| `DOT` | `.`——取成员：`a.b`、`$meta.id`、`g[T].value` |
| `INT`、`FLOAT` | `42`、`3.14`、`.5` |
| `STRING`、`SINGLE_STRING` | `"one line"`、`'one line'`——转义可以，裸换行不行 |
| `TRIPLE_STRING` | `'''keeps newlines, spacing and quotes'''` |
| `BOOLEAN` | `true`、`false` |
| `NIL` | `nil`、`void`、`None`——同一个单位元，三种写法 |
| `NEVER`、`ANY`、`ELLIPSIS` | `never`、`Any`、`...` |
| `HINT_BLOCK` | `{ ... }`——一段交给宿主的文字，花括号可以嵌套 |
| `IMPORT`、`AS`、`PATTERN` | `import`、`as`、`pattern` |
| `ASSIGN` | `=`——一个定义 |
| `SUM_OP`、`PROD_OP`、`EXP_OP`、`APPLY_OP` | `\|`、`*`、`<-`、`<<` |
| `LBRACKET`、`RBRACKET`、`LPAREN`、`RPAREN`、`COMMA` | `[`、`]`、`(`、`)`、`,` |

## 运算符

| 运算符 | 形式 | 意思 |
|--------|------|------|
| 定义 | `Name = body` | 一个类型定义 |
| 和 | `A \| B` | A 或 B |
| 积 | `A * B` | A 和 B 都要 |
| 指数 | `B <- A` | 从 A 到 B 的函数 |
| 部分计算 | `T << $a A` | 把那个实参在源码里给掉：`(B <- $a A) << $a A` 就是 `B`；给进去的要落在那个参数位上（`A' <: A`） |
| 泛型 | `Name[T]` | 带形参的类型 |
| 模式 | `Name[T]` | 一个**泛型**：`Name` 是一个目录，一条 `pattern` 行一个形参，按源码里实参的样子决断选一个文件——[`viba-pattern.md`](viba-pattern.md) |
| tag | `$label T` | 带名字的成员 / 分支 |
| 单位元（积） | `nil` | 积的单位元（`A * nil = A`）；`void`、`None`、`Object` 是别名——`Object` 是同一个单位元，写在块状积的头 |
| 单位元（和） | `never` | 和的单位元（`A \| never = A`），底：它是所有类型的子类型；`Oneof` 是同一个单位元，写在块状和的头 |
| 顶 | `Any` | 所有类型都是它的子类型，只有它自己（或与它相等的，如 `Any \| int`）在它下面 |
| 省略 | `...` | 开着的、可变参数的类型 |
| 元组 | `(A, B, C)` | 按位置的积（顺序有意义），不是带 tag 的 `*` 的语法糖 |
| 提示块 | `{ ... }` | 任意文字，可以嵌套——类型的备注，或某一步实现来处的提示 |
| import | `import a.b [as c]` | 模块引用（顶层） |

## 定义约定

一个定义有它自己的约定：每个字段和形参都带 tag、块状的头写第一行、一支不成和、内建容器按它被造的
方式用。[`viba-style.md`](viba-style.md) 把这些写成条目，还带一份刚写完定义时的自检清单。

## 求值

viba 是**声明式**的，不是命令式：定义是**绑定**，不是语句，文件不按行执行。求值按需求驱动——用到
才求，求一次就记住（call-by-need，带缓存）——所以源码里的顺序不是求值的顺序，一个定义可以引用写在
它后面的定义（前向引用），一个模块不能把同一个名字定义两次，没人用的绑定永远不求。一个绑定在自己
求值时又被用到，是循环定义，报错，不会让它把栈撑爆。

这是策略的一半；另一半在一次调用怎么发生。实参按值传，只有函数类型的形参是非严格的：源码里写下的
那次调用带着它的环境一起交过去，被调用方要了才求。[`viba/exec/viba-interpreter.md`](../exec/viba-interpreter.md)
把两半都写出来，并点名能看出每一半的那些情形；这样的形参放在哪个位置，见 [`viba-style.md`](viba-style.md) 第 9 节。

## 字符串

- `"double quoted"`——普通字符串
- `'single quoted'`——单引号字符串
- `'''triple quoted'''`——保留换行和空白

## 注释

`#` 到行尾。

## 运算符优先级（低到高）

优先级由上面的语法定，从最松的开始：部分计算最松，然后是和、积、指数——应用和 tag 绑得最紧。

1. `<<`（部分计算）——左结合，最松
2. `|`（和）——左结合
3. `*`（积）——左结合
4. `<-`（指数）——左结合

所有二元运算符都是左结合。`T << X` 不是构造子：它当场把一个函数约掉，实参给全了就剩下结果本身。
每个 `<<` 是一次应用，一次应用返回的可能又是一个带自己 tag 的函数，所以一串 `<<` 不会被折成一次
带扁平实参表的调用：同一次调用里同一个 tag 给两次是程序错误（[`viba-style.md`](viba-style.md) 第 9 节）。

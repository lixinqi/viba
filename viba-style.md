# Viba 编写规范

[`README.md`](README.md) 是语言参考：语法、算子、例子。这一份讲**怎么把定义写对、写清楚**——写给
写 viba 的人，也写给照着写 viba 的 agent。每条都以"读的人一眼看懂"为准。

## 1. 标签就是语义

积的每个成员、函数的每个实参，都写 tag：

```viba
Point = $x float * $y float
Handler = Response <- $request Request
```

不要写成 `Point = float * float`（读的人不知道哪个是横、哪个是竖）或 `Handler = Response <-
Request`（这个参数是什么）。

不带标签的成员只留给"本来就没有名字"的东西：

- **元组**：`(A, B)`——顺序本身就是语义；
- **空**：`nil` 这样的分支（这一支什么都不是）；
- **裸值分支**：`int | str`——按值的种类区分的叶子。

## 2. 一行不写头，多行块头在第一行

`Object` 是积的单位元（就是 `nil`），`Oneof` 是和的单位元（就是 `never`）。写不写都是同一个类型
（`Object * $x int` 与 `$x int` 互为子类型），所以这是**版式**上的事：

- **一行**：不写头。整份定义一眼看全，头是多余的：写 `Parent = nil | $parent_of Node`，**不是**
  `Parent = Oneof | nil | $parent_of Node`。
- **摊成多行块**：头写第一行。块的第一行就是"这一块是积还是和"的说明书；块以字段或分支开头，等于
  让读的人自己去猜。

```viba
# one line: the shape is plain, no head
Option[T] = $some T | nil
Config = $mode "fast" * $threads 42 * $ratio 3.14
Parent = nil | $parent_of Node
UserId = int

# laid out as a block: the head says which shape the block is
Entry =
  Object
  * $name str
  * $state State

Result =
  Oneof
  | $ok Entry
  | $missing UserId
```

链**内部**的单位仍写 `nil` / `never`——`Object` / `Oneof` 只是块头的拼法。**指数永远不带头**：
它的第一个元素就是结果，每个实参带自己的 tag。

## 3. 一支不成和

和的链只有一支时，头和这一层都是多余的：`Oneof | T` 就是 `T`，`Oneof | $x T` 就是 `$x T`。所以：

- 要表达"二选一"，至少两支：`Result = $ok Entry | $missing UserId`；
- 只有一种情况就别写和：直接写那个成员，或者（想给它一个名字）写成一个带标签的积
  `Box = Object * $value T`。

积这边有个对称的坑：**不带标签的单成员不是那个成员**。`Object * int` 与 `int` 不是同一个类型
（前者是"一个位置成员的积"），而且读不出这个成员是什么——这正是第 1 条要求成员带 tag 的原因。

## 4. 容器：`list` / `set` / `dict`

三个容器是内建的，用法就三件事：**形状、字面量、寻址**。

```viba
Items = $items list[int]
Tags = $tags set[str]
Table = $table dict[str, int]
Nested = $nested list[dict[str, int]]
```

- **形状**：`list[T]` 有序、按位置寻址；`set[T]` 无序（枚举顺序由实现定）；`dict[K, V]` 按键寻址。
  键写 `str`——访问协议里按键那一步就是 `$at_key str`。
- **没有 `repeated` 这回事**：一个字段可重复就把类型写成 `list[T]`，不要再包一层。
- **字面量有自己的写法**：`ListLiteral[1, "x"]`、`SetLiteral[1, 2]`、`DictLiteral[("k", 1)]`，
  空的是 `ListLiteral[]`。它们是 `list[...]` 之类的居民，写在**材料**那一侧（呈证、Prepare）。
- **可以任意嵌套**：`list[dict[str, int]]`、`dict[str, list[$x int]]`。
- **这三个名字（连 `ListLiteral` / `SetLiteral` / `DictLiteral`）不能拿来定义**：定义名和泛型形参
  里出现它们，解析器当场拒；它们只在类型表达式里出现。

## 5. 函数：结果在前，实参带 tag，提示收尾

```viba
Distance =
    int
  <- $env Environment
  <- $a int
  <- $b int
  <- { how far apart the two were }
```

- 链的第一个元素是**结果**，然后是实参，按给的顺序排；每个实参带 tag。
- `{...}` 是**提示**：写给照着它补实现的人（常常是 agent），说这一步要干什么。它不是参数，也不被
  执行——`<<` 会把说明跳过去。
- 可执行函数依赖 environ：签名里有 `$env Environment` 这一格，调用时也得给。

## 6. 其它

- **内建标量是 `bool` / `int` / `float` / `str`。** `string` 不是内建名：它什么都解析不到，而且
  语法层不会报——类型层会（`module_get_type`、`is_sub_type`）。
- **一份定义一个表达式**，各自一行：没有逗号，也没有语句分隔符。
- **有 `__ret__` 才是程序**：没有它的文件是设计，不是程序，跑它是错。

## 7. 写完怎么查

语法检查器就是解析器，一次调用：

```python
from pathlib import Path
from viba import viba_ast

viba_ast.parse(Path("store.viba").read_text())   # SyntaxError: what is wrong, and which line
```

语法里没有词的字符、没闭合的代码块、拿内建容器当定义名，都会报，而且一定给出行号。设计本身另有一份
检查（一个积里 tag 不重复、内联链要摊到底）：

```python
from viba.check_tag_and_inline import check_tag_and_inline
from viba.viba_type_descriptor import empty_pool, parse_viba_file, pool_add_file

pool = empty_pool()
file = parse_viba_file(pool, source, "store.viba", "store")
check_tag_and_inline(pool_add_file(pool, file.ok_value).ok_value)   # Ok(nil), or what is wrong
```

语言自己的用例是两条命令：

```bash
python -m viba.parser     # 语法：131 份能编过、13 份必须编不过
python -m viba.viba_ast   # 打印器写出来的，再解析回去是同一棵树
```

## 8. 照着写一份

一份小东西，按上面的规矩写完整：叶子、一行不带头、块带头、函数、容器字段。这份也由
`python -m viba.parser` 检查。

```viba
# --- the leaves: a number, and a small set of states ---
UserId = int
State = "draft" | "live"          # told apart by the value: no tags to give

# --- a tree: the root has no parent ---
Parent = nil | $parent_of Node

# --- a one-line product: no head ---
Index = $by_name dict[str, UserId]

# --- products and sums laid out as blocks: the head says which shape it is ---
Entry =
  Object
  * $name str
  * $state State

Node =
  Object
  * $entry Entry
  * $parent Parent

Result =
  Oneof
  | $ok Entry
  | $missing UserId

# --- a function: the result first, every argument tagged, the hint last ---
Lookup =
    Result
  <- $env Environment
  <- $id UserId
  <- { find the entry that id names }
```

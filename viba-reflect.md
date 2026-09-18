# Viba 反射协议

## 1. 范围

本文档定义 Viba 的反射协议：用 Viba 类型描述一份 `.viba` 源码的结构（描述符），并规定具体实现如何按照这些描述符被只读地访问（访问协议）。

协议分两半：

- **描述侧**：把 `.viba` 源码编译成描述符，并在描述符上查询。它只依赖源码，与具体实现无关。
- **访问侧**：向一份具体实现的数据取值。它由实现的提供方完成。

两侧的边界是一条语法上的线：**访问侧的定义一律带 `[Data]` 形参，描述侧一个都不带。** 凡是带 `[Data]` 形参的定义，就是实现方必须完整实现的协议。

## 2. 与 protobuf 的对应

| protobuf | 本协议 | 负责方 |
|---|---|---|
| `descriptor.proto` | 描述符（第 3、4 节） | 描述侧 |
| `DescriptorPool` | `VibaPool`（第 4 节） | 描述侧 |
| `reflection.h` 的 `Reflection` | 访问协议（第 7 节） | 访问侧 |
| 无 | `VibaFileHashes`：数据侧声明所依据的设计版本（第 7 节） | 访问侧 |
| `Reflection::Set*` / `Clear*` / `Mutable*` | 无：只读 | —— |
| `FieldMask` | `VibaPath`（读的那一半） | 描述侧 |

有一处类型上的差别需要说明：protobuf 的字段类型要么是标量、要么是命名的 message 或 enum，所以它有 `type` 与 `type_name` 两栏；Viba 的字段类型本身就是一段类型表达式，且允许内联（`$x (int | str)`），因此描述符里一栏即可装下整段语法。

## 3. 描述符

描述符的类型定义、逐字段的 protobuf 对照、以及描述侧查询，都写在 `viba/viba_type_descriptor.viba`。这里只说要点。

一份 `.viba` 源码的语法全在描述符里：文件与 import、定义（含泛型形参）、成员、类型表达式（引用 / 应用 / 元组 / 带标签 / 和链 / 积链 / 指数链 / 字面量 / 单位元 / 零元 / 省略号 / 代码块），以及叶子常量，外加一个把若干文件放在一起的池子。

两条形状上的要点：

- **描述符是自足的**：每个描述符都持有 `$pool`。拿到任何一个描述符，它自己的细节都问得出来，调用方不必再从外面递上下文进来。
- **积与和地位相同**：定义体是积，成员就是字段；是和，成员就是分支。两者都是"tag 加一段类型"，因此只有一个成员描述符，字段与分支不分成两个类。

## 4. 池子

池子只是索引，不做名字解析。三张名字表的口径：

- 文件名：`$file_name` 原文。
- 定义：`模块名.定义名`。
- 成员：`模块名.定义名.$tag`。

不带标签的成员没有名字，不进名字表，只能按位置查。

## 5. 描述侧查询

查询清单在 `viba/viba_type_descriptor.viba`。它们有一个共同点：签名里没有 `Data`，全是拿描述符换描述符；查不到或形状不符一律 `Err`，不返回空值。按用途分四组：

- **编译与池子**：`ParseVibaFile`、`PoolAddFile`、`PoolFindFile`、`PoolFindDefinition`、`PoolFindMember`、`FileFindImportByLocalName`。
- **定义上**：`DefinitionMembers`、`DefinitionFindMemberByTag`、`DefinitionFindMemberByIndex`、`DefinitionFile`。
- **成员上**：`MemberTypeName`、`MemberResolvedDefinition`、`MemberContainingDefinition`。
- **类型上**：`TypeIsOptional`。可选就是 `T | nil` 这个和形状，所以问类型，不问成员。

有一条口径在这里点出，它跟第 7 节直接相关：**形态查询只看写出来的形状**。`TypeIsOptional` 就是这样——成员写 `$xs Names` 而 `Names := A | nil` 时，它答"不是可选"；想刨到底，得顺着 `MemberResolvedDefinition` 一层层问。

## 6. 寻址

```viba
VibaStep :=
    Oneof
  | $by_tag str
  | $by_field_index int
  | $at_index int
  | $at_key str

VibaPath := list[VibaStep]
```

| step | 含义 |
|---|---|
| `$by_tag` | 当前这一段里某个带标签的字段 |
| `$by_field_index` | 当前这一段里第 i 个位置字段 |
| `$at_index` | 容器的第 i 个元素（`list` / `set` / tuple） |
| `$at_key` | `dict` 的某个键，键为字符串 |

坐标完全由设计侧的语言写成，不含 `Data`。

一步落在哪一段上，先要把那一段展开到能寻址的形状。展开只按池子里的定义走，不做别的猜测：

- **名字**：引用一个定义时，落到那个定义的体上。
- **泛型应用**：构造子指向一个泛型定义时，把实参代进形参，落到实例化的体上。材料那边写成同一个名字或同一个应用时，也按同一条规矩展开——设计里的成员与材料里的那一段因此一一对得上。
- **和 / 积 / 指数**：三种链一个形状，成员就是 `$elements`；链头是单位元就不算成员。指数的 `$elements[0]` 是结果，其余是参数——`never <- $not_operand A` 的成员就是被禁止的那个操作数，`by_tag not_operand` / `get_not_operand()` 取到的就是它。
- **写不出成员的形状**：字面量、代码块、单位元与零元，到此为止，成员为空。

展开只动"这一段是什么形状"，不动数据：数据里没有的坐标照样答 `false`；反过来，实现里的数据也可以多过设计（见第 8 节）。

## 7. 访问协议

### 7.1 接口

```viba
VibaNode[Data] :=
    Object
  * $descriptor VibaTypeDescriptor
  * $data Data
  * $by_tag (VibaNode[Data] <- $tag str)
  * $by_field_index (VibaNode[Data] <- $i int)
  * $at_index (VibaNode[Data] <- $i int)
  * $at_key (VibaNode[Data] <- $key str)
  * $leaf VibaConstantValue
  * $is_list bool
  * $is_set bool
  * $is_dict bool
  * $len int
  * $keys list[str]

VibaRoot[Data] :=
    Result[VibaNode[Data]]
  <- $definition VibaDefinitionDescriptor
  <- $data Data

VibaHas[Data] :=
    Result[bool]
  <- $node VibaNode[Data]
  <- $step VibaStep

VibaGet[Data] :=
    Result[VibaNode[Data]]
  <- $node VibaNode[Data]
  <- $step VibaStep

VibaLeaf[Data] :=
    Result[VibaConstantValue]
  <- $node VibaNode[Data]

VibaLength[Data] :=
    Result[int]
  <- $node VibaNode[Data]

VibaKeys[Data] :=
    Result[list[str]]
  <- $node VibaNode[Data]

VibaFileHashes[Data] :=
    Result[set[str]]
  <- $data Data
```

| 接口 | 说明 | protobuf |
|---|---|---|
| `VibaNode[Data]` | 一个节点：设计侧的一段描述符，加上实现侧的一份数据；其余格子都是第 7.4 节的访问器 | `Message*` 与 `Descriptor*` 这一对 |
| `VibaRoot` | 起点：这份数据对应设计里的哪个定义。内部必须先核版本，见下 | —— |
| `VibaHas` | 这一步有没有。缺席是正常答案，答 `false` | `Reflection::HasField()` |
| `VibaGet` | 往里走一步，得到一个新的节点，因此可以继续往下走 | `Reflection::Get*()` |
| `VibaLeaf` | 走到叶子，取值 | `GetInt32` / `GetString` / `GetMessage` 那一族 |
| `VibaLength` | 容器的元素个数：`list` / `set` / tuple 的长度，`dict` 的键数。不是容器就是 `Err` | `Reflection::FieldSize()` / `RepeatedFieldSize()` |
| `VibaKeys` | `dict` 的键，按实现给的顺序。不是 `dict` 就是 `Err` | `Reflection::MapBegin()` 那一族 |
| `VibaFileHashes` | 这份数据是按哪些设计文件做出来的：那些文件的 `$file_hash` 集合。用于校验版本 | 无对应物 |

`Data` 是形参，实现方把它绑到自己那份数据表示上。数据全程不透明，只有 `VibaLeaf` 会把它翻成设计侧的五种字面量。

**起点必须核版本**：实现方在 `$root` 内部务必调一次 `$file_hashes`，看 `$definition.$file_hash` 在不在返回的集合里。

- 在，才给出起点节点；
- 不在（或者 `$file_hashes` 自己就是 `Err`），返回 `Err`，不给节点。

这样起点一旦成立，后面沿链读到的每一段都属于同一版设计，不必每步再核。版本对不上是 `Err` 而不是 `false`：这不是"数据里缺了这一段"，而是"这份数据压根不是对着这版设计做的"，继续读没有意义。

### 7.2 实现方交付的类型

```viba
VibaAccess[Data] :=
    Object
  * $root (Result[VibaNode[Data]] <- $definition VibaDefinitionDescriptor <- $data Data)
  * $has (Result[bool] <- $node VibaNode[Data] <- $step VibaStep)
  * $get (Result[VibaNode[Data]] <- $node VibaNode[Data] <- $step VibaStep)
  * $leaf (Result[VibaConstantValue] <- $node VibaNode[Data])
  * $length (Result[int] <- $node VibaNode[Data])
  * $keys (Result[list[str]] <- $node VibaNode[Data])
  * $file_hashes (Result[set[str]] <- $data Data)
```

实现方交付一份 `VibaAccess[自己那份数据类型]`：七格函数齐全，才算把第 7.1 节的接口实现完整。`$root` 里那一次版本核对是强制的（第 7.1 节）。节点上的访问器由框架补（第 7.4 节），不是实现方的事。

`$file_hashes` 是数据侧对自己版本的声明：这份数据是按哪些设计文件、哪一版做出来的。它是只读的。`$root` 必须拿它核一次版本（第 7.1 节）；第 7.3 节的 `VibaVersionMatches` 是同一个核对的另一种用法，用于不建起点、先问一句的场合。

### 7.3 便利函数

```viba
VibaResolve[Data] :=
    Result[VibaNode[Data]]
  <- $node VibaNode[Data]
  <- $path VibaPath

VibaRead[Data] :=
    Result[VibaConstantValue]
  <- $node VibaNode[Data]
  <- $path VibaPath

VibaListFields[Data] :=
    Result[list[VibaNode[Data]]]
  <- $node VibaNode[Data]
  <- $definition VibaDefinitionDescriptor

VibaVersionMatches[Data] :=
    Result[bool]
  <- $pool VibaPool
  <- $data Data
```

- `VibaResolve`：逐个 step 走 `VibaGet`。
- `VibaRead`：先 `VibaResolve` 再 `VibaLeaf`，一条路径直接读出细节。
- `VibaListFields`：按 `DefinitionMembers` 逐个 `VibaGet`，取到的收进表；缺的字段不进表。对应 `Reflection::ListFields()`。
- `VibaVersionMatches`：把池子里所有文件的 `$file_hash` 收成一个集合，看数据的 `VibaFileHashes` 是否都在里面。都在就是 `true`；有找不到的，说明数据是基于已被改写或已删除的设计做的，需要重建。反方向不管：池子里多出来的文件不影响判断，数据可以只覆盖设计的一部分。

这四个由第 7.1 节的接口拼出，不需要实现方单独提供。

### 7.4 节点上的便捷访问

节点上挂了便捷函数，人可以顺着点下去，不必每次写 `VibaGet` 和 `VibaStep`。链式调用时把字段名的 `$` 去掉，tag 参数也写不带 `$` 的名字：

```
root.by_tag('len').leaf()
root.by_tag('counter').at_key('value').leaf()
```

每一步回来的还是 `VibaNode[Data]`，所以能一直点；`leaf` 是终点。它们全是第 7.1 节的糖。节点上这些格子是**访问器**：取的时候才算，算不出来就抛异常，不是造节点时就算好：

| 便捷函数 | 等价于 |
|---|---|
| `by_tag name` | `VibaGet(node, $by_tag name)` |
| `by_field_index i` | `VibaGet(node, $by_field_index i)` |
| `at_index i` | `VibaGet(node, $at_index i)` |
| `at_key key` | `VibaGet(node, $at_key key)` |
| `leaf` | `VibaLeaf(node)` |
| `is_list` / `is_set` / `is_dict` | 看 `$descriptor` 的链头是不是内建的 `list` / `set` / `dict` |
| `len` | `VibaLength(node)` |
| `keys` | `VibaKeys(node)` |

**容器**：`list` / `set` / `dict` 是内建的三种容器，各有形态识别与取值。

- `is_list` / `is_set` / `is_dict` 只看**写出来的形状**：描述符的链头写着 `list[...]` 才是 `list`；如果这一段写的是一个名字（`$xs Names`），形态识别不算，要刨到底就顺着 `MemberResolvedDefinition` 一层层问。这条口径与第 5 节的 `TypeIsOptional` 一致。
- `len` 对三种容器都成立：`list` / `set` / tuple 给元素个数，`dict` 给键数；不是容器就是 `Err`。
- `keys` 只对 `dict` 成立，给键，顺序由实现定；不是 `dict` 就是 `Err`。
- 枚举一个序列是 `len` 加 `at_index` 走一遍（`set` 没有固定顺序，从头到尾的顺序由实现定）；读 `dict` 的值是 `keys` 加 `at_key` 走一遍。

**`Err` 直接抛异常**：第 7.1 节的接口返回 `Result[...]`，链式写法里每一步都判一次 `Ok` 太啰嗦。便捷函数只在 `Ok(v)` 时给出 `v`；一旦拿到的不是 `Ok(.)`（也就是 `Err`），直接抛异常，把 `Err` 的那句话带出去。所以链上不会出现 `Result`——`Err` 会打断整条链，这也是链式写法的代价。要自己处理失败，就用第 7.1 节的接口逐层判。

会抛的只有"取值"那一类：`by_tag` / `by_field_index` / `at_index` / `at_key` / `leaf` / `len` / `keys`。**问"有没有"和"试着取"的那一类不抛**，见下面 Python 侧那一段。

**落到 Python 对象上**：`Data` 本身就是一个 Python 对象时，节点在 Python 侧用魔术方法合成接口，链式写法就是原生 Python 写法：

| 便捷函数 | Python 侧 |
|---|---|
| `by_tag name` | `node.get_{name}()`：`__getattr__` 合成，名字就是 tag 去掉 `$` |
| `by_field_index i` | `node.get_field_{i}()`：同上，按位置取 |
| `at_index i` / `at_key key` | `node[i]` / `node[key]`：`__getitem__` |
| `leaf` | `node.value`：原生标量，bool / int / float / str / None |
| `is_list` / `is_set` / `is_dict` | 同名属性 |
| `len` | `len(node)`：`__len__` |
| 序列枚举 | `for x in node:`：`__iter__`，内部就是 `len` 加 `at_index` |
| `dict` 的键 / 值 / 键值对 | `node.keys()` / `node.values()` / `node.items()` |
| 有没有 `by_tag name` | `node.has_{name}()`：等价 `VibaHas(node, $by_tag name)`，返回 `bool`，**永不抛** |
| 有没有 `by_field_index i` | `node.has_field_{i}()`：同上，按位置 |
| 有没有（同一件事） | `'name' in node`：`__contains__`；对容器节点就是"有没有这个下标 / 键" |
| 试着取 | `node.try_get_{name}()` / `node.try_get_field_{i}()`：返回 `Result`，**不抛** |
| 看得到合成出来的名字 | `node.__dir__`：把 `get_{...}` / `has_{...}` 列出来，`dir()` 和补全才看得见 |

问"有没有"的一类**永远不抛**：设计里有、数据里没有就是 `False`（这正是"实现没做到"的证据），实现自己坏了也是 `False`——对问话的人来说两者一样是"这里没有可用的东西"。要区分这两种，用 `try_get_{name}()`：它返回 `Result`，`Ok(node)` 是取到了，`Ok(nil)` 是没有，`Err` 才是问不出来。它对应规格里的 `Result[VibaNode[Data] | nil]`。

名字由 `__getattr__` 兜底合成（Python 没有 `__hasattr__` 这种协议；`hasattr(x, n)` 就是 `getattr(x, n)` 加上吃掉 `AttributeError`），`__contains__` 与 `__dir__` 是真的魔术方法。

`set` 在 Python 里可以迭代，但没有下标，`node[i]` 对它是 `Err`（抛异常）。

```python
root.get_len().value
root.get_counter()['value'].value
if 'items' in root:
    for item in root.get_items():
        print(item.get_name().value)
print(root.try_get_missing())     # Ok(None)：这份数据里没有这一段
```

便捷函数由框架按第 7.1 节的接口补上，不是实现方的额外义务：实现方只承诺 `VibaAccess` 那七格。

## 8. 契约与口径

**契约**：实现必须覆盖设计。设计里列得出的每一条坐标，在数据里都要走得通，并落得到叶子。

- 走得通时，`VibaRead` 直接给出细节。
- 走不通时，`VibaHas` 答 `false`。这是"实现未覆盖设计"的事实记录。
- 实现中的数据可以多于设计。超出设计的部分不在本协议范围内：坐标只能来自描述符，没有走到它的路径。
- 版本要对得上：数据声明的每个 `$file_hash` 都要能在当前设计里找到（`VibaVersionMatches`）。找不到，说明数据依据的是已被改写或已删除的设计。

`Err` 只表示实现侧取不到（句柄失效、远端超时等），不表示缺席。

**口径**：

- **规范化（链式）**：反射描述规范化后的形状。主链压成链：`A | B | C` 与 `(A | B) | C` 是同一条描述，二元写法不再出现。支链不压：`A | (B | C)` 是元素 `A` 加一条次级链，和 `A | B | C` 不同形，`$right` 侧的分组一直留着。
- **定义平铺**：定义都在文件这一层，全名即 `模块名.定义名`。
- **单位元不是字段**：积链链头的 `Object` / `nil`、和链链头的 `Oneof` / `never` 是单位元，不进 `$members`。
- **两种对位**：带标签的成员按 tag 对位，tag 写出来是什么就记什么（含 `$`）；不带标签的按位置对位，没有名字。
- **名字就是名字**：描述符里的引用不会在这里被替换；解析是另一条查询。
- **叶子五种**：bool / int / float / str / nil。

## 9. 实现要求

- 交付一份 `VibaAccess[自己那份数据类型]`，七格函数齐全。
- 容器要能数、能枚举：`$length` 给元素个数，`$keys` 给 `dict` 的键；枚举序列靠它们加 `$at_index`。
- 起点核版本：`$root` 内部先调 `$file_hashes`，`$definition.$file_hash` 不在集合里就返回 `Err`，不给节点。
- 声明版本：`$file_hashes` 给出这份数据所依据的设计文件的哈希集合。
- 节点上的便捷函数不用实现方提供，由框架按接口拼（第 7.4 节）。
- 只读：没有 `Set` / `Clear` / `Mutable`。
- 不报名字、不自省、不上报 schema：实现只需在内部把描述符里的 tag 对到自己那边的键上。设计侧的结构一律以 `.viba` 源码为准。
- 不限定语言与数据表示：协议不预设实现用什么语言、数据内部长什么样。

## 10. 附：protobuf 有而本协议没有的

- 写操作：`Reflection::Set*` / `Clear*` / `Mutable*` / `Swap` / `Add*`、unknown fields、`FieldMask` 的写的那一半。
- 按类型取值：`GetInt32` / `GetInt64` / `GetFloat` / `GetDouble` / `GetBool` / `GetString` / `GetEnum` / `GetMessage` / `GetRepeated*` / `GetMap*`。
- 服务与扩展：`FindServiceByName` / `FindMethodByName` / `FindExtensionByName` / `FindExtensionByNumber`，以及 `ServiceDescriptor` / `MethodDescriptor`。
- 字段细节：`cpp_type`、`has_default_value` / `default_value`、`json_name`、`is_extension`、`is_packed`、`containing_oneof`。
- 文件细节：各类 options、`syntax` / `edition`、`source_code_info`（行号列号）、`public_dependency` / `weak_dependency`。
- 结构：`nested_type`、extension range、reserved range、枚举值的显式编号。

反过来，本协议有而 protobuf 没有的：源码文件的哈希、数据侧的版本声明、泛型形参、标签与位置两套对位、任意内联的类型表达式、单位元与省略号、代码块、以及未解析的引用也是合法的。

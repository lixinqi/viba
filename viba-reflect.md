# Viba 反射协议

## 1. 为什么需要反射

我们手上有两样东西：

- **设计**：`.viba` 源码，写清了要什么结构。比如

  ```viba
  Len := int
  Rule :=
      RuleObject
    * $len Metric[Len]
    * $check Predicate[{len under 50}, $python_code {...}]
  ```

- **材料**：实现方那边的数据，实际是什么。它可以是 Python 对象、JSON、一段二进制。

写工具、做校验、查问题的时候，都要求同一件事：**不预先知道设计长什么样，也能按设计把材料里的数据读出来**。具体就是给个地址——"`$len` 这一段"、"第 2 个元素"、"键 `k`"——问它在不在、把它取出来、走到里面接着问，直到取到叶子上的值。

麻烦的地方有四条：

- **材料长什么样由实现方定**。协议不规定语言和数据结构，也不要求材料长得像设计；两边对得上就行。
- **设计会改版**。材料可能是先前的设计做出来的；一份材料对着哪一版设计，是上层自己记的事，协议不管。
- **实现必须覆盖设计**。设计里列得出的每一条地址，实现那边都要走得通、落得到叶子；数据比设计多一点没关系（多出来的部分不在协议范围内）。反过来，设计里有而实现那边没有，就是不满足契约——问出来是 `false`，那是在说"没做到"。
- **读是有步骤的**。先问这一段有没有，再往里走一步，走到叶子才取值；容器还要能数、能枚举。

所以协议分成两半：**描述侧**把 `.viba` 源码编译成**描述符**（地图），只跟源码有关；**访问侧**按地图从材料里取值，由实现方交付。

## 2. 怎么用

一句话：先立起点，再顺着地图一步一步点下去，点到叶子取值。

```python
root = access(rule).root(witness)      # 起点：给根节点
root.get_len().value                   # $len 这一段，取叶子：17
root.get_counter()['value'].value      # 走两步：$counter → 键 'value'
if 'items' in root:                    # 先问有没有，再迭代
    for item in root.get_items():
        print(item.get_name().value)
root.try_get_missing()                 # Ok(None)：实现那边没有这一段
```

每一步回来的还是同一个节点类型，所以能一直点；点不出来就抛异常，问有没有的那一类永不抛。`set` 没有下标，对它是 `Err`。

不想一步步点，就写一条路径交给便利函数：

- 一条路径走到底，直接读值（走不通给 `Err`）；
- 列出一个定义的所有字段，取到的收进表；
- 只问一句"这份材料是不是对着这版设计做的"，不建起点。

接口、实现方要交什么、Python 侧有哪些写法，第 5 节。

## 3. 它大概怎么做

三件东西：

1. **描述符**：`.viba` 源码编译出来的结构——文件、定义、成员、类型表达式各是什么。它就是那张地图。地图只依赖源码，跟具体实现无关（第 6–8 节）。
2. **地址**：一串步子（按 tag、按位置、按下标、按键）。每一步落在设计图的一段上，走之前先把那一段展开到能寻址的形状（第 4 节）。
3. **取值接口**：起点、问有没有、走一步、读叶子、数个数、取键，一共六格。实现方只交这六格；其余（一条路径走到底、列字段、链式写法）全由这六格拼出来，是糖（第 5 节）。

材料那边的两种对位、容器怎么数、契约是什么，第 9 节。

## 4. 地址

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

地址完全由设计侧的语言写成，不含材料那边的东西。

一步落在哪一段上，先要把那一段展开到能寻址的形状。展开只按设计里的定义走，不做别的猜测：

- **名字**：引用一个定义时，落到那个定义的体上。
- **泛型应用**：构造子指向一个泛型定义时，把实参代进形参，落到实例化的体上。材料那边写成同一个名字或同一个应用时，也按同一条规矩展开——设计里的成员与材料里的那一段因此一一对得上。
- **和 / 积 / 指数**：三种链一个形状，成员就是 `$elements`；链头是单位元就不算成员。指数的 `$elements[0]` 是结果，其余是参数，例如：`never <- $not_operand A` 的成员就是被禁止的那个操作数，`by_tag not_operand` / `get_not_operand()` 取到的就是它。
- **无标签的和可以省层**：和的分支都没有标签、且至多一个内节点（叶子不算内节点）时，材料可以直接写那一支的内容，不必套上和这一层。这时 `[$x, 0]` 得到那一支的实例，`[$x, $a]`（标签落在那一支里）也穿得过去；叶子分支按值的种类对位——`int | str` 写 `7` 落在 `int` 上，`int | nil` 写 `nil` 落在 `nil` 上。两个以上内节点的和不能省（分不清是哪一支），带标签的和按 tag 寻址。
- **写不出成员的形状**：字面量、代码块、单位元与零元，到此为止，成员为空。

展开只动"这一段是什么形状"，不动数据：数据里没有的地址照样答 `false`；反过来，实现里的数据也可以多过设计（第 9 节）。

## 5. 访问协议

本节的定义都属于访问侧：它们都带 `[Data]` 形参，`Data` 是"材料本身的类型"这个空位，实现方把自己的数据类型填进去。第 6–8 节那些不带 `[Data]` 的定义属于描述侧。**凡是带 `[Data]` 的定义，实现方必须一条不落都实现。**

### 5.1 接口

```viba
VibaNode[Data] :=
    Object
  * $descriptor VibaTypeDescriptor
  * $data Data
  * $by_tag (VibaNode[Data] <- $tag str)
  * $by_field_index (VibaNode[Data] <- $i int)
  * $at_index (VibaNode[Data] <- $i int)
  * $at_key (VibaNode[Data] <- $key str)
  * $leaf VibaConstant
  * $is_list bool
  * $is_set bool
  * $is_dict bool
  * $len int
  * $keys list[str]

VibaReflectConfig :=
		Object
	* $never_equivalent_terminators set[str]
	* $nil_equivalent_terminators set[str]

Result[T] :=
		Oneof
	| $ok ($ok_value T)
	| $err ($err_msg str)

VibaRoot[Data] :=
    Result[VibaNode[Data]]
  <- $config VibaReflectConfig
  <- $definition VibaDefinitionDescriptor
  <- $data Data


VibaHas[Data] :=
    Result[bool]
  <- $config VibaReflectConfig
  <- $node VibaNode[Data]
  <- $step VibaStep

VibaGet[Data] :=
    Result[VibaNode[Data]]
  <- $config VibaReflectConfig
  <- $node VibaNode[Data]
  <- $step VibaStep

VibaLeaf[Data] :=
    Result[VibaConstant]
  <- $config VibaReflectConfig
  <- $node VibaNode[Data]

VibaLength[Data] :=
    Result[int]
  <- $config VibaReflectConfig
  <- $node VibaNode[Data]

VibaKeys[Data] :=
    Result[list[str]]
  <- $config VibaReflectConfig
  <- $node VibaNode[Data]

```

| 接口 | 说明 |
|---|---|
| `VibaNode[Data]` | 一个节点：设计侧的一段描述符，加上实现侧的一份数据；其余格子都是第 5.4 节的访问器 |
| `VibaRoot` | 起点：这份数据对应设计里的哪个定义 |
| `VibaHas` | 这一步有没有。答 `false`：设计里没这个地址，或者实现那边没有这一段——后者是没覆盖设计 |
| `VibaGet` | 往里走一步，得到一个新的节点，因此可以继续往下走 |
| `VibaLeaf` | 走到叶子，取值 |
| `VibaLength` | 容器的元素个数：`list` / `set` / tuple 的长度，`dict` 的键数。不是容器就是 `Err` |
| `VibaKeys` | `dict` 的键，按实现给的顺序。不是 `dict` 就是 `Err` |

`Data` 是形参，实现方把它绑到自己那份数据表示上。数据全程不透明，只有 `VibaLeaf` 会把它翻成设计侧的五种字面量。

**起点只配成对**：`$root` 拿到数据就给节点。版本不归协议管：一份材料对着哪一版设计做的，由上层自己记、自己核。

### 5.2 便利函数

```viba
VibaResolve[Data] :=
    Result[VibaNode[Data]]
  <- $config VibaReflectConfig
  <- $node VibaNode[Data]
  <- $path VibaPath

VibaGetByPath[Data] :=
    Result[VibaConstant]
  <- $config VibaReflectConfig
  <- $node VibaNode[Data]
  <- $path VibaPath

VibaListFields[Data] :=
    Result[list[VibaNode[Data]]]
  <- $config VibaReflectConfig
  <- $node VibaNode[Data]
  <- $definition VibaDefinitionDescriptor

```

- `VibaResolve`：逐个 step 走 `VibaGet`。
- `VibaGetByPath`：先 `VibaResolve` 再 `VibaLeaf`，一条路径直接读出细节。
- `VibaListFields`：按 `DefinitionMembers` 逐个 `VibaGet`，取到的收进表；缺的字段不进表。

这三个由第 5.1 节的接口拼出，不需要实现方单独提供。

### 5.3 实现方交付的类型

```viba
VibaAccess[Data] :=
    Object
  * $root (Result[VibaNode[Data]] <- $definition VibaDefinitionDescriptor <- $data Data)
  * $has (Result[bool] <- $node VibaNode[Data] <- $step VibaStep)
  * $get (Result[VibaNode[Data]] <- $node VibaNode[Data] <- $step VibaStep)
  * $leaf (Result[VibaConstant] <- $node VibaNode[Data])
  * $length (Result[int] <- $node VibaNode[Data])
  * $keys (Result[list[str]] <- $node VibaNode[Data])
  * Assert[{
  	其构造方法必须接受VibaReflectConfig参数。
  }]
```

实现方交付一份 `VibaAccess[自己那份数据类型]`：六格函数齐全，才算把第 5.1 节的接口实现完整。节点上的访问器由框架补（第 5.4 节），不是实现方的事。

### 5.4 节点上的便捷访问

节点上挂了便捷函数，人可以顺着点下去，不必每次写 `VibaGet` 和 `VibaStep`。链式调用时把字段名的 `$` 去掉，tag 参数也写不带 `$` 的名字：

```
root.by_tag('len').leaf()
root.by_tag('counter').at_key('value').leaf()
```

每一步回来的还是 `VibaNode[Data]`，所以能一直点；`leaf` 是终点。它们全是第 5.1 节的糖。节点上这些格子是**访问器**：取的时候才算，算不出来就抛异常，不是造节点时就算好：

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

- `is_list` / `is_set` / `is_dict` 只看**写出来的形状**：描述符的链头写着 `list[...]` 才是 `list`；如果这一段写的是一个名字（`$xs Names`），形态识别不算，要刨到底就顺着 `MemberResolvedDefinition` 一层层问。
- `len` 对三种容器都成立：`list` / `set` / tuple 给元素个数，`dict` 给键数；不是容器就是 `Err`。
- `keys` 只对 `dict` 成立，给键，顺序由实现定；不是 `dict` 就是 `Err`。
- 枚举一个序列是 `len` 加 `at_index` 走一遍（`set` 没有固定顺序，从头到尾的顺序由实现定）；读 `dict` 的值是 `keys` 加 `at_key` 走一遍。

**`Err` 直接抛异常**：第 5.1 节的接口返回 `Result[...]`，链式写法里每一步都判一次 `Ok` 太啰嗦。便捷函数只在 `Ok(v)` 时给出 `v`；一旦拿到的不是 `Ok(.)`（也就是 `Err`），直接抛异常，把 `Err` 的那句话带出去。所以链上不会出现 `Result`——异常会打断整条链，这也是链式写法的代价。要自己处理失败，就用第 5.1 节的接口逐层判。

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

**取值失败抛 `VibaReflectError`**：上面会抛的那一类，抛出来的异常都叫 `VibaReflectError`，`Err` 的那句话原样在异常信息里；catch 它就能接住反射一路上所有取不出来的情况。

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

便捷函数由框架按第 5.1 节的接口补上，不是实现方的额外义务：实现方只承诺 `VibaAccess` 那六格。

## 6. 描述符

描述符的类型定义、逐字段说明与描述侧查询，都写在 `viba/viba_type_descriptor.viba`。这里只说要点。

一份 `.viba` 源码的语法全在描述符里：文件与 import、定义（含泛型形参）、成员、类型表达式（引用 / 应用 / 元组 / 带标签 / 和链 / 积链 / 指数链 / 字面量 / 单位元 / 零元 / 省略号 / 代码块），以及叶子常量。

两条形状上的要点：

- **描述符是自足的**：拿到任何一个描述符，它自己的细节都问得出来，调用方不必再从外面递上下文进来。
- **积与和地位相同**：定义体是积，成员就是字段；是和，成员就是分支。两者都是"tag 加一段类型"，因此只有一种成员描述符，字段与分支不分成两个类。

## 7. 名字与位置

带标签的东西按名字找，不带标签的按位置找。三张名字表的约定：

- 文件名：`$file_name` 原文。
- 定义：`模块名.定义名`。
- 成员：`模块名.定义名.$tag`。

不带标签的成员没有名字，不进名字表，只能按位置查。

## 8. 描述侧查询

查询清单在 `viba/viba_type_descriptor.viba`。它们有一个共同点：签名里没有 `Data`，全是拿描述符换描述符；查不到或形状不符一律 `Err`，不返回空值。按用途分三组：

- **编译与索引**：`ParseVibaFile`、`PoolAddFile`、`PoolFindFile`、`PoolFindDefinition`、`PoolFindMember`、`FileFindImportByLocalName`。
- **定义上**：`DefinitionMembers`、`DefinitionFindMemberByTag`、`DefinitionFindMemberByIndex`、`DefinitionFile`。
- **成员上**：`MemberTypeName`、`MemberResolvedDefinition`、`MemberContainingDefinition`。

有一条约定在这里点出，它跟第 5.4 节直接相关：**形态查询只看写出来的形状**。成员写的是一个名字（`$xs Names`）时，形态识别不算；想刨到底，得顺着 `MemberResolvedDefinition` 一层层问。

## 9. 契约与约定

**契约**：实现必须覆盖设计。设计里列得出的每一条地址，在数据里都要走得通，并落得到叶子。

- 走得通时，`VibaGetByPath` 直接给出细节。
- 走不通时，`VibaHas` 答 `false`。这是"实现未覆盖设计"的事实记录。
- 实现中的数据可以多于设计。超出设计的部分不在本协议范围内：地址只能来自描述符，没有走到它的路径。

`Err` 只表示实现侧取不到（句柄失效、远端超时等），不表示缺席。

**约定**：

- **规范化（链式）**：反射描述规范化后的形状。主链压成链：`A | B | C` 与 `(A | B) | C` 是同一条描述，二元写法不再出现。支链不压：`A | (B | C)` 是元素 `A` 加一条次级链，和 `A | B | C` 不同形，`$right` 侧的分组一直留着。
- **定义平铺**：定义都在文件这一层，全名即 `模块名.定义名`。
- **单位元不是成员**：和链 / 积链 / 指数链链头的 `Object` / `nil` / `Oneof` / `never` 是单位元，不进 `$members`。
- **两种对位**：带标签的成员按 tag 对位，tag 写出来是什么就记什么（含 `$`）；不带标签的按位置对位，没有名字。
- **名字就是名字**：描述符里的引用不会在这里被替换；解析是另一条查询。
- **叶子五种**：bool / int / float / str / nil。

## 10. 实现要求

- 交付一份 `VibaAccess[自己那份数据类型]`，六格函数齐全。
- 容器要能数、能枚举：`$length` 给元素个数，`$keys` 给 `dict` 的键；枚举序列靠它们加 `$at_index`。
- 起点只给节点：`$root` 不做别的检查；版本由上层自己管。
- 节点上的便捷函数不用实现方提供，由框架按接口拼（第 5.4 节）。
- 只读：协议里没有写的操作。
- 不报名字、不自省、不上报 schema：实现只需在内部把描述符里的 tag 对到自己那边的键上。设计侧的结构一律以 `.viba` 源码为准。
- 不限定语言与数据表示：协议不预设实现用什么语言、数据内部长什么样。

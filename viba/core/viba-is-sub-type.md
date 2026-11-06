# is_sub_type：sub 能不能当 sup 用

`is_sub_type(sub, sup)` 判定 `sub << sup`：`sub` 的每一个值是不是也都能当 `sup` 的值用。它吃的是
`viba.type` 的 `Type` 值——`AstNodeType` 包着源码里的结构，或者叶子类型——不看数据、不跑程序。

## 1. 类型

```viba
IsSubType =
    Result[bool]
  <- $sub Type
  <- $sup Type
  <- $terminators set[str]
  <- $config VibaReflectConfig
  <- Hint[{
      $sub、$sup 是 Type 值：AstNodeType 包着源码里的结构，或叶子类型（nil、never、字面量、标量、Any）。
      $terminators 是一组类型名，给 `never <- A` 那种链用；默认空。
      $config 说哪些名字算单位元，默认 reflect.language_config。
      答 Ok(true / false)；$err 说的是输入不成立（同一积里同一个 tag 两次、内联环、名字解析不到），
      不是判定结果。
    }]
```

## 2. 结构判定

没有名义类型：一个名字就是它源码形式的别名，处处按结构比。

按结构判定是有意的选择，代价和收益都清楚：

- **名字做身份要另立一套规矩**：哪个名字才算数（模块全名？从哪一段开始算？）、改名或挪文件算不算
  换了一个类型、两个 repo 合到一起重名怎么办。这些都得先答，而且每答一条都会漏进数据里。
- **地址做身份最快，但出不了进程**：序列化写出去的是字节，不是指针；反序列化回来要对上号，还得再
  起一套名字。
- **结构加各级 tag 已经把语义带上了**：一个值本身就是一棵带 tag 的树，它的数据路径（描述符加上每
  一步的 tag，[`data/viba-reflect.md`](data/viba-reflect.md)）就是它的地址，写回源码、跨实现、跨进程
  都还是同一份。判定这一处要做的事更多，但类型是不可变的：同一次比较的结果可以缓存，再进一步还能
  让同一份源码形式驻留成同一个对象。那是实现上的功课，不是这条路本身的价格；换来的是序列化与反序列化
  只需照着结构走，以及开着的和、泛型目录这类扩展不必回头改已有的值。

- 定义展开成它的 body 再比：`B = A * $find bool` 里，`B << A` 成立；`X[T] = list[T]` 的
  `X[int]` 拿 `list[int]` 的 body 比。
- 展不开的定义引用按名字比：泛型名字不带实参时就只剩名字。
- 一边是 `AstNodeType`（和、积、元组、指数链、应用）时按结构递归；递归到叶子时两边都抬成 `Type`
  再比。
- `TypeRef` 在它自己的容器模块里经 `module_get_type` 解析（词法作用域）；解析不到时查 `AstNodeType`
  上挂的 `env_get` 绑定，由内向外——泛型形参就是这样拿到意思的。真的解析不到是输入不成立，答
  `$err`，不是 `false`。

## 3. 叶子

- 字面量和标量：`literal(v) << base` 当且仅当同一族（`7 << int`、`"a" << str`）；两个字面量比
  `<<` 当且仅当值相等。
- `never << T` 恒成立（底类型）。
- `T << never` 当且仅当 `T` 就是 `never`：只有 `never` 能落在 `never` 那一支里。
- `T << Any` 恒成立；`Any << T` 当且仅当 `T` 也是 `Any`（或与它相等，比如 `Any | int`）。

## 4. 积

积按内联比：不带 tag 的成员展开后是积时，把它的成员摊进来，可以一直摊；展开后是积的单位元
（`nil`、提示块 `{...}`，或落在它们上面的名字——`Assert` / `Hint` / `Appendix` 这类名字由配置交进来）
时什么都不贡献——所以 `A * $z T` 带着 A 的 tag，`Object * $a int`
两个方向都等于 `$a int`。别的不带 tag 的成员是按位置对位的成员，按顺序与另一边对位。

tag 把成员绑在一起：同一个积里同一个 tag 两次（内联出来的、或源码里就写重的）是输入不成立。内联链
回到正在展开的定义（`A = A * $x int`，绕别名或泛型都算）也答 `$err`。

## 5. 和

`sub` 的每一支都要落在 `sup` 的某一支里。带 tag 的一支先按 tag 认；`sup` 那一支没有 tag 时，比的是
tag 底下的类型——所以 `$b str << int | str` 成立。`sup` 是 `never` 时只有 `never` 落得进去。

## 6. 元组与函数

- 元组按位置逐个比，长度不同就不成立。
- 函数（指数链）是函数对函数：返回位协变，参数位逆变，从 `$arg0` 起逐个对位。`sub` 先对齐到 `sup`
  的参数个数：短了在末尾补 `never`，长了截掉，`sup` 不动。所以
  `(int <- $a int <- $b str) << (int <- $a int)` 成立（多出来的参数 `sup` 不会问），
  `(int <- $a int) << (int <- $a int <- never)` 也成立（补上的 `never` 正是 `sup` 要的），
  但 `(int <- $a int) << (int <- $a int <- $b bool)` 不成立：补 `never` 答不了一个 `bool`。
- 返回位是 `never` 的链（`never <- A`）照样按函数比；指定了 `terminators` 时，这种链的分支还多两条
  取法：一个分支由某个 terminator 或一个算作 `never <- 分支` 的成员接下，而只照抄这个应用本身的
  `sub` 什么都接不下。没给 `terminators` 时只有函数规则，`never <- B << never <- B` 成立。

## 7. 容器字面量

`ListLiteral[a, b, c]` 落在 `list[a | b | c]` 里：每个元素都落在那个并集里；`SetLiteral` 同理；
`DictLiteral[(k, v), …]` 落在 `dict[k 并集, v 并集]` 里。反过来不成立，混用容器家族也不成立。

## 8. 共归纳

环按共归纳取，假定的是最大不动点：一对正在比的 (sub, sup) 再遇到就算成立。所以
`Tree[T] = $leaf T * $kids list[Tree[T]]` 是自己的展开；`MyList[T] = $head T * $tail MyList[T] | nil`
和换了名字的同一个类型互为子类型；只回到自己的定义（`Loop[T] = Loop[T]`）是最大的类型，跟谁比都是
两边的子类型——谁写出这么一份定义，就是给设计一个什么都能装下的类型。

## 9. 错误

`$err` 只说输入不成立，不说判定：

- `the tag $a appears twice in one product`
- `the inline chain comes back to 'A'`
- 名字解析不到时的 `type … unresolved …`

## 10. 这一版不做的

- **名字的含义**：`TypeRef` 只按它所在模块和 `env_get` 解析，没有别的上下文；同一个名字在两份设计里
  可以指不同的东西。
- **性能上的保证**：每次判定从头走，没有跨调用的缓存；成对出现的工作表只管共归纳。
- **数据**：只看类型。

## 11. 测试

[`tests/test_is_sub_type.py`](../../tests/test_is_sub_type.py)：叶子、和、积（含内联）、元组、函数（含
参数个数对齐）、容器字面量、`Any` 与 `...`、共归纳、两类错误的 `$err`，以及 `tests/data/is_sub_type/`
下成对的用例文件。

# check_tag_and_inline：摊开之后一个积里没有重复的 tag，内联链不会绕回来

积里不带 tag 的成员是内联位：它展开后是积（或一整条带 tag 的链）时，那些成员摊进这个积里
（[`viba-reflect.md`](../core/data/viba-reflect.md) 第 4 节）。摊开之后有两处错只有摊开才看得见：同一个积里
同一个 tag 出现两次；内联链绕回正在摊的定义。这份检查从每个定义出发走一遍，把走得到的每个积都
摊开查一遍，跟“这次拿它跟谁比”没有关系。

## 1. 类型

```viba
CheckTagAndInline =
    Result[void]
  <- $design VibaPool
  <- $config VibaReflectConfig
  <- Hint[{
      $design 是描述符池（viba_type_descriptor.viba 的 VibaPool）。
      $config 说哪些名字算单位元，默认 reflect.language_config；单位元没有 tag、也不内联，
      换一份名字不影响结论。访问器在这里按它建，调用方不持有。
      答 Ok(nil)；不成立就答 $err 一句话，说的是第一处错，按定义在源码里的顺序。
    }]
```

## 2. 池子怎么来

吃的是描述符池，不是源码：

```python
from viba.viba_type_descriptor import empty_pool, parse_viba_file, pool_add_file
from viba.check_tag_and_inline import check_tag_and_inline

pool = empty_pool()
file = parse_viba_file(pool, source, "kb.viba", "kb").ok_value
pool = pool_add_file(pool, file).ok_value
check_tag_and_inline(pool)          # Ok(None)，或者第一处错
```

一份设计有几个文件就 `parse_viba_file` 几次，每次用它的模块名；池子按模块名和全名索引
（[`viba-reflect.md`](../core/data/viba-reflect.md) 第 3 节）。

## 3. 查什么

- **同一个积里同一个 tag 两次**，内联出来的算进来：`$x` 写在源里，内联进来的基又带一个 `$x`；
  或者同一个基内联两次。
- **内联链摊不到底**：摊开时又回到正在摊的那个定义（`A = A * $x int`，绕别名或两个定义互指都算），
  它没有展开可取。

一层里就写重的 tag 到不了这里：编译那份文件时已经拒了，`PoolAddFile` 答 `duplicate member`。
留给这一遍的，只有摊开之后才重复的 tag，和一条永远摊不完的链。

## 4. 和判定、serialize 的关系

`is_sub_type` 与 `serialize` 也拒这两种，但只在它们自己的走法碰到的时候——嵌在带 tag 的成员里、
或容器元素里的积，只有比较真正走下去才会被取到。这一遍不看比较：每个定义出发，走得通的每个积都查。
所以它是“发现”的地方，那两处是兜底（[`viba-reflect.md`](../core/data/viba-reflect.md) 第 4 节）。

判定与转换两个 JSON 工具先跑这一遍（[`viba-check-convertable-to-json-schema.md`](viba-check-convertable-to-json-schema.md) 第 3 节）。

## 5. 例子

一个积里每个 tag 只出现一次，这不成问题：

```viba
A =
  Object
  * $x int
  * $y int
```

一个基内联进同一个积，`$x` 就重复了：

```viba
A =
  Object
  * $x int
  * $y int

B = A * $x int
```

链绕回它自己，摊不完：

```viba
C = C * $x int
```

带 tag 的成员不摊开，它里面的积自己成一个积，照常查：

```viba
A =
  Object
  * $x int

B = Object * $b A
```

## 6. 错误

- `the tag $x appears twice in one product`
- `the inline chain comes back to 'C'`

## 7. 这一版不做的

- **和里的重复 tag**：这一遍只查积。和的两个选项带同一个 tag 是判定的事，写在一层里时编译就拒了
  （`duplicate member`）。
- **按位置对位的积成员**：内联位之外不带 tag 的成员不带 tag 也能出现，这一遍不管它们对不对位；
  那是反射协议的事（[`viba-reflect.md`](../core/data/viba-reflect.md) 第 4 节）。
- **数据**：只看设计，不看实例。

## 8. 测试

[`tests/test_check_tag_and_inline.py`](../../tests/test_check_tag_and_inline.py)：一个 tag、内联出来的重复、
两种内联环、带 tag 的成员、容器里的积，以及包内几份 `.viba` 设计本身编得过。

两个实参能不能互相当成对方：两条模式依次对位，都对得上就答 `true`。编号小的先试。

## `__generic__.viba`

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## 100.viba

两个都是单位元（`()` 或 `nil`）：没有内容可比，算合得上。

```viba
pattern () | nil
pattern () | nil

value = true
```

## 150.viba

两个都是标量里的一种（`bool` 或 `int`）：也合得上。

```viba
pattern bool | int
pattern bool | int

value = true
```

## 200.viba

两条模式都叫 `A`：一个名字要求两处是同一个类型。

```viba
pattern A
pattern A

value = true
```

同一个类型当然能互相当成对方。

## 300.viba

两个名字不同：`A` 和 `B` 不要求是同一个类型。

```viba
pattern A
pattern B

value = false
```

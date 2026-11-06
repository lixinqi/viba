## `colors.viba`

三个颜色：结构不同（tag 不同），所以是三个类型。`__type__` 是这份定义自己的名字：对 `Red` 来说，
`tagged[__type__, Object]` 与 `$Red Object` 是同一个类型；`Blue`、`Green` 各自同理。

```viba
Red   = tagged[__type__, Object]
Blue  = tagged[__type__, Object]
Green = tagged[__type__, Object]
```

## `units.viba`

三个名字，同一个类型：都是单位元。

```viba
Red   = Object
Blue  = Object
Green = Object
```

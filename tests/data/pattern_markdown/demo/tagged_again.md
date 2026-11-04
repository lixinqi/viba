把抓到的 tag 原样装回一个类型里。

## `__generic__.viba`

标记：这份文档是一个泛型。内容本来就不看，空着也行。

## 100.viba

最简单的装法：重新装一个 `tagged[arg_name, T]`。

```viba
# The same tag, built again from the symbol the pattern extracted.
pattern tagged[arg_name, T]

type = tagged[arg_name, T]
```

## 200.viba

整条链里带着同一个 tag：模式写成一条链，答的也是这条链。

```viba
# The same tag, inside the chain it was given in: one `pattern` line, however
# many positions it takes, and the chain it gives is what tells this file from
# 100.viba.
pattern A <- int <- tagged[arg_name, T]

type = A <- int <- tagged[arg_name, T]
```

两条模式都收下，靠的是链的长度——200 只在链更长时才轮到。

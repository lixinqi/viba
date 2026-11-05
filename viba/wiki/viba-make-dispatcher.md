# make_dispatcher：把几个 handler 拼成一个函数

同一件事，参数类型不同的时候，各写一个 handler；`make_dispatcher[f0, f1]` 把它们拼成一个函数：调用
时按这次给的参数挑一个 handler，把调用交给它；一个 handler 都收不下，它答一个 `DispatchError`。

```viba
# a0.viba
__decl__ =
    int
  <- $env Env
  <- $a int
  <- $b bool

args = __get_args__ << __decl__

__impl__ = args.a
```

```viba
# a1.viba
__decl__ =
    int
  <- $env Env
  <- $a bool
  <- $b str

__impl__ = 2
```

```viba
# dispatch.viba
import a0
import a1
import make_dispatcher

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ =
    make_dispatcher[a0, a1]
  << args.env << $a 1 << $b true
```

## 1. 调用与答案

```viba
make_dispatcher[a0, a1] << args.env << $a 1 << $b true       # 1（a0 收下）
make_dispatcher[a0, a1] << args.env << $a true << $b "x"    # 2（a1 收下）
make_dispatcher[a0, a1] << args.env << $a 1 << $b "x"       # $dispatch_error_message "no handler matches"
```

- 收得下，答案就是那个 handler 的答案；
- 一个都不收，答案是一个 `DispatchError`，`$dispatch_error_message` 是 `"no handler matches"`。

## 2. 类型

`make_dispatcher[f0, f1].type` 是拼出来的那个函数的类型。几个 handler 之间一致的东西，全由同一个名字
钉住：

- **结果类型**：同一个 `Ret`；
- **参数个数**：一样多；
- **参数名**：按位置一样（`$a`、`$b`……）。

每个参数位是各 handler 在该位的类型之并。两个 handler、两个参数时：

```
Either[Ret, DispatchError] <- (A0_0 | A1_0) <- (A0_1 | A1_1)
```

- `Either[Ret, DispatchError]`：两个分支一样时塌成那个类型（`Either[int, int]` 就是 `int`），不一样时
  是两者的和（`Either[int, str]` 就是 `int | str`）；
- `DispatchError = Object * $dispatch_error_message str`。

为什么这样拼：一个 handler 说的是「这个位置的参数是这几种类型里的一种」；几个 handler 的同一个位置
并起来，就是「这个位置收这几种」。调用时参数落进哪个 handler 的那一组，就把调用交给它——所以参数位
是并，结果位是各个结果的和再加上「都不收」那一支。

`.type` 给出这个类型，写出来形如 `make_dispatcher_impl[a0, a0, a1, a1].type`（`make_dispatcher_impl`
是拼出它的那一层，名字里的几个 handler 就是这次给的几个）；拿去判定、比较，认的都是上面那条链。

## 3. 约束

- **1 到 32 个 handler**，每个 **1 到 8 个参数**。
- 每个 handler 的第一个参数是 `$env Env`；它不算 dispatcher 的参数，给环境的那一位不算进参数。
- 参数名按位置一致：第一个 handler 写 `$a`、`$b`，别的 handler 在同样的位置也得是 `$a`、`$b`。
- 结果类型一致。
- **顺序有意义**：先问前面的，第一个收得下的赢。所以窄的放前面（`$a int` 在 `$a Any` 前面）。

## 4. 怎么选

按 handler 的顺序，把这次给的参数（`$env` 不算）折成一份积，问每个 handler 的参数积收不收得下；
第一个收得下的就是答案。

判定按类型，不按值：

- `1` 落进 `int`，`true` 落进 `bool`，`"x"` 落进 `str`；
- 所以同一个位置写 `$a int` 的 handler 先收下 `1`，写 `$a bool` 的收下 `true`；
- 参数个数不对（少给或多给）是程序错误，不是 `DispatchError`。

## 5. 检测几个 handler 能不能并列

参数名或结果类型不一致，决断会失败。**取 `.type` 就报**：

```
no pattern of 'make_dispatcher_impl' matches [h0, h0, h1, h1]: the decision failed
```

所以不用真的调用一次来试：

```viba
make_dispatcher[a0, a1].type     # 能取到：a0、a1 可以并列成派遣项
make_dispatcher[h0, h1].type     # 报上面那条：不能并列
```

调用时也一样，报同一条。

## 6. 已知限制

- `make_dispatcher[f0, f1]` 作为值，声明出来的是 `Any <- $env Env <- $args ...`：参数名和参数类型只在
  `.type` 里是精确的。要类型就用 `.type`，不要指望这个值本身的声明。

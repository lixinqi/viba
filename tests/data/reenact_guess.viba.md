猜数字的录制与重演。范围由 entry 从宿主取（`range_of`），所以同一份程序可以录 0..100，也可以录
0..2^30。参考侧 `reference_c` 的返回值由 `cached_reference_c` 在录制时存下；重演不跑 `reference_c`，
只读录下来的那一份。玩家 `candidate_b` 每次重演都重新跑，`compare_b_c` 拿两个返回值比较。

## `reenacted_func_a_impl.viba`

猜数字的实现：`reference_c` 是谜底，`candidate_b` 是玩家，`compare_b_c` 比较大小；`cached_reference_c`
是录制时固定下来的谜底。

```viba reenacted_func_a_impl.viba
# The guessing game: reference_c is the secret (0..range), candidate_b is the
# player's guess, cached_reference_c is the secret fixed for the replay, and
# compare_b_c answers -1 when the guess is low, 0 when it is the secret, and 1
# when it is high.
reference_c =
    int
  <- $env Env
  <- $range int
  <- { the secret, 0..range }

candidate_b =
    int
  <- $env Env
  <- $range int
  <- { the player's guess }

compare_b_c =
    int
  <- $env Env
  <- $b int
  <- $c int
  <- { -1 when b is low, 0 when b is c, 1 when b is high }

__decl__ =
    int
  <- $env Env
  <- $range int

args = __get_args__ << __decl__

cached_reference_c =
    reenact_with_reference[reference_c, "cached_reference_c"]
  << $env args.env
  << $range args.range

candidate_b_ret = candidate_b << $env args.env << $range args.range

__impl__ = compare_b_c << $env args.env << $b candidate_b_ret << $c cached_reference_c
```

## `entry.viba`

录制：范围由宿主给，一个猜测被记下来。

```viba entry.viba
# The recording: the game's range comes from the host; one guess is recorded.
import reenacted_func_a_impl

range_of =
    int
  <- $env Env
  <- { the game's range }

reenacted_func_a = reenact_entry[reenacted_func_a_impl, "guess_game"]

__decl__ =
    int
  <- $env Env

args = __get_args__ << __decl__

range = range_of << $env args.env
__impl__ = reenacted_func_a << $env args.env << $range range
```

## `replay.viba`

重演：把录下来的每一次拿出来重跑。

```viba replay.viba
# The replay: the recorded guesses, with the entry handed over.
import reenacted_func_a_impl

__decl__ =
    list[Any]
  <- $env Env

args = __get_args__ << __decl__

__impl__ =
    reenact
  << $env (args.env.sub_env << args.env << "replay")
  << $f reenacted_func_a_impl
  << $tag "guess_game"
```

"""一次调用里的实参树：嵌着的调用一层层跑出来，环境按树套下去。

一步的实参可以是一次调用，那一次调用的实参又可以是一次调用 —— 每一层都要自己那几样先算出来，
再在**自己那一层的子环境里**跑掉，跑出来的值才是上一层的实参。所以：

- 嵌 d 层 `mul`（最里面是 1，每层乘 2）给出 `2**d`，宿主恰好被问 d 次；
- 第 k 层（0 是最外那层）跑在自己的子环境里，路径里恰有 k 个 `arg0/a/mul` 段 —— 每一段都写着
  源码里的 tag 和它调的函数，路径因此能跟代码对上；两层不共用一条路径。

用例是造出来的（存临时目录，不进 `tests/data/`）：深度 20 起。

    python3 tests/test_interpreter_tree_apply.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, is_ok, value_of

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret)

checks = Checks("interpreter_tree_apply")
check = checks.check

# 深度：造出来的用例一层层嵌这么多次调用。20 起，是这一层的下界（每一层都比上一层多欠一次
# 环境，一层算错就整棵树错）。
DEPTHS = (20, 24)

# 一步的实参里嵌着的调用：`mul << $a (….) << $b 2`，最里面是 1。
HOST_STEPS = """mul =
    int
  <- $env Env
  <- $a Any
  <- $b Any
  <- { 相乘 }

echo =
    int
  <- $env Env
  <- $x Any
  <- { 原样 }
"""


def nested_call(depth: int) -> str:
    """嵌 depth 层的 `mul ... << $b 2`，最里面是 1：深度 d 给出 2**d。"""
    if depth == 0:
        return "1"
    return f"mul << $a ({nested_call(depth - 1)}) << $b 2"


def a_chain_of(depth: int) -> str:
    """一份用例：一步的实参里嵌 depth 层调用，后接一步 `echo` 取出这一步记下的值。"""
    return (
        HOST_STEPS
        + """
__decl__ =
    int
  <- $env Env

args = __get_args__ << __decl__

__impl__ =
    sequential
  << $s (mul << $a"""
        + f" ({nested_call(depth - 1)})"
        + """ << $b 2)
  << (echo << $x ($var "s"))
  << args.env
"""
    )


def a_chain_with_factors(depth: int) -> tuple:
    """(那份用例, 每一层的因子)：第 k 层（0 是最里面）乘 factors[k]。

    因子各不相同，答案就不是 `2**d` 那种"跟生成它的式子同一个样"的数：它等于各因子的积，
    可以另算一遍。
    """
    factors = [k + 2 for k in range(depth)]          # 2, 3, 4, …

    def nested(level: int) -> str:
        if level == 0:
            return "1"
        return f"mul << $a ({nested(level - 1)}) << $b {factors[level - 1]}"

    return (
        HOST_STEPS
        + """
__decl__ =
    int
  <- $env Env

args = __get_args__ << __decl__

__impl__ =
    sequential
  << $s (mul << $a"""
        + f" ({nested(depth - 1)})"
        + f" << $b {factors[depth - 1]})"
        + """
  << (echo << $x ($var "s"))
  << args.env
"""
    ), factors


def host_for(calls):
    """宿主：算 mul，原样给出实参的 echo；每次都记下（名字，跑在哪个环境）。

    `mul` 的实参是数（`a.value`）：嵌着的那一层没算出来就取不到值，所以这一记同时也说
    "每一层都真的跑了"。
    """
    def get_func(module_path, func_name):
        if func_name == "mul":
            def mul(env, a, b):
                calls.append((func_name, env.storage.cur_storage_path))
                return a.value * b.value
            return mul
        if func_name in ("echo", "builtin.echo"):
            return lambda env, x: x
        return None
    return get_func


def environ_for(store, calls):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls)), viba_path=str(store))


def _a_nested_call_in_a_step(tmp: Path):
    """一步的实参里嵌一次调用：那一次先跑出来，才是这一步的实参。"""
    cases = tmp / "nested"
    cases.mkdir(parents=True, exist_ok=True)
    path = cases / "one_nested_call.viba"
    path.write_text(a_chain_of(2))
    calls = []
    result = interpret(str(path), environ_for(tmp / "nested-one", calls))
    check(is_ok(result) and value_of(result) == 4,
          f"实参里嵌一次 mul：给出 4，{result!r}")
    check([name for name, _path in calls] == ["mul", "mul"],
          f"两层 mul 都跑了，各一次：{calls!r}")
    check(calls[0][1] != calls[1][1],
          f"两层不共用一条数据路径：{[path for _n, path in calls]!r}")
    # 数据路径上的名字跟代码对得上：这一步站在 `step0`、带 tag `s`、调的是 `mul`；它实参里那次
    # 调用站在 `arg0`、带 tag `a`、调的也是 `mul`（`viba-interpreter.md`「`sub_env` 的数据路径名字」）。
    check("/step0/s/mul/" in calls[1][1],
          f"这一步的名字写着它的 tag 和它调的函数：{calls[1][1]!r}")
    check("/arg0/a/mul/" in calls[0][1],
          f"实参里那次调用照样写着：{calls[0][1]!r}")
    check("/arg0/a/mul/" not in calls[1][1],
          f"外面那层没有这一层：{calls[1][1]!r}")
    # 解释器自己造的那几层也写着它是什么：模块的名（`sequential_impl.200`）、选中的模式文件的整名
    # （`sequential_step.300`），上一层已经写着整名时只写文件号（`sequential_impl.200/200`）。
    check("/sequential_impl.200/200/" in calls[1][1],
          f"泛型模块那一层写着它选中的文件，它自己的子环境只用文件号：{calls[1][1]!r}")
    check("/sequential_step.300/" in calls[1][1],
          f"选中的模式文件写它的整名：{calls[1][1]!r}")
    check("/tree_apply_closure_only_without_env.300/" in calls[0][1],
          f"实参那一层的模式文件也写整名：{calls[0][1]!r}")


def _each_level_gets_a_value(tmp: Path, depth: int):
    """每一层收到的都是内层算出来的**数**，不是那份闭包；因子各不相同，答案就是它们的积。

    宿主被问的次序是最里面一层先、往外一层层收，所以第 k 次收到的第一位实参应当是
    `factors[0] * ... * factors[k-1]`（另算一遍），第二位是 `factors[k]`。
    """
    cases = tmp / f"factors-{depth}"
    cases.mkdir(parents=True, exist_ok=True)
    path = cases / f"{depth}_distinct_factors.viba"
    text, factors = a_chain_with_factors(depth)
    path.write_text(text)
    seen = []

    def host_get_func(module_path, func_name):
        if func_name == "mul":
            def mul(env, a, b):
                seen.append((a.value, b.value))
                return a.value * b.value
            return mul
        if func_name in ("echo", "builtin.echo"):
            return lambda env, x: x
        return None

    env = Environment(EnvironmentStorage("root", None, str(tmp / f"store-factors-{depth}")),
                      EnvironmentCompute(host_get_func),
                      viba_path=str(cases))
    result = interpret(str(path), env)
    want, running = 1, 1
    for factor in factors:
        want *= factor
    check(is_ok(result) and value_of(result) == want,
          f"嵌 {depth} 层、因子各不同，给出各因子的积 {want}：{result!r}")
    expected = []
    for factor in factors:
        expected.append((running, factor))
        running *= factor
    check(seen == expected,
          f"第 k 层收到的是上一层的数与自己的因子：{seen!r}")


def _the_tree_of_environments(tmp: Path, depth: int):
    """深度 depth 的链：值、跑过几次、每一层跑在哪，都对一遍。"""
    cases = tmp / f"depth-{depth}"
    cases.mkdir(parents=True, exist_ok=True)
    path = cases / f"{depth}_nested_calls.viba"
    path.write_text(a_chain_of(depth))
    calls = []
    result = interpret(str(path), environ_for(tmp / f"store-{depth}", calls))
    check(is_ok(result) and value_of(result) == 2 ** depth,
          f"嵌 {depth} 层给出 {2 ** depth}：{result!r}")
    check(len(calls) == depth,
          f"嵌 {depth} 层就是 {depth} 次调用：{len(calls)}")
    paths = [where for _name, where in calls]
    check(len(set(paths)) == depth,
          f"每一层跑在自己的数据路径上，没有两条一样：{paths!r}")
    # 最里面那层先被问，往外一层层收；第 k 层（0 最外）的路径里恰有 k 个 `arg0` 段：
    # 实参树每往里一层就多套一个子环境。这些段上写着 tag 和函数名（`arg0/a/mul`），所以路径
    # 跟代码对得上；最外那层是这一步自己（`step0/s/mul`）。
    levels = list(reversed(paths))
    for level, where in enumerate(levels):
        before_run = where.split("/run/", 1)[0]
        check(before_run.count("/arg0/") == level,
              f"第 {level} 层的路径里应恰有 {level} 个 arg0 段：{where!r}")
        check(before_run.count("/arg0/a/mul") == level,
              f"第 {level} 层的路径里应恰有 {level} 个 `arg0/a/mul` 段：{where!r}")
        check("/step0/s/mul/" in where,
              f"第 {level} 层都在这一步的数据路径下面，那一段写着 tag 和函数名：{where!r}")


def _the_last_step_carries_no_tag(tmp: Path):
    """最后那一步没有 tag：它的名字就是 `last` 加上它调的函数（`last/mul`）。"""
    cases = tmp / "last-step"
    cases.mkdir(parents=True, exist_ok=True)
    path = cases / "last_step.viba"
    path.write_text(HOST_STEPS + """
__decl__ =
    int
  <- $env Env

args = __get_args__ << __decl__

__impl__ =
    sequential
  << $x (mul << $a 3 << $b 4)
  << (mul << $a ($var "x") << $b 5)
  << args.env
""")
    calls = []
    result = interpret(str(path), environ_for(tmp / "store-last-step", calls))
    check(is_ok(result) and value_of(result) == 60,
          f"第一步 12 记在 `x` 下，最后一步取它乘 5：{result!r}")
    check([name for name, _path in calls] == ["mul", "mul"],
          f"两步各问一次 mul：{calls!r}")
    check("/step0/x/mul/" in calls[0][1],
          f"第一步的名字写着它的 tag 和它调的函数：{calls[0][1]!r}")
    check("/last/mul/" in calls[1][1],
          f"最后那一步没有 tag，名字是 `last` 加它调的函数：{calls[1][1]!r}")


def run(tmp: Path):
    _a_nested_call_in_a_step(tmp)
    _the_last_step_carries_no_tag(tmp)
    for depth in DEPTHS:
        _the_tree_of_environments(tmp, depth)
        _each_level_gets_a_value(tmp, depth)


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-tree-apply-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

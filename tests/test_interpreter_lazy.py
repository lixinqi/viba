"""函数类型的槽：写在那一格上的实参不在这里算，宿主叫它的时候才算。

每条用例是一份可以打开的文件（`tests/data/lazy/*.viba`），这里只列它该跑出什么。宿主那一边
只叫它想叫的那个实参，所以没走的那一支既不做副作用、也不会因为没有实现而挡路。同一份分支文件
在门槛不同时走不同的那一支，那十份 poison 文件把毒放在走不到的那一支里。

    python3 tests/test_interpreter_lazy.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, value_of

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import NoImplementationException, Ok, UnderlyingVibaOpFailed, VibaProgramErr

checks = Checks("interpreter_lazy")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "lazy"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def host_for(calls, knobs):
    """计数宿主：每被算一次就记一笔；门槛那一步把一个旋钮交出来。"""
    def get_func(module_path, func_name):
        if func_name == "threshold_of":
            return lambda env: knobs["threshold"]
        if func_name == "tick":
            def tick(env):
                calls.append("tick")
                return 1
            return tick
        if func_name == "tock":
            def tock(env):
                calls.append("tock")
                return 2
            return tock
        if func_name == "ignore_x":
            # 完全不碰 get_x：那个实参不该被算
            return lambda env, get_x: 7
        if func_name == "take_x":
            # 叫了 get_x：那个实参这时才算
            return lambda env, get_x: get_x(env).value
        if func_name == "eager_pair":
            # 没有函数类型的槽：两个实参都要算
            return lambda env, x, y: 1
        if func_name == "ge":
            return lambda env, x, y: x.value >= y.value
        if func_name == "condition_holds":
            return lambda env: True
        if func_name == "builtin.echo":
            return lambda env, x: x
        if func_name == "inner_lambda_record":
            # 分支值本身也是一次调用：它被算过就说明那一支走了
            def inner_lambda_record(env, label):
                calls.append(label.value)
                return 0
            return inner_lambda_record
        if func_name == "ask_twice":
            # 同一个实参问两次：只该算一次
            def ask_twice(env, get_x):
                return get_x(env).value + get_x(env).value
            return ask_twice
        if func_name == "ask_nothing":
            # 两个实参都不问
            return lambda env, get_a, get_b: 0
        if func_name == "positional":
            # 两个函数类型的槽都收，按写下来的顺序给
            def positional(env, get_cond, get_v):
                return get_v(env).value if get_cond(env).value else 0
            return positional
        if func_name == "poison_raises_when_asked":
            def poison_raises_when_asked(env):
                calls.append("boom")
                raise ZeroDivisionError("boom")
            return poison_raises_when_asked
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(store, calls, knobs=None):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls, knobs or {"threshold": 0})),
                       viba_path=str(REPOSITORY_ROOT))


# (文件, 该跑出什么, 要看住的副作用, 门槛)：
#   value   Ok，叶子是这个值
#   error   VibaProgramErr，话里含这个片段
#   no_implementation   没有实现：宿主没实现那一步
#   fail    UnderlyingVibaOpFailed，话里含这个片段
CASES_TO_RUN = [
    # 只有选中那一支的实参被算：门槛由宿主给，同一个文件跑两个方向
    ("branches", "value", 1, ["tick"], 0),
    ("branches", "value", 2, ["tock"], 5),
    # 分支值写成一次调用：只有走的那一支留下记录
    ("recorded_branches", "value", 0, ["true_branch"], 0),
    ("recorded_branches", "value", 0, ["false_branch"], 5),
    # 分支背后那一步没有实现：报的就是那一步，而且它没被算过
    ("recorded_missing", "no_implementation", None, [], 0),
    # 宿主不叫那个实参：它一次都不算，没有实现也不挡路
    ("ignored_argument", "value", 7, [], 0),
    # 那一格还没给、别的先给了：这个调用存不下来
    ("half_given_slot", "error", "was given 1 of its 2 arguments", [], 0),
    # 一个实参只算一次：问两次不等于做两遍
    ("ask_twice", "value", 2, ["tick"], 0),
    ("ask_nothing", "value", 0, [], 0),
    ("ask_twice_failing", "fail", "poison_raises_when_asked raised", ["boom"], 0),
    # 两个函数类型的槽：位置实参与乱序 tag 都按参数位置交给宿主
    ("positional", "value", 1, ["tick"], 0),
    ("out_of_order", "value", 1, ["tick"], 0),
    # 没有函数类型的槽：一切照旧，实参先算
    ("eager_pair", "value", 1, ["tick"], 0),
    # 叫了那个实参，算它时出的事照常报出来
    ("called_missing", "no_implementation", None, [], 0),
    ("called_boom", "error", "is not a function", [], 0),
    ("no_environment", "error", "was not given an Environment", [], 0),
]


def run(tmp: Path):
    check(len(CASES_TO_RUN) == 16, f"sixteen cases: {len(CASES_TO_RUN)}")
    for index, (name, kind, want, calls_wanted, threshold) in enumerate(CASES_TO_RUN):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        calls = []
        result = interpret(str(program),
                           environ_for(tmp / f"store-{index}", calls,
                                       {"threshold": threshold}))
        if kind == "value":
            check(isinstance(result, Ok) and value_of(result) == want,
                  f"{name}: expected {want!r}, got {result!r}")
        elif kind == "error":
            check(isinstance(result, VibaProgramErr) and want in result.err_msg,
                  f"{name}: expected an error saying {want!r}, got {result!r}")
        elif kind == "no_implementation":
            check(isinstance(result, NoImplementationException),
                  f"{name}: expected a step with no implementation, got {result!r}")
        elif kind == "fail":
            check(isinstance(result, UnderlyingVibaOpFailed) and want in result.msg,
                  f"{name}: expected a failure saying {want!r}, got {result!r}")
        if calls_wanted is not None:
            check(calls == calls_wanted,
                  f"{name}: expected the side effects {calls_wanted}, got {calls}")

    # 走不到的那一支放毒：那十份文件可以直接打开，Ok(42) 就是"那一支没被算"的证明
    poison = sorted(CASES.glob("poison_*.viba"))
    check(len(poison) == 10, f"ten poison cases are on disk: {len(poison)}")
    for index, program in enumerate(poison):
        result = interpret(str(program), environ_for(tmp / f"store-poison{index}", []))
        check(isinstance(result, Ok) and value_of(result) == 42,
              f"poison in the branch that is not taken ({program.name}): {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-lazy-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

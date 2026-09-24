"""惰性参数：`ParametersLazyEvaluated` 标记的函数，实参不先算。

被标记的函数（`branch.viba` 的两个开关就是一个例子）不按平常的方式调用：`interpret` 把每个
写下来的实参包成一个无参 lambda 交给宿主，宿主叫哪个才算哪个。于是"没走的那一支"不会被求值——
这才是 if/else；不然两条都算完再丢掉一条，只是结果一样。

    python3 tests/test_interpreter_lazy.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, Host, value_of, write

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import NotMyDutyException, Ok, UnderlyingVibaOpFailed, VibaProgramErr

checks = Checks("interpreter_lazy")
check = checks.check
labelled = checks.labelled


def host_for(calls):
    """计数宿主：`tick` / `tock` 每被算一次就记一笔。"""
    def get_func(module_path, func_name):
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
            return lambda get_env, get_x: 7
        if func_name == "take_x":
            # 叫了 get_x：那个实参这时才算
            return lambda get_env, get_x: get_x().value
        if func_name == "eager_pair":
            # 没被标记的函数：两个实参都要算（老行为）
            return lambda env, x, y: 1
        if func_name == "ge":
            return lambda env, x, y: x.value >= y.value
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(calls, store):
    # `import branch` finds branch.viba the way the branch suite does: the
    # checkout root is on the module search path.
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls)),
                       viba_path=str(Path(__file__).resolve().parent.parent))


BRANCHES = """
import branch

tick =
    int <- $env Environment <- { the branch that is taken first }
tock =
    int <- $env Environment <- { the branch that is taken second }
ge =
    bool <- $env Environment <- $x int <- $y int <- { x >= y }

condition = ge << $env environ << $x 1 << $y THRESHOLD
__ret__ =
  Oneof
  | (branch.id_or_never << $env environ << $condition condition << $v (tick << $env environ))
  | (branch.never_or_nil << $env environ << $condition condition << $v (tock << $env environ))
"""


def run(tmp: Path):
    _only_the_taken_branch_is_computed(tmp)
    _an_ignored_argument_is_never_computed(tmp)
    _an_unmarked_function_is_still_eager(tmp)
    _a_called_getter_carries_what_stopped(tmp)
    _the_marker_marks_functions(tmp)


def _only_the_taken_branch_is_computed(tmp: Path):
    """branch.viba 的开关：只有选中那一支的实参被算。"""
    taken_first = write(tmp, "taken_first.viba", BRANCHES.replace("THRESHOLD", "0"))
    calls = []
    result = interpret(taken_first, environ_for(calls, tmp / "store-a"))
    check(isinstance(result, Ok) and value_of(result) == 1, f"the first branch: {result!r}")
    check(calls == ["tick"],
          f"and only the first branch's value was computed: {calls}")

    taken_second = write(tmp, "taken_second.viba", BRANCHES.replace("THRESHOLD", "5"))
    calls = []
    result = interpret(taken_second, environ_for(calls, tmp / "store-b"))
    check(isinstance(result, Ok) and value_of(result) == 2, f"the second branch: {result!r}")
    check(calls == ["tock"],
          f"and only the second branch's value was computed: {calls}")


def _an_ignored_argument_is_never_computed(tmp: Path):
    """宿主不叫那个 getter，实参就不算——哪怕它根本没有实现。"""
    program = write(tmp, "ignored.viba", """
ignore_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer seven without looking at x }
    ]

ghost =
    int <- $env Environment <- { nothing implements this }

__ret__ = ignore_x << $env environ << $x (ghost << $env environ)
""")
    calls = []
    result = interpret(program, environ_for(calls, tmp / "store-c"))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"an argument nobody asks for is not computed, so its missing "
          f"implementation never shows: {result!r}")

    # 先给一半、再给另一半：惰性跟着走
    half = write(tmp, "half.viba", """
ignore_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer seven without looking at x }
    ]

ghost =
    int <- $env Environment <- { nothing implements this }

half = ignore_x << $env environ
__ret__ = half << $x (ghost << $env environ)
""")
    calls = []
    result = interpret(half, environ_for(calls, tmp / "store-d"))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a marked function stays lazy through a partial application: {result!r}")


def _an_unmarked_function_is_still_eager(tmp: Path):
    """没标记的函数一切照旧：每个实参都先算出来。"""
    program = write(tmp, "eager.viba", """
eager_pair =
    int
  <- $env Environment
  <- $x int
  <- $y int
  <- { ignore both arguments, but they are computed first }

tick =
    int <- $env Environment <- { a value with a side effect }

__ret__ = eager_pair << $env environ << $x (tick << $env environ) << $y 2
""")
    calls = []
    result = interpret(program, environ_for(calls, tmp / "store-e"))
    check(isinstance(result, Ok) and value_of(result) == 1,
          f"an unmarked call still answers: {result!r}")
    check(calls == ["tick"],
          f"and its arguments were computed eagerly, the way they always were: {calls}")


def _a_called_getter_carries_what_stopped(tmp: Path):
    """宿主叫了那个 getter，实参算的时候出的事就照常报出来。"""
    missing = write(tmp, "called_missing.viba", """
take_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer whatever x is }
    ]

ghost =
    int <- $env Environment <- { nothing implements this }

__ret__ = take_x << $env environ << $x (ghost << $env environ)
""")
    calls = []
    result = interpret(missing, environ_for(calls, tmp / "store-f"))
    check(isinstance(result, NotMyDutyException),
          f"a getter that is called and has no implementation defers, as ever: {result!r}")

    boomed = write(tmp, "called_boom.viba", """
take_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer whatever x is }
    ]

__ret__ = take_x << $env environ << $x (1 << $x 2)
""")
    calls = []
    labelled(interpret(boomed, environ_for(calls, tmp / "store-g")), "is not a function",
             "an argument that is broken when the getter runs -> the same err, "
             "not a failure of the host that asked for it")


def _the_marker_marks_functions(tmp: Path):
    """标记只标函数；漏了 environ 也照旧是错。"""
    not_a_function = write(tmp, "not_a_function.viba", """
wrong =
    ParametersLazyEvaluated[int]

__ret__ = wrong << $env environ
""")
    calls = []
    result = interpret(not_a_function, environ_for(calls, tmp / "store-h"))
    check(isinstance(result, VibaProgramErr) and
          "ParametersLazyEvaluated marks a function" in result.err_msg,
          f"the marker around a non-function says so: {result!r}")

    no_env = write(tmp, "no_env.viba", """
ignore_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer seven without looking at x }
    ]

__ret__ = ignore_x << 5 << 1
""")
    calls = []
    labelled(interpret(no_env, environ_for(calls, tmp / "store-i")),
             "was not given an Environment",
             "a marked function still needs its environment")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-lazy-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

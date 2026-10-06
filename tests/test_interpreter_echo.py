"""echo 开关：分支值是一个函数，由拿到它的一方在自己挑的环境里跑。

`branch.echo_or_never` 收 `$get_v (Any <- $env Environment)`。拿到它的一方决定算不算、在哪个
环境下算 —— `branch.py` 给它一个子环境（`sub_env(env, "echo_or_never")`），所以那一支是在自己的
路径下算出来的，`get_func` 看到的 `module_path` 就是它跑的那条路径。

    python3 tests/test_interpreter_echo.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import error_of, message_of, Checks, value_of

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import Ok

CASES = Path(__file__).resolve().parent / "data" / "echo"

checks = Checks("interpreter_echo")
check = checks.check
labelled = checks.labelled


def host_for(calls):
    def get_func(module_path, func_name):
        if func_name == "leaf":
            def leaf(env):
                calls.append((module_path, "leaf"))
                return 42
            return leaf
        if func_name == "builtin.echo":
            def echo(env, x):
                calls.append((module_path, "echo"))
                return x
            return echo
        return branch.get_func(module_path, func_name)
    return Environment(EnvironmentStorage("root", None, None),
                       EnvironmentCompute(get_func),
                       viba_path=str(CASES.parents[2]))


def run(tmp: Path):
    _never_or_echo_takes_the_other_side()
    _the_branch_runs_in_its_own_environment()
    _the_other_branch_is_not_computed()
    _echo_hands_back_a_written_piece()


def _the_branch_runs_in_its_own_environment():
    """条件成立：那一支跑了，而且跑在宿主给它造的子环境里。"""
    calls = []
    result = interpret(str(CASES / "branch_in_its_own_environment.viba"), host_for(calls))
    check(isinstance(result, Ok) and value_of(result) == 42, f"结果来自那一支：{result!r}")
    check([where for where, _ in calls] == ["root/echo_or_never"],
          f"它在自己的子环境里跑：{calls}")


def _the_other_branch_is_not_computed():
    """条件不成立：那一支一次都不算，和里只剩另一支。"""
    calls = []
    result = interpret(str(CASES / "the_other_branch.viba"), host_for(calls))
    check(isinstance(result, Ok) and value_of(result) == 7, f"和里只剩另一支：{result!r}")
    check(calls == [], f"没走的那一支没有被算：{calls}")


def _never_or_echo_takes_the_other_side():
    """反过来的那个：条件不成立时才跑，也在自己的子环境里。"""
    calls = []
    result = interpret(str(CASES / "never_or_echo_takes_it.viba"), host_for(calls))
    check(isinstance(result, Ok), f"那一支跑起来了：{result!r}")
    check(isinstance(result, Ok) and value_of(result) == 42, f"结果来自那一支：{result!r}")
    check([where for where, _ in calls] == ["root/never_or_echo"],
          f"它也在自己的子环境里跑：{calls}")


def _echo_hands_back_a_written_piece():
    """`builtin.echo` 交回写下来的那份东西：宿主跑它，交回来的就是它。"""
    calls = []
    result = interpret(str(CASES / "echo_hands_back_a_written_piece.viba"), host_for(calls))
    check(isinstance(result, Ok) and value_of(result) == 7, f"给出的是那份写下来的东西：{result!r}")
    check([name for _, name in calls] == ["echo"], f"跑的是 echo 那一步：{calls}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-echo-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

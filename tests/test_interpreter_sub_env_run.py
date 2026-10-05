"""内建目录里的 sub_env_run（`viba/sub_env_run.viba`）：把一个调用跑到它自己的子环境里。

    sub_env_run << $sub_env_name << env << f << ...
        ===  f << ($sub_env << env << sub_env_name) << ...

`sub_env_run` 是内建目录里的一个模块，跟 `Y`、`apply` 一样：名字对每个模块可见，所以写它不用
import（带前缀的 `builtin.sub_env_run` 叫的是同一个模块）。它的 `$args ...` 把链上剩下的实参收成
一份积，交给 `apply` 摊开——所以 `f` 拿到几个实参就写几个。

用例在 `tests/data/sub_env_run/`：

    helper.viba                  f：两个整数相加
    sum3.viba                    f：三个整数相加
    at_the_named_child.viba      名字与环境的那个位置：`sub_env_name`、环境、f、剩下的实参
    the_qualified_name.viba      `builtin.sub_env_run` 叫的是同一个模块
    the_rest_is_f_s_arguments.viba `...` 收下的那几个摊开就是那次调用的实参
    the_arguments_are_computed_here.viba 实参算在调用方那一层，f 跑在子环境里
    the_tagged_call.viba         每个实参都写全 tag
    a_nameless_parent.viba       给它的环境叫什么无所谓
    a_module_of_its_own.viba     自己模块写了这个名字，内建目录里那个就被压住了

宿主记下每一步的 `module_path` 与名字，套件拿数据路径当证据：`f` 的那一步跑在给它的环境的名字底下
（`root/run/low/...`），而调用方写的实参算在调用方那一层（`root`）。

    python3 tests/test_interpreter_sub_env_run.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, value_of

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret)
from viba.type import Ok

checks = Checks("interpreter_sub_env_run")
check = checks.check
labelled = checks.labelled

CASES = Path(__file__).resolve().parent / "data" / "sub_env_run"

# 跑在子环境名字底下的那几步（f 里的 `builtin.add`）
UNDER_THE_NAME = "root/run/low"


def host_for(calls):
    """宿主：`builtin.add`、用例自己写的 `here`、以及那个遮蔽用例的 `sub_env_run`。"""
    def get_func(module_path, func_name):
        calls.append((module_path, func_name))
        if func_name == "builtin.add":
            return lambda env, x, y: x.value + y.value
        if func_name == "here":
            return lambda env: 1
        if func_name == "sub_env_run":
            return lambda env: 1
        return None
    return get_func


def environ_for(store, calls):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls)),
                       viba_path=str(CASES))


def run(tmp: Path):
    _under_the_name(tmp, "at_the_named_child", "the bare name", 9)
    _under_the_name(tmp, "the_qualified_name", "the qualified name", 9)
    _under_the_name(tmp, "the_tagged_call", "every argument tagged", 9)
    _a_nameless_parent(tmp)
    _the_rest_is_f_s_arguments(tmp)
    _the_arguments_are_computed_here(tmp)
    _a_module_may_shadow_it(tmp)


def _a_module_may_shadow_it(tmp: Path):
    """自己模块里写了这个名字，读到的就是它：内建目录里那个排在任何别的名字之后。"""
    calls = []
    result = interpret(str(CASES / "a_module_of_its_own.viba"),
                       environ_for(tmp / "shadow", calls))
    check(isinstance(result, Ok) and value_of(result) == 1,
          f"its own sub_env_run answered 1: {result!r}")
    check(("root", "sub_env_run") in calls and
          not [one for one in calls if one[1].startswith("builtin.")],
          f"the module's own definition is what ran: {calls}")


def _under_the_name(tmp: Path, name: str, label: str, want):
    """`f` 跑在给它的环境的那名字底下，跑出来的就是 f 给出的那个值。"""
    calls = []
    result = interpret(str(CASES / f"{name}.viba"), environ_for(tmp / name, calls))
    check(isinstance(result, Ok) and value_of(result) == want,
          f"{name}: {label} answers {want!r}, got {result!r}")
    places = [where for where, what in calls if what == "builtin.add"]
    check(bool(places) and all(one.startswith(UNDER_THE_NAME) for one in places),
          f"{name}: f's own step ran under the name it was given: {calls}")


def _a_nameless_parent(tmp: Path):
    """给它的环境叫什么无所谓：`low` 只是那个环境的孩子。"""
    calls = []
    result = interpret(str(CASES / "a_nameless_parent.viba"),
                       environ_for(tmp / "nameless", calls))
    check(isinstance(result, Ok) and value_of(result) == 9,
          f"4 + 5 = 9 under a nameless parent: {result!r}")
    places = [where for where, what in calls if what == "builtin.add"]
    check(bool(places) and all(one.startswith("root/tmp_") and "/low/" in one
                              for one in places),
          f"the name is a child of the environment given, whatever it is: {calls}")


def _the_rest_is_f_s_arguments(tmp: Path):
    """`...` 收下的那份积摊开就是 f 的实参：写三个，f 就收到三个。"""
    calls = []
    result = interpret(str(CASES / "the_rest_is_f_s_arguments.viba"),
                       environ_for(tmp / "rest", calls))
    check(isinstance(result, Ok) and value_of(result) == 6,
          f"1 + 2 + 3 = 6: {result!r}")
    places = [where for where, what in calls if what == "builtin.add"]
    check(len(places) == 2 and all(one.startswith(UNDER_THE_NAME) for one in places),
          f"a three-argument call is spread into the call it makes: {calls}")


def _the_arguments_are_computed_here(tmp: Path):
    """实参算在调用方那一层；f 跑在给它的环境的子环境里。"""
    calls = []
    result = interpret(str(CASES / "the_arguments_are_computed_here.viba"),
                       environ_for(tmp / "here", calls))
    check(isinstance(result, Ok) and value_of(result) == 6,
          f"the argument answered 1, and 1 + 5 = 6: {result!r}")
    check(("root", "here") in calls,
          f"the argument was worked out where it was written: {calls}")
    places = [where for where, what in calls if what == "builtin.add"]
    check(bool(places) and all(one.startswith(UNDER_THE_NAME) for one in places),
          f"and the call itself ran under the name: {calls}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-sub-env-run-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

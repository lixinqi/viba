"""`<<`：给参数这一件事——给几个、给谁、什么时候算。

写出来的实参按书写顺序算；说明块不是实参；给多了、给错了、给的不是函数都是 VibaProgramErr。
每条用例是一份可以打开的文件（`tests/data/application/*.viba`），被调的那个函数在
`arithmetic.viba` 里，宿主的实现按名字认它，所以这里只列每份文件该跑出什么。

    python3 tests/test_interpreter_application.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import (error_of, full_name_of, message_of,
                                  module_of, module_path_of, Checks, Host, value_of)

from viba.reflect import access as reflect_access

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import VibaProgramErr, Ok

checks = Checks("interpreter_application")
check = checks.check
labelled = checks.labelled

CASES = Path(__file__).resolve().parent / "data" / "application"


def run(tmp: Path):
    _arguments(tmp)
    _order_and_slots(tmp)
    _argument_types(tmp)
    _lazy_argument_types(tmp)
    _long_chain(tmp)


# (文件, 该跑出什么)：want 为 None 表示 Ok。
ARGUMENT_CASES = [
    ("all_given", None, "all three given"),
    ("one_short", "was given 2 of its 3 arguments", "one argument short (the environment is in)"),
    ("one_too_many", "takes no more arguments", "one argument too many (every slot is filled)"),
    ("no_such_tag", "takes no $z", "an argument the function does not have"),
    ("note_after", None, "documentation after the arguments"),
    ("note_between", None, "documentation between the arguments"),
    ("bare_name", None, "a bare function name is the closure it stands for"),
    ("code_block_only", "documentation", "a code block is not a value"),
    ("undefined_name", "no definition named", "a name nothing defines"),
    ("tagged_note", "documentation", "a tagged code block where an argument goes"),
    ("number_given_an_argument", "is not a function", "giving an argument to a number"),
]


def _arguments(tmp: Path):
    """少给、多给、给错、说明块、分两步。"""
    host = Host()
    environ = host.environ()
    for name, want, label in ARGUMENT_CASES:
        labelled(interpret(str(CASES / f"{name}.viba"), environ), want, label)

    result = interpret(str(CASES / "two_steps.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"a partially applied function kept in a definition: {result!r}")

    result = interpret(str(CASES / "positional.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"arguments written without tags bind in order: {result!r}")


def _order_and_slots(tmp: Path):
    """书写顺序与参数位：tag 的次序不影响结果；没有参数位的函数给不了实参。"""
    host = Host()
    environ = host.environ()

    result = interpret(str(CASES / "out_of_order.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"tags may be given in any order: {result!r}")

    labelled(interpret(str(CASES / "no_slots.viba"), environ),
             "takes no $env Env parameter",
             "a function with no environment slot can never run -> VibaProgramErr")

    # 参数出错：那个函数根本不会被调用
    host.calls.clear()
    exploded = interpret(str(CASES / "argument_boom.viba"), environ)
    checks.failed(exploded, "raised", "an argument that blows up")
    check(error_of(exploded).module_path == module_path_of(CASES / "argument_boom")
          and error_of(exploded).full_qualified_func_name ==
          full_name_of(CASES / "argument_boom", "explode"),
          f"the step that stopped is the argument's, not the call's: {exploded!r}")
    check((module_of(CASES / "argument_boom"), "add") not in host.calls,
          f"the call itself never happens: {host.calls}")


# (文件, 该跑出什么)：want 为 None 表示 Ok。
SLOT_CASES = [
    ("slot_0", "does not fit $a int", "a string in an int slot"),
    ("slot_1", "does not fit $b int", "and in the second slot"),
    ("slot_2", "does not fit $a int", "a bool is no int"),
    ("slot_3", "does not fit $a int", "the environment in an int slot"),
    ("slot_4", "does not fit $a int", "nil in an int slot"),
    ("slot_5", "does not fit $p", "a product whose member does not fit"),
    ("slot_6", None, "a literal that does fit goes through"),
    ("slot_7", None, "and so does a product that fits"),
]


def _argument_types(tmp: Path):
    """实参要装得下那个参数：装不下是**程序错**，宿主还没看见这个实参。

    值层现在也用得上判定的那套 <:：字面量、可序列化数据、环境都有写下来的类型。
    判不出类型的（宿主自己的值、判定层 settle 不了的）照旧放过去，交给实现那一步的人。
    """
    def get_func(path, func_name):
        if func_name == "x_of":
            return lambda env, point: reflect_access.leaf(point.by_tag("x")).ok_value
        return Host().get_func(path, func_name)

    environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))
    for name, want, label in SLOT_CASES:
        labelled(interpret(str(CASES / f"{name}.viba"), environ), want, label)


def _lazy_argument_types(tmp: Path):
    """函数类型的那个实参不先算：宿主叫它的时候才核，叫了才错，不叫就不错。"""
    def get_func(path, func_name):
        if func_name == "watch":
            return lambda env, x: x(env)      # 叫了那个实参：这时才算、才核
        if func_name == "ignore":
            return lambda env, x: 7           # 不叫它：那个实参一次都不算
        if path == "builtin" and func_name == "echo":
            return lambda env, x: x
        return Host().get_func(path, func_name)

    host = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))
    labelled(interpret(str(CASES / "lazy_asked.viba"), host), "does not fit int",
             "a function-typed slot is checked when the host asks")
    result = interpret(str(CASES / "lazy_ignored.viba"), host)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"and an argument nobody asks for is never checked: {result!r}")


def _long_chain(tmp: Path):
    """二十个实参的一条链。"""
    host = Host()
    original = host.get_func

    def summing(path, func_name):
        if func_name == "sum":
            return lambda env, *rest: sum(v.value for v in rest)
        return original(path, func_name)

    host.get_func = summing
    environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(host.get_func))
    result = interpret(str(CASES / "long_chain.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 210,
          f"a twenty-argument chain: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-application-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

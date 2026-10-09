"""`<<`：给参数这一件事——给几个、给谁、什么时候算。

给出的实参按源码里的顺序算；说明块不是实参；给多了、给的不是函数都是 VibaProgramErr。给实参这一步
不核类型：`$a int` 那里给 `"x"` 也照给，撞上的是那一步的实现。
每条用例是一份可以打开的文件（`tests/data/application/*.viba`），被调的那个函数在
`arithmetic.viba` 里，宿主的实现按名字认它，所以这里只列每份文件该跑出什么。

    python3 tests/test_interpreter_application.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import (full_name_of, module_of, module_path_of,
                                  stop_text, Checks, Host, is_ok, value_of)

from viba.reflect import access as reflect_access

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret

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


# (文件, 该跑出什么)：want 为 None 表示跑出一个值。
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
    check(is_ok(result) and value_of(result) == 42,
          f"a partially applied function kept in a definition: {result!r}")

    result = interpret(str(CASES / "positional.viba"), environ)
    check(is_ok(result) and value_of(result) == 42,
          f"arguments given without tags bind in order: {result!r}")


def _order_and_slots(tmp: Path):
    """源码里的顺序与参数位：tag 的次序不影响结果；没有参数位的函数给不了实参。"""
    host = Host()
    environ = host.environ()

    result = interpret(str(CASES / "out_of_order.viba"), environ)
    check(is_ok(result) and value_of(result) == 3,
          f"tags may be given in any order: {result!r}")

    labelled(interpret(str(CASES / "no_slots.viba"), environ),
             "takes no $env Env parameter",
             "a function with no environment slot can never run -> VibaProgramErr")

    # 参数出错：那个函数根本不会被调用
    host.calls.clear()
    exploded = interpret(str(CASES / "argument_boom.viba"), environ)
    checks.failed(exploded, "raised", "an argument that blows up")
    check(stop_text(exploded, "$module_path") == module_path_of(CASES / "argument_boom")
          and stop_text(exploded, "$full_qualified_func_name") ==
          full_name_of(CASES / "argument_boom", "explode"),
          f"the step that stopped is the argument's, not the call's: {exploded!r}")
    check((module_of(CASES / "argument_boom"), "add") not in host.calls,
          f"the call itself never happens: {host.calls}")


# (文件, 该跑出什么)：给实参这一步不核类型，所以装不下的那些照给 —— 那一步的实现拿它去做原来
# 那件事，撞上了就在那里报（`$underlying_viba_op_err`，话里是 raised）。
SLOT_CASES = [
    ("slot_0", "raised", "a string in an int slot reaches the step and breaks it there"),
    ("slot_1", "raised", "and in the second slot"),
    ("slot_3", "raised", "the environment in an int slot"),
    ("slot_4", "raised", "nil in an int slot"),
]

# (文件, 该跑出什么值)：装不下、但照用得下去的几种
SLOT_VALUES = [
    ("slot_2", 2, "a bool in an int slot adds as one (true + 1)"),
    ("slot_5", "s", "a product whose member does not fit is handed over as it is"),
    ("slot_6", 7, "a literal that fits"),
    ("slot_7", 1, "and so does a product that fits"),
]


def _argument_types(tmp: Path):
    """给一个实参不核类型：设计说 `$a int`，给一个字符串也照给。

    判定层那套 <: 判的是**设计** —— 把一个函数链判成某个类型、给一份设计取实参 —— 不是这一次
    调用给的这一个值。所以装不下的实参也交到那一步的实现手里：它拿它做整数运算就在那里撞上
    （`$underlying_viba_op_err`），照用得下去的（`true + 1`）就照答。
    """
    def get_func(path, func_name):
        if func_name == "x_of":
            return lambda env, point: reflect_access.leaf(point.by_tag("x")).ok_value
        return Host().get_func(path, func_name)

    environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))
    for name, want, label in SLOT_CASES:
        checks.failed(interpret(str(CASES / f"{name}.viba"), environ), want, label)
    for name, want, label in SLOT_VALUES:
        result = interpret(str(CASES / f"{name}.viba"), environ)
        check(is_ok(result) and value_of(result) == want, f"{label}, got {result!r}")


def _lazy_argument_types(tmp: Path):
    """函数类型的那个实参不先算：宿主叫它的时候才算，算了也不核类型。"""
    def get_func(path, func_name):
        if func_name == "watch":
            return lambda env, x: x(env)      # 叫了那个实参：这时才算
        if func_name == "ignore":
            return lambda env, x: 7           # 不叫它：那个实参一次都不算
        if path == "builtin" and func_name == "echo":
            return lambda env, x: x
        return Host().get_func(path, func_name)

    host = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))
    result = interpret(str(CASES / "lazy_asked.viba"), host)
    check(is_ok(result) and value_of(result) == "x",
          f"what the call in a function-typed slot answers is handed over as it is: "
          f"{result!r}")
    result = interpret(str(CASES / "lazy_ignored.viba"), host)
    check(is_ok(result) and value_of(result) == 7,
          f"and an argument nobody asks for is never computed: {result!r}")


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
    check(is_ok(result) and value_of(result) == 210,
          f"a twenty-argument chain: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-application-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

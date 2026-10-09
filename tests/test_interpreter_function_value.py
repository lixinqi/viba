"""函数值：一个值知道自己是不是函数、叫什么、定义在哪个模块里。

Python 解释器见到 `twice` 不能只看到一段文字。`twice` 是值系统里的一份函数数据：它带着自己的
名字，带着定义它的那个模块（连同那个模块的名字、模块被认得的路径、模块取自哪个文件）。这些由
`VObject.function`（`viba.reflect.VibaFunction`）带着，宿主拿到 `$f Any` 这样的槽位时看到的就是它。

只有函数与闭包这么处理：`1 * 2` 是构造出来的值，它没有"定义在哪个模块"这回事，`function` 是 None。

    python3 tests/test_interpreter_function_value.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, is_ok, value_of

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret)
from viba.reflect import VibaFunction

CASES = Path(__file__).resolve().parent / "data" / "function_value"

checks = Checks("interpreter_function_value")
check = checks.check


def what_it_is(value) -> str:
    """一句话说清宿主编到的这份值：`名字@模块|路径|文件`，构造出来的值就说 `constructed`。"""
    function = getattr(value, "function", None)
    if function is None:
        return "constructed"
    return "|".join([f"{function.name}@{function.module_name}",
                     function.relative_path, function.absolute_path])


def environ_for():
    """宿主：`about` 把拿到的值说成一句话。"""
    def get_func(module_path, func_name):
        if func_name == "about":
            return lambda env, value: what_it_is(value)
        return None

    return Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))


def said(name: str) -> str:
    """跑一个用例文件，把它答的那句话取回来。"""
    result = interpret(str(CASES / name), environ_for())
    check(is_ok(result), f"{name} answers: {result!r}")
    return value_of(result) if is_ok(result) else str(result)


def _a_named_function():
    """`twice` 交给宿主：名字是 `twice`，模块是它自己被定义的那份文件。"""
    said_it = said("names_a_function.viba")
    parts = said_it.split("|")
    check(parts[0] == "twice@names_a_function",
          f"the value names the function and the module that defines it: {said_it!r}")
    check(parts[1] == "/names_a_function",
          f"and the path that module is known by: {said_it!r}")
    check(parts[2] == str(CASES / "names_a_function.viba"),
          f"and the file the module was taken from: {said_it!r}")


def _a_closure_is_the_same_function():
    """`twice << $x 5` 是同一个函数：闭包也带着定义它的模块。"""
    said_it = said("names_a_closure.viba")
    check(said_it == "twice@names_a_closure|/names_a_closure|"
          + str(CASES / "names_a_closure.viba"),
          f"a closure names the function it is a call of: {said_it!r}")


def _a_constructed_value_has_no_module():
    """构造出来的值没有函数这回事：`1 * 2` 谁也不定义。"""
    said_it = said("a_constructed_value.viba")
    check(said_it == "constructed",
          f"a product is built, not defined, so it names no module: {said_it!r}")


def _the_defining_module_travels():
    """函数值走进另一份文件，它还是原来那份文件里的函数。"""
    said_it = said("crosses.viba")
    check(said_it == "twice@crosses|/crosses|" + str(CASES / "crosses.viba"),
          f"a function value keeps the module that defines it: {said_it!r}")


def _a_member_is_its_own_module_function():
    """`reader.twice` 是 `reader` 的函数，不是取它的那份文件的函数。"""
    said_it = said("takes_a_member.viba")
    check(said_it == "twice@reader|/reader|" + str(CASES / "reader.viba"),
          f"a member taken from another module belongs to that module: {said_it!r}")


def _the_value_says_it_itself():
    """值本身认得这件事：`VibaFunction` 上写的就是它的名字与它的模块。"""
    said_it = said("names_a_function.viba").split("|")
    function = VibaFunction("twice", None)
    check((function.name, function.module_name, function.relative_path,
           function.absolute_path) == ("twice", "", "", ""),
          "a function over a module that records nothing answers empty strings")
    check(said_it[0] == "twice@names_a_function",
          f"and a real one answers what the module records: {said_it!r}")


def run():
    _a_named_function()
    _a_closure_is_the_same_function()
    _a_constructed_value_has_no_module()
    _the_defining_module_travels()
    _a_member_is_its_own_module_function()
    _the_value_says_it_itself()


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-function-value-"))
    try:
        run()
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

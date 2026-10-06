"""值：宿主给出什么、__impl__ 能写成什么、一个名字算几次。

不纯的东西从这里出去（宿主函数），回来的必须是叶子或可序列化数据；函数、模块、泛型应用都不是值。
每条用例是一份可以打开的文件（`tests/data/values/*.viba`），这里只列它该跑出什么。

    python3 tests/test_interpreter_values.py
"""

import sys
import tempfile
from pathlib import Path
from typing import get_args

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import error_of, message_of, Checks, Host, value_of

from viba import viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret)
from viba.reflect import access as reflect_access
from viba.type import (FAILURE_TAG, NOT_IMPLEMENTED_TAG, Err, InterpretResult, VibaProgramErr,
                       Ok, UnderlyingOpErr)

checks = Checks("interpreter_values")
check = checks.check
labelled = checks.labelled

CASES = Path(__file__).resolve().parent / "data" / "values"


def _case(name: str) -> str:
    return str(CASES / f"{name}.viba")


def run(tmp: Path):
    _the_branches(tmp)
    _host_answers(tmp)
    _written_as_ret(tmp)
    _written(tmp)
    _names_and_repeats(tmp)
    _crossing_the_host_boundary(tmp)


# (文件, 宿主给出什么, 说明)
HOST_ANSWER_CASES = [
    ("leaf", 7, "an int"),
    ("text", "hi", "a str"),
    ("flag", True, "a bool"),
    ("ratio", 0.5, "a float"),
    ("nothing", None, "None lands as nil"),
]

# 假值但不是 nil：0、空串、false 都是叶子
FALSY_CASES = [
    ("falsy_zero", 0, "0"),
    ("falsy_empty", "", "an empty str"),
    ("falsy_falsey", False, "false"),
]

# 给不了叶子的容器
NO_LEAF_CASES = [
    ("answer_a_list", "a list"),
    ("answer_a_tuple", "a tuple"),
    ("answer_a_dict", "a dict"),
]


def _the_branches(tmp: Path):
    """`interpret` 的答案就是那几支：类型少一个、tag 改一个，按类型分的支就变了。"""
    branches = get_args(InterpretResult)
    check(set(branches) == {Ok, Err},
          f"the two branches interpret answers with: {branches!r}")
    check(FAILURE_TAG == "$underlying_viba_op_err"
          and NOT_IMPLEMENTED_TAG == "$not_implemented_err",
          f"the two tags a UnderlyingOpErr answers under: {FAILURE_TAG!r}, {NOT_IMPLEMENTED_TAG!r}")


def _host_answers(tmp: Path):
    """宿主返回什么，__impl__ 就是什么；宿主出错就是 VibaProgramErr，不是崩。"""
    host = Host()
    environ = host.environ()
    for name, want, label in HOST_ANSWER_CASES:
        result = interpret(_case(name), environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"a host function returning {label}: {result!r}")

    for name, want, label in FALSY_CASES:
        result = interpret(_case(name), environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"a host answer of {label} is a leaf, not nil: {result!r}")

    for name, label in NO_LEAF_CASES:
        checks.failed(interpret(_case(name), environ), "no leaf",
                      f"a host answer that is {label}")

    boom = _case("boom")
    exploded = interpret(boom, environ)
    checks.failed(exploded, "ZeroDivision", "a host function that raises")
    check(error_of(exploded).module_path == "root"
          and error_of(exploded).func_name == "explode"
          and error_of(exploded).msg.startswith("raised"),
          f"and the failure names the step and why: {exploded!r}")
    checks.failed(interpret(boom, Host(get_func_raises=True).environ()), "raised",
                  "a get_func that raises")

    stopped = interpret(_case("no_impl"), environ)
    check(isinstance(error_of(stopped), UnderlyingOpErr)
          and error_of(stopped).tag == NOT_IMPLEMENTED_TAG
          and error_of(stopped).msg == "no implementation",
          "get_func says None: the run stops with no implementation, not a VibaProgramErr")
    check(not isinstance(error_of(stopped), VibaProgramErr),
          "and that stop is not a VibaProgramErr: nothing broke, one step has no implementation")
    check(error_of(stopped).module_path == "root"
          and error_of(stopped).func_name == "ghost"
          and error_of(stopped).msg == "no implementation",
          f"the stop names the step and why: {stopped!r}")
    check(isinstance(error_of(stopped).call.data, viba_ast.TypeRef)
          and error_of(stopped).call.data.name == "ghost",
          f"a call written with no argument of its own is the name alone: {error_of(stopped).call!r}")

    result = interpret(_case("echo"), environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"a host function echoing its argument: {result!r}")

    checks.failed(interpret(_case("arity"), environ), "raised",
                  "a host function of the wrong arity")

    result = interpret(_case("made"), environ)
    check(isinstance(result, Ok) and value_of(result) == 11,
          f"a node the host built itself: {result!r}")

    checks.failed(interpret(_case("afunc"), environ), "no leaf",
                  "a host answer that is a function")


# (文件, 该跑出什么)
LITERAL_CASES = [("lit_42", 42, "a literal int"), ("lit_hi", "hi", "a literal str"),
                 ("lit_true", True, "a literal bool")]
UNIT_CASES = [("unit_nil", "nil"), ("unit_never", "never"), ("unit_Any", "Any")]
NOT_A_VALUE_CASES = [("written_application", "a generic application"),
                     ("written_exponent", "an exponent")]


def _written_as_ret(tmp: Path):
    """__impl__ 写成字面量、单位、和/积、名字：哪些是值。"""
    host = Host()
    environ = host.environ()

    for name, want, label in LITERAL_CASES:
        result = interpret(_case(name), environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"__impl__ written as {label}: {result!r}")
    for name, label in UNIT_CASES:
        result = interpret(_case(name), environ)
        check(isinstance(result, Ok), f"__impl__ written as {label}: {result!r}")
    for name, label in NOT_A_VALUE_CASES:
        labelled(interpret(_case(name), environ), "cannot compute",
                 f"__impl__ written as {label} -> VibaProgramErr")

    # 写下来的和值是 viba 数据：没给出的那几支留下的都是 never
    result = interpret(_case("written_sum"), environ)
    check(isinstance(result, Ok),
          f"__impl__ written as a sum is viba data, not an error: {result!r}")

    check(isinstance(interpret(_case("module_value"), environ), Ok),
          "__impl__ written as a module is the closure it stands for")

    labelled(interpret(_case("builtin_value"), environ), "no definition named",
             "a builtin type name used as a value -> VibaProgramErr")


# (文件, 有几个成员)
DATA_CASES = [("written_tuple", 2, "a tuple of two"),
              ("written_empty_tuple", 0, "the empty tuple")]


def _written(tmp: Path):
    """写在值位置上的数据就是可序列化数据：元组、tag、积。"""
    host = Host()
    environ = host.environ()
    for name, want, label in DATA_CASES:
        result = interpret(_case(name), environ)
        check(isinstance(result, Ok) and len(result.ok_value) == want,
              f"__impl__ written as {label} is viba data: {result!r}")

    # 一份按 tag 与积写下来的可序列化数据，写法和它的类型写法一样
    result = interpret(_case("tags_and_product"), environ)
    check(isinstance(result, Ok), f"data written as tags and a product: {result!r}")
    if isinstance(result, Ok):
        node = result.ok_value
        check(reflect_access.leaf(node.by_tag("victim").by_tag("x")).ok_value == 0 and
              reflect_access.leaf(node.by_tag("suspect").by_tag("y")).ok_value == 4,
              f"and its members read back: {node!r}")


def _names_and_repeats(tmp: Path):
    """名字与次序：定义写在用之后也算、一个定义只算一次、同一个名字不许写两次。"""
    host = Host()
    environ = host.environ()

    labelled(interpret(_case("twice_defined"), environ), "is defined twice",
             "one name defined twice in one module -> VibaProgramErr")

    result = interpret(_case("defined_after"), environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a definition written after its use: {result!r}")

    # `__impl__` first, everything it uses after it: the same answer either way
    result = interpret(_case("impl_written_first"), environ)
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"__impl__ written before everything it uses: {result!r}")

    result = interpret(_case("nested"), environ)
    check(isinstance(result, Ok) and value_of(result) == 3 and
          ("root", "unused") not in host.calls,
          f"a definition nobody asks for is never run: {result!r}")

    # 一个定义算一次：两处用它，宿主只被叫一次
    host.calls.clear()
    result = interpret(_case("memo"), environ)
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"a definition used twice: {result!r}")
    check(host.calls.count(("root", "leaf")) == 1,
          f"a definition is computed once: {host.calls}")


def _crossing_the_host_boundary(tmp: Path):
    """宿主手里拿到 viba 函数：只是数据——拿着、存着、原样递回来，调不动。"""
    host = Host()
    environ = host.environ()

    higher = _case("higher")
    original = host.get_func

    def hand_back(path, func_name):
        if func_name == "show":
            return lambda env, f: f            # 拿到的就是那个节点，原样交回来
        return original(path, func_name)

    host.get_func = hand_back
    host2 = Environment(EnvironmentStorage("root"), EnvironmentCompute(host.get_func))
    result = interpret(higher, host2)
    check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.TypeRef),
          f"a viba function handed to the host is viba data (its written name): {result!r}")
    check(result.ok_value.data.name == "inc",
          f"and what it holds is that name: {result.ok_value.data!r}")
    host.get_func = original

    # 宿主自己说：get_func 抛出它，等于这次调用没有实现
    host.knobs["refuse"] = ("twice",)
    checks.not_implemented(interpret(higher, environ),
                             "a get_func that refuses the call")
    host.knobs.pop("refuse")

    # 函数类型的实参交给宿主的是"它代表的那次调用"：宿主可以带上自己的实参叫它
    # （wrapper 就是这么转发的），多喂的那一个由 viba 这边说清楚 —— 这次调用不
    # 再有槽位收它，所以是程序错，而不是把 Python 的 TypeError 报成实现坏了。
    checks.labelled(interpret(_case("overfeed"), environ), "takes no more arguments",
                    "a host that gives the viba function too many arguments")

    checks.failed(interpret(_case("fed"), environ), "raised",
                  "a host handing a viba function something with no leaf")

    # 那个参数声明的是 int：把函数递进去是程序错，判定层当场拦下，走不到宿主
    checks.labelled(interpret(_case("handed"), environ), "does not fit $x int",
                    "a viba function handed where an int is declared: a program error")

    # 那个参数声明的是 Any：函数装得下，宿主拿到它再交回来，才轮到"没有叶子"
    result = interpret(_case("handed_any"), environ)
    check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.TypeRef),
          f"a viba function handed where Any is declared comes back as data: {result!r}")

    # 宿主里面再跑一次 interpret：两个 run 互不干扰
    host.knobs["inner_file"] = _case("inner_module")
    result = interpret(_case("outer_run"), environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a host function that runs interpret itself: {result!r}")
    host.knobs.pop("inner_file")

    # 宿主抛的不是 Exception：interpret 不吞
    try:
        interpret(_case("interrupt"), environ)
        check(False, "a host raising KeyboardInterrupt is not swallowed")
    except KeyboardInterrupt:
        check(True, "a host raising KeyboardInterrupt is not swallowed")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-values-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

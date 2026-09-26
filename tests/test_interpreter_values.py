"""值：宿主答什么、__ret__ 能写成什么、一个名字算几次。

不纯的东西从这里出去（宿主函数），回来的必须是叶子或材料；函数、模块、泛型应用都不是值。

    python3 tests/test_interpreter_values.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import ADD, LEAF, Checks, Host, value_of, write

from viba import viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret)
from viba.reflect import access as reflect_access
from viba.type import VibaProgramErr, NotMyDutyException, Ok, Step

checks = Checks("interpreter_values")
check = checks.check
labelled = checks.labelled


def run(tmp: Path):
    _host_answers(tmp)
    _written_as_ret(tmp)
    _written(tmp)
    _names_and_repeats(tmp)
    _crossing_the_host_boundary(tmp)


def _host_answers(tmp: Path):
    """宿主返回什么，__ret__ 就是什么；宿主出错就是 VibaProgramErr，不是崩。"""
    host = Host()
    environ = host.environ()
    for func, want, label in (("leaf", 7, "an int"), ("text", "hi", "a str"),
                              ("flag", True, "a bool"), ("ratio", 0.5, "a float"),
                              ("nothing", None, "None lands as nil")):
        path = write(tmp, f"{func}.viba", f"""
{func} =
	int
	<- $env Environment
	<- {{ inline }}
__ret__ = {func} << $env environ
""")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"a host function returning {label}: {result!r}")

    # 假值但不是 nil：0、空串、false 都是叶子
    for func, want, label in (("zero", 0, "0"), ("empty", "", "an empty str"),
                              ("falsey", False, "false")):
        path = write(tmp, f"falsy_{func}.viba", f"""
{func} =
	int
	<- $env Environment
	<- {{ inline }}
__ret__ = {func} << $env environ
""")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"a host answer of {label} is a leaf, not nil: {result!r}")

    # 答不出叶子的容器
    for func, label in (("answer_a_list", "a list"), ("answer_a_tuple", "a tuple"),
                        ("answer_a_dict", "a dict")):
        path = write(tmp, f"{func}.viba", f"""
{func} =
	int
	<- $env Environment
	<- {{ inline }}
__ret__ = {func} << $env environ
""")
        checks.failed(interpret(path, environ), "no leaf",
                      f"a host answer that is {label}")

    boom = write(tmp, "boom.viba", """
explode =
	int
	<- $env Environment
	<- { go }
__ret__ = explode << $env environ
""")
    exploded = interpret(boom, environ)
    checks.failed(exploded, "ZeroDivision", "a host function that raises")
    check(exploded.step == Step("root", "explode") and exploded.reason == "raised",
          f"and the failure names the step and why: {exploded!r}")
    checks.failed(interpret(boom, Host(get_func_raises=True).environ()), "raised",
                  "a get_func that raises")

    missing = write(tmp, "no_impl.viba", """
ghost =
	int
	<- $env Environment
	<- { nothing implements this }
__ret__ = ghost << $env environ
""")
    stopped = interpret(missing, environ)
    check(isinstance(stopped, NotMyDutyException),
          "get_func says None: the run stops with the deferral, not a VibaProgramErr")
    check(not isinstance(stopped, VibaProgramErr),
          "and that deferral is not a VibaProgramErr: it says 'not mine', not 'broke'")
    check(stopped.step == Step("root", "ghost") and stopped.reason == "no implementation",
          f"the deferral names the step and why: {stopped!r}")
    check(stopped.call is None,
          f"a step given no material carries none: {stopped.call!r}")

    echo = write(tmp, "echo.viba", """
echo =
	int
	<- $env Environment
	<- $x int
	<- { pass it through }
__ret__ = echo << $env environ << $x 42
""")
    result = interpret(echo, environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"a host function echoing its argument: {result!r}")

    arity = write(tmp, "arity.viba", """
wrong_arity =
	int
	<- $env Environment
	<- $a int
	<- $b int
	<- { takes two }
__ret__ = wrong_arity << $env environ << $a 1 << $b 2
""")
    checks.failed(interpret(arity, environ), "raised", "a host function of the wrong arity")

    made = write(tmp, "made.viba", """
make_node =
	int
	<- $env Environment
	<- { make a node }
__ret__ = make_node << $env environ
""")
    result = interpret(made, environ)
    check(isinstance(result, Ok) and value_of(result) == 11,
          f"a node the host built itself: {result!r}")

    afunc = write(tmp, "afunc.viba", """
answer_a_function =
	int
	<- $env Environment
	<- { answer a function }
__ret__ = answer_a_function << $env environ
""")
    checks.failed(interpret(afunc, environ), "no leaf",
                  "a host answer that is a function")


def _written_as_ret(tmp: Path):
    """__ret__ 写成字面量、单位、和/积、名字：哪些是值。"""
    host = Host()
    environ = host.environ()

    for body, want, label in (("42", 42, "a literal int"), ('"hi"', "hi", "a literal str"),
                              ("true", True, "a literal bool")):
        path = write(tmp, f"lit_{want}.viba", f"__ret__ = {body}\n")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"__ret__ written as {label}: {result!r}")
    for body, label in (("nil", "nil"), ("never", "never"), ("Any", "Any")):
        path = write(tmp, f"unit_{label}.viba", f"__ret__ = {body}\n")
        result = interpret(path, environ)
        check(isinstance(result, Ok), f"__ret__ written as {label}: {result!r}")
    for body, label in (("list[int]", "a generic application"),
                        ("int <- $x int", "an exponent")):
        path = write(tmp, f"written_{label.split()[-1]}.viba", f"__ret__ = {body}\n")
        labelled(interpret(path, environ), "cannot compute",
                 f"__ret__ written as {label} -> VibaProgramErr")

    # A written sum is material: every branch that did not answer never stays.
    sum_value = write(tmp, "written_sum.viba", "__ret__ = (1 | 2)\n")
    result = interpret(sum_value, environ)
    check(isinstance(result, Ok),
          f"__ret__ written as a sum is material, not an error: {result!r}")

    write(tmp, "late_lib.viba", LEAF + "__ret__ = leaf << $env environ\n")
    module_value = write(tmp, "module_value.viba",
                         "import late_lib as lib\n__ret__ = lib\n")
    check(isinstance(interpret(module_value, environ), Ok),
          "__ret__ written as a module is the closure it stands for")

    builtin_value = write(tmp, "builtin_value.viba", "__ret__ = str\n")
    labelled(interpret(builtin_value, environ), "no definition named",
             "a builtin type name used as a value -> VibaProgramErr")


def _written(tmp: Path):
    """写在值位置上的数据就是材料：元组、tag、积。"""
    host = Host()
    environ = host.environ()
    for body, want, label in (("(1, 2)", 2, "a tuple of two"),
                              ("()", 0, "the empty tuple")):
        path = write(tmp, f"written_{abs(hash(body))}.viba", f"__ret__ = {body}\n")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and len(result.ok_value) == want,
              f"__ret__ written as {label} is material: {result!r}")

    # 一份 witness 的写法：tag 与积是材料，和它的类型写法一样
    witness = write(tmp, "witness.viba",
                    '__ret__ = $victim ($x 0 * $y 0) * $suspect ($x 3 * $y 4)\n')
    result = interpret(witness, environ)
    check(isinstance(result, Ok), f"a witness written as tags and a product: {result!r}")
    if isinstance(result, Ok):
        node = result.ok_value
        check(reflect_access.leaf(node.by_tag("victim").by_tag("x")).ok_value == 0 and
              reflect_access.leaf(node.by_tag("suspect").by_tag("y")).ok_value == 4,
              f"and its members read back: {node!r}")


def _names_and_repeats(tmp: Path):
    """名字与次序：同名取先写的、定义写在用之后也算、一个定义只算一次。"""
    host = Host()
    environ = host.environ()

    twice_defined = write(tmp, "twice_defined.viba", "x = 1\nx = 2\n__ret__ = x\n")
    result = interpret(twice_defined, environ)
    check(isinstance(result, Ok) and value_of(result) == 1,
          f"a name defined twice: the first definition stands: {result!r}")

    defined_after = write(tmp, "defined_after.viba", "__ret__ = x\nx = 7\n")
    result = interpret(defined_after, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a definition written after its use: {result!r}")

    nested = write(tmp, "nested.viba", ADD + """
unused =
	int
	<- $env Environment
	<- { never asked for }
__ret__ = add << $env environ << $a 1 << $b 2
""")
    result = interpret(nested, environ)
    check(isinstance(result, Ok) and value_of(result) == 3 and
          ("root", "unused") not in host.calls,
          f"a definition nobody asks for is never run: {result!r}")

    # 一个定义算一次：两处用它，宿主只被叫一次
    memo = write(tmp, "memo.viba", ADD + LEAF + """
half = leaf << $env environ
__ret__ = add << $env environ << $a half << $b half
""")
    host.calls.clear()
    result = interpret(memo, environ)
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"a definition used twice: {result!r}")
    check(host.calls.count(("root", "leaf")) == 1,
          f"a definition is computed once: {host.calls}")


def _crossing_the_host_boundary(tmp: Path):
    """宿主手里拿到 viba 函数：只是数据——拿着、存着、原样递回来，调不动。"""
    host = Host()
    environ = host.environ()

    higher = write(tmp, "higher.viba", """
inc =
	int
	<- $env Environment
	<- $x int
	<- { add one }
show =
	int
	<- $env Environment
	<- $f (int <- $env Environment <- $x int)
	<- { hand the argument back }
__ret__ = show << $env environ << $f inc
""")
    original = host.get_func

    def hand_back(path, func_name):
        if func_name == "show":
            return lambda env, f: f            # 拿到的就是那个节点，原样交回来
        return original(path, func_name)

    host.get_func = hand_back
    host2 = Environment(EnvironmentStorage("root"), EnvironmentCompute(host.get_func))
    result = interpret(higher, host2)
    check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.TypeRef),
          f"a viba function handed to the host is material (its written name): {result!r}")
    check(result.ok_value.data.name == "inc",
          f"and what it holds is that name: {result.ok_value.data!r}")
    host.get_func = original

    # 宿主自己说"不是我的事"：get_func 抛递延，等于回答递延
    host.knobs["refuse"] = ("twice",)
    checks.deferred(interpret(higher, environ),
                    "a get_func that refuses the call")
    host.knobs.pop("refuse")

    overfeed = write(tmp, "overfeed.viba", """
overfeed =
	int
	<- $env Environment
	<- $f (int <- $env Environment <- $x int)
	<- $x int
	<- { give f one argument too many }
inc =
	int
	<- $env Environment
	<- $x int
	<- { add one }
__ret__ = overfeed << $env environ << $f inc << $x 10
""")
    checks.failed(interpret(overfeed, environ), "raised",
                  "a host that gives the viba function too many arguments")

    fed = write(tmp, "fed.viba", """
feed_a_list =
	int
	<- $env Environment
	<- $f (int <- $env Environment <- $x int)
	<- { hand f a list }
inc =
	int
	<- $env Environment
	<- $x int
	<- { add one }
__ret__ = feed_a_list << $env environ << $f inc
""")
    checks.failed(interpret(fed, environ), "raised",
                  "a host handing a viba function something with no leaf")

    # 那个参数声明的是 int：把函数递进去是程序错，判定层当场拦下，走不到宿主
    handed = write(tmp, "handed.viba", LEAF + """
echo =
	int
	<- $env Environment
	<- $x int
	<- { hand the argument back }
__ret__ = echo << $env environ << $x leaf
""")
    checks.labelled(interpret(handed, environ), "does not fit $x int",
                    "a viba function handed where an int is declared: a program error")

    # 那个参数声明的是 Any：函数装得下，宿主拿到它再交回来，才轮到"没有叶子"
    handed_any = write(tmp, "handed_any.viba", LEAF + """
echo =
	int
	<- $env Environment
	<- $x Any
	<- { hand the argument back }
__ret__ = echo << $env environ << $x leaf
""")
    result = interpret(handed_any, environ)
    check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.TypeRef),
          f"a viba function handed where Any is declared comes back as data: {result!r}")

    # 宿主里面再跑一次 interpret：两个 run 互不干扰
    inner = write(tmp, "inner_module.viba", LEAF + "__ret__ = leaf << $env environ\n")
    outer = write(tmp, "outer_run.viba", """
inner_value =
	int
	<- $env Environment
	<- { run another module from inside the host }
__ret__ = inner_value << $env environ
""")
    host.knobs["inner_file"] = inner
    result = interpret(outer, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a host function that runs interpret itself: {result!r}")
    host.knobs.pop("inner_file")

    # 宿主抛的不是 Exception：interpret 不吞
    interrupt = write(tmp, "interrupt.viba", """
interrupt =
	int
	<- $env Environment
	<- { interrupt }
__ret__ = interrupt << $env environ
""")
    try:
        interpret(interrupt, environ)
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

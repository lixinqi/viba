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

from viba.interpreter import interpret
from viba.type import Err, Ok

checks = Checks("interpreter_values")
check = checks.check
labelled = checks.labelled


def run(tmp: Path):
    _host_answers(tmp)
    _written_as_ret(tmp)
    _shapes(tmp)
    _names_and_repeats(tmp)
    _crossing_the_host_boundary(tmp)


def _host_answers(tmp: Path):
    """宿主返回什么，__ret__ 就是什么；宿主出错就是 Err，不是崩。"""
    host = Host()
    environ = host.environ()
    for func, want, label in (("leaf", 7, "an int"), ("text", "hi", "a str"),
                              ("flag", True, "a bool"), ("ratio", 0.5, "a float"),
                              ("nothing", None, "None lands as nil")):
        path = write(tmp, f"{func}.viba", f"""
{func} :=
	int
	<- $env Environment
	<- {{ inline }}
__ret__ := {func} << $env environ
""")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"a host function returning {label}: {result!r}")

    # 假值但不是 nil：0、空串、false 都是叶子
    for func, want, label in (("zero", 0, "0"), ("empty", "", "an empty str"),
                              ("falsey", False, "false")):
        path = write(tmp, f"falsy_{func}.viba", f"""
{func} :=
	int
	<- $env Environment
	<- {{ inline }}
__ret__ := {func} << $env environ
""")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"a host answer of {label} is a leaf, not nil: {result!r}")

    # 答不出叶子的容器
    for func, label in (("answer_a_list", "a list"), ("answer_a_tuple", "a tuple"),
                        ("answer_a_dict", "a dict")):
        path = write(tmp, f"{func}.viba", f"""
{func} :=
	int
	<- $env Environment
	<- {{ inline }}
__ret__ := {func} << $env environ
""")
        labelled(interpret(path, environ), "no leaf",
                 f"a host answer that is {label} -> Err")

    boom = write(tmp, "boom.viba", """
explode :=
	int
	<- $env Environment
	<- { go }
__ret__ := explode << $env environ
""")
    labelled(interpret(boom, environ), "ZeroDivision", "a host function that raises -> Err")
    labelled(interpret(boom, Host(get_func_raises=True).environ()), "raised",
             "a get_func that raises -> Err")

    missing = write(tmp, "no_impl.viba", """
ghost :=
	int
	<- $env Environment
	<- { nothing implements this }
__ret__ := ghost << $env environ
""")
    labelled(interpret(missing, environ), "no implementation", "get_func says None -> Err")

    echo = write(tmp, "echo.viba", """
echo :=
	int
	<- $env Environment
	<- $x int
	<- { pass it through }
__ret__ := echo << $env environ << $x 42
""")
    result = interpret(echo, environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"a host function echoing its argument: {result!r}")

    arity = write(tmp, "arity.viba", """
wrong_arity :=
	int
	<- $env Environment
	<- $a int
	<- $b int
	<- { takes two }
__ret__ := wrong_arity << $env environ << $a 1 << $b 2
""")
    labelled(interpret(arity, environ), "raised", "a host function of the wrong arity -> Err")

    made = write(tmp, "made.viba", """
make_node :=
	int
	<- $env Environment
	<- { make a node }
__ret__ := make_node << $env environ
""")
    result = interpret(made, environ)
    check(isinstance(result, Ok) and value_of(result) == 11,
          f"a node the host built itself: {result!r}")

    afunc = write(tmp, "afunc.viba", """
answer_a_function :=
	int
	<- $env Environment
	<- { answer a function }
__ret__ := answer_a_function << $env environ
""")
    labelled(interpret(afunc, environ), "no leaf",
             "a host answer that is a function -> Err")


def _written_as_ret(tmp: Path):
    """__ret__ 写成字面量、单位、和/积、名字：哪些是值。"""
    host = Host()
    environ = host.environ()

    for body, want, label in (("42", 42, "a literal int"), ('"hi"', "hi", "a literal str"),
                              ("true", True, "a literal bool")):
        path = write(tmp, f"lit_{want}.viba", f"__ret__ := {body}\n")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"__ret__ written as {label}: {result!r}")
    for body, label in (("nil", "nil"), ("never", "never"), ("Any", "Any")):
        path = write(tmp, f"unit_{label}.viba", f"__ret__ := {body}\n")
        result = interpret(path, environ)
        check(isinstance(result, Ok), f"__ret__ written as {label}: {result!r}")
    for body, label in (("(1 | 2)", "a sum"), ("(1 * 2)", "a product"),
                        ("list[int]", "a generic application"),
                        ("int <- $x int", "an exponent")):
        path = write(tmp, f"shape_{label.split()[-1]}.viba", f"__ret__ := {body}\n")
        labelled(interpret(path, environ), "cannot compute",
                 f"__ret__ written as {label} -> Err")

    write(tmp, "late_lib.viba", LEAF + "__ret__ := leaf << $env environ\n")
    module_value = write(tmp, "module_value.viba",
                         "import late_lib as lib\n__ret__ := lib\n")
    labelled(interpret(module_value, environ), "still waiting for arguments",
             "__ret__ written as a module -> Err")

    builtin_value = write(tmp, "builtin_value.viba", "__ret__ := str\n")
    labelled(interpret(builtin_value, environ), "no definition named",
             "a builtin type name used as a value -> Err")


def _shapes(tmp: Path):
    """元组是材料（空元组也是一个值）。"""
    host = Host()
    environ = host.environ()
    for body, want, label in (("(1, 2)", 2, "a tuple of two"),
                              ("()", 0, "the empty tuple")):
        path = write(tmp, f"shape_{abs(hash(body))}.viba", f"__ret__ := {body}\n")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and len(result.ok_value) == want,
              f"__ret__ written as {label} is material: {result!r}")


def _names_and_repeats(tmp: Path):
    """名字与次序：同名取先写的、定义写在用之后也算、一个定义只算一次。"""
    host = Host()
    environ = host.environ()

    twice_defined = write(tmp, "twice_defined.viba", "x := 1\nx := 2\n__ret__ := x\n")
    result = interpret(twice_defined, environ)
    check(isinstance(result, Ok) and value_of(result) == 1,
          f"a name defined twice: the first definition stands: {result!r}")

    defined_after = write(tmp, "defined_after.viba", "__ret__ := x\nx := 7\n")
    result = interpret(defined_after, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a definition written after its use: {result!r}")

    nested = write(tmp, "nested.viba", ADD + """
unused :=
	int
	<- $env Environment
	<- { never asked for }
__ret__ := add << $env environ << $a 1 << $b 2
""")
    result = interpret(nested, environ)
    check(isinstance(result, Ok) and value_of(result) == 3 and
          ("root", "unused") not in host.calls,
          f"a definition nobody asks for is never run: {result!r}")

    # 一个定义算一次：两处用它，宿主只被叫一次
    memo = write(tmp, "memo.viba", ADD + LEAF + """
half := leaf << $env environ
__ret__ := add << $env environ << $a half << $b half
""")
    host.calls.clear()
    result = interpret(memo, environ)
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"a definition used twice: {result!r}")
    check(host.calls.count(("root", "leaf")) == 1,
          f"a definition is computed once: {host.calls}")


def _crossing_the_host_boundary(tmp: Path):
    """宿主手里拿到 viba 函数：能调、给多了是 Err、喂不进没叶子的东西。"""
    host = Host()
    environ = host.environ()

    higher = write(tmp, "higher.viba", """
inc :=
	int
	<- $env Environment
	<- $x int
	<- { add one }
twice :=
	int
	<- $env Environment
	<- $f (int <- $env Environment <- $x int)
	<- $x int
	<- { call f twice }
__ret__ := twice << $env environ << $f inc << $x 10
""")
    result = interpret(higher, environ)
    check(isinstance(result, Ok) and value_of(result) == 22,
          f"a viba function handed to the host is callable: {result!r}")

    overfeed = write(tmp, "overfeed.viba", """
overfeed :=
	int
	<- $env Environment
	<- $f (int <- $env Environment <- $x int)
	<- $x int
	<- { give f one argument too many }
inc :=
	int
	<- $env Environment
	<- $x int
	<- { add one }
__ret__ := overfeed << $env environ << $f inc << $x 10
""")
    labelled(interpret(overfeed, environ), "raised",
             "a host that gives the viba function too many arguments -> Err")

    fed = write(tmp, "fed.viba", """
feed_a_list :=
	int
	<- $env Environment
	<- $f (int <- $env Environment <- $x int)
	<- { hand f a list }
inc :=
	int
	<- $env Environment
	<- $x int
	<- { add one }
__ret__ := feed_a_list << $env environ << $f inc
""")
    labelled(interpret(fed, environ), "raised",
             "a host handing a viba function something with no leaf -> Err")

    # 没给全参数的函数当实参：宿主拿到的就是那个函数，回手就被拒
    handed = write(tmp, "handed.viba", LEAF + """
echo :=
	int
	<- $env Environment
	<- $x int
	<- { hand the argument back }
__ret__ := echo << $env environ << $x leaf
""")
    labelled(interpret(handed, environ), "no leaf",
             "a viba function handed where a value is expected, echoed back -> Err")

    # 宿主里面再跑一次 interpret：两个 run 互不干扰
    inner = write(tmp, "inner_module.viba", LEAF + "__ret__ := leaf << $env environ\n")
    outer = write(tmp, "outer_run.viba", """
inner_value :=
	int
	<- $env Environment
	<- { run another module from inside the host }
__ret__ := inner_value << $env environ
""")
    host.knobs["inner_file"] = inner
    result = interpret(outer, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a host function that runs interpret itself: {result!r}")
    host.knobs.pop("inner_file")

    # 宿主抛的不是 Exception：interpret 不吞
    interrupt = write(tmp, "interrupt.viba", """
interrupt :=
	int
	<- $env Environment
	<- { interrupt }
__ret__ := interrupt << $env environ
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

"""`<<`：给参数这一件事——给几个、给谁、什么时候算。

写出来的实参按书写顺序算；说明块不是实参；给多了、给错了、给的不是函数都是 VibaProgramErr。

    python3 tests/test_interpreter_application.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import ADD, Checks, Host, value_of, write

from viba.reflect import access as reflect_access

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import VibaProgramErr, Ok, Step

checks = Checks("interpreter_application")
check = checks.check
labelled = checks.labelled


def run(tmp: Path):
    _arguments(tmp)
    _order_and_slots(tmp)
    _argument_types(tmp)
    _by_need_argument_types(tmp)
    _long_chain(tmp)


def _arguments(tmp: Path):
    """少给、多给、给错、说明块、分两步。"""
    host = Host()
    environ = host.environ()
    cases = [
        ("__ret__ = add << $env environ << $a 1 << $b 2\n", None, "all three given"),
        ("__ret__ = add << $env environ << $a 1\n",
         "was given 2 of its 3 arguments", "one argument short (the environment is in)"),
        ("__ret__ = add << $env environ << $a 1 << $b 2 << $b 3\n",
         "takes no more arguments", "one argument too many (every slot is filled)"),
        ("__ret__ = add << $env environ << $a 1 << $z 2\n",
         "takes no $z", "an argument the function does not have"),
        ("__ret__ = add << $env environ << $a 1 << $b 2 << { trailing note }\n", None,
         "documentation after the arguments"),
        ("__ret__ = add << $env environ << { a note } << $a 1 << $b 2\n", None,
         "documentation between the arguments"),
        ("__ret__ = add\n", None, "a bare function name is the closure it stands for"),
        ("__ret__ = { just a note }\n", "documentation", "a code block is not a value"),
        ("__ret__ = Nope\n", "no definition named", "a name nothing defines"),
        ("__ret__ = add << $env environ << $a { note } << $b 2\n", "documentation",
         "a tagged code block where an argument goes"),
        ("__ret__ = 1 << $x 2\n", "is not a function", "giving an argument to a number"),
    ]
    for index, (body, want, label) in enumerate(cases):
        path = write(tmp, f"apply{index}.viba", ADD + body)
        labelled(interpret(path, environ), want, label)

    two_steps = write(tmp, "two_steps.viba", ADD + """
half = add << $a 40
__ret__ = half << $b 2 << environ
""")
    result = interpret(two_steps, environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"a partially applied function kept in a definition: {result!r}")

    positional = write(tmp, "positional.viba", ADD + """
__ret__ = add << environ << 40 << 2
""")
    result = interpret(positional, environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"arguments written without tags bind in order: {result!r}")


def _order_and_slots(tmp: Path):
    """书写顺序与参数位：tag 的次序不影响结果；没有参数位的函数给不了实参。"""
    host = Host()
    environ = host.environ()

    out_of_order = write(tmp, "out_of_order.viba", ADD + """
__ret__ = add << $b 2 << $env environ << $a 1
""")
    result = interpret(out_of_order, environ)
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"tags may be given in any order: {result!r}")

    # 一个参数位都没有的函数：不带 tag 的实参就是给多了
    no_slots = write(tmp, "no_slots.viba", """
f =
	int
	<- { a function with no argument slot }
__ret__ = f << 1
""")
    labelled(interpret(no_slots, environ), "takes no $env Environment argument",
             "a function with no environment slot can never run -> VibaProgramErr")

    # 参数出错：那个函数根本不会被调用
    argument_boom = write(tmp, "argument_boom.viba", ADD + """
explode =
	int
	<- $env Environment
	<- { go }
__ret__ = add << $env environ << $a (explode << $env environ) << $b 2
""")
    host.calls.clear()
    exploded = interpret(argument_boom, environ)
    checks.failed(exploded, "raised", "an argument that blows up")
    check(exploded.step == Step("root", "explode"),
          f"the step that stopped is the argument's, not the call's: {exploded.step!r}")
    check(("root", "add") not in host.calls,
          f"the call itself never happens: {host.calls}")


def _argument_types(tmp: Path):
    """实参要装得下它那一格：装不下是**程序错**，宿主还没看见这个实参。

    值层现在也用得上判定的那套 <:：字面量、材料、环境都有写下来的类型。
    判不出类型的（宿主自己的值、判定层settle不了的）照旧放过去，交给实现那一步的人。
    """
    def get_func(path, func_name):
        if func_name == "x_of":
            return lambda env, point: reflect_access.leaf(point.by_tag("x")).ok_value
        return Host().get_func(path, func_name)

    environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))
    product_slot = """
x_of =
	int
	<- $env Environment
	<- $p ($x int * $y int)
	<- { the x of that point }
"""
    cases = [
        ('__ret__ = add << $env environ << $a "x" << $b 1\n',
         "does not fit $a int", "a string in an int slot"),
        ('__ret__ = add << $env environ << $a 1 << $b "x"\n',
         "does not fit $b int", "and in the second slot"),
        ('__ret__ = add << $env environ << $a true << $b 1\n',
         "does not fit $a int", "a bool is no int"),
        ('__ret__ = add << $env environ << $a environ << $b 1\n',
         "does not fit $a int", "the environment in an int slot"),
        ('__ret__ = add << $env environ << $a nil << $b 1\n',
         "does not fit $a int", "nil in an int slot"),
        (product_slot + '__ret__ = x_of << $env environ << $p ($x "s" * $y 1)\n',
         "does not fit $p", "a product whose member does not fit"),
        ('__ret__ = add << $env environ << $a 3 << $b 4\n', None,
         "a literal that does fit goes through"),
        (product_slot + '__ret__ = x_of << $env environ << $p ($x 1 * $y 2)\n', None,
         "and so does a product that fits"),
    ]
    for index, (body, want, label) in enumerate(cases):
        path = write(tmp, f"slot{index}.viba", ADD + body)
        labelled(interpret(path, environ), want, label)


def _by_need_argument_types(tmp: Path):
    """按需的实参不先算，所以那一格在宿主叫它的时候才核：叫了才错，不叫就不错。"""
    watch = """
watch =
    int
  <- $env Environment
  <- $x CalledByNeed[int]
  <- { hand x back }
"""
    asked = write(tmp, "by_need_asked.viba",
                  watch + '__ret__ = watch << $env environ << $x "x"\n')
    ignored = write(tmp, "by_need_ignored.viba", watch + """
ignore =
    int
  <- $env Environment
  <- $x CalledByNeed[int]
  <- { answer seven without looking at x }
__ret__ = ignore << $env environ << $x "x"
""")

    def get_func(path, func_name):
        if func_name == "watch":
            return lambda env, get_x: get_x().value
        if func_name == "ignore":
            return lambda env, get_x: 7
        return Host().get_func(path, func_name)

    host = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))
    labelled(interpret(asked, host), "does not fit $x int",
             "a marked function asks for the argument: the slot is checked then")
    result = interpret(ignored, host)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"and an argument nobody asks for is never checked: {result!r}")


def _long_chain(tmp: Path):
    """二十个实参的一条链。"""
    host = Host()
    environ = host.environ()
    many = "".join(f" << $a{i} {i}" for i in range(1, 21))
    long_chain = write(tmp, "long_chain.viba", """
sum =
	int
	<- $env Environment
	<- $a1 int <- $a2 int <- $a3 int <- $a4 int <- $a5 int
	<- $a6 int <- $a7 int <- $a8 int <- $a9 int <- $a10 int
	<- $a11 int <- $a12 int <- $a13 int <- $a14 int <- $a15 int
	<- $a16 int <- $a17 int <- $a18 int <- $a19 int <- $a20 int
	<- { add them all }
__ret__ = sum << $env environ""" + many + "\n")
    original = host.get_func

    def summing(path, func_name):
        if func_name == "sum":
            return lambda env, *rest: sum(v.value for v in rest)
        return original(path, func_name)

    host.get_func = summing
    host2 = Environment(EnvironmentStorage("root"), EnvironmentCompute(host.get_func))
    result = interpret(long_chain, host2)
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

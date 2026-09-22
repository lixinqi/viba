"""`<<`：给参数这一件事——给几个、给谁、什么时候算。

写出来的实参按书写顺序算；说明块不是实参；给多了、给错了、给的不是函数都是 Err。

    python3 tests/test_interpreter_application.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import ADD, Checks, Host, value_of, write

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import Err, Ok, Step

checks = Checks("interpreter_application")
check = checks.check
labelled = checks.labelled


def run(tmp: Path):
    _arguments(tmp)
    _order_and_slots(tmp)
    _long_chain(tmp)


def _arguments(tmp: Path):
    """少给、多给、给错、说明块、分两步。"""
    host = Host()
    environ = host.environ()
    cases = [
        ("__ret__ = add << $env environ << $a 1 << $b 2\n", None, "all three given"),
        ("__ret__ = add << $env environ << $a 1\n",
         "waiting for arguments", "one argument short"),
        ("__ret__ = add << $env environ << $a 1 << $b 2 << $b 3\n",
         "is not a function", "one argument too many (the call already answered)"),
        ("__ret__ = add << $env environ << $a 1 << $z 2\n",
         "takes no $z", "an argument the function does not have"),
        ("__ret__ = add << $env environ << $a 1 << $b 2 << { trailing note }\n", None,
         "documentation after the arguments"),
        ("__ret__ = add << $env environ << { a note } << $a 1 << $b 2\n", None,
         "documentation between the arguments"),
        ("__ret__ = add\n", "waiting for arguments", "the function itself is not a value"),
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
half = add << $env environ << $a 40
__ret__ = half << $b 2
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
    labelled(interpret(no_slots, environ), "takes no more arguments",
             "an untagged argument to a function with no slots -> Err")

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

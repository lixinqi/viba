"""interpreter 各套件共用的东西：计数器、宿主。

不是套件本身（没有 `__main__` 的跑法），`test_interpreter_*.py` 从这里取：

    from interpreter_support import Checks, Host, value_of

`Checks` 管每个套件自己的通过/失败计数与那一行汇总；失败会打出来。用例本身是
`tests/data/` 下的 `.viba` 文件，不写在这里。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                              interpret)
from viba.reflect import VibaNode, access as reflect_access
from viba.type import VibaProgramErr, UnderlyingVibaOpFailed, NoImplementationException, Ok

CASES = Path(__file__).resolve().parent / "data" / "interpreter"


class Checks:
    """One suite's counter: `check`, `labelled`, and the summary line."""

    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failures = 0

    def check(self, ok: bool, label: str):
        if ok:
            self.passed += 1
        else:
            self.failures += 1
            print(f"FAIL: {label}")

    def labelled(self, result, want, label: str):
        """`want` is a substring of the VibaProgramErr, or None for Ok."""
        if want is None:
            self.check(isinstance(result, Ok), f"{label}: {result!r}")
        else:
            self.check(isinstance(result, VibaProgramErr) and want in result.err_msg,
                       f"{label}: expected VibaProgramErr({want!r}), got {result!r}")

    def no_implementation(self, result, label: str):
        """The run stopped at a step `get_func` does not implement."""
        self.check(isinstance(result, NoImplementationException),
                   f"{label}: expected a step with no implementation, got {result!r}")

    def failed(self, result, want: str, label: str):
        """A step's implementation broke: `want` is part of its message."""
        self.check(isinstance(result, UnderlyingVibaOpFailed) and want in result.msg,
                   f"{label}: expected UnderlyingVibaOpFailed({want!r}), got {result!r}")

    def report(self) -> int:
        print(f"{self.name}: {self.passed} passed, {self.failures} failed")
        return 1 if self.failures else 0


def value_of(result):
    """The leaf a run answered: None for a nil piece, the stop itself otherwise.

    A suite writes `isinstance(result, Ok) and value_of(result) == 7` and the
    left side is false when the run stopped, so this must not assume a value:
    a stop has no leaf to read.
    """
    if not isinstance(result, Ok):
        return result
    value = result.ok_value
    if isinstance(value, (Ok, VibaProgramErr)):
        return value.err_msg if isinstance(value, VibaProgramErr) else value.ok_value
    if not isinstance(value, VibaNode):
        return value
    leaf = reflect_access.leaf(value)
    if not isinstance(leaf, Ok):
        return leaf
    return leaf.ok_value


class Host:
    """The host side: a few functions, a record of the calls, and knobs."""

    def __init__(self, **knobs):
        self.calls = []
        self.printed = []
        self.knobs = knobs

    def get_func(self, module_path, func_name):
        self.calls.append((module_path, func_name))
        if self.knobs.get("get_func_raises"):
            raise RuntimeError("host broke")
        if self.knobs.get("refuse_with") is not None:
            raise self.knobs["refuse_with"]     # get_func says so itself, in its own words
        if func_name in self.knobs.get("refuse", ()):
            # a host that refuses the call without saying why
            raise NoImplementationException()
        if func_name in self.knobs.get("missing", ()):
            return None
        if func_name == "add":
            return lambda env, a, b: a.value + b.value
        if func_name == "join":
            return lambda env, a, b: a.value + b.value
        if func_name == "print":
            def show(env, x):
                self.printed.append(x)
                return None
            return show
        if func_name == "explode":
            return lambda env: 1 / 0
        if func_name == "inc":
            return lambda env, x: x.value + 1
        if func_name == "twice":
            def twice(env, f, x):
                return f(env, x).value + f(env, x).value
            return twice
        if func_name == "echo":
            return lambda env, x: x
        if func_name == "leaf":
            return lambda env: 7
        if func_name == "text":
            return lambda env: "hi"
        if func_name == "flag":
            return lambda env: True
        if func_name == "ratio":
            return lambda env: 0.5
        if func_name == "nothing":
            return lambda env: None
        if func_name == "zero":
            return lambda env: 0
        if func_name == "empty":
            return lambda env: ""
        if func_name == "falsey":
            return lambda env: False
        if func_name == "answer_a_list":
            return lambda env: [1, 2]
        if func_name == "answer_a_tuple":
            return lambda env: (1, 2)
        if func_name == "answer_a_dict":
            return lambda env: {"a": 1}
        if func_name == "wrong_arity":
            return lambda env: 1
        if func_name == "make_node":
            def make_node(env):
                from viba import viba_ast
                from viba.reflect import VibaNode, access
                from viba.viba_type_descriptor import descriptor_of
                from viba.type import AstNodeType, custom_module
                node = viba_ast.Constant(11)
                return VibaNode(access,
                                descriptor_of(AstNodeType(node, custom_module(""))), node)
            return make_node
        if func_name == "answer_a_function":
            return lambda env: (lambda: 1)
        if func_name == "overfeed":
            def overfeed(env, f, x):
                return f(env, x, 1)
            return overfeed
        if func_name == "inner_value":
            def inner_value(env):
                result = interpret(self.knobs["inner_file"], env)
                if isinstance(result, VibaProgramErr):
                    raise RuntimeError(result.err_msg)
                return result.ok_value
            return inner_value
        if func_name == "feed_a_list":
            def feed_a_list(env, f):
                return f(env, [1, 2])
            return feed_a_list
        if func_name == "interrupt":
            def interrupt(env):
                raise KeyboardInterrupt
            return interrupt
        return None

    def environ(self, path="root", viba_path=None, store_root_dir=None):
        return Environment(EnvironmentStorage(path, None, store_root_dir),
                           EnvironmentCompute(self.get_func), viba_path)


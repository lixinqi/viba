"""interpreter 各套件共用的东西：计数器、宿主、语料片段。

不是套件本身（没有 `__main__` 的跑法），`test_interpreter_*.py` 从这里取：

    from interpreter_support import ADD, LEAF, TEXT, Checks, Host, value_of, write

`Checks` 管每个套件自己的通过/失败计数与那一行汇总；失败会打出来。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                              interpret)
from viba.reflect import access as reflect_access
from viba.type import Err, Ok

CASES = Path(__file__).resolve().parent / "data" / "interpreter"


class Checks:
    """One suite's counter: `check`, `labelled`, and the summary line."""

    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0

    def check(self, ok: bool, label: str):
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            print(f"FAIL: {label}")

    def labelled(self, result, want, label: str):
        """`want` is a substring of the Err, or None for Ok."""
        if want is None:
            self.check(isinstance(result, Ok), f"{label}: {result!r}")
        else:
            self.check(isinstance(result, Err) and want in result.err_msg,
                       f"{label}: expected Err({want!r}), got {result!r}")

    def report(self) -> int:
        print(f"{self.name}: {self.passed} passed, {self.failed} failed")
        return 1 if self.failed else 0


def value_of(result):
    """The leaf a run answered: None for a nil piece."""
    if not isinstance(result, Ok):
        return result
    return reflect_access.leaf(result.ok_value).ok_value


def write(tmp: Path, name: str, source: str) -> str:
    """Write one module into the suite's scratch directory; answer its path."""
    path = tmp / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return str(path)


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
                if isinstance(result, Err):
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


# 三份常写的片段：一个只答 7 的函数、一个答文本的、一个两数相加的。
LEAF = """
leaf =
	int
	<- $env Environment
	<- { answer seven }
"""

TEXT = """
text =
	str
	<- $env Environment
	<- { answer some text }
"""

ADD = """
add =
	int
	<- $env Environment
	<- $a int
	<- $b int
	<- { add two integer }
"""

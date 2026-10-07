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
from viba.type import (FAILURE_TAG, NOT_IMPLEMENTED_TAG,
                       EnvironmentApiInvalidArgumentErr, Err, InterpretError,
                       VibaProgramErr, UnderlyingOpErr, Ok)

CASES = Path(__file__).resolve().parent / "data" / "interpreter"


def module_of(path) -> str:
    """The module a case file is known by: its stem.

    A module records the name it was loaded as (`CustomModuleType.name`), and a file
    handed to `interpret` is bound under its stem, so that is what a step of it is
    asked under: `get_func("memo", "leaf")` (viba-interpreter.md, "`get_func` 与
    `func_name`").
    """
    return Path(str(path)).stem


def module_path_of(path) -> str:
    """The same module read as a path: `/memo`."""
    return "/" + module_of(path)


def full_name_of(path, name: str) -> str:
    """The whole name of a step declared in that file: `memo.leaf`."""
    return f"{module_of(path)}.{name}"


def error_of(result):
    """The error a result carries: an `Err`'s, or the error itself.

    `interpret` answers `Err(error)`; the other APIs hand the error back as it
    is. Both go through here, so a suite can check either without knowing which.
    """
    if isinstance(result, Err):
        return result.error
    return result if isinstance(result, InterpretError) else None


def message_of(result) -> str:
    """What the run stopped with, as a sentence; '' when it answered a value."""
    return getattr(error_of(result), "msg", "")


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
        error = error_of(result)
        if want is None:
            self.check(isinstance(result, Ok), f"{label}: {result!r}")
        else:
            self.check(isinstance(error, VibaProgramErr) and want in error.msg,
                       f"{label}: expected VibaProgramErr({want!r}), got {result!r}")

    def not_implemented(self, result, label: str):
        """The run stopped at a step `get_func` does not implement: that tag."""
        error = error_of(result)
        self.check(isinstance(error, UnderlyingOpErr)
                   and error.tag == NOT_IMPLEMENTED_TAG,
                   f"{label}: expected a step with no implementation, got {result!r}")

    def failed(self, result, want: str, label: str):
        """A step's implementation broke: that tag, and `want` in its message."""
        error = error_of(result)
        self.check(isinstance(error, UnderlyingOpErr)
                   and error.tag == FAILURE_TAG and want in error.msg,
                   f"{label}: expected UnderlyingOpErr({want!r}), got {result!r}")

    def environment_api(self, result, want: str, api_name: str, label: str):
        """An environment api refused what it was given: that error, that api, that message."""
        error = error_of(result)
        self.check(isinstance(error, EnvironmentApiInvalidArgumentErr)
                   and api_name == error.api_name and want in error.msg,
                   f"{label}: expected EnvironmentApiInvalidArgumentErr("
                   f"{api_name!r}, {want!r}), got {result!r}")

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
        return value.msg if isinstance(value, VibaProgramErr) else value.ok_value
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
        # This host serves one module's steps, so it reads them by their own names.
        if self.knobs.get("get_func_raises"):
            raise RuntimeError("host broke")
        if self.knobs.get("refuse_with") is not None:
            raise self.knobs["refuse_with"]     # get_func says so itself, in its own words
        if func_name in self.knobs.get("refuse", ()):
            # a host that refuses the call without saying why
            raise UnderlyingOpErr(tag=NOT_IMPLEMENTED_TAG)
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
                    raise RuntimeError(result.msg)
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


class PlacedHost(Host):
    """The shared host, plus where each step actually ran.

    `get_func` is handed the module a step is declared in; where a call *runs* is
    what the environment it receives carries, so a step records that here
    (viba-interpreter.md, "`get_func` 与 `func_name`").
    """

    def __init__(self, **knobs):
        super().__init__(**knobs)
        self.ran_at = []                 # (data path, name), in the order steps ran

    def get_func(self, module_path, func_name):
        step = super().get_func(module_path, func_name)
        if step is None:
            return None

        def ran(env, *args):
            self.ran_at.append((env.storage.cur_storage_path, func_name))
            return step(env, *args)
        return ran

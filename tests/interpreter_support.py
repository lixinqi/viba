"""interpreter 各套件共用的东西：计数器、宿主、读结果的那几个小函数。

不是套件本身（没有 `__main__` 的跑法），`test_interpreter_*.py` 从这里取：

    from interpreter_support import Checks, Host, is_ok, value_of

`Checks` 管每个套件自己的通过/失败计数与那一行汇总；失败会打出来。用例本身是
`tests/data/` 下的 `.viba` 文件，不写在这里。

`interpret` 与 `exec` 回答的是一份 viba 数据（声明里的 `InterpretResult`），所以这里读它
也按数据读：`is_ok`、`value_of`、`stop_tag`、`stop_text`、`stop_node`。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                              interpret, not_implemented)
from viba.reflect import VObject, access as reflect_access, by_tag
from viba.type import (ENVIRONMENT_API_TAG, ERR_TAG, FAILURE_TAG,
                       NOT_IMPLEMENTED_TAG, OK_TAG, PROGRAM_ERR_TAG, Ok,
                       VibaProgramErr)

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


def branch_of(result) -> str:
    """Which branch the run answered on: `$ok`, or `$err` when it stopped."""
    return result.data.tag


def is_ok(result) -> bool:
    """Did the run answer a value, rather than stop?"""
    return branch_of(result) == OK_TAG


def stop_of(result):
    """What the run stopped with, as viba data (`$err`'s payload); None for a value."""
    return None if is_ok(result) else result.by_tag(ERR_TAG)


def stop_tag(result) -> str:
    """Which kind of stop it was: `$not_implemented_err`, `$viba_program_err`, ..."""
    stop = stop_of(result)
    return "" if stop is None else stop.data.tag


def stop_members(result):
    """The members of the stop: its payload's object — `$msg`, `$call`, `$api_name`."""
    stop = stop_of(result)
    return None if stop is None else stop.by_tag(stop.data.tag)


def _member(result, tag):
    members = stop_members(result)
    if members is None:
        return None
    given = reflect_access.get(members, by_tag(tag))
    if not isinstance(given, Ok) or given.ok_value is None:
        return None
    return given.ok_value


def stop_text(result, tag: str) -> str:
    """The str one member of the stop carries; '' when it carries none."""
    given = _member(result, tag)
    if given is None:
        return ""
    leaf = reflect_access.leaf(given)
    if isinstance(leaf, Ok) and isinstance(leaf.ok_value, str):
        return leaf.ok_value
    return ""


def stop_node(result, tag: str):
    """One member of the stop, as viba data; None when it carries none."""
    return _member(result, tag)


def error_of(result):
    """The error an object-form `Result` carries, or the error itself.

    The other APIs of this layer (`parse`, `serialize`, `load_generic`, the
    descriptor builders) answer `Ok`/error objects, and their suites read the
    error with this. `interpret` and `exec` answer viba data instead: a suite that
    ran one reads it with `stop_tag`, `stop_text` and `stop_node`.
    """
    if isinstance(result, VibaProgramErr):
        return result
    return getattr(result, "error", None)


def message_of(result) -> str:
    """What a stop says, as a sentence; '' when it answered a value.

    Both forms go through here: the node `interpret` answers, and the object form
    the other APIs of this layer answer.
    """
    if isinstance(result, VObject):
        return stop_text(result, "$msg")
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
        """`want` is a substring of the `$viba_program_err` message, or None for a value."""
        if want is None:
            self.check(is_ok(result), f"{label}: {result!r}")
        else:
            self.check(stop_tag(result) == PROGRAM_ERR_TAG
                       and want in message_of(result),
                       f"{label}: expected a $viba_program_err with {want!r}, "
                       f"got {result!r}")

    def not_implemented(self, result, label: str):
        """The run stopped at a step `get_func` does not implement: that tag."""
        self.check(stop_tag(result) == NOT_IMPLEMENTED_TAG,
                   f"{label}: expected a step with no implementation, got {result!r}")

    def failed(self, result, want: str, label: str):
        """A step's implementation broke: that tag, and `want` in its message."""
        self.check(stop_tag(result) == FAILURE_TAG and want in message_of(result),
                   f"{label}: expected a $underlying_viba_op_err with {want!r}, "
                   f"got {result!r}")

    def environment_api(self, result, want: str, api_name: str, label: str):
        """An environment api refused what it was given: that stop, that api, that message."""
        self.check(stop_tag(result) == ENVIRONMENT_API_TAG
                   and stop_text(result, "$api_name") == api_name
                   and want in message_of(result),
                   f"{label}: expected a $environment_api_invalid_argument_err"
                   f"({api_name!r}, {want!r}), got {result!r}")

    def report(self) -> int:
        print(f"{self.name}: {self.passed} passed, {self.failures} failed")
        return 1 if self.failures else 0


def value_of(result):
    """The leaf a run answered: None for a nil piece, the stop node otherwise.

    A suite writes `is_ok(result) and value_of(result) == 7`, and the left side is
    false when the run stopped, so this must not assume a value: a stop has no leaf
    to read.
    """
    if not is_ok(result):
        return result
    return leaf_of(answer_of(result))


def answer_of(result):
    """The answer a run gave, as viba data: the `$ok` branch's payload."""
    return result.by_tag(OK_TAG)


def leaf_of(node):
    """The leaf one piece of viba data carries, or the error the accessor gave."""
    leaf = reflect_access.leaf(node)
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
        if self.knobs.get("hands_back") is not None:
            return self.knobs["hands_back"]   # get_func hands the failure back itself
        if func_name in self.knobs.get("refuse", ()):
            # a host that hands back the failure without saying anything more
            return not_implemented()
        if func_name in self.knobs.get("missing", ()):
            return None
        if func_name in self.knobs.get("says_no", ()):
            # the implementation itself says the same thing, without words of its own
            def says_no(env, *args):
                return not_implemented()
            return says_no
        said = self.knobs.get("says_no_with", {}).get(func_name)
        if said is not None:
            # ... or hands back the failure it wrote itself
            def says_no_with(env, *args):
                return said
            return says_no_with
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
                from viba.reflect import VObject, access
                from viba.viba_type_descriptor import descriptor_of
                from viba.type import AstNodeType, custom_module
                node = viba_ast.Constant(11)
                return VObject(access,
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
                if not is_ok(result):
                    raise RuntimeError(message_of(result))
                return result.by_tag(OK_TAG)
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
        if step is None or not callable(step):
            return step                  # nothing implements it, or it said so itself

        def ran(env, *args):
            self.ran_at.append((env.storage.cur_storage_path, func_name))
            return step(env, *args)
        return ran

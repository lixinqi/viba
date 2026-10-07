"""一步是按哪两个名字问的：它在哪个模块、在那个模块里叫什么。

viba 里怎么 import，就怎么传给 `get_func` —— `import foo.bar`（或者 `import foo.bar as foo_bar`）
之后写 `foo.bar.b_scale << …` 也好、写别名 `b_scale << …` 也好，问的都是
`get_func("foo.bar", "b_scale")`（`viba-interpreter.md`「`get_func` 与 `func_name`」）。就地声明的
步骤问的是跑起来的那份模块，和它在那儿的名字。

结果里记的是同一件事的另外两种读法：`$module_path` 是那个模块读成路径（`/foo/bar`），`$full_qualified_func_name` 是
整名（`foo.bar.b_scale`），而 `$call` 里带的就是这个整名 —— 拿它再跑一遍时按最后一个点切开，前面是模块、
后面是名字。

    python3 tests/test_interpreter_step_names.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Environment, EnvironmentCompute, EnvironmentStorage
from interpreter_support import (error_of, full_name_of, module_path_of,
                                  value_of)

from viba import viba_ast
from viba.interpret import interpret
from viba.type import Ok, UnderlyingOpErr

CASES = Path(__file__).resolve().parent / "data" / "step_names"

checks = Checks("interpreter_step_names")
check = checks.check


class Recorder:
    """A host that records what it was asked and implements nothing.

    Every case here stops at the same step with no implementation, which is what
    carries the name it was asked under.
    """

    def __init__(self):
        self.calls = []
        self.environ = Environment(EnvironmentStorage("root"),
                                   EnvironmentCompute(self.get_func), str(CASES))

    def get_func(self, module_path, func_name):
        self.calls.append((module_path, func_name))
        return None


def _asked(name: str):
    """(the names the host was asked, the error) for one case."""
    host = Recorder()
    result = interpret(str(CASES / f"{name}.viba"), host.environ)
    return host.calls, error_of(result), result


def run():
    _a_dotted_call_keeps_its_whole_name()
    _an_alias_asks_the_name_it_stands_for()
    _an_own_step_keeps_its_own_name()
    _the_call_carries_the_same_name()


def _a_dotted_call_keeps_its_whole_name():
    """`foo.bar.b_scale << …`：问的就是这个名字，这一步声明在 `/foo/bar`。"""
    calls, error, result = _asked("direct")
    check(calls == [("foo.bar", "b_scale")],
          f"the module as it was imported, then the name it has there: {calls}")
    check(isinstance(error, UnderlyingOpErr) and error.full_qualified_func_name == "foo.bar.b_scale",
          f"and the whole name is what the result reports: {result!r}")
    check(error.module_path == "/foo/bar",
          f"the file it is declared in is the module path: {error.module_path!r}")


def _an_alias_asks_the_name_it_stands_for():
    """别名不是一层新的名字：`b_scale = foo.bar.b_scale` 之后问的还是后者。"""
    calls, error, result = _asked("alias")
    check(calls == [("foo.bar", "b_scale")],
          f"the alias is the call it stands for, asked the same way: {calls}")
    check(isinstance(error, UnderlyingOpErr) and error.full_qualified_func_name == "foo.bar.b_scale",
          f"and that name is what the result reports: {result!r}")
    check(error.module_path == "/foo/bar",
          f"the alias stands for that module too: {error.module_path!r}")


def _an_own_step_keeps_its_own_name():
    """就地声明的步骤问就地写的名字（它没有前缀可带）。"""
    calls, error, result = _asked("own")
    check(calls == [("own", "b_scale")],
          f"a step of this file is asked from this file's module: {calls}")
    check(isinstance(error, UnderlyingOpErr) and
          error.full_qualified_func_name == full_name_of(CASES / "own", "b_scale"),
          f"the whole name is the module and the name: {result!r}")
    check(error.module_path == module_path_of(CASES / "own"),
          f"and it is declared in this file, which is the module path: "
          f"{error.module_path!r}")


def _the_call_carries_the_same_name():
    """`$call` 里的名字与问的名字是同一个（`$call` 写的就是被问的那次调用）。"""
    calls, error, result = _asked("alias")
    check(isinstance(error, UnderlyingOpErr), f"this case stops at a step: {result!r}")
    written = " ".join(viba_ast.unparse_type(error.call.data).split())
    check(written == '__dyn_call__ << "foo.bar.b_scale" << $x 2',
          f"the call the error carries says the whole name: {written!r}")


if __name__ == "__main__":
    run()
    sys.exit(checks.report())

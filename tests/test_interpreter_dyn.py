"""`__dyn_call__` 与 `__dyn_method__`：名字当数据的那种调用。

一步是按名字实现的，把那个名字当**数据**写下来，这次调用就从任何模块都读得起来 ——
不用写它的那个模块在场：`__dyn_call__ << env << "a.b.c" << $a1` 就是 `a.b.c << env << $a1`。
成员是某份值的成员时，名字不再是名字路径，走 `__dyn_method__`：
`__dyn_method__ << env << "f" << value << …` 就是 `$f << value << …`。

`UnderlyingOpErr` 的 `$call` 就是这两种写法：名字是字符串数据，环境剔掉，只留下闭包
（`viba-interpreter.md`「把一次调用写成可执行的」）。

    python3 tests/test_interpreter_dyn.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import (answer_of, Checks, full_name_of, Host, is_ok,
                                  stop_node, stop_tag, stop_text, value_of)

from viba import viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            exec, interpret)
from viba.is_sub_type import is_sub_type
from viba.type import (AstNodeType, ENVIRONMENT_API_TAG, FAILURE_TAG,
                       NOT_IMPLEMENTED_TAG, Ok, custom_module)
from viba.viba_type_descriptor import descriptor_of

CASES = Path(__file__).resolve().parent / "data" / "dyn"

checks = Checks("interpreter_dyn")
check = checks.check


def _written(node) -> str:
    """A piece of viba data as written, with the laying out flattened away."""
    return " ".join(viba_ast.unparse_type(node).split()).replace("( ", "(")


def _host(**knobs):
    """An environment whose steps are the shared host's."""
    return Host(**knobs)


def run(tmp: Path):
    _what_the_name_spells()
    _what_the_member_name_spells()
    _a_call_without_an_environment_is_a_value()
    _the_call_the_error_carries(tmp)
    _the_member_the_error_carries()
    _what_a_call_hands_on()
    _what_the_environment_api_error_records()
    _what_the_design_says()


def _what_the_name_spells():
    """`__dyn_call__ << env << "add" << …` 与 `add << env << …` 是同一次调用。"""
    host = _host()
    by_name = interpret(str(CASES / "call_by_name.viba"), host.environ())
    written = interpret(str(CASES / "call_written.viba"), host.environ())
    check(is_ok(by_name) and value_of(by_name) == 3,
          f"a call whose name is data runs: {by_name!r}")
    check(value_of(by_name) == value_of(written),
          f"and it is the call the same name at the head makes: {written!r}")
    check(("call_by_name", "add") in host.calls and
          ("call_written", "add") in host.calls,
          f"the host was asked for that name, from the module that declares it: "
          f"{host.calls}")


def _what_the_member_name_spells():
    """`__dyn_method__ << env << "f" << box << …` 与 `$f << box << …` 是同一次调用。"""
    def get_func(path, func_name):
        if func_name == "inc":
            # 成员要 owner、环境、x 三样（`$f << box << args.env << 1`）。
            return lambda box, environ, x: x.value + 1
        return Host().get_func(path, func_name)

    env = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))
    by_name = interpret(str(CASES / "method_by_name.viba"), env)
    by_tag = interpret(str(CASES / "method_by_tag.viba"), env)
    check(is_ok(by_name) and value_of(by_name) == 2,
          f"a member whose name is data is taken from the value and run: {by_name!r}")
    check(value_of(by_name) == value_of(by_tag),
          f"and it is the call the tag at the head makes: {by_tag!r}")


def _a_call_without_an_environment_is_a_value():
    """环境没给，`__dyn_call__` 还是一个闭包；给它环境就执行。"""
    closure = interpret(str(CASES / "closure.viba"), _host().environ())
    check(is_ok(closure)
          and isinstance(answer_of(closure).data, viba_ast.Partial),
          f"no environment: the call is a value: {closure!r}")
    check(_written(answer_of(closure).data) == '__dyn_call__ << "add" << $a 1',
          f"written as the same call, the environment left out: "
          f"{_written(answer_of(closure).data)!r}")
    ran = interpret(str(CASES / "closure_run.viba"), _host().environ())
    check(is_ok(ran) and value_of(ran) == 3,
          f"and giving that call an environment runs it: {ran!r}")


def _the_call_the_error_carries(tmp: Path):
    """没实现的那一步带回来的 `$call`：名字是数据，环境不在里面，照它能把这次调用做一遍。"""
    stopped = interpret(str(CASES / "missing_step.viba"), _host(missing=("add",)).environ())
    check(stop_tag(stopped) == NOT_IMPLEMENTED_TAG
          and stop_text(stopped, "$full_qualified_func_name")
          == full_name_of(CASES / "missing_step", "add"),
          f"the stop names the step: {stopped!r}")
    call = _written(stop_node(stopped, "$call").data)
    check(call == '__dyn_call__ << "missing_step.add" << $a 1 << $b 2',
          f"the call is the same call with the name as data: {call!r}")

    # 照它再做一遍：把这段写法放进一份新模块，就地给上环境。
    replay = tmp / "replay.viba"
    replay.write_text("__decl__ =\n    int\n  <- $env Env\n\n"
                      "args = __get_args__ << __decl__\n\n"
                      "__impl__ = " + call.replace(
                          "__dyn_call__ << ", "__dyn_call__ << args.env << ", 1) + "\n")
    again = interpret(str(replay), _host().environ())
    check(is_ok(again) and value_of(again) == 3,
          f"the call the error carried runs again, from another module: {again!r}")


def _the_member_the_error_carries():
    """成员那一支：`$call` 保留「取这份值的成员」，那份值里的调用按能跑的形式写。

    成员的值是一次调用（`box` 的 `$f` 是 `inc`）。成员那一层保留（`__dyn_method__`），而那份值里
    写着这个调用的那个成员按**能再跑一遍的形式**写：`$f (__dyn_call__ << "inc")` —— 名字当数据走，
    读回来时不必解析 `inc` 这个名字，也就不必让写它的那个模块在场。环境不在里面。
    """
    stopped = interpret(str(CASES / "missing_member.viba"), _host(missing=("inc",)).environ())
    check(stop_tag(stopped) == NOT_IMPLEMENTED_TAG
          and stop_text(stopped, "$full_qualified_func_name")
          == full_name_of(CASES / "missing_member", "inc"),
          f"the stop names the step the member's value stands for: {stopped!r}")
    call = _written(stop_node(stopped, "$call").data)
    check(call == '__dyn_method__ << "f" << ($f (__dyn_call__ << "inc") * $y 2) << 1',
          f"the member layer is kept, the value comes first, the call in it is "
          f"written the way a call travels, the environment is dropped: {call!r}")


def _what_a_call_hands_on():
    """`$call` 里凡是调用都按能跑的形式写：名字当数据走，读的人不必解析它。

    函数类型的参数就是这样（`$f inc`）：写下来的是那个名字，写进 `$call` 的是它的调用
    （`$f (__dyn_call__ << "inc")`）。另起一个运行，把这一步补上，照着它就能把这次调用做一遍。
    """
    stopped = interpret(str(CASES / "fn_argument.viba"), _host(missing=("apply",)).environ())
    check(stop_tag(stopped) == NOT_IMPLEMENTED_TAG
          and stop_text(stopped, "$full_qualified_func_name")
          == full_name_of(CASES / "fn_argument", "apply"),
          f"the stop names the step that is missing: {stopped!r}")
    call = _written(stop_node(stopped, "$call").data)
    check(call == '__dyn_call__ << "fn_argument.apply" << $f (__dyn_call__ << "inc")',
          f"the function argument is the call it stands for, written the way a call "
          f"travels: {call!r}")

    def get_func(path, func_name):
        if func_name == "apply":
            return lambda environ, f: 7
        if func_name == "inc":
            return lambda environ, x: x.value + 1
        return None

    env = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))
    again = exec(stop_node(stopped, "$call"), env)
    check(is_ok(again) and value_of(again) == 7,
          f"and that call runs again with no module to resolve the name in: {again!r}")


def _what_the_environment_api_error_records():
    """环境上的 api 收不下给它的东西：这一停里记下是哪一个 api、拿到了什么参数。"""
    headless = Environment(None, EnvironmentCompute(lambda path, name: None))

    stopped = interpret(str(CASES / "api_error_args.viba"), headless)
    checks.environment_api(stopped, "raised", "Environment.sub_env",
                           "an environment api that refused what it was given")
    check(_written(stop_node(stopped, "$args").data) == '"kid"',
          f"what that api was given is recorded, the environment not among it: "
          f"{_written(stop_node(stopped, '$args').data)!r}")
    check(stop_tag(stopped) not in (FAILURE_TAG, NOT_IMPLEMENTED_TAG),
          f"an environment api is no step of the program: {stopped!r}")
    check(stop_node(stopped, "$stack") is None,
          "and it names no stack: it is not about the program")

    nothing = interpret(str(CASES / "api_error_no_args.viba"), headless)
    check(stop_tag(nothing) == ENVIRONMENT_API_TAG
          and stop_text(nothing, "$api_name") == "Environment.tmp_env"
          and isinstance(stop_node(nothing, "$args").data, viba_ast.Nil),
          f"an api given nothing that travels records nothing: {nothing!r}")


def _what_the_design_says():
    """设计层读同一段写法：名字是数据，它说的是 `Any` —— 设计说不出更多。"""
    source = '__impl__ = __dyn_call__ << "add" << $a 1'
    module = custom_module(source)
    node = viba_ast.parse(source).body[0].body
    got = is_sub_type(AstNodeType(node, module), AstNodeType(viba_ast.Any(), module))
    check(isinstance(got, Ok) and got.ok_value is True,
          f"a call whose name is data is `Any` to the design: {got!r}")
    # 同一段写法也要能当一份数据带着走（交给宿主、再递回来那条路）
    check(descriptor_of(AstNodeType(node, module)) is not None,
          "and it has a descriptor to travel with")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-dyn-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

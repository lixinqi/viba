"""模块当函数：__impl__ 进出的那一层，以及模块调用的数据路径。

模块就是函数，一条数据路径上只该有一次调用——所以两次模块调用不许用同一条路径。

    python3 tests/test_interpreter_modules.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import (full_name_of, message_of, module_of,
                                  module_path_of, stop_node, stop_tag, stop_text,
                                  CASES, Checks, Host, PlacedHost, is_ok, value_of)

from viba import serialize
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret, not_implemented)
from viba.type import PROGRAM_ERR_TAG, Ok

checks = Checks("interpreter_modules")
check = checks.check
labelled = checks.labelled

MODULES = Path(__file__).resolve().parent / "data" / "modules"


def _case(name: str) -> str:
    return str(MODULES / f"{name}.viba")


def run(tmp: Path):
    _spec_modules()
    _nested_modules(tmp)
    _cycles(tmp)
    _storage_paths(tmp)
    _not_implemented(tmp)


def _not_implemented(tmp: Path):
    """没实现的那一步不是失败：整条程序停在那一步上，把这一步带回来。

    这一步要能穿过模块调用（被调的模块里那一步没实现，整个 run 的结果就是「没有实现」），
    也要能穿过宿主自己交回来的那份数据（`get_func` 交回它，或者这一步的实现自己交回它，都等于
    这次调用没有实现）——宿主没说这一步是哪一个时，run 把这一步与这次调用补上；说了就留着它的。
    穿过去之后**带回来的那一步是里面那一次调用**，不是外面那一层。同一份 store 上补上那一步，
    同一个 run 就跑完了——这个结果什么都没落下。
    """
    host = Host()
    environ = host.environ()

    outer = _case("not_implemented_outer")

    host.knobs["missing"] = ("add",)
    stopped = interpret(outer, environ)
    checks.not_implemented(stopped, "a step with no implementation inside a called module")
    check(stop_text(stopped, "$module_path") == module_path_of(CASES / "not_implemented_module")
          and stop_text(stopped, "$full_qualified_func_name") ==
          full_name_of(CASES / "not_implemented_module", "add"),
          f"the step is the call that stopped, not the module that called it: {stopped!r}")
    check(message_of(stopped) == "no implementation",
          f"and why: {message_of(stopped)!r}")
    check(_text_of(stop_node(stopped, "$call")) ==
          '__dyn_call__ << "not_implemented_module.add" << $a 1 << $b 2',
          f"the call as it was written, a name that travels: "
          f"{_text_of(stop_node(stopped, '$call'))!r}")
    host.knobs.pop("missing")

    result = interpret(outer, environ)
    check(is_ok(result) and value_of(result) == 3,
          f"the same run finishes once the step is implemented: {result!r}")

    host.knobs["refuse"] = ("add",)
    stopped = interpret(outer, environ)
    checks.not_implemented(stopped, "a get_func that hands back the bare failure data")
    check(stop_text(stopped, "$module_path") == module_path_of(CASES / "not_implemented_module")
          and stop_text(stopped, "$full_qualified_func_name") ==
          full_name_of(CASES / "not_implemented_module", "add")
          and message_of(stopped) == "no implementation",
          f"the bare data is completed with the step the run knows: {stopped!r}")
    host.knobs.pop("refuse")

    # 宿主自己带了话：run 不覆盖它说了什么
    host.knobs["hands_back"] = not_implemented(
        msg="no add here", module_path="root/elsewhere",
        full_qualified_func_name="add")
    stopped = interpret(outer, environ)
    checks.not_implemented(stopped, "a get_func that hands back a failure in its own words")
    check(stop_text(stopped, "$module_path") == "root/elsewhere"
          and stop_text(stopped, "$full_qualified_func_name") == "add"
          and message_of(stopped) == "no add here",
          f"and what it said is kept: {stopped!r}")
    host.knobs.pop("hands_back")

    # 实现自己交回它：不说别的，run 一样把这一步与这次调用补上，tag 留着它给的
    host.knobs["says_no"] = ("add",)
    stopped = interpret(outer, environ)
    checks.not_implemented(stopped, "an implementation that says it has none")
    check(stop_text(stopped, "$module_path") == module_path_of(CASES / "not_implemented_module")
          and stop_text(stopped, "$full_qualified_func_name") ==
          full_name_of(CASES / "not_implemented_module", "add")
          and message_of(stopped) == "no implementation",
          f"the step the run knows is filled in, and why: {stopped!r}")
    check(_text_of(stop_node(stopped, "$call")) ==
          '__dyn_call__ << "not_implemented_module.add" << $a 1 << $b 2',
          f"the call as well, so this result alone can be taken up again: "
          f"{_text_of(stop_node(stopped, '$call'))!r}")
    host.knobs.pop("says_no")

    # ... 它自己带了话：话留着，步名与这次调用照补
    host.knobs["says_no_with"] = {"add": not_implemented(msg="no add here")}
    stopped = interpret(outer, environ)
    checks.not_implemented(stopped, "an implementation that says so in its own words")
    check(message_of(stopped) == "no add here"
          and stop_text(stopped, "$module_path") == module_path_of(CASES / "not_implemented_module")
          and stop_text(stopped, "$full_qualified_func_name") ==
          full_name_of(CASES / "not_implemented_module", "add"),
          f"and what it said is kept beside the step: {stopped!r}")
    host.knobs.pop("says_no_with")


def _text_of(node):
    """One piece of viba data as written, with the laying out flattened away."""
    written = serialize.serialize("call", node)
    if not isinstance(written, Ok):
        return repr(node)
    return " ".join(written.ok_value.split("=", 1)[1].split())


def _spec_modules():
    """设计里那两份：模块当函数、import、sub_env、print。"""
    host = PlacedHost()
    environ = host.environ()

    result = interpret(str(CASES / "add_demo.viba"), environ)
    check(is_ok(result) and value_of(result) == 1000000,
          f"add_demo's __impl__ is what add answered: {result!r}")
    check((module_of(CASES / "add_demo"), "add") in host.calls,
          f"a step is asked from the module that declares it: {host.calls}")

    result = interpret(str(CASES / "main.viba"), environ)
    check(is_ok(result), f"main runs: {result!r}")
    check(("add_demo", "add") in host.calls,
          f"a called module's own step is asked from that module: {host.calls}")
    check(len(host.printed) == 1 and getattr(host.printed[0], "value", None) == 1000000,
          f"print got the value main computed: {host.printed!r}")


def _nested_modules(tmp: Path):
    """模块里 import 模块、dotted import、设计模块、模块成员。"""
    host = PlacedHost()
    environ = host.environ()

    result = interpret(_case("outer"), environ)
    check(is_ok(result) and value_of(result) == 3,
          f"a module imported by a module: {result!r}")

    result = interpret(_case("dotted"), environ)
    check(is_ok(result) and value_of(result) == "ab",
          f"a dotted import finds pkg/mod.viba: {result!r}")

    labelled(interpret(_case("use_design"), environ), "has no __impl__",
             "calling a module that is design only -> $viba_program_err")

    labelled(interpret(_case("dotted_member"), environ), "has no 'Only.More'",
             "a dotted rest that names no definition -> $viba_program_err")

    # 同一个模块两次调用：两条自己的路径，跑两次
    host.calls.clear()
    host.ran_at.clear()
    result = interpret(_case("twice_module"), environ)
    check(is_ok(result) and value_of(result) == 14,
          f"one module called twice: {result!r}")
    leaves = [path for path, name in host.ran_at if name == "leaf"]
    check(leaves == ["root/first", "root/second"],
          f"each call runs the module again, under a path of its own: {host.ran_at}")


def _cycles(tmp: Path):
    """自调用与 A→B→A：环按名字抓。"""
    host = PlacedHost()
    environ = host.environ()

    labelled(interpret(_case("loop"), environ), "already running",
             "a module that calls itself -> $viba_program_err")

    labelled(interpret(_case("cycle_a"), environ), "already running",
             "a module call cycle A->B->A -> $viba_program_err")


def _storage_paths(tmp: Path):
    """一条数据路径上只该有一次调用。

    同一条数据路径上：正在跑的调用不能再进去（环）；已经处理过的**同一个**模块就是同一个
    子计算，把它给出的那份交回去；换个模块挤同一条数据路径才是错的。
    """
    host = PlacedHost()
    environ = host.environ()
    # 主文件自己也占着它那个路径：直接拿 environ 调模块就是撞车
    same_env = _case("same_env")
    labelled(interpret(same_env, environ), "storage path",
             "a module handed the caller's func_name environment -> $viba_program_err")

    # 两次调用给同一个子环境（同名子环境就是同一个 storage）：同一个模块、同一条
    # 数据路径 = 同一个子计算，第二次拿的是第一次给出的那份
    result = interpret(_case("repeated_path"), environ)
    check(is_ok(result) and value_of(result) == 14,
          f"the same call at one storage path is answered once: {result!r}")

    # 两个不同的模块，用同一个名字的子环境 → 也撞车
    labelled(interpret(_case("two_modules"), environ), "storage path",
             "two modules under one storage path -> $viba_program_err")

    # 各给各的名字：两次都跑得起来，宿主看到两个路径，按书写顺序
    host.calls.clear()
    host.ran_at.clear()
    result = interpret(_case("two_names"), environ)
    check(is_ok(result) and value_of(result) == 14,
          f"two modules, two storage paths: {result!r}")
    check([path for path, name in host.ran_at if name == "leaf"]
          == ["root/one", "root/two"],
          f"each module call runs under a path of its own: {host.ran_at}")

    # 没有 storage 的环境：主文件先占下那条空路径，模块再用它就是撞车
    headless = Environment(None, EnvironmentCompute(host.get_func))
    labelled(interpret(same_env, headless), "storage path",
             "a storage-less environment: the module call collides too")

    # 撞车的错误说得清楚：给每次调用一个自己的子环境
    result = interpret(same_env, environ)
    check(stop_tag(result) == PROGRAM_ERR_TAG and "sub_env" in message_of(result),
          f"and the message says what to do: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-modules-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

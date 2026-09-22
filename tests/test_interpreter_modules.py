"""模块当函数：__ret__ 进出的那一层，以及模块调用的 storage 路径。

模块就是函数，它的 storage 路径是它的身份——所以两次模块调用不许用同一条路径。

    python3 tests/test_interpreter_modules.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import ADD, CASES, LEAF, Checks, Host, value_of, write

from viba import serialize
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import VibaProgramErr, NotMyDutyException, Ok, Step

checks = Checks("interpreter_modules")
check = checks.check
labelled = checks.labelled


def run(tmp: Path):
    _spec_modules()
    _nested_modules(tmp)
    _cycles(tmp)
    _storage_paths(tmp)
    _deferral(tmp)


def _deferral(tmp: Path):
    """没实现的那一步不是失败：整条程序停在递延上，等别人来补。

    递延要能穿过模块调用（被调的模块里那一步没实现，整个 run 的答案就是递延），
    也要能穿过宿主的拒绝（`get_func` 抛递延，等于回答递延），而且穿过去之后**不被改写**：
    带回来的那一步是里面那一次调用，不是外面那一层。同一份 store 上补上那一步，同一个 run
    就跑完了——递延什么都没落下。
    """
    host = Host()
    environ = host.environ()

    write(tmp, "deferred_module.viba",
          ADD + "__ret__ = add << $env environ << $a 1 << $b 2\n")
    outer = write(tmp, "deferred_outer.viba", """
import deferred_module as inner
__ret__ = inner << (environ.sub_env << "deferred_module")
""")

    host.knobs["missing"] = ("add",)
    stopped = interpret(outer, environ)
    checks.deferred(stopped, "a step with no implementation inside a called module")
    check(stopped.step == Step("root/deferred_module", "add"),
          f"the step is the call that stopped, not the module that called it: {stopped.step!r}")
    check(stopped.reason == "no implementation", f"and why: {stopped.reason!r}")
    check(_text_of(stopped.call) == "$a 1 * $b 2",
          f"with the material it was given, as it was written: {_text_of(stopped.call)!r}")
    host.knobs.pop("missing")

    result = interpret(outer, environ)
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"the same run finishes once the step is implemented: {result!r}")

    host.knobs["refuse"] = ("add",)
    stopped = interpret(outer, environ)
    checks.deferred(stopped, "a get_func that refuses the call it cannot serve")
    check(stopped.step == Step("root/deferred_module", "add") and
          stopped.reason == "refused",
          f"a bare refusal is completed with the step the run knows: {stopped!r}")
    host.knobs.pop("refuse")

    # 宿主自己带了话：run 不覆盖它说了什么
    host.knobs["refuse_with"] = NotMyDutyException(
        Step("root/elsewhere", "add"), None, "not this shard")
    stopped = interpret(outer, environ)
    checks.deferred(stopped, "a get_func that refuses in its own words")
    check(stopped.step == Step("root/elsewhere", "add") and
          stopped.reason == "not this shard",
          f"and what it said is kept: {stopped!r}")
    host.knobs.pop("refuse_with")


def _text_of(node):
    """One piece of material as written, with the laying out flattened away."""
    written = serialize.serialize("call", node)
    if not isinstance(written, Ok):
        return repr(node)
    return " ".join(written.ok_value.split("=", 1)[1].split())


def _spec_modules():
    """设计里那两份：模块当函数、import、sub_env、print。"""
    host = Host()
    environ = host.environ()

    result = interpret(str(CASES / "add_demo.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 1000000,
          f"add_demo's __ret__ is what add answered: {result!r}")
    check(("root", "add") in host.calls,
          f"the implementation is looked up at the environment's path: {host.calls}")

    result = interpret(str(CASES / "main.viba"), environ)
    check(isinstance(result, Ok), f"main runs: {result!r}")
    check(("root/add_demo", "add") in host.calls,
          f"a module called through a sub-environment looks its functions up there: {host.calls}")
    check(len(host.printed) == 1 and getattr(host.printed[0], "value", None) == 1000000,
          f"print got the value main computed: {host.printed!r}")


def _nested_modules(tmp: Path):
    """模块里 import 模块、dotted import、设计模块、模块成员。"""
    host = Host()
    environ = host.environ()

    write(tmp, "inner.viba", ADD + "__ret__ = add << $env environ << $a 1 << $b 2\n")
    outer = write(tmp, "outer.viba", """
import inner as inner
__ret__ = inner << (environ.sub_env << "inner")
""")
    result = interpret(outer, environ)
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"a module imported by a module: {result!r}")

    write(tmp, "pkg/mod.viba", """
join =
	str
	<- $env Environment
	<- $a str
	<- $b str
	<- { join two strings }
__ret__ = join << $env environ << $a "a" << $b "b"
""")
    dotted = write(tmp, "dotted.viba", """
import pkg.mod as mod
__ret__ = mod << (environ.sub_env << "mod")
""")
    result = interpret(dotted, environ)
    check(isinstance(result, Ok) and value_of(result) == "ab",
          f"a dotted import finds pkg/mod.viba: {result!r}")

    write(tmp, "design_only.viba", "Only = $x int\n")
    design = write(tmp, "use_design.viba", """
import design_only as d
__ret__ = d << (environ.sub_env << "d")
""")
    labelled(interpret(design, environ), "has no __ret__",
             "calling a module that is design only -> VibaProgramErr")

    write(tmp, "late_lib.viba", LEAF + "__ret__ = leaf << $env environ\n")
    dotted_member = write(tmp, "dotted_member.viba",
                          "import late_lib as lib\n__ret__ = lib.Only.More\n")
    labelled(interpret(dotted_member, environ), "has no 'Only.More'",
             "a dotted rest that names no definition -> VibaProgramErr")

    # 同一个模块两次调用：两条自己的路径，跑两次
    host.calls.clear()
    twice_module = write(tmp, "twice_module.viba", ADD + """
import late_lib as lib
__ret__ = add << $env environ
  << $a (lib << (environ.sub_env << "first"))
  << $b (lib << (environ.sub_env << "second"))
""")
    result = interpret(twice_module, environ)
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"one module called twice: {result!r}")
    leaf_calls = [call for call in host.calls if call[1] == "leaf"]
    check([path for path, _ in leaf_calls] == ["root/first", "root/second"],
          f"each call runs the module again, under its own path: {host.calls}")


def _cycles(tmp: Path):
    """自调用与 A→B→A：环按名字抓。"""
    host = Host()
    environ = host.environ()

    write(tmp, "loop.viba", "import loop as loop\n__ret__ = loop << environ\n")
    labelled(interpret(str(tmp / "loop.viba"), environ), "already running",
             "a module that calls itself -> VibaProgramErr")

    write(tmp, "cycle_a.viba",
          "import cycle_b as b\n__ret__ = b << (environ.sub_env << \"b\")\n")
    write(tmp, "cycle_b.viba",
          "import cycle_a as a\n__ret__ = a << (environ.sub_env << \"a\")\n")
    labelled(interpret(str(tmp / "cycle_a.viba"), environ), "already running",
             "a module call cycle A->B->A -> VibaProgramErr")


def _storage_paths(tmp: Path):
    """模块调用的 storage 路径是它的身份：两次调用不可以用同一个。"""
    host = Host()
    environ = host.environ()
    write(tmp, "lib.viba", LEAF + "__ret__ = leaf << $env environ\n")

    # 主文件自己也占着它那个路径：直接拿 environ 调模块就是撞车
    same_env = write(tmp, "same_env.viba",
                     "import lib as lib\n__ret__ = lib << environ\n")
    labelled(interpret(same_env, environ), "storage path",
             "a module handed the caller's own environment -> VibaProgramErr")

    # 两次调用给同一个子环境（同名子环境就是同一个 storage）→ 第二次撞车
    repeated = write(tmp, "repeated_path.viba", ADD + """
import lib as lib
__ret__ = add << $env environ
  << $a (lib << (environ.sub_env << "one"))
  << $b (lib << (environ.sub_env << "one"))
""")
    labelled(interpret(repeated, environ), "storage path",
             "two calls to one storage path -> VibaProgramErr")

    # 两个不同的模块，用同一个名字的子环境 → 也撞车
    write(tmp, "other.viba", LEAF + "__ret__ = leaf << $env environ\n")
    two_modules = write(tmp, "two_modules.viba", ADD + """
import lib as one
import other as two
__ret__ = add << $env environ
  << $a (one << (environ.sub_env << "m"))
  << $b (two << (environ.sub_env << "m"))
""")
    labelled(interpret(two_modules, environ), "storage path",
             "two modules under one storage path -> VibaProgramErr")

    # 各给各的名字：两次都跑得起来，宿主看到两个路径，按书写顺序
    host.calls.clear()
    two_names = write(tmp, "two_names.viba", ADD + """
import lib as one
import other as two
__ret__ = add << $env environ
  << $a (one << (environ.sub_env << "one"))
  << $b (two << (environ.sub_env << "two"))
""")
    result = interpret(two_names, environ)
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"two modules, two storage paths: {result!r}")
    check([path for path, func in host.calls if func == "leaf"]
          == ["root/one", "root/two"],
          f"the host sees each module under its own path: {host.calls}")

    # 没有 storage 的环境：主文件先占下那条空路径，模块再用它就是撞车
    headless = Environment(None, EnvironmentCompute(host.get_func))
    labelled(interpret(same_env, headless), "storage path",
             "a storage-less environment: the module call collides too")

    # 撞车的错误说得清楚：给每次调用一个自己的子环境
    result = interpret(same_env, environ)
    check(isinstance(result, VibaProgramErr) and "sub_env" in result.err_msg,
          f"and the message says what to do: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-modules-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

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

from viba.interpreter import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import Err, Ok

checks = Checks("interpreter_modules")
check = checks.check
labelled = checks.labelled


def run(tmp: Path):
    _spec_modules()
    _nested_modules(tmp)
    _cycles(tmp)
    _storage_paths(tmp)


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

    write(tmp, "inner.viba", ADD + "__ret__ := add << $env environ << $a 1 << $b 2\n")
    outer = write(tmp, "outer.viba", """
import inner as inner
__ret__ := inner << (environ.sub_env << "inner")
""")
    result = interpret(outer, environ)
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"a module imported by a module: {result!r}")

    write(tmp, "pkg/mod.viba", """
join :=
	str
	<- $env Environment
	<- $a str
	<- $b str
	<- { join two strings }
__ret__ := join << $env environ << $a "a" << $b "b"
""")
    dotted = write(tmp, "dotted.viba", """
import pkg.mod as mod
__ret__ := mod << (environ.sub_env << "mod")
""")
    result = interpret(dotted, environ)
    check(isinstance(result, Ok) and value_of(result) == "ab",
          f"a dotted import finds pkg/mod.viba: {result!r}")

    write(tmp, "design_only.viba", "Only := $x int\n")
    design = write(tmp, "use_design.viba", """
import design_only as d
__ret__ := d << (environ.sub_env << "d")
""")
    labelled(interpret(design, environ), "has no __ret__",
             "calling a module that is design only -> Err")

    write(tmp, "late_lib.viba", LEAF + "__ret__ := leaf << $env environ\n")
    dotted_member = write(tmp, "dotted_member.viba",
                          "import late_lib as lib\n__ret__ := lib.Only.More\n")
    labelled(interpret(dotted_member, environ), "has no 'Only.More'",
             "a dotted rest that names no definition -> Err")

    # 同一个模块两次调用：两条自己的路径，跑两次
    host.calls.clear()
    twice_module = write(tmp, "twice_module.viba", ADD + """
import late_lib as lib
__ret__ := add << $env environ
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

    write(tmp, "loop.viba", "import loop as loop\n__ret__ := loop << environ\n")
    labelled(interpret(str(tmp / "loop.viba"), environ), "already running",
             "a module that calls itself -> Err")

    write(tmp, "cycle_a.viba",
          "import cycle_b as b\n__ret__ := b << (environ.sub_env << \"b\")\n")
    write(tmp, "cycle_b.viba",
          "import cycle_a as a\n__ret__ := a << (environ.sub_env << \"a\")\n")
    labelled(interpret(str(tmp / "cycle_a.viba"), environ), "already running",
             "a module call cycle A->B->A -> Err")


def _storage_paths(tmp: Path):
    """模块调用的 storage 路径是它的身份：两次调用不可以用同一个。"""
    host = Host()
    environ = host.environ()
    write(tmp, "lib.viba", LEAF + "__ret__ := leaf << $env environ\n")

    # 主文件自己也占着它那个路径：直接拿 environ 调模块就是撞车
    same_env = write(tmp, "same_env.viba",
                     "import lib as lib\n__ret__ := lib << environ\n")
    labelled(interpret(same_env, environ), "storage path",
             "a module handed the caller's own environment -> Err")

    # 两次调用给同一个子环境（同名子环境就是同一个 storage）→ 第二次撞车
    repeated = write(tmp, "repeated_path.viba", ADD + """
import lib as lib
__ret__ := add << $env environ
  << $a (lib << (environ.sub_env << "one"))
  << $b (lib << (environ.sub_env << "one"))
""")
    labelled(interpret(repeated, environ), "storage path",
             "two calls to one storage path -> Err")

    # 两个不同的模块，用同一个名字的子环境 → 也撞车
    write(tmp, "other.viba", LEAF + "__ret__ := leaf << $env environ\n")
    two_modules = write(tmp, "two_modules.viba", ADD + """
import lib as one
import other as two
__ret__ := add << $env environ
  << $a (one << (environ.sub_env << "m"))
  << $b (two << (environ.sub_env << "m"))
""")
    labelled(interpret(two_modules, environ), "storage path",
             "two modules under one storage path -> Err")

    # 各给各的名字：两次都跑得起来，宿主看到两个路径，按书写顺序
    host.calls.clear()
    two_names = write(tmp, "two_names.viba", ADD + """
import lib as one
import other as two
__ret__ := add << $env environ
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
    check(isinstance(result, Err) and "sub_env" in result.err_msg,
          f"and the message says what to do: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-modules-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

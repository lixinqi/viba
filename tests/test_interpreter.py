"""viba.interpreter 的验收：跑那份可执行 viba 模块。

语料是设计里那两份：add_demo.viba（add / print / __ret__）与 main.viba
（import add_demo as demo，把子 environment 交给模块调用）。

    python tests/test_interpreter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.interpreter import (Environment, EnvironmentCompute, EnvironmentStorage,
                              interpret)
from viba.type import Err, Ok

CASES = Path(__file__).resolve().parent / "data" / "interpreter"

PASS = FAIL = 0


def check(ok: bool, label: str):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def _compute(printed, seen=None):
    """The host side: two functions, and the paths they were asked for."""
    seen = seen if seen is not None else []

    def get_func(module_path, func_name):
        seen.append((module_path, func_name))
        if func_name == "add":
            return lambda env, a, b: a.value + b.value
        if func_name == "print":
            def show(env, x):
                printed.append(x)
                return None
            return show
        return None
    return EnvironmentCompute(get_func)


def _environ(printed, seen=None):
    return Environment(EnvironmentStorage("root"), _compute(printed, seen))


def _write(tmp: Path, name: str, source: str) -> str:
    path = tmp / name
    path.write_text(source)
    return str(path)


def run() -> int:
    printed = []
    seen = []
    environ = _environ(printed, seen)

    # 1. 模块就是函数：environ 进，__ret__ 出
    result = interpret(str(CASES / "add_demo.viba"), environ)
    check(isinstance(result, Ok), f"add_demo runs: {result!r}")
    check(isinstance(result, Ok) and result.ok_value.value == 1000000,
          "its __ret__ is what add answered")
    check(("root", "add") in seen,
          f"the implementation was looked up at the environment's path: {seen}")

    # 2. import 进来的模块可以调用，也可以取它的函数
    result = interpret(str(CASES / "main.viba"), environ)
    check(isinstance(result, Ok), f"main runs: {result!r}")
    check(len(printed) == 1 and getattr(printed[0], "value", None) == 1000000,
          f"print got the value main computed: {printed!r}")
    check(("root/add_demo", "add") in seen,
          f"a module called through a sub-environment looks its functions up there: {seen}")

    # 3. 子 environment 带着父级的 compute
    child = environ.sub_env("add_demo")
    check(isinstance(child, Environment) and child.compute is environ.compute,
          "a sub-environment holds the parent's compute")
    check(child.storage.cur_storage_path == "root/add_demo",
          f"and its own storage path: {child.storage.cur_storage_path!r}")

    # 4. 没有 __ret__ 的文件是设计，不是程序
    tmp = Path(__file__).resolve().parent / "data" / "interpreter" / "_tmp"
    tmp.mkdir(exist_ok=True)
    design = _write(tmp, "design.viba", "A := int\n")
    check(isinstance(interpret(design, environ), Err),
          "a file without __ret__ is design, not a program")

    # 5. 没有实现就是 Err，而不是猜一个
    bare = Environment(EnvironmentStorage("root"), EnvironmentCompute(lambda p, n: None))
    check(isinstance(interpret(str(CASES / "add_demo.viba"), bare), Err),
          "no implementation for add -> Err")

    # 6. 每个可执行函数都要依赖 environ
    no_env = _write(tmp, "no_env.viba", """
double := int <- $a int <- { double it }
__ret__ := double << $a 21
""")
    check(isinstance(interpret(no_env, environ), Err),
          "a function without $env Environment -> Err")

    # 7. 没有给 environ 也是 Err
    not_given = _write(tmp, "not_given.viba", """
add :=
	int
	<- $env Environment
	<- $a int
	<- $b int
	<- { add }
__ret__ := add << $a 1 << $b 2
""")
    check(isinstance(interpret(not_given, environ), Err),
          "a call that was not given the environment -> Err")

    # 8. 找不到模块
    missing = _write(tmp, "missing.viba", "import nope as n\n__ret__ := n << environ\n")
    check(isinstance(interpret(missing, environ), Err),
          "an import that names no file -> Err")

    print(f"interpreter: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())

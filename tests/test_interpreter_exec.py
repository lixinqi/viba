"""`exec`：跑写出来的一份代码（`interpret` 那个入口，第一个参数换成代码本身）。

参数形式和 `interpret` 一样（环境、`get_file`、`list_files`），只是第一份不是文件路径而是模块文本：
主模块不从文件读，所以它**没有文件、也没有名字** —— `import` 只按环境的搜索路径找（它旁边没有目录），
`$stack` 最外那一帧的 `$file_path` 是空串，编译不过时那条话说的是 `<viba_code>`
（`viba-interpreter.md`）。

    python3 tests/test_interpreter_exec.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import error_of, Checks, Host, value_of

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            exec, interpret)
from viba.type import Ok, VibaProgramErr

checks = Checks("interpreter_exec")
check = checks.check

ADD = """__decl__ =
    int
  <- $env Env

args = __get_args__ << __decl__

add =
    int
  <- $env Env
  <- $a int
  <- $b int
  <- { add two integer }

__impl__ = add << $env args.env << $a 1 << $b 2
"""


def run(tmp: Path):
    _the_code_runs()
    _the_entry_points_agree(tmp)
    _no_name_and_no_file()
    _imports_come_from_the_search_path(tmp)
    _what_it_refuses()


def _the_code_runs():
    """写出来的代码就是主模块：跑出它的 `__impl__`。"""
    host = Host()
    result = exec(ADD, host.environ())
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"the code written out runs: {result!r}")
    check(("root", "add") in host.calls,
          f"and the host is asked the way it always is: {host.calls}")


def _the_entry_points_agree(tmp: Path):
    """同一份文本，走 `exec` 和走一份同名的文件，答案一样。"""
    written = tmp / "same.viba"
    written.write_text(ADD)
    host = Host()
    by_code = exec(ADD, host.environ())
    by_file = interpret(str(written), host.environ())
    check(value_of(by_code) == value_of(by_file) == 3,
          f"the code and a file holding it answer the same: {by_code!r}, {by_file!r}")


def _no_name_and_no_file():
    """主模块没有名字、没有文件：栈里那一帧的 `$file_path` 是空串。"""
    host = Host()
    stopped = exec("__decl__ =\n    int\n  <- $env Env\n\n"
                   "args = __get_args__ << __decl__\n\n"
                   "ghost =\n\tint\n\t<- $env Env\n"
                   "\t<- { nothing implements this }\n"
                   "__impl__ = ghost << $env args.env\n", host.environ())
    error = error_of(stopped)
    check(error.func_name == "ghost", f"the step is the one that stopped: {stopped!r}")
    # 名字解析不了是程序错，而 `$stack` 只在程序错那一支上：那一帧就是这份代码
    named = error_of(exec("__impl__ = nope", host.environ()))
    check(isinstance(named, VibaProgramErr) and "in module ''" in named.msg,
          f"the module has no name to report either: {named!r}")
    check(len(named.stack) == 1 and named.stack[0].file_path == ""
          and named.stack[0].lineno == 0,
          f"and the outermost frame has no file: {named.stack!r}")


def _imports_come_from_the_search_path(tmp: Path):
    """`import` 只按搜索路径找：它旁边没有目录。"""
    place = tmp / "modules"
    place.mkdir()
    (place / "helper.viba").write_text(
        "__decl__ =\n    int\n  <- $env Env\n\nargs = __get_args__ << __decl__\n\n"
        "inc =\n    int\n  <- $env Env\n  <- $x int\n  <- { add one }\n\n"
        "__impl__ = inc << $env args.env << $x 41\n")
    code = ("__decl__ =\n    int\n  <- $env Env\n\nargs = __get_args__ << __decl__\n\n"
            "import helper\n\n__impl__ = helper << (args.env.tmp_env << args.env)\n")
    found = exec(code, Host().environ(viba_path=str(place)))
    check(isinstance(found, Ok) and value_of(found) == 42,
          f"an import is looked up on the search path: {found!r}")
    missing = exec(code, Host().environ(viba_path=str(tmp)))
    check("not found" in getattr(error_of(missing), "msg", ""),
          f"and nowhere else — there is no directory beside the code: {missing!r}")


def _what_it_refuses():
    """环境、`get_file`、`list_files` 和代码本身：说法和 `interpret` 那套一样。"""
    host = Host()
    check("needs an Environment" in getattr(error_of(exec(ADD, "nope")), "msg", ""),
          "an environment is an environment")
    check("get_file is a function" in
          getattr(error_of(exec(ADD, host.environ(), get_file=7)), "msg", ""),
          "get_file is a function or None")
    check("list_files is a function" in
          getattr(error_of(exec(ADD, host.environ(), list_files=7)), "msg", ""),
          "list_files is a function or None")
    check("exec needs the module's text" in
          getattr(error_of(exec(7, host.environ())), "msg", ""),
          "the code is a str")

    # 编不过：报的是程序错，说的名字是那份代码的标签
    bad = error_of(exec("__impl__ = )(", host.environ()))
    check(isinstance(bad, VibaProgramErr) and "cannot parse <viba_code>" in bad.msg,
          f"code that does not compile names the label it was compiled as: {bad!r}")

    # 没有 storage 的环境：什么环境上的 api 都不碰时照样跑得起来
    headless = Environment(None, EnvironmentCompute(host.get_func))
    plain = exec("__impl__ = 7", headless)
    check(isinstance(plain, Ok) and value_of(plain) == 7,
          f"an environment with no storage still runs code that needs nothing: {plain!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-exec-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

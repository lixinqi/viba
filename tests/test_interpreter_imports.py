"""源从哪来、名字怎么解析：get_file、VIBA_PATH、编译不过的源、只编一次。

一次运行里的文件有三种来处：主文件、import 旁边、VIBA_PATH 按顺序；`get_file` 在场时全都从
它那儿来。名字按 import 绑定的名字解析（带 as 绑别名，不带 as 绑全名，最长前缀赢）。

用例都在 `tests/data/imports/` 下：能找到的、找不到的、编译不过的、目录结构本身就是搜索路径的，
一份文件一个地方；这里只列每一次运行该跑出什么。

    python3 tests/test_interpreter_imports.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import CASES, Checks, Host, value_of

from viba.interpret import interpret, sub_env, tmp_env
from viba.type import VibaProgramErr, Ok

checks = Checks("interpreter_imports")
check = checks.check
labelled = checks.labelled

MODULES = Path(__file__).resolve().parent / "data" / "imports"
ONE = MODULES / "one"
TWO = MODULES / "two"
FLAT = MODULES / "flat"
DOTTED_TREE = MODULES / "dotted_tree"
FLAT_DOTTED = MODULES / "flatdotted"
DEEP = MODULES / "deep"
VFS = MODULES / "vfs"


def _case(name: str) -> str:
    return str(MODULES / f"{name}.viba")


def _text_of(path: Path) -> str:
    """A file served out of the dict by `get_file` still comes from a file."""
    return path.read_text()


def run(tmp: Path):
    _names(tmp)
    _paths(tmp)
    _virtual_files(tmp)
    _bad_sources(tmp)
    _compiled_once(tmp)


# (文件, 错误话里的片段, 说明)
NAME_CASES = [
    ("names_no_such_member", "has no 'nope'", "a name the imported module does not have"),
    ("names_no_such_env_member", "environment has no 'nope'",
     "a name the environment does not have"),
    ("names_env_without_name", "still waiting for arguments",
     "environ.sub_env with no module name"),
    ("names_nobody", "no definition named", "a name nobody defined"),
]


def _names(tmp: Path):
    """名字解析：设计模块、没写 as 的 import、本地定义压过别名、import 写在最后。"""
    host = Host()
    environ = host.environ()

    for name, want, label in NAME_CASES:
        labelled(interpret(_case(name), environ), want, f"{label} -> VibaProgramErr")

    labelled(interpret(_case("wrong_arg"), environ), "needs an Environment",
             "a module called without an Environment -> VibaProgramErr")

    # import 没写 as：绑定的就是模块全名
    result = interpret(_case("plain_user"), environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"an import without as binds its whole name: {result!r}")

    # 本地定义压过 import 的别名
    result = interpret(_case("shadow"), environ)
    check(isinstance(result, Ok) and value_of(result) == 5,
          f"a local definition shadows an import alias: {result!r}")

    # import 写在定义之后也算
    result = interpret(_case("late_import"), environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"an import written at the end of the file: {result!r}")

    labelled(interpret(_case("missing_import"), environ), "not found",
             "an import that names no file -> VibaProgramErr")

    labelled(interpret(_case("nothing_here"), environ), "no such file",
             "a main file that is not there -> VibaProgramErr")


def _paths(tmp: Path):
    """VIBA_PATH：它挂在 environment 上，按顺序找，import 旁边的先赢。"""
    host = Host()
    environ = host.environ()

    on_path = host.environ(viba_path=f"{ONE}:{TWO}")
    result = interpret(_case("uses_path"), on_path)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"the first directory on VIBA_PATH wins: {result!r}")

    result = interpret(str(TWO / "near.viba"), on_path)
    check(isinstance(result, Ok) and value_of(result) == "hi",
          f"a module next to the importer beats VIBA_PATH: {result!r}")

    # 空条目、不存在的目录：跳过，不炸
    ragged = f":{ONE}:{MODULES / 'missing'}::{FLAT}:"
    result = interpret(_case("dotted"), host.environ(viba_path=ragged))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a dotted module found on VIBA_PATH, empty and missing entries skipped: {result!r}")

    result = interpret(_case("aliased"), host.environ(viba_path=ragged))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"the same module under an alias: {result!r}")

    # 相对路径按当前目录算
    relative = os.path.relpath(str(FLAT), os.getcwd())
    result = interpret(_case("dotted"), host.environ(viba_path=relative))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a relative VIBA_PATH entry: {result!r}")

    # 仓库里现成的两份 .viba：main.viba 旁边没有 pkg/，模块只在给进来的目录里
    paths_case = CASES / "paths"
    main_file = str(paths_case / "main.viba")
    result = interpret(main_file, host.environ(viba_path=paths_case / "elsewhere"))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a dotted module on a VIBA_PATH that is one Path: {result!r}")
    labelled(interpret(main_file, host.environ(viba_path=7)), "viba_path is a string",
             "a VIBA_PATH that is not a string or a path -> VibaProgramErr")
    labelled(interpret(main_file, environ), "not found",
             "the same module with no VIBA_PATH at all -> VibaProgramErr")

    # 点分 import 的最长前缀赢：a.b 与 a.b.c 各是各的模块
    result = interpret(_case("both_dotted"), host.environ(viba_path=str(DOTTED_TREE)))
    check(isinstance(result, Ok) and value_of(result) == 2,
          f"the longest import prefix wins: {result!r}")

    # 点分名也可以是一个带点的平面文件：pkg.inner.viba
    result = interpret(_case("flat_dotted"), host.environ(viba_path=str(FLAT_DOTTED)))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a dotted import that is one file named pkg.inner.viba: {result!r}")

    # 主文件：相对路径、Path、都行
    here = os.getcwd()
    try:
        os.chdir(MODULES)
        result = interpret("relative_main.viba", environ)
    finally:
        os.chdir(here)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a main file named by a relative path: {result!r}")
    result = interpret(str(MODULES / "relative_main.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a main file given as a Path: {result!r}")

    # 深 import：五层，每层给下一个一条自己的路径
    result = interpret(str(DEEP / "leaf1.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a five-deep import chain: {result!r}")
    check(sub_env(on_path, "child").viba_path == on_path.viba_path and
          tmp_env(on_path).viba_path == on_path.viba_path,
          "a child environment keeps the parent's module search path")


def _virtual_files(tmp: Path):
    """get_file：源从宿主手里来，一个字节都不碰文件系统。

    `get_file` 交回来的那一份也是磁盘上的文件（`tests/data/imports/vfs/`），
    只是路径换成虚拟的 `/vfs/...`，所以这里没有写在 Python 里的 viba 源。
    """
    host = Host()
    environ = host.environ()

    files = {
        "/vfs/main.viba": _text_of(VFS / "main.viba"),
        "/vfs/pkg/inner.viba": _text_of(VFS / "pkg" / "inner.viba"),
    }
    asked = []

    def get_file(path):
        asked.append(path)
        return files.get(path)

    result = interpret("/vfs/main.viba", environ, get_file=get_file)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a run served out of a dict: {result!r}")
    check("/vfs/pkg/inner.viba" in asked,
          f"the hook is asked for the module next to the importer: {asked}")
    check(all(isinstance(path, str) for path in asked),
          f"every path the hook sees is written as a string: {asked}")
    check(not Path("/vfs/main.viba").exists(),
          "the paths the hook serves are not on this filesystem")

    # 找不到的名字：每个地方都问过，然后说 not found
    asked.clear()
    missing = "/vfs/wants_nope.viba"
    files[missing] = _text_of(VFS / "wants_nope.viba")
    labelled(interpret(missing, environ, get_file=get_file), "not found",
             "a name the hook does not serve -> not found")
    check(asked == ["/vfs/wants_nope.viba", "/vfs/nope.viba",
                    "/vfs/nope/__generic__.viba"],
          f"the main file, then every place the name could be: {asked}")

    # 模块已经加载过就不再问
    asked.clear()
    twice = "/vfs/twice.viba"
    lib_path = "/vfs/lib.viba"
    files[twice] = _text_of(VFS / "twice.viba")
    files[lib_path] = _text_of(VFS / "lib.viba")
    result = interpret(twice, environ, get_file=get_file)
    check(isinstance(result, Ok) and value_of(result) == 14 and
          asked.count(lib_path) == 1,
          f"a module already loaded is not asked for again: {result!r} {asked}")

    # "这里没有"的三种说法都行：None、FileNotFoundError，以及别的报错
    def with_main(behaviour):
        def hook(path):
            return files[missing] if path == missing else behaviour(path)
        return hook

    def raising(path):
        raise FileNotFoundError(path)

    labelled(interpret(missing, environ, get_file=with_main(raising)), "not found",
             "a hook that raises FileNotFoundError -> not found")
    labelled(interpret(missing, environ, get_file=with_main(
        lambda path: (_ for _ in ()).throw(ValueError("数据库连不上")))), "raised",
        "a hook that raises something else -> VibaProgramErr")
    labelled(interpret(missing, environ, get_file=with_main(lambda path: b"bytes")),
             "not the file's text", "a hook that answers bytes -> VibaProgramErr")
    labelled(interpret(missing, environ, get_file=with_main(lambda path: "X = (")),
             "cannot parse",
             "a hook that answers something that does not compile -> VibaProgramErr")

    # 主文件也要走 hook
    labelled(interpret("/vfs/nowhere.viba", environ, get_file=get_file), "no such file",
             "a main file the hook does not serve -> no such file")

    # hook 在场时不用文件系统：磁盘上那份按虚拟路径给的内容读
    real = str(MODULES / "relative_main.viba")
    served = dict(files)
    served[real] = _text_of(MODULES / "two" / "lib.viba")
    result = interpret(real, environ, get_file=served.get)
    check(isinstance(result, Ok) and value_of(result) == "hi",
          f"get_file wins over the filesystem: {result!r}")

    labelled(interpret(missing, environ, get_file=7), "get_file is a function",
             "a get_file that is not callable -> VibaProgramErr")


def _bad_sources(tmp: Path):
    """编译不过的源、坏的实现、坏的主路径。"""
    host = Host()
    environ = host.environ()

    labelled(interpret(_case("broken"), environ), "cannot parse",
             "a main file that does not compile -> VibaProgramErr")
    labelled(interpret(_case("bad_import"), environ), "cannot parse",
             "an imported module that does not compile -> VibaProgramErr")

    # 词法上就没有这个词：'-' 不能被悄悄跳过，否则 -5 会跑成 5
    result = interpret(_case("negative"), environ)
    check(isinstance(result, VibaProgramErr) and "cannot parse" in result.err_msg
          and "illegal character" in result.err_msg,
          f"a character with no token of its own -> VibaProgramErr: {result!r}")

    # CRLF 只是行尾：写得跟 LF 一样读
    result = interpret(_case("crlf"), environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a module written with CRLF line endings: {result!r}")

    labelled(interpret(str(MODULES), environ), "cannot read",
             "the main path is a directory -> VibaProgramErr")
    labelled(interpret(_case("gone"), environ), "no such file",
             "no such file -> VibaProgramErr")

    bad = Host()
    bad.get_func = lambda p, n: "not callable"
    checks.failed(interpret(_case("weird"), bad.environ()), "raised",
                  "a non-callable implementation")


def _compiled_once(tmp: Path):
    """一个文件只编一次；同一个 tag 给两次，后给的算。"""
    import viba.interpret as interpreter_module

    host = Host()
    environ = host.environ()

    top = _case("top")
    parsed = []
    original = interpreter_module.custom_module

    def counting(source):
        parsed.append(1)
        return original(source)

    interpreter_module.custom_module = counting
    try:
        result = interpret(top, environ)
    finally:
        interpreter_module.custom_module = original
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"one module reached by two importers runs for both: {result!r}")
    check(len(parsed) == 4,
          f"each file is parsed once, however many importers it has: {len(parsed)}")

    result = interpret(_case("twice_tag"), environ)
    check(isinstance(result, Ok) and value_of(result) == 5,
          f"a tag given twice: the later value stands: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-imports-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

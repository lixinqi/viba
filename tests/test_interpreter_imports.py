"""源从哪来、名字怎么解析：get_file、VIBA_PATH、编译不过的源、只编一次。

一次运行里的文件有三种来处：主文件、import 旁边、VIBA_PATH 按顺序；`get_file` 在场时全都从
它那儿来。名字按 import 绑定的名字解析（带 as 绑别名，不带 as 绑全名，最长前缀赢）。

    python3 tests/test_interpreter_imports.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import ADD, CASES, LEAF, TEXT, Checks, Host, value_of, write

from viba.interpret import interpret
from viba.type import VibaProgramErr, Ok

checks = Checks("interpreter_imports")
check = checks.check
labelled = checks.labelled


def run(tmp: Path):
    _names(tmp)
    _paths(tmp)
    _virtual_files(tmp)
    _bad_sources(tmp)
    _compiled_once(tmp)


def _names(tmp: Path):
    """名字解析：设计模块、没写 as 的 import、本地定义压过别名、import 写在最后。"""
    host = Host()
    environ = host.environ()

    write(tmp, "design_two.viba", "Only = int <- $env Environment\n")
    for body, want, label in (
            ("d.nope", "has no 'nope'", "a name the imported module does not have"),
            ("environ.nope", "environment has no 'nope'",
             "a name the environment does not have"),
            ("environ.sub_env", "still waiting for arguments",
             "environ.sub_env with no module name"),
            ("nope << $env environ", "no definition named", "a name nobody defined")):
        path = write(tmp, f"names_{abs(hash(body))}.viba",
                     f"import design_two as d\n__ret__ = {body}\n")
        labelled(interpret(path, environ), want, f"{label} -> VibaProgramErr")

    write(tmp, "plain_two.viba", LEAF + "__ret__ = leaf << $env environ\n")
    wrong = write(tmp, "wrong_arg.viba", "import plain_two as p\n__ret__ = p << 7\n")
    labelled(interpret(wrong, environ), "needs an Environment",
             "a module called without an Environment -> VibaProgramErr")

    # import 没写 as：绑定的就是模块全名
    write(tmp, "plain.viba", LEAF + "__ret__ = leaf << $env environ\n")
    plain = write(tmp, "plain_user.viba", """
import plain
__ret__ = plain << (environ.sub_env << "plain")
""")
    result = interpret(plain, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"an import without as binds its whole name: {result!r}")

    # 本地定义压过 import 的别名
    shadow = write(tmp, "shadow.viba",
                   "import plain as plain\nplain = 5\n__ret__ = plain\n")
    result = interpret(shadow, environ)
    check(isinstance(result, Ok) and value_of(result) == 5,
          f"a local definition shadows an import alias: {result!r}")

    # import 写在定义之后也算
    late = write(tmp, "late_import.viba",
                 "__ret__ = plain << (environ.sub_env << \"plain\")\n"
                 "import plain\n")
    result = interpret(late, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"an import written at the end of the file: {result!r}")

    missing = write(tmp, "missing_import.viba",
                    "import nope as n\n__ret__ = n << environ\n")
    labelled(interpret(missing, environ), "not found", "an import that names no file -> VibaProgramErr")

    labelled(interpret(str(tmp / "nothing_here.viba"), environ), "no such file",
             "a main file that is not there -> VibaProgramErr")


def _paths(tmp: Path):
    """VIBA_PATH：它挂在 environment 上，按顺序找，import 旁边的先赢。"""
    host = Host()
    environ = host.environ()
    first = tmp / "one"
    second = tmp / "two"
    first.mkdir(exist_ok=True)
    second.mkdir(exist_ok=True)
    write(first, "lib.viba", LEAF + "__ret__ = leaf << $env environ\n")
    write(second, "lib.viba", TEXT + "__ret__ = text << $env environ\n")

    on_path = host.environ(viba_path=f"{first}:{second}")
    user = write(tmp, "uses_path.viba",
                 "import lib as lib\n__ret__ = lib << (environ.sub_env << \"lib\")\n")
    result = interpret(user, on_path)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"the first directory on VIBA_PATH wins: {result!r}")

    near = write(second, "near.viba",
                 "import lib as lib\n__ret__ = lib << (environ.sub_env << \"lib\")\n")
    result = interpret(near, on_path)
    check(isinstance(result, Ok) and value_of(result) == "hi",
          f"a module next to the importer beats VIBA_PATH: {result!r}")

    # 空条目、不存在的目录：跳过，不炸
    flat = tmp / "flat"
    flat.mkdir(exist_ok=True)
    write(flat, "pkg/inner.viba", LEAF + "__ret__ = leaf << $env environ\n")
    ragged = f":{first}:{tmp / 'missing'}::{flat}:"
    dotted = write(tmp, "dotted.viba",
                   "import pkg.inner\n"
                   "__ret__ = pkg.inner << (environ.sub_env << \"inner\")\n")
    result = interpret(dotted, host.environ(viba_path=ragged))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a dotted module found on VIBA_PATH, empty and missing entries skipped: {result!r}")

    aliased = write(tmp, "aliased.viba",
                    "import pkg.inner as inner\n"
                    "__ret__ = inner << (environ.sub_env << \"inner\")\n")
    result = interpret(aliased, host.environ(viba_path=ragged))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"the same module under an alias: {result!r}")

    # 相对路径按当前目录算
    relative = os.path.relpath(str(flat), os.getcwd())
    result = interpret(dotted, host.environ(viba_path=relative))
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
    dotted_dir = tmp / "dotted"
    dotted_dir.mkdir(exist_ok=True)
    write(dotted_dir, "a/b.viba", "X = 1\n__ret__ = 1\n")
    write(dotted_dir, "a/b/c.viba", "X = 2\n__ret__ = 2\n")
    both = write(tmp, "both_dotted.viba",
                 "import a.b\nimport a.b.c\n"
                 "__ret__ = a.b.c << (environ.sub_env << \"c\")\n")
    result = interpret(both, host.environ(viba_path=str(dotted_dir)))
    check(isinstance(result, Ok) and value_of(result) == 2,
          f"the longest import prefix wins: {result!r}")

    # 点分名也可以是一个带点的平面文件：pkg.inner.viba
    flat_dir = tmp / "flatdotted"
    flat_dir.mkdir(exist_ok=True)
    write(flat_dir, "pkg.inner.viba", LEAF + "__ret__ = leaf << $env environ\n")
    flat_use = write(tmp, "flat_dotted.viba",
                     "import pkg.inner\n"
                     "__ret__ = pkg.inner << (environ.sub_env << \"inner\")\n")
    result = interpret(flat_use, host.environ(viba_path=str(flat_dir)))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a dotted import that is one file named pkg.inner.viba: {result!r}")

    # 主文件：相对路径、Path、都行
    write(tmp, "relative_main.viba", LEAF + "__ret__ = leaf << $env environ\n")
    here = os.getcwd()
    try:
        os.chdir(tmp)
        result = interpret("relative_main.viba", environ)
    finally:
        os.chdir(here)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a main file named by a relative path: {result!r}")
    result = interpret(Path(tmp) / "relative_main.viba", environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a main file given as a Path: {result!r}")

    # 深 import：五层，每层给下一个一条自己的路径
    depth = tmp / "deep"
    depth.mkdir(exist_ok=True)
    write(depth, "leaf5.viba", LEAF + "__ret__ = leaf << $env environ\n")
    previous = "leaf5"
    for level in range(4, 0, -1):
        name = f"leaf{level}"
        write(depth, f"{name}.viba",
              f"import {previous} as down\n"
              f"__ret__ = down << (environ.sub_env << \"{previous}\")\n")
        previous = name
    result = interpret(str(depth / "leaf1.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a five-deep import chain: {result!r}")
    check(on_path.sub_env("child").viba_path == on_path.viba_path and
          on_path.tmp_sub_env().viba_path == on_path.viba_path,
          "a child environment keeps the parent's module search path")


def _virtual_files(tmp: Path):
    """get_file：源从宿主手里来，一个字节都不碰文件系统。"""
    host = Host()
    environ = host.environ()

    files = {
        "/vfs/main.viba": ("import pkg.inner\n"
                           "__ret__ = pkg.inner << (environ.sub_env << \"inner\")\n"),
        "/vfs/pkg/inner.viba": LEAF + "__ret__ = leaf << $env environ\n",
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
    files[missing] = "import nope\n__ret__ = nope << environ\n"
    labelled(interpret(missing, environ, get_file=get_file), "not found",
             "a name the hook does not serve -> not found")
    check(asked == ["/vfs/wants_nope.viba", "/vfs/nope.viba"],
          f"the main file, then one place for the name that is not there: {asked}")

    # 模块已经加载过就不再问
    asked.clear()
    twice = "/vfs/twice.viba"
    lib_path = "/vfs/lib.viba"
    files[twice] = ADD + ("import lib as one\nimport lib as two\n"
                          "__ret__ = add << $env environ"
                          " << $a (one << (environ.sub_env << \"one\"))"
                          " << $b (two << (environ.sub_env << \"two\"))\n")
    files[lib_path] = LEAF + "__ret__ = leaf << $env environ\n"
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

    # hook 在场时不用文件系统：磁盘上那份真的不再被读
    real = write(tmp, "real_on_disk.viba", LEAF + "__ret__ = leaf << $env environ\n")
    served = dict(files)
    served[real] = TEXT + "__ret__ = text << $env environ\n"
    result = interpret(real, environ, get_file=served.get)
    check(isinstance(result, Ok) and value_of(result) == "hi",
          f"get_file wins over the filesystem: {result!r}")

    labelled(interpret(missing, environ, get_file=7), "get_file is a function",
             "a get_file that is not callable -> VibaProgramErr")


def _bad_sources(tmp: Path):
    """编译不过的源、坏的实现、坏的主路径。"""
    host = Host()
    environ = host.environ()

    broken = write(tmp, "broken.viba", "add =\n\tint\n\t<- )\n")
    labelled(interpret(broken, environ), "cannot parse",
             "a main file that does not compile -> VibaProgramErr")
    bad_import = write(tmp, "bad_import.viba", "import broken as b\n__ret__ = b << environ\n")
    labelled(interpret(bad_import, environ), "cannot parse",
             "an imported module that does not compile -> VibaProgramErr")

    # 词法上就没有这个词：'-' 不能被悄悄跳过，否则 -5 会跑成 5
    negative = write(tmp, "negative.viba", "__ret__ = -5\n")
    result = interpret(negative, environ)
    check(isinstance(result, VibaProgramErr) and "cannot parse" in result.err_msg
          and "illegal character" in result.err_msg,
          f"a character with no token of its own -> VibaProgramErr: {result!r}")

    # CRLF 只是行尾：写得跟 LF 一样读
    crlf = write(tmp, "crlf.viba",
                 (LEAF + "__ret__ = leaf << $env environ\n").replace("\n", "\r\n"))
    result = interpret(crlf, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a module written with CRLF line endings: {result!r}")

    # 冒号不是这门语言的字符：定义只有 `=`，写错了就停在词法上
    colon = write(tmp, "colon.viba", "__ret__ := 5\n")
    result = interpret(colon, environ)
    check(isinstance(result, VibaProgramErr) and "illegal character ':'" in result.err_msg,
          f"a colon where a definition writes `=` -> VibaProgramErr: {result!r}")

    labelled(interpret(str(tmp), environ), "cannot read", "the main path is a directory -> VibaProgramErr")
    labelled(interpret(str(tmp / "gone.viba"), environ), "no such file", "no such file -> VibaProgramErr")

    bad = Host()
    bad.get_func = lambda p, n: "not callable"
    weird = write(tmp, "weird.viba", LEAF + "__ret__ = leaf << $env environ\n")
    checks.failed(interpret(weird, bad.environ()), "raised",
                  "a non-callable implementation")


def _compiled_once(tmp: Path):
    """一个文件只编一次；同一个 tag 给两次，后给的算。"""
    import viba.interpret as interpreter_module

    host = Host()
    environ = host.environ()

    write(tmp, "shared.viba", LEAF + "__ret__ = leaf << $env environ\n")
    write(tmp, "left.viba",
          "import shared as s\n__ret__ = s << (environ.sub_env << \"s\")\n")
    write(tmp, "right.viba",
          "import shared as r\n__ret__ = r << (environ.sub_env << \"r\")\n")
    top = write(tmp, "top.viba", ADD + """
import left as l
import right as r
__ret__ = add << $env environ << $a (l << (environ.sub_env << "l")) << $b (r << (environ.sub_env << "r"))
""")

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

    twice = write(tmp, "twice_tag.viba", ADD + """
__ret__ = add << $env environ << $a 1 << $a 2 << $b 3
""")
    result = interpret(twice, environ)
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

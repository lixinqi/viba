"""Environment 上那几个走数据路径的成员：`$get_root`、`$get_relative_path`、
`$find_by_relative_path`、`$convert_sub_to_sibling`、`$uncompress_relative_path`。

两边的规矩都测：宿主那一侧（`viba/interpret.py` 里那几个函数，以及子环境记着的父级），
和 viba 那一侧（`tests/data/environment/env_*.viba`）。每个成员自己有哪几条规矩、写错时
报什么，以及它们合起来的用法 —— 根 → 相对路径 → 按那条路径找回来，以及把一个很深的数据路径
压成 sup 旁边的 `名字_sha1`、原数据路径记在返回值上。

    python3 tests/test_interpreter_env_paths.py
"""

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import error_of, message_of, Checks, Host, value_of

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            convert_sub_to_sibling, find_by_relative_path,
                            get_relative_path, get_root, interpret,
                            sub_env, tmp_env)
from viba.interpret import BUILTIN_DIR
from viba.type import Ok

checks = Checks("interpreter_env_paths")
check = checks.check
failed = checks.failed
environment_api = checks.environment_api

CASES = Path(__file__).resolve().parent / "data" / "environment"

# builtin.viba 里 Environment 的成员：设计层与运行层要同一份
MEMBERS = ("get_root", "get_relative_path", "find_by_relative_path",
           "convert_sub_to_sibling", "uncompress_relative_path")


def raised(call, want: str, label: str):
    """The host-side method refused the call: its RuntimeError says `want`."""
    try:
        call()
    except RuntimeError as exc:
        check(want in str(exc), f"{label}: expected RuntimeError({want!r}), got {exc!r}")
    else:
        check(False, f"{label}: expected RuntimeError({want!r}), nothing was raised")


def walk(environ, *names):
    """The environments along the way, the starting one first: root, root/a, …"""
    walked = [environ]
    for name in names:
        environ = sub_env(environ, name)
        walked.append(environ)
    return walked


def chain(environ, *names):
    """`environ` walked down by those names: the last one, root/a/b/c and the like."""
    return walk(environ, *names)[-1]


def run(tmp: Path):
    _the_chain(tmp)
    _get_root(tmp)
    _get_relative_path(tmp)
    _find_by_relative_path(tmp)
    _convert_sub_to_sibling(tmp)
    _uncompress_relative_path(tmp)
    _combinations(tmp)
    _the_declaration(tmp)
    _written_in_viba(tmp)


def _the_chain(tmp: Path):
    """子环境记着它是从哪个环境来的：那四个成员走的就是这条链。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "chain"))
    _, a, b, c = walk(environ, "a", "b", "c")

    check([one.storage.cur_storage_path for one in (environ, a, b, c)] ==
          ["root", "root/a", "root/a/b", "root/a/b/c"],
          "a child's path is its parent's plus its name")
    check(a.parent is environ and b.parent is a and c.parent is b,
          "every child names the environment it was made from")
    check(environ.parent is None, "the root was made from nothing")
    check(tmp_env(c).parent is c, "tmp_env's child names it too")
    check(Environment(EnvironmentStorage("root"), environ.compute).parent is None,
          "an environment the host made by hand has no parent")


def _get_root(tmp: Path):
    """`get_root`：链顶那一个；根自己是自己的根；没有环境就没有根。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "root"))
    _, a, b, c = walk(environ, "a", "b", "c")

    check(get_root(c) is environ, "get_root walks up to the top of the chain")
    check(get_root(a) is get_root(c), "one chain, one root")
    check(get_root(environ) is environ, "the root is its own root")
    check(get_root(None) is None, "no environment, no root")
    check(c.get_root(c) is environ, "the member hung on an environment reads the same")

    raised(lambda: get_root(7), "takes an Environment", "a root asked of a number")
    raised(lambda: get_root("root"), "takes an Environment", "a root asked of a string")
    raised(lambda: c.get_root(7), "takes an Environment", "the member asked of a number")


def _get_relative_path(tmp: Path):
    """`get_relative_path`：数据路径去掉 root 那段前缀；不给 root 就用链顶；不是祖先就报错。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "relative"))
    _, a, b, c = walk(environ, "a", "b", "c")

    check(get_relative_path(c, environ) == "a/b/c", "the whole path from the root")
    check(get_relative_path(c, a) == "b/c", "from a middling ancestor")
    check(get_relative_path(c, c) == "", "a storage path seen from itself is empty")
    check(get_relative_path(environ, environ) == "", "and so is the root's")
    check(get_relative_path(c, None) == "a/b/c", "with nil for the root the chain's own is used")
    check(get_relative_path(environ, None) == "", "the root seen from a nil root is empty")

    check(c.get_relative_path(c, environ) == "a/b/c", "the member reads the same")
    check(c.get_relative_path(c, None) == "a/b/c", "and reads a nil root the same way")

    raised(lambda: get_relative_path(c, chain(a, "x")), "is no child of",
           "a root in another branch")
    raised(lambda: get_relative_path(a, c), "is no child of", "a root below the storage path")
    raised(lambda: get_relative_path(None, None), "takes an Environment",
           "no environment at all")
    raised(lambda: get_relative_path(c, 7), "takes an Environment", "a root that is no environment")
    raised(lambda: c.get_relative_path(7, None), "takes an Environment",
           "the member asked for a number")

    # 名字相近的两条路径不算祖先：root/a 与 root/ab
    check(get_relative_path(chain(environ, "ab"), environ) == "ab",
          "a name that starts the same is still its own name")
    raised(lambda: get_relative_path(chain(environ, "ab"), chain(environ, "a")), "is no child of",
           "root/a is no ancestor of root/ab")


def _find_by_relative_path(tmp: Path):
    """`find_by_relative_path`：按相对路径走下来；空路径就是 root 自己；一条条段都得是目录名。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "find"))
    _, a, b, c = walk(environ, "a", "b", "c")

    check(find_by_relative_path("a/b", environ).storage is b.storage,
          "a relative path reaches the same storage sub_env hands back")
    check(find_by_relative_path("a", environ).storage is a.storage, "one segment, one child")
    check(find_by_relative_path("", c).storage is c.storage, "the empty path is the root itself")
    check(find_by_relative_path("x/y", c).storage.cur_storage_path == "root/a/b/c/x/y",
          "every segment walks one child down")
    check(find_by_relative_path("a//b/", environ).storage is b.storage,
          "empty segments are skipped")
    check(find_by_relative_path("a/b", environ).parent.storage is a.storage,
          "what was found knows the parent it was reached through")
    check(get_root(find_by_relative_path("a/b", environ)) is environ,
          "so the chain it belongs to is the root's")

    # 成员读法：不写 root 就从读到它那个环境往下走
    check(c.find_by_relative_path("x", None).storage.cur_storage_path == "root/a/b/c/x",
          "with a nil root the member searches from the environment it was read off")
    check(c.find_by_relative_path("", None).storage is c.storage,
          "and the empty path is that environment itself")
    check(c.find_by_relative_path("a", environ).storage is a.storage, "a written root wins")

    raised(lambda: find_by_relative_path("a/b", None), "takes an Environment",
           "a relative path with no root to read it from")
    raised(lambda: find_by_relative_path("../x", c), "is no relative path", "a .. segment")
    raised(lambda: find_by_relative_path("./x", c), "is no relative path", "a . segment")
    raised(lambda: find_by_relative_path(c, environ), "takes a relative path",
           "an environment written where the path goes")
    raised(lambda: find_by_relative_path(7, environ), "takes a relative path",
           "a number written where the path goes")


def _convert_sub_to_sibling(tmp: Path):
    """`convert_sub_to_sibling`：把 sub 的数据路径压成 sup 旁边一个名字，原数据路径记在返回值上。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "convert"))
    _, a, b, c = walk(environ, "a", "b", "c")
    child = sub_env(c, "child")

    made = convert_sub_to_sibling(a, child)
    digest = hashlib.sha1(b"root/a/b/c/child").hexdigest()
    check(made.storage.cur_storage_path == f"root/a_{digest}",
          "the storage path is the sup's path, an underscore, and the sha1 of the sub's path")
    check(made.parent is environ, "it is made beside the sup: the same parent")
    check(made.uncompress_relative_path == "root/a/b/c/child",
          "the path that was compressed is recorded on the answer")
    check(a.uncompress_relative_path is None,
          "and a plain environment records nothing")

    again = convert_sub_to_sibling(a, child)
    check(again.storage is made.storage,
          "the same sup and sub answer the same path")

    other = convert_sub_to_sibling(a, sub_env(c, "other"))
    check(other.storage.cur_storage_path != made.storage.cur_storage_path,
          "a different sub answers a different path")
    check(other.uncompress_relative_path == "root/a/b/c/other",
          "and records its own path")

    # 深数据路径压成一个名字：长度不再跟着层数长
    deep = chain(a, "x", "y", "z", "w")
    pressed = convert_sub_to_sibling(a, deep)
    check(pressed.storage.cur_storage_path ==
          f"root/a_{hashlib.sha1(b'root/a/x/y/z/w').hexdigest()}",
          "however deep the sub is, the compressed path is one hash long")
    check(pressed.uncompress_relative_path == "root/a/x/y/z/w",
          "and the deep path is the one recorded")

    # sup 的路径必须是 sub 的路径的前缀，而且要落在名字边界上
    other_root = Host().environ(store_root_dir=str(tmp / "elsewhere"))
    raised(lambda: convert_sub_to_sibling(a, chain(other_root, "a")),
           "does not begin with", "a sub from another chain")
    raised(lambda: convert_sub_to_sibling(a, sub_env(environ, "ab")),
           "does not begin with", "a directory whose name merely starts the same")
    raised(lambda: convert_sub_to_sibling(a, a), "does not begin with",
           "the sup itself")
    pressed_again = convert_sub_to_sibling(a, made)
    check(pressed_again.uncompress_relative_path == made.storage.cur_storage_path,
          "a path already pressed from the sup is a sub like any other")
    check(pressed_again.storage.cur_storage_path.startswith("root/a_"),
          "and the sup's path is the prefix of what comes out")
    raised(lambda: convert_sub_to_sibling(environ, b), "is the root",
           "a sup that is the chain's own root")
    raised(lambda: convert_sub_to_sibling(a, None), "takes an Environment",
           "no sub at all")


def _uncompress_relative_path(tmp: Path):
    """`uncompress_relative_path`：压出来的那一层记着原数据路径，别的层是 nil。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "uncompress"))
    _, a, b, c = walk(environ, "a", "b", "c")

    for one, where in ((environ, "the root"), (a, "a child"), (c, "a deeper child")):
        check(one.uncompress_relative_path is None, f"{where} records nothing")

    pressed = convert_sub_to_sibling(a, c)
    check(pressed.uncompress_relative_path == "root/a/b/c",
          "the compressed layer records the path it stands for")
    check(sub_env(pressed, "again").uncompress_relative_path is None,
          "a child of it records nothing of its own")


def _combinations(tmp: Path):
    """合起来用：根 → 相对路径 → 找回来，以及压过的数据路径还能按路径走。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "combination"))
    _, a, b, c = walk(environ, "a", "b", "c")

    pressed = convert_sub_to_sibling(a, c)
    for target in (environ, a, b, c, pressed):
        relative = get_relative_path(target, environ)
        check(find_by_relative_path(relative, environ).storage is target.storage,
              f"{relative!r} found again is the same path")
        check(get_relative_path(find_by_relative_path(relative, environ), environ) == relative,
              f"{relative!r} written out again is the same path")

    check(get_relative_path(c, b) == "c",
          "a child's own path is written from its parent")
    raised(lambda: get_relative_path(pressed, c), "is no child of",
           "a sibling is no child of the one beside it")
    check(find_by_relative_path(get_relative_path(pressed, environ), environ).storage
          is pressed.storage,
          "a compressed layer is reachable like any other directory")

    # 一层压一层：同一个 sup 下压多少次，数据路径都只有它自己的名字加一个哈希那么长
    deeper = convert_sub_to_sibling(a, sub_env(pressed, "low"))
    check(len(deeper.storage.cur_storage_path) == len(pressed.storage.cur_storage_path),
          "compressing again under the same sup keeps the same path length")
    check(deeper.uncompress_relative_path == f"{pressed.storage.cur_storage_path}/low",
          "and records the path it stands for")


def _the_declaration(tmp: Path):
    """设计层那一份（`viba/builtin.viba`）与宿主那一份（Environment 上的方法）对得上。"""
    declared = (BUILTIN_DIR / "builtin.viba").read_text()
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "declared"))
    for name in MEMBERS:
        check(f"  * ${name} (" in declared,
              f"{name} is declared in builtin.viba's Environment")
    for name in MEMBERS[:-1]:
        check(callable(getattr(environ, name)), f"{name} is a method of an environment")
    check(environ.uncompress_relative_path is None,
          "uncompress_relative_path is a member that is a value: nil until something is pressed")
    check("$convert_sub_to_sibling (Env <- $sup Env <- $sub Env)" in declared,
          "convert_sub_to_sibling takes the sup and then the sub")
    check("$uncompress_relative_path (str | nil)" in declared,
          "uncompress_relative_path is a string or nil")
    check("$find_by_relative_path (Env <- $relative_path str" in declared,
          "find_by_relative_path takes the relative path and then the root")
    check("$get_root ((Env | nil) <- $current (Env | nil))" in declared,
          "get_root takes an environment and may answer none")


def _written_in_viba(tmp: Path):
    """viba 那边写出来的调用：每一个成员，加上写错的那几种。"""
    cases = tmp / "cases"

    # 根：read off 这一层，用 tag 写在链头
    host = Host()
    top = host.environ(store_root_dir=str(cases / "root"))
    here = chain(top, "a", "b", "c")
    result = interpret(str(CASES / "env_root.viba"), here)
    check(isinstance(result, Ok) and result.ok_value is top,
          f"$get_root << args.env answers the chain's root: {result!r}")

    result = interpret(str(CASES / "env_root_nil.viba"), here)
    check(isinstance(result, Ok) and value_of(result) is None,
          f"args.env.get_root << nil answers nil: {result!r}")

    # 相对路径：从根往下，以及不给根
    result = interpret(str(CASES / "env_relative_path.viba"), here)
    check(isinstance(result, Ok) and value_of(result) == "a/b/c",
          f"the relative path from the root: {result!r}")
    result = interpret(str(CASES / "env_relative_path.viba"), top)
    check(isinstance(result, Ok) and value_of(result) == "",
          f"the root's own relative path is empty: {result!r}")
    result = interpret(str(CASES / "env_relative_path_no_root.viba"), here)
    check(isinstance(result, Ok) and value_of(result) == "a/b/c",
          f"with nil written for the root, the chain's own root is used: {result!r}")

    environment_api(interpret(str(CASES / "env_relative_path_wrong_root.viba"), here),
                    "is no child of", "Environment.get_relative_path",
                    "a root written as this layer's child")

    # 按相对路径找回来
    result = interpret(str(CASES / "env_find_by_path.viba"), here)
    check(isinstance(result, Ok) and result.ok_value.storage is chain(top, "a", "b").storage,
          f"find_by_relative_path reaches root/a/b: {result!r}")
    result = interpret(str(CASES / "env_find_from_me.viba"), here)
    check(isinstance(result, Ok) and
          result.ok_value.storage.cur_storage_path == "root/a/b/c/x",
          f"with no root it searches from the environment it was read off: {result!r}")
    environment_api(interpret(str(CASES / "env_find_by_path_bad_path.viba"), here),
                    "is no relative path", "Environment.find_by_relative_path",
                    "a path with a .. segment")
    environment_api(interpret(str(CASES / "env_find_by_path_wrong_tag.viba"), here),
                    "takes a relative path", "Environment.find_by_relative_path",
                    "a tag at the chain head: the environment lands where the path goes")

    # convert_sub_to_sibling 与 uncompress_relative_path
    result = interpret(str(CASES / "env_uncompress_relative_path.viba"), here)
    check(isinstance(result, Ok) and value_of(result) is None,
          f"a plain layer records nothing: {result!r}")
    result = interpret(str(CASES / "env_uncompress_relative_path_tag.viba"), here)
    check(isinstance(result, Ok) and value_of(result) is None,
          f"the tag form reads the same member: {result!r}")

    result = interpret(str(CASES / "env_convert_sub_to_sibling.viba"), here)
    check(isinstance(result, Ok) and value_of(result) == "root/a/b/c/child",
          f"the pressed layer records the path it stands for: {result!r}")

    environment_api(interpret(str(CASES / "env_convert_sub_to_sibling_root.viba"), here),
                    "is the root", "Environment.convert_sub_to_sibling",
                    "a sup that is the chain's own root")
    environment_api(interpret(str(CASES / "env_convert_sub_to_sibling_nil.viba"), here),
                    "takes an Environment", "Environment.convert_sub_to_sibling",
                    "nil written for the sub")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-env-paths-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

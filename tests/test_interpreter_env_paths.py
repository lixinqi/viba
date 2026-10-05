"""Environment 上那四个走地址的成员：`$get_root`、`$get_relative_path`、
`$find_by_relative_path`、`$next_sibling`。

两边的规矩都测：宿主那一侧（`viba/interpret.py` 里那四个函数，以及子环境记着的父级），
和 viba 那一侧（`tests/data/environment/env_*.viba`）。每个成员自己有哪几条规矩、写错时
报什么，以及它们合起来的用法 —— 根 → 相对路径 → 按那条路径找回来、`next_sibling` 的编号
一直长下去、拿到的名字已经被占住。

    python3 tests/test_interpreter_env_paths.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host, value_of

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            find_by_relative_path, get_relative_path, get_root,
                            interpret, next_sibling, sub_env, tmp_env)
from viba.interpret import BUILTIN_DIR
from viba.type import Ok

checks = Checks("interpreter_env_paths")
check = checks.check
failed = checks.failed

CASES = Path(__file__).resolve().parent / "data" / "environment"

# builtin.viba 里 Environment 的成员：设计层与运行层要同一份
MEMBERS = ("get_root", "get_relative_path", "find_by_relative_path", "next_sibling")


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
    _next_sibling(tmp)
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
    """`get_relative_path`：地址去掉 root 那段前缀；不给 root 就用链顶；不是祖先就报错。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "relative"))
    _, a, b, c = walk(environ, "a", "b", "c")

    check(get_relative_path(c, environ) == "a/b/c", "the whole path from the root")
    check(get_relative_path(c, a) == "b/c", "from a middling ancestor")
    check(get_relative_path(c, c) == "", "an address seen from itself is empty")
    check(get_relative_path(environ, environ) == "", "and so is the root's")
    check(get_relative_path(c, None) == "a/b/c", "with nil for the root the chain's own is used")
    check(get_relative_path(environ, None) == "", "the root seen from a nil root is empty")

    check(c.get_relative_path(c, environ) == "a/b/c", "the member reads the same")
    check(c.get_relative_path(c, None) == "a/b/c", "and reads a nil root the same way")

    raised(lambda: get_relative_path(c, chain(a, "x")), "is no child of",
           "a root in another branch")
    raised(lambda: get_relative_path(a, c), "is no child of", "a root below the address")
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


def _next_sibling(tmp: Path):
    """`next_sibling`：名字里的号加一；名字被占了、或者根要兄弟，都当场报错。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "sibling"))
    _, a, b, c = walk(environ, "a", "b", "c")

    first = next_sibling(c)
    check(first.storage.cur_storage_path == "root/a/b/c0", "c answers c0")
    check(first.parent is b, "and it is a child of the same parent c is in")
    second = next_sibling(first)
    check(second.storage.cur_storage_path == "root/a/b/c1", "c0 answers c1")
    check(get_root(second) is environ, "a sibling stays in the same chain")

    raised(lambda: next_sibling(c), "is already taken", "c0 asked for twice")
    raised(lambda: next_sibling(first), "is already taken", "c1 asked for twice")
    raised(lambda: next_sibling(environ), "is the root", "the root has no sibling to make")

    # 名字里的号：有号就加一，没有就从 0 起，只有号也一样（每条一支，名字互不打扰）
    for branch, name, want in (("n1", "c", "c0"), ("n2", "c7", "c8"),
                               ("n3", "c007", "c8"), ("n4", "c9", "c10"),
                               ("n5", "1", "2"), ("n6", "d", "d0")):
        where = chain(environ, branch)
        check(next_sibling(chain(where, name)).storage.cur_storage_path ==
              f"root/{branch}/{want}",
              f"{name!r} answers {want!r}")

    # 名字已经被占住：同一个名字再要一次不行（sub_env 给过也算占住）
    handed = chain(b, "e0")
    raised(lambda: next_sibling(chain(b, "e")), "is already taken",
           "a name sub_env already handed out")
    check(next_sibling(handed).storage.cur_storage_path == "root/a/b/e1",
          "the next number is free again")

    # store 底下已经有那个目录：这是复用地址，报错
    store = tmp / "sibling-store"
    host = Host()
    top = host.environ(store_root_dir=str(store))
    where = chain(top, "a", "b")
    (store / "root/a/b/d0").mkdir(parents=True)
    raised(lambda: next_sibling(chain(where, "d")), "already a directory under the store root",
           "a directory already under the store root")
    (store / "root/a/b/d0").rmdir()
    check(next_sibling(chain(where, "d")).storage.cur_storage_path == "root/a/b/d0",
          "with the directory gone the name is free")


def _combinations(tmp: Path):
    """合起来用：根 → 相对路径 → 找回来，以及兄弟的编号一直长下去。"""
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "combination"))
    _, a, b, c = walk(environ, "a", "b", "c")

    first = next_sibling(c)
    second = next_sibling(first)
    for target in (environ, a, b, c, first, second):
        relative = get_relative_path(target, environ)
        check(find_by_relative_path(relative, environ).storage is target.storage,
              f"{relative!r} found again is the same address")
        check(get_relative_path(find_by_relative_path(relative, environ), environ) == relative,
              f"{relative!r} written out again is the same path")

    check(get_relative_path(second, b) == "c1",
          "a sibling's own path is written from the parent both are in")
    raised(lambda: get_relative_path(second, c), "is no child of",
           "a sibling is no child of the one beside it")
    check(find_by_relative_path("a/b/c0", environ).storage is first.storage,
          "a sibling is addressable like any other directory")

    deep = second
    for number in range(2, 8):
        deep = next_sibling(deep)
        check(deep.storage.cur_storage_path == f"root/a/b/c{number}",
              f"the numbering keeps going: c{number}")
    check(next_sibling(deep).storage.cur_storage_path == "root/a/b/c8",
          "and one more after seven of them")


def _the_declaration(tmp: Path):
    """设计层那一份（`viba/builtin.viba`）与宿主那一份（Environment 上的方法）对得上。"""
    declared = (BUILTIN_DIR / "builtin.viba").read_text()
    host = Host()
    environ = host.environ(store_root_dir=str(tmp / "declared"))
    for name in MEMBERS:
        check(f"  * ${name} (" in declared,
              f"{name} is declared in builtin.viba's Environment")
        check(callable(getattr(environ, name)), f"{name} is a method of an environment")
    check("$find_by_relative_path (Environment <- $relative_path str" in declared,
          "find_by_relative_path takes the relative path and then the root")
    check("$get_root ((Environment | nil) <- $current (Environment | nil))" in declared,
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

    failed(interpret(str(CASES / "env_relative_path_wrong_root.viba"), here), "is no child of",
           "a root written as this layer's child")

    # 按相对路径找回来
    result = interpret(str(CASES / "env_find_by_path.viba"), here)
    check(isinstance(result, Ok) and result.ok_value.storage is chain(top, "a", "b").storage,
          f"find_by_relative_path reaches root/a/b: {result!r}")
    result = interpret(str(CASES / "env_find_from_me.viba"), here)
    check(isinstance(result, Ok) and
          result.ok_value.storage.cur_storage_path == "root/a/b/c/x",
          f"with no root it searches from the environment it was read off: {result!r}")
    failed(interpret(str(CASES / "env_find_by_path_bad_path.viba"), here), "is no relative path",
           "a path with a .. segment")
    failed(interpret(str(CASES / "env_find_by_path_wrong_tag.viba"), here),
           "takes a relative path",
           "a tag at the chain head: the environment lands where the path goes")

    # next_sibling：编号、连着要、以及名字被占住
    result = interpret(str(CASES / "env_next_sibling.viba"), here)
    check(isinstance(result, Ok) and result.ok_value.storage.cur_storage_path == "root/a/b/c0",
          f"$next_sibling << args.env answers c0: {result!r}")
    failed(interpret(str(CASES / "env_next_sibling.viba"), here), "is already taken",
           "the same case run twice under one environment")
    failed(interpret(str(CASES / "env_next_sibling.viba"), top), "is the root",
           "the root asked for a sibling")

    result = interpret(str(CASES / "env_next_sibling_twice.viba"), chain(top, "a", "b", "c0"))
    check(isinstance(result, Ok) and result.ok_value.storage.cur_storage_path == "root/a/b/c2",
          f"a sibling of a sibling goes on to c2: {result!r}")

    failed(interpret(str(CASES / "env_next_sibling_taken.viba"), chain(top, "a", "b", "d")),
           "is already taken", "two calls for one name in one run")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-env-paths-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

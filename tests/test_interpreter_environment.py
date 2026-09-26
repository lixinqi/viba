"""Environment：$env 那个参数、子环境、以及宿主给自己加的东西。

每个可执行函数都要 $env Environment，而且那个参数必须写成 $env；sub_env 按名字给子环境，
tmp_sub_env 每次给一个新的；子环境带着父级的 compute。

    python3 tests/test_interpreter_environment.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import ADD, LEAF, Checks, Host, value_of, write

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret, sub_env, tmp_sub_env)
from viba.type import VibaProgramErr, Ok

checks = Checks("interpreter_environment")
check = checks.check
labelled = checks.labelled


def run(tmp: Path):
    _the_env_slot(tmp)
    _children(tmp)
    _written_in_viba(tmp)
    _host_members(tmp)


def _the_env_slot(tmp: Path):
    """必须给、必须是 Environment、那个参数必须写成 $env。"""
    host = Host()
    environ = host.environ()

    no_env = write(tmp, "no_env.viba", """
double = int <- $a int <- { double it }
__ret__ = double << $a 21
""")
    labelled(interpret(no_env, environ), "$env Environment",
             "a function without $env -> VibaProgramErr")

    not_given = write(tmp, "not_given.viba", ADD + "__ret__ = add << $a 1 << $b 2\n")
    check(isinstance(interpret(not_given, environ), Ok),
          "a call that was not given the environment is a closure, not an error")

    wrong = write(tmp, "wrong_env.viba",
                  ADD + "__ret__ = add << $env 7 << $a 1 << $b 2\n")
    labelled(interpret(wrong, environ), "was not given an Environment",
             "an environment argument that is not an Environment -> VibaProgramErr")

    # 环境那个参数必须写成 $env：不带 tag 的 Environment 位不算数
    positional = write(tmp, "positional_env.viba", """
f =
	int
	<- Environment
	<- $x int
	<- { inline }
__ret__ = f << environ << $x 1
""")
    labelled(interpret(positional, environ), "takes no $env Environment",
             "an environment slot written without a tag -> VibaProgramErr")

    labelled(interpret(str(tmp / "not_given.viba"), object()), "needs an Environment",
             "interpret with something that is not an Environment -> VibaProgramErr")

    no_compute = write(tmp, "no_compute.viba", LEAF + "__ret__ = leaf << $env environ\n")
    labelled(interpret(no_compute, Environment(None, None)), "compute",
             "an environment with no compute side -> VibaProgramErr")


def _children(tmp: Path):
    """sub_env：路径是 <父>/<名>，同名同一个；tmp_sub_env：每次一个新的。"""
    host = Host()
    environ = host.environ()

    child = sub_env(environ, "a")
    grand = sub_env(child, "b")
    check(child.storage.cur_storage_path == "root/a",
          "a sub-environment's path is <parent>/<name>")
    check(grand.storage.cur_storage_path == "root/a/b", "and it nests")
    check(child.compute is environ.compute and grand.compute is environ.compute,
          "every sub-environment holds the parent's compute")
    check(sub_env(environ, "a").storage is child.storage,
          "the same name hands back the same sub-storage")
    check(isinstance(child, Environment), "sub_env answers an Environment")

    child_a = sub_env(environ, "a")
    child_b = sub_env(environ, "b")
    check(sub_env(child_a, "x").storage.cur_storage_path == "root/a/x" and
          sub_env(child_b, "x").storage.cur_storage_path == "root/b/x",
          "the same name under two parents is two storages")

    host.calls.clear()
    paths = write(tmp, "paths.viba",
                  ADD + "__ret__ = add << $env environ << $a 1 << $b 2\n")
    labelled(interpret(paths, child), None, "a module runs under a sub-environment")
    check(("root/a", "add") in host.calls,
          f"its path is what get_func sees: {host.calls}")

    # 子环境的名来自 viba 那边写下的东西：可序列化数据取它的叶子，别的就 str 一下
    seeded = EnvironmentStorage("root", {"a": EnvironmentStorage("root/a")})
    check(sub_env(Environment(seeded, environ.compute), "a").storage is
          seeded.sub_storage["a"],
          "a storage handed in ready-made is the one sub_env hands back")
    check(sub_env(environ, 7).storage.cur_storage_path == "root/7",
          "a name that is not a string still lands in the path")

    # tmp_sub_env：名字不用起，每次都是新的一个
    first = tmp_sub_env(environ)
    second = tmp_sub_env(environ)
    check(isinstance(first, Environment) and first.compute is environ.compute,
          "tmp_sub_env answers a child with the parent's compute")
    check(first.storage.cur_storage_path.startswith("root/tmp_") and
          second.storage.cur_storage_path.startswith("root/tmp_"),
          f"and its path says it is temporary: {first.storage.cur_storage_path}")
    check(first.storage is not second.storage and
          first.storage.cur_storage_path != second.storage.cur_storage_path,
          "two calls are two children, unlike sub_env with one name")
    check(first.storage.cur_storage_path in
          [s.cur_storage_path for s in environ.storage.sub_storage.values()],
          "the child it made is remembered under its own name")


def _written_in_viba(tmp: Path):
    """viba 那边调 environ.sub_env / tmp_sub_env，以及 __ret__ 就是环境。"""
    host = Host()
    environ = host.environ()

    # 两次模块调用各拿一个临时环境，直接过唯一性那一关
    host.calls.clear()
    write(tmp, "tmp_lib.viba", LEAF + "__ret__ = leaf << $env environ\n")
    tmp_calls = write(tmp, "tmp_calls.viba", ADD + """
import tmp_lib as lib
__ret__ = add << $env environ
  << $a (lib << (environ.tmp_sub_env << environ) << ())
  << $b (lib << (environ.tmp_sub_env << environ) << ())
""")
    result = interpret(tmp_calls, environ)
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"two module calls, each under a temporary environment: {result!r}")
    leaf_paths = [path for path, func in host.calls if func == "leaf"]
    check(len(leaf_paths) == 2 and len(set(leaf_paths)) == 2 and
          all(path.startswith("root/tmp_") for path in leaf_paths),
          f"each call under a path of its own: {host.calls}")

    # 它收的就是那个环境：写别的值不成立（判定层拦下，见 test_is_sub_type.py）
    wrong = write(tmp, "tmp_wrong.viba", "__ret__ = environ.tmp_sub_env << nil\n")
    check(not isinstance(interpret(wrong, environ), Ok),
          "a temporary child is asked for with the environment it belongs to")

    sub = write(tmp, "sub.viba", """
go =
	int
	<- $env Environment
	<- { nobody calls this }
__ret__ = environ.sub_env << environ << "child"
""")
    result = interpret(sub, environ)
    check(isinstance(result, Ok) and isinstance(result.ok_value, Environment) and
          result.ok_value.storage.cur_storage_path == "root/child",
          f"environ.sub_env written in viba names a child: {result!r}")

    the_env = write(tmp, "the_env.viba", "__ret__ = environ\n")
    result = interpret(the_env, environ)
    check(isinstance(result, Ok) and result.ok_value is environ,
          f"a module whose __ret__ is the environment: {result!r}")

    # 已经答完的 sub_env 再给参数：那不是函数
    answered = write(tmp, "env_answered.viba",
                     '__ret__ = environ.sub_env << environ << "a" << "b"\n')
    labelled(interpret(answered, environ), "is not a function",
             "another argument given to an answered sub_env -> VibaProgramErr")


def _host_members(tmp: Path):
    """environ 上的成员：宿主加的方法能调，不是函数的报 VibaProgramErr，没 storage 也是 VibaProgramErr。"""
    host = Host()
    environ = host.environ()

    class Shouting(Environment):
        __slots__ = ()

        def shout(self, x):
            return f"{x.value}!"

    shouting = Shouting(EnvironmentStorage("root"), EnvironmentCompute(host.get_func))
    said = write(tmp, "shout.viba", "__ret__ = environ.shout << \"hi\"\n")
    result = interpret(said, shouting)
    check(isinstance(result, Ok) and value_of(result) == "hi!",
          f"a method the host hung on its environment: {result!r}")

    storage = write(tmp, "env_member.viba", "__ret__ = environ.storage\n")
    labelled(interpret(storage, environ), "environment has no 'storage'",
             "an environment member that is not callable -> VibaProgramErr")

    # 把 import 绑到 environ 上，也压不过内建的那个环境
    write(tmp, "late_lib.viba", LEAF + "__ret__ = leaf << $env environ\n")
    env_alias = write(tmp, "env_alias.viba",
                      "import late_lib as environ\n__ret__ = environ\n")
    result = interpret(env_alias, environ)
    check(isinstance(result, Ok) and isinstance(result.ok_value, Environment),
          f"environ stays the built-in environment even imported as one: {result!r}")

    # 没有 storage 的环境：宿主那边崩了也是 VibaProgramErr，不是把异常扔出来
    headless = Environment(None, EnvironmentCompute(host.get_func))
    headless_use = write(tmp, "headless.viba", '__ret__ = environ.sub_env << environ << "a"\n')
    checks.failed(interpret(headless_use, headless), "raised",
                  "an environment with no storage, and a step that needs one")
    headless_tmp = write(tmp, "headless_tmp.viba",
                         "__ret__ = environ.tmp_sub_env << environ\n")
    checks.failed(interpret(headless_tmp, headless), "raised",
                  "tmp_sub_env on an environment with no storage")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-environment-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

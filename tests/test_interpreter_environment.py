"""Environment：$env 那个参数、子环境、宿主给自己加的东西，以及 store 里的一次落盘。

每个可执行函数都要 $env Environment，而且那个参数在源码里必须是 $env；sub_env 按名字给子环境，
tmp_env 每次给一个新的；子环境带着父级的 compute。环境不是值：一次运行把环境当答案交回来，
就以 `$viba_program_err` 停下（环境是这次调用的规则）。store 的一次存进整份落完整（另一个进程可能
同时在取和存同一份 store）。

    python3 tests/test_interpreter_environment.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import (Checks, Host, PlacedHost, is_ok, module_of,
                                  value_of)

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret, sub_env, tmp_env)

checks = Checks("interpreter_environment")
check = checks.check
labelled = checks.labelled
environment_api = checks.environment_api

CASES = Path(__file__).resolve().parent / "data" / "environment"


def run(tmp: Path):
    _the_env_slot(tmp)
    _the_env_is_no_answer(tmp)
    _children(tmp)
    _given_in_viba_source(tmp)
    _host_members(tmp)
    _a_store_put_lands_whole(tmp)


def _the_env_is_no_answer(tmp: Path):
    """环境不是结果：只有内建函数能把 Env 声明成返回值，别的函数一律当场报错。"""
    host = Host()
    environ = host.environ()

    # the module's own __decl__
    labelled(interpret(str(CASES / "env_as_a_result.viba"), environ),
             "answers the environment", "__decl__ answering Env -> VibaProgramErr")
    # 模块里的一个函数定义
    labelled(interpret(str(CASES / "env_as_a_result_function.viba"), environ),
             "answers the environment", "a definition answering Env -> VibaProgramErr")
    # 积的成员里的一个函数定义
    labelled(interpret(str(CASES / "env_as_a_result_in_a_product.viba"), environ),
             "answers the environment", "a member chain answering Env -> VibaProgramErr")

    # 收一个 Environment 当参数不在此列：那不是结果；但把它当答案交回来，这一次运行
    # 也停下 —— 环境是这次调用的规则，不是值
    labelled(interpret(str(CASES / "the_env.viba"), environ), "answered the environment",
             "a module that declares Any and answers the environment -> VibaProgramErr")


def _the_env_slot(tmp: Path):
    """必须给、必须是 Environment、那个参数在源码里必须是 $env。"""
    host = Host()
    environ = host.environ()

    labelled(interpret(str(CASES / "no_env.viba"), environ), "$env Env",
             "a function without $env -> VibaProgramErr")

    check(is_ok(interpret(str(CASES / "not_given.viba"), environ)),
          "a call that was not given the environment is a closure, not an error")

    labelled(interpret(str(CASES / "env_not_env_function.viba"), environ),
             "the $env parameter must be Env",
             "a function whose $env parameter is not Env -> VibaProgramErr")
    labelled(interpret(str(CASES / "wrong_env.viba"), environ),
             "was not given an Environment",
             "an environment argument that is not an Environment -> VibaProgramErr")

    # 环境那个参数在源码里必须是 $env：不带 tag 的 Environment 位不算数
    labelled(interpret(str(CASES / "positional_env.viba"), environ), "takes no $env Env",
             "an environment slot in the source without a tag -> VibaProgramErr")

    labelled(interpret(str(CASES / "not_given.viba"), object()), "needs an Environment",
             "interpret with something that is not an Environment -> VibaProgramErr")

    labelled(interpret(str(CASES / "no_compute.viba"), Environment(None, None)), "compute",
             "an environment with no compute side -> VibaProgramErr")


def _children(tmp: Path):
    """sub_env：路径是 <父>/<名>，同名同一个；tmp_env：每次一个新的。"""
    host = PlacedHost()
    environ = host.environ()
    ran_at = host.ran_at

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
    ran_at.clear()
    labelled(interpret(str(CASES / "paths.viba"), child), None,
             "a module runs under a sub-environment")
    check((module_of(CASES / "paths"), "add") in host.calls,
          f"the module path is what get_func sees: {host.calls}")
    check(ran_at == [("root/a", "add")],
          f"and the place it ran is what the environment carries: {ran_at}")

    # 子环境的名来自 viba 那边落盘的东西：可序列化数据取它的叶子，别的就 str 一下
    seeded = EnvironmentStorage("root", {"a": EnvironmentStorage("root/a")})
    check(sub_env(Environment(seeded, environ.compute), "a").storage is
          seeded.sub_storage["a"],
          "a storage handed in ready-made is the one sub_env hands back")
    check(sub_env(environ, 7).storage.cur_storage_path == "root/7",
          "a name that is not a string still lands in the path")

    # tmp_env：名字不用起，每次都是新的一个
    first = tmp_env(environ)
    second = tmp_env(environ)
    check(isinstance(first, Environment) and first.compute is environ.compute,
          "tmp_env answers a child with the parent's compute")
    check(first.storage.cur_storage_path.startswith("root/tmp_") and
          second.storage.cur_storage_path.startswith("root/tmp_"),
          f"and its path says it is temporary: {first.storage.cur_storage_path}")
    check(first.storage is not second.storage and
          first.storage.cur_storage_path != second.storage.cur_storage_path,
          "two calls are two children, unlike sub_env with one name")
    check(first.storage.cur_storage_path in
          [s.cur_storage_path for s in environ.storage.sub_storage.values()],
          "the child it made is remembered under its own name")


def _given_in_viba_source(tmp: Path):
    """viba 那边调 environ.sub_env / tmp_env，以及 __impl__ 就是环境。

    viba 源码里的 sub_env 交出的那个孩子不是值，所以让宿主实现的那一步把它拿到的环境
    报回来，看的就是它的数据路径（`PlacedHost.ran_at`）；而把环境当答案交回来的运行，
    以 `$viba_program_err` 停下。
    """
    host = PlacedHost()
    environ = host.environ()
    ran_at = host.ran_at

    # 两次模块调用各拿一个临时环境，直接过唯一性那一关
    host.calls.clear()
    ran_at.clear()
    result = interpret(str(CASES / "tmp_calls.viba"), environ)
    check(is_ok(result) and value_of(result) == 14,
          f"two module calls, each under a temporary environment: {result!r}")
    check([name for _, name in host.calls].count("leaf") == 2,
          f"the leaf step is asked twice, under its module's path: {host.calls}")
    leaves = [path for path, name in ran_at if name == "leaf"]
    check(len(leaves) == 2 and len(set(leaves)) == 2 and
          all(path.startswith("root/tmp_") for path in leaves),
          f"and each call runs at a temporary environment of its own: {ran_at}")

    # 它收的就是那个环境：给别的值不成立（那个 api 当场拒绝）
    check(not is_ok(interpret(str(CASES / "tmp_wrong.viba"), environ)),
          "a temporary child is asked for with the environment it belongs to")

    # sub_env 交出的那个孩子：让宿主实现的那一步报出它跑在哪条数据路径上
    host.calls.clear()
    ran_at.clear()
    result = interpret(str(CASES / "sub_env_child.viba"), environ)
    check(is_ok(result) and value_of(result) == 7,
          f"a host step run under the child of a sub_env in the source: {result!r}")
    check(ran_at == [("root/child", "leaf")],
          f"environ.sub_env in viba source names the child it should: {ran_at}")

    # 名字也可以按源码里的几份给：站在哪一位、带哪个 tag、调的是哪个函数，拼出来是
    # `step0/a/add`；几份都是叶子时什么都写不出来，那个 api 当场拒绝
    host.calls.clear()
    ran_at.clear()
    result = interpret(str(CASES / "named_child.viba"), environ)
    check(is_ok(result) and value_of(result) == 7,
          f"a host step run under a child named in pieces: {result!r}")
    check(ran_at == [("root/step0/a/add", "leaf")],
          f"the pieces name the child, so the path reads back against the code: {ran_at}")
    environment_api(interpret(str(CASES / "named_nothing.viba"), environ), "names nothing",
                    "Environment.sub_env",
                    "a name of pieces that name nothing")

    # 答案就是环境：环境是这次调用的规则，不是值，运行以 $viba_program_err 停下
    labelled(interpret(str(CASES / "sub.viba"), environ), "answered the environment",
             "environ.sub_env in viba source, answered -> VibaProgramErr")
    labelled(interpret(str(CASES / "the_env.viba"), environ), "answered the environment",
             "a module whose __impl__ is the environment -> VibaProgramErr")

    # 已经处理完的 sub_env 再给参数：那不是函数
    labelled(interpret(str(CASES / "env_answered.viba"), environ), "is not a function",
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
    result = interpret(str(CASES / "shout.viba"), shouting)
    check(is_ok(result) and value_of(result) == "hi!",
          f"a method the host hung on its environment: {result!r}")

    # 成员是一个值时，取出来就是那个值；值本身不是标量（这里是一个 storage 对象）才报错
    # （那是这份程序问错了，不是环境上的 api 拒绝了什么）。
    labelled(interpret(str(CASES / "env_member.viba"), environ),
             "which is no leaf",
             "an environment member that is a value is handed over, "
             "and a host object that is no leaf is refused")

    # 名字就是名字：import 绑到 env 上，`args.env` 仍是这次调用收到的那份环境
    result = interpret(str(CASES / "env_alias.viba"), environ)
    check(is_ok(result) and value_of(result) == 7,
          f"a name bound as an import is a name, not the environment: {result!r}")

    # 没有 storage 的环境：那个 api 拿到的环境它收不下，报的是
    # EnvironmentApiInvalidArgumentErr，带上 api 的名字和它拿到的实参（环境不在里面）
    headless = Environment(None, EnvironmentCompute(host.get_func))
    environment_api(interpret(str(CASES / "headless.viba"), headless), "raised",
                    "Environment.sub_env",
                    "an environment with no storage, and a step that needs one")
    environment_api(interpret(str(CASES / "headless_tmp.viba"), headless),
                    "raised", "Environment.tmp_env",
                    "tmp_env on an environment with no storage")


def _a_store_put_lands_whole(tmp: Path):
    """store 里的一次存进是整份落完整的：先存在旁边再改名过去，不留半份、不留下临时文件。

    一份 store 可以被几个进程同时取用和存进，所以取的那一方只该看到「还没有」或者「完整的一份」。
    """
    storage = EnvironmentStorage("root", None, str(tmp / "store"))
    at = "root/case/value.viba"

    storage.put_text(at, "value =\n  1\n")
    on_disk = sorted(str(one.relative_to(tmp / "store"))
                     for one in (tmp / "store").rglob("*") if one.is_file())
    check(on_disk == [at], f"one store put leaves the one file behind: {on_disk!r}")
    check(storage.get_text(at) == "value =\n  1\n",
          f"and the text comes back whole: {storage.get_text(at)!r}")

    storage.put_text(at, "value =\n  2\n")
    check(storage.get_text(at) == "value =\n  2\n",
          f"a second put replaces the whole text: {storage.get_text(at)!r}")
    check(sorted(one.name for one in (tmp / "store" / "root" / "case").iterdir()) ==
          ["value.viba"],
          f"and leaves no half-saved file behind: "
          f"{sorted(one.name for one in (tmp / 'store' / 'root' / 'case').iterdir())!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-environment-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

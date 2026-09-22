"""幂等：结果要能回放，不纯的宿主函数靠 EnvironmentStorage 的快照做到这一点。

一次运行的结果可回放，靠的是同一条 storage 路径上留着上一次的值。快照是序列化的 viba 数据，
放在 `store_root_dir` 底下；`tmp_sub_env` 的路径每次都不同，所以挂在它底下的调用回放不到——
这不是缺陷，而是"这个函数需要显名保存"的信号。

    python3 tests/test_interpreter_idempotence.py
"""

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, value_of, write

from viba import viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                              interpret, read_snapshot, replayed, snapshot_path,
                              write_snapshot)
from viba.reflect import VibaNode, access as reflect_access
from viba.type import AstNodeType, Err, Ok, custom_module
from viba.viba_type_descriptor import descriptor_of

checks = Checks("interpreter_idempotence")
check = checks.check
labelled = checks.labelled

ROLL = """
roll =
	int
	<- $env Environment
	<- $n int
	<- { roll a die: not a pure function, so its answer is snapshotted }
__ret__ = roll << $env environ << $n 1
"""


def run(tmp: Path):
    _snapshots_and_replay(tmp)
    _store_text(tmp)
    _tmp_paths_never_replay(tmp)


def _snapshots_and_replay(tmp: Path):
    """同一个 store 跑两遍：值相同，不纯的那条路只走一次。"""
    store = tmp / "store"
    calls = []

    def roll(env, n):
        def compute():
            calls.append(n.value)
            return random.randint(1, 10 ** 6)
        return replayed(env, compute, f"roll-{n.value}")

    compute = EnvironmentCompute(lambda path, func: roll if func == "roll" else None)
    source = write(tmp, "roll.viba", ROLL)

    def fresh_environ(root=store):
        return Environment(EnvironmentStorage("root", None, str(root)), compute)

    first = interpret(source, fresh_environ())
    check(isinstance(first, Ok), f"the first run computes: {first!r}")
    first_value = value_of(first)
    check(calls == [1], f"and the impure function ran once: {calls}")

    second = interpret(source, fresh_environ())
    check(isinstance(second, Ok) and value_of(second) == first_value,
          f"the second run answers the first run's value: {second!r}")
    check(calls == [1], f"the impure function is not called again: {calls}")

    # 快照是序列化的 viba 数据，摆在那里，读得回来
    environ = fresh_environ()
    snapshot = Path(environ.storage.store_root_dir) / snapshot_path(environ, "roll-1")
    check(snapshot.is_file(), f"the snapshot is a file: {snapshot}")
    text = snapshot.read_text()
    check(text.startswith("value ="), f"and it is viba source: {text!r}")
    stored = viba_ast.parse(text).body[0].body
    check(isinstance(stored, viba_ast.Constant) and stored.value == first_value,
          f"carrying the value that was answered: {stored!r}")

    # 换一个 store：没有快照可回放，那条路又走了一次
    calls.clear()
    other = interpret(source, fresh_environ(tmp / "other-store"))
    check(isinstance(other, Ok) and calls == [1],
          f"another store has nothing to replay: {other!r} {calls}")


def _store_text(tmp: Path):
    """read_snapshot / write_snapshot 的边角，以及坏掉的快照。"""
    store = tmp / "store"
    calls = []

    def roll(env, n):
        def compute():
            calls.append(n.value)
            return random.randint(1, 10 ** 6)
        return replayed(env, compute, f"roll-{n.value}")

    compute = EnvironmentCompute(lambda path, func: roll if func == "roll" else None)
    source = write(tmp, "roll.viba", ROLL)

    def fresh_environ():
        return Environment(EnvironmentStorage("root", None, str(store)), compute)

    empty = fresh_environ()
    check(read_snapshot(empty, "nothing-here") is None,
          "an empty store answers None")
    given = viba_ast.ProductChain([
        viba_ast.TypeRef("Object"),
        viba_ast.Tagged("$a", viba_ast.Constant(1)),
        viba_ast.Tagged("$b", viba_ast.Constant("x"))])
    written = VibaNode(reflect_access, descriptor_of(
        AstNodeType(given, custom_module(""))), given)
    write_snapshot(empty, written, "product")
    again = read_snapshot(empty, "product")
    check(isinstance(again, VibaNode) and
          reflect_access.leaf(again.by_tag("a")).ok_value == 1 and
          reflect_access.leaf(again.by_tag("b")).ok_value == "x",
          f"a material goes out and comes back: {again!r}")

    # 快照坏了：宿主抛，interpret 答 Err，不是崩
    broken = fresh_environ()
    broken.storage.write_text(snapshot_path(broken, "roll-1"), "value = (")
    checks.failed(interpret(source, broken), "raised", "a snapshot that does not parse")
    nameless = fresh_environ()
    nameless.storage.write_text(snapshot_path(nameless, "roll-1"), "other = 1\n")
    checks.failed(interpret(source, nameless), "no value", "a snapshot with no value")

    # 子 storage 带着同一个 store root；read_text/write_text 是纯文本那一层
    child = empty.storage.sub("a")
    check(child.store_root_dir == empty.storage.store_root_dir,
          "a child storage keeps the parent's store root")
    child.write_text("notes/one.txt", "hello")
    check(child.read_text("notes/one.txt") == "hello", "and reads back what it wrote")
    check(child.read_text("notes/two.txt") is None, "nothing there answers None")


def _tmp_paths_never_replay(tmp: Path):
    """tmp_sub_env 的随机路径不许用来存要回放的东西：它是给纯函数和无返回值用的。

    一个需要幂等的函数如果挂在 tmp_sub_env 底下，回放永远命中不了——每次的路径
    都是新的，快照会一次次写进不同的 tmp_ 目录。幂等检查因此失败，这正是给用户
    的信号：换成显名子环境。"""
    store = tmp / "tmp-store"
    calls = []

    def roll(env, n):
        def compute():
            calls.append(n.value)
            return random.randint(1, 10 ** 6)
        return replayed(env, compute, "roll")

    compute = EnvironmentCompute(lambda path, func: roll if func == "roll" else None)
    write(tmp, "dice.viba", """
roll =
	int
	<- $env Environment
	<- $n int
	<- { roll a die: not pure }
__ret__ = roll << $env environ << $n 1
""")
    named = write(tmp, "dice_named.viba",
                  "import dice as d\n__ret__ = d << (environ.sub_env << \"dice\")\n")
    temporary = write(tmp, "dice_tmp.viba",
                      "import dice as d\n__ret__ = d << (environ.tmp_sub_env << ())\n")

    def fresh_environ():
        return Environment(EnvironmentStorage("root", None, str(store)), compute)

    calls.clear()
    first = interpret(named, fresh_environ())
    second = interpret(named, fresh_environ())
    check(isinstance(first, Ok) and isinstance(second, Ok) and
          value_of(first) == value_of(second) and calls == [1],
          f"a named path replays across runs: {first!r} {second!r} {calls}")

    calls.clear()
    before = sorted(path.name for path in (store / "root").glob("tmp_*"))
    first = interpret(temporary, fresh_environ())
    second = interpret(temporary, fresh_environ())
    after = sorted(path.name for path in (store / "root").glob("tmp_*"))
    check(isinstance(first, Ok) and isinstance(second, Ok) and
          value_of(first) is not None and value_of(second) is not None and
          calls == [1, 1],
          f"a temporary path cannot replay: {first!r} {second!r} {calls}")
    check(len(after) == len(before) + 2,
          f"and every run leaves its own snapshot behind: {before} -> {after}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-idempotence-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

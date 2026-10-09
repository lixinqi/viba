"""解释器自己的栈太长时：把系统栈变成工作清单（`Env.$try_compact`）。

环境的父子链（`$get_parent`）就是解释器自己的栈：一次模块调用拿到一层自己的环境，链上就多一个
环。Y 的每一层都给下一层一层子环境，所以一个够深的递归会把解释器自己的 Python 栈压爆（`fib_acc`
跑到两千层时，链上一万多个环，解释器自己的栈跟着长同样多）。`viba/y_helper.viba` 因此在模块体之前
给出一个压缩点：

    env = $try_compact << args.env << 32

`$try_compact` 答的是给它的那个环境，所以这一步可以起个名字（`env`）在下面用；定义按需求值，
用它的那一处算的时候它才算，压缩点因此排在这一层跑起来之前。过了 32 个环，它就把此刻正在跑的
调用交出来（`cur_stack`），由这次运行最外面那一次调用一个一个地再做（`global_resumable_stack`）：
每次都在当下的 Python 深度上做，所以系统栈不再跟着递归长。已经答过的调用不会再做一遍 —— 答案
按数据路径记着（`_Runner.done`），再做一次模块体时它就在那个已答的调用上停住。

用例在 `tests/data/compaction/`：

    main/at_1.viba               两层，链最长 12，一个压缩点都用不上（对照）
    main/at_100.viba             101 层，链最长 37，链被压缩 14 次，答案是 100
    main/limit_from_a_step.viba  上限是一条定义算出来的，且给 0：模块体做两次
    main/elsewhere.viba          给的环境不在跑的那条链上：什么都不压
    main/impure_at_100.viba      101 层，每层都问一次宿主那个不纯的步骤

`steps/count_down.viba` 是那个递归步骤：一层算 1（由 `note` 给）加下一层，所以每算一次模块体就
正好问一次 `note`（宿主给出，并记下这一笔）。宿主记下的笔数因此就是"这一层跑了几次"：链太长时
模块体会被再做一次，而已经答过的调用不会再做，笔数是一个有界的倍数，不会随重做层层放大。

`steps/impure_count.viba` 是一样的步骤，只是每层先问一次宿主那个不纯的 `tick`：它把自己交给
`replayed` 落快照，所以问了很多次、真正算过的只有每个数据路径一次 —— 重做给的是同一个答案，
再跑一次连算都不必算。

    python3 tests/test_interpreter_compaction.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, is_ok, value_of

from viba.interpret import (BUILTIN_DIR, Environment, EnvironmentCompute,
                            EnvironmentStorage, _Compact, env_chain_length,
                            get_parent, interpret, replayed, sub_env, try_compact)
checks = Checks("interpreter_compaction")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "compaction"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# 压缩点给的那个上限（`viba/y_helper.viba` 里是 32），以及用例自己的那点预期
LIMIT = 32
LAYERS_AT_100 = 101        # n 从 100 走到 0：一层一次调用
RUNS_AT_100 = 100          # 这一步算出来的值：走过的层数


def host_for(notes, asked, computed):
    """每一步的实现：`branch` 的两个开关、内建算子里这几个、`note`，以及 `tick`。

    `note` 记下这一笔，并答出那个上限（用例文件里它给的是 0）。`tick` 是不纯的那一步：
    它把自己交给 `replayed` 落快照，所以每问一次有一笔"问过"（`asked`），
    而真正算过的只有落下快照的那一次（`computed`）。
    """
    def get_func(module_path, func_name):
        if func_name == "note":
            # 每层的那 1：记下这一层跑起来了，并把 1 交给这一层的答案
            def note(env, what):
                notes.append((module_path, what.value))
                return 1
            return note
        if func_name == "limit":
            # 压缩点那个上限：记一笔，并答 0，让链当场过线
            def limit(env, what):
                notes.append((module_path, what.value))
                return 0
            return limit
        if func_name == "tick":
            # 不纯的那一步：问一次记一次，真正算过的走快照
            def tick(env):
                asked.append(env.storage.cur_storage_path)
                return replayed(env, lambda: computed.append(
                    env.storage.cur_storage_path) or 1)
            return tick
        if module_path == "branch":
            return branch.get_func(module_path, func_name)
        if module_path == "builtin":
            if func_name == "echo":
                return lambda env, x: x
            if func_name == "lt":
                return lambda env, x, y: x.value < y.value
            if func_name == "sub":
                return lambda env, x, y: x.value - y.value
            if func_name == "add":
                return lambda env, x, y: x.value + y.value
        return None
    return get_func


def environ_for(store, notes, asked=None, computed=None):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(
                           host_for(notes, asked if asked is not None else [],
                                    computed if computed is not None else [])),
                       viba_path=f"{CASES}:{REPOSITORY_ROOT}")


class Counting:
    """一次运行，外加它压了几次链、同时有几个调用在跑。

    `_hold` 就是每次压缩把调用放回工作清单的那一下；`len(runner.running)` 是此刻在系统栈上的
    模块调用数 —— 压缩要办的正是这个数不跟着递归长。
    """

    def __init__(self):
        self.compactions = 0
        self.chain_lengths = []
        self.running_counts = []

    def install(self):
        import viba.interpret as interpreter
        self._hold = interpreter._hold
        self._run_module = interpreter._run_module
        counter = self

        def hold(runner, tasks):
            counter.compactions += 1
            return counter._hold(runner, tasks)

        def run_module(runner, module, environ, name, file, *rest, **knobs):
            counter.chain_lengths.append(env_chain_length(environ))
            counter.running_counts.append(len(runner.running))
            return counter._run_module(runner, module, environ, name, file,
                                       *rest, **knobs)
        interpreter._hold = hold
        interpreter._run_module = run_module

    def remove(self):
        import viba.interpret as interpreter
        interpreter._hold = self._hold
        interpreter._run_module = self._run_module

    @property
    def deepest_chain(self) -> int:
        return max(self.chain_lengths or [0])

    @property
    def deepest_run(self) -> int:
        """The most module calls that were on the system stack at one moment."""
        return max(self.running_counts or [0])

    @property
    def calls(self) -> int:
        return len(self.running_counts)


def run(tmp: Path):
    _the_members()
    _a_shallow_recursion(tmp)
    _a_deep_recursion(tmp)
    _the_compaction_point(tmp)
    _an_environment_the_run_is_not_in(tmp)
    _an_impure_step_replays(tmp)


def _the_members():
    """`$get_parent` 与 `$try_compact`：设计层那一份与宿主那一份对得上。"""
    declared = (BUILTIN_DIR / "builtin.viba").read_text()
    check("$get_parent ((Env | nil) <- $env Env)" in declared,
          "get_parent is declared in builtin.viba's Environment")
    check("$try_compact (Env <- $env Env <- $env_chain_length_limit int)" in declared,
          "try_compact takes the environment and the chain length limit, and answers "
          "the environment")

    environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(lambda m, f: None))
    _, child = environ, sub_env(environ, "child")
    check(get_parent(child) is environ, "the host's get_parent is the environment it was made from")
    check(get_parent(environ) is None, "a chain's root has no parent")
    check(child.get_parent(child) is environ, "the member hung on an environment takes the same")
    check(env_chain_length(environ) == 1 and env_chain_length(child) == 2,
          "the chain counts the environment itself and every step up to the root")

    # 答的是给它的那个环境：名字可以拿到别处当环境用
    check(try_compact(child, 5) is child,
          "a chain inside the limit is no reason to compact: the environment it was "
          "given comes back")
    # 不在一次运行里：链没过线就答得出来，过线了没有运行可压，当场说清楚
    deep = environ
    for _ in range(40):
        deep = sub_env(deep, "step")
    try:
        try_compact(deep, 5)
    except RuntimeError as exc:
        check("no run is here" in str(exc),
              f"outside a run the member says so, got {exc!r}")
    else:
        check(False, "compacting outside a run should say there is no run")


def _a_shallow_recursion(tmp: Path):
    """链没到上限：一个压缩点都不用，答案还是走出来的那个。"""
    notes: list = []
    counting = Counting()
    counting.install()
    try:
        result = interpret(str(CASES / "main" / "at_1.viba"),
                           environ_for(tmp / "at-1", notes))
    finally:
        counting.remove()
    check(is_ok(result) and value_of(result) == 1,
          f"at_1: two layers of Y answer 1, got {result!r}")
    check(counting.compactions == 0,
          f"at_1: the chain stays below {LIMIT}, so nothing is compacted "
          f"(compacted {counting.compactions} times, deepest chain "
          f"{counting.deepest_chain})")
    check(counting.deepest_chain <= LIMIT,
          f"at_1: the chain never passes the limit, got {counting.deepest_chain}")


def _a_deep_recursion(tmp: Path):
    """101 层：链被压缩好多次，而答案还是走出来的那个。"""
    notes: list = []
    counting = Counting()
    counting.install()
    try:
        result = interpret(str(CASES / "main" / "at_100.viba"),
                           environ_for(tmp / "at-100", notes))
    finally:
        counting.remove()
    check(is_ok(result) and value_of(result) == RUNS_AT_100,
          f"at_100: 101 layers of Y answer {RUNS_AT_100}, got {result!r}")
    check(counting.compactions > 0,
          f"at_100: a chain past {LIMIT} is compacted, got {counting.compactions}")
    check(counting.deepest_chain <= LIMIT + 8,
          f"at_100: no chain is left longer than the limit, got "
          f"{counting.deepest_chain}")
    check(counting.calls > 10 * LAYERS_AT_100,
          f"at_100: a 101-layer recursion makes many calls, got {counting.calls}")
    check(counting.deepest_run <= 3 * LIMIT,
          f"at_100: those {counting.calls} calls never put more than "
          f"{3 * LIMIT} on the stack at once, got {counting.deepest_run}")

    # 每一层都跑过；而一层不会跑很多次 —— 已经答过的调用不会再做，所以重做是有界的
    layers = [what for _module, what in notes if what == "a layer ran"]
    check(len(layers) >= LAYERS_AT_100,
          f"at_100: every one of the {LAYERS_AT_100} layers ran, got {len(layers)}")
    check(len(layers) <= 3 * LAYERS_AT_100,
          f"at_100: replaying a module body stops at the calls already answered, "
          f"so no layer runs over and over, got {len(layers)} runs for "
          f"{LAYERS_AT_100} layers")


def _the_compaction_point(tmp: Path):
    """上限是一次调用算出来的，且给 0：模块体做两次，第二次走完。

    那个上限是一条定义（`env`）算出来的，而 `env` 又给 `$env` 那一处用 —— 定义按需求值，用它的
    那一处算的时候它才算，所以压缩点排在那次调用之前：这就是顺序化。
    """
    notes: list = []
    result = interpret(str(CASES / "main" / "limit_from_a_step.viba"),
                       environ_for(tmp / "limit-zero", notes))
    check(is_ok(result) and value_of(result) == 7,
          f"limit_from_a_step: the run answers 7 through a compaction, got {result!r}")
    runs = [what for _module, what in notes if what == "the body ran"]
    check(len(runs) == 2,
          f"limit_from_a_step: the body starts once, is taken off the stack, and is "
          f"made again, got {len(runs)} starts")


def _an_environment_the_run_is_not_in(tmp: Path):
    """给 `$try_compact` 一个解释器不在其中的环境：环境答回来，一个压缩点都不发生。"""
    notes: list = []
    counting = Counting()
    counting.install()
    try:
        result = interpret(str(CASES / "main" / "elsewhere.viba"),
                           environ_for(tmp / "elsewhere", notes))
    finally:
        counting.remove()
    check(is_ok(result) and value_of(result) == 7,
          f"elsewhere: the environment comes back and the run goes on, got {result!r}")
    check(counting.compactions == 0,
          f"elsewhere: a chain the interpreter is not inside is no reason to compact, "
          f"got {counting.compactions}")


def _an_impure_step_replays(tmp: Path):
    """重做一份模块体时，不纯的那一步由它的快照回答：问过很多次，算过一次。"""
    asked: list = []
    computed: list = []
    store = tmp / "impure"
    environ = environ_for(store, [], asked, computed)
    result = interpret(str(CASES / "main" / "impure_at_100.viba"), environ)
    check(is_ok(result) and value_of(result) == RUNS_AT_100,
          f"impure_at_100: 101 layers answer {RUNS_AT_100}, got {result!r}")
    check(len(computed) == LAYERS_AT_100,
          f"impure_at_100: {LAYERS_AT_100} layers, {LAYERS_AT_100} computations — one "
          f"snapshot per data path, got {len(computed)}")
    check(len(asked) > len(computed),
          f"impure_at_100: the body is made again, so the step is asked more often "
          f"than it computes ({len(asked)} asked, {len(computed)} computed)")
    check(len(set(computed)) == len(computed),
          "impure_at_100: one computation per data path")

    # 再跑一次：同一批数据路径，快照在那里，一次都不必再算
    again = environ_for(store, [], asked, computed)
    result = interpret(str(CASES / "main" / "impure_at_100.viba"), again)
    check(is_ok(result) and value_of(result) == RUNS_AT_100,
          f"impure_at_100 again: the same answer, got {result!r}")
    check(len(computed) == LAYERS_AT_100,
          f"impure_at_100 again: the snapshots answer every layer, so nothing is "
          f"computed again, got {len(computed)} computations")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-compaction-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

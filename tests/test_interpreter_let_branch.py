"""`:=` 与 `branch` 联合：绑定和那两个开关一起用的时候，谁先算、谁不算。

    python3 tests/test_interpreter_let_branch.py

每个 case 是一份可以打开的文件（`tests/data/let_branch/*.viba`），这里只列它该跑出什么。
`tests/data/let_branch/answer.viba` 不是 case，是被 import 的辅助模块。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, value_of

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.reflect import access as reflect_access
from viba.type import NotMyDutyException, Ok, UnderlyingVibaOpFailed, VibaProgramErr

checks = Checks("interpreter_let_branch")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "let_branch"


def host_for(calls):
    """宿主：两个开关交给 branch，tick/tock 记一次副作用，ghost 没人实现。"""
    def get_func(module_path, func_name):
        if func_name in ("tick", "tock"):
            def counted(env):
                calls.append(func_name)
                return len(calls)
            return counted
        if func_name == "add":
            return lambda env, a, b: a.value + b.value
        if func_name == "ge":
            return lambda env, x, y: x.value >= y.value
        if func_name == "explode":
            def explode(env):
                raise ZeroDivisionError("poison")
            return explode
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(calls, store):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls)),
                       viba_path=str(Path(__file__).resolve().parent.parent))


# (文件, 该跑出什么)：
#   ("value", 叶子)   Ok，且叶子是这个值
#   ("error", 片段)   VibaProgramErr，话里含这个片段
#   ("defer", None)   递延：宿主没实现那一步
#   ("fail", 片段)    UnderlyingVibaOpFailed，话里含这个片段
# 最后一列是要看住的副作用调用；None 表示不看。
CASES_TO_RUN = [
    ("condition_is_a_let_block", "value", 10, None),
    ("condition_bound_once", "value", 20, ["tick"]),
    ("value_uses_an_outer_binding", "value", 7, None),
    ("value_is_a_let_block", "value", 7, None),
    ("value_let_block_when_untaken", "value", 42, []),
    ("value_let_block_counts_when_taken", "value", 1, ["tick"]),
    ("untaken_branch_binding_is_poison", "value", 42, []),
    ("taken_branch_binding_defers", "defer", None, []),
    ("taken_branch_binding_fails", "fail", "explode raised", []),
    ("binding_takes_the_switch_result", "value", 7, None),
    ("binding_takes_a_never_result", "value", 5, None),
    ("two_switches_bound_then_added", "value", 30, None),
    ("condition_binding_invisible_to_the_value", "error", "no definition named 'c'", None),
    ("binding_inside_the_taken_branch_is_scoped", "error", "no definition named 'b'", None),
    ("branch_with_a_deferred_condition", "defer", None, []),
    ("nested_switch_in_the_taken_branch", "value", 9, None),
    ("nested_switch_in_the_untaken_branch", "value", 42, []),
    ("outer_binding_reaches_the_inner_switch", "value", 7, None),
    ("inner_binding_shadows_outer_in_a_branch", "value", 3, None),
    ("branch_binds_a_module_answer", "value", 41, None),
]


def _leaf_of(result):
    return reflect_access.leaf(result.ok_value).ok_value


def run(tmp: Path):
    check(len(CASES_TO_RUN) == 20, f"twenty cases: {len(CASES_TO_RUN)}")
    for index, (name, kind, want, calls_wanted) in enumerate(CASES_TO_RUN):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        calls = []
        result = interpret(str(program), environ_for(calls, tmp / f"store-{index}"))
        if kind == "value":
            check(isinstance(result, Ok) and _leaf_of(result) == want,
                  f"{name}: expected {want!r}, got {result!r}")
        elif kind == "error":
            check(isinstance(result, VibaProgramErr) and want in result.err_msg,
                  f"{name}: expected an error saying {want!r}, got {result!r}")
        elif kind == "defer":
            check(isinstance(result, NotMyDutyException),
                  f"{name}: expected the deferral, got {result!r}")
        elif kind == "fail":
            check(isinstance(result, UnderlyingVibaOpFailed) and want in result.msg,
                  f"{name}: expected a failure saying {want!r}, got {result!r}")
        if calls_wanted is not None:
            check(calls == calls_wanted,
                  f"{name}: expected the side effects {calls_wanted}, got {calls}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-let-branch-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

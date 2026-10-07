"""branch 的开关 + 独立文件里的那些调用 + 部分计算：边角案例。

每条用例是一份可以打开的文件（`tests/data/branch_switch/*.viba`）。它自己只写 `__impl__` 那一条链：
分支值总是别的文件里写下的一次调用（`kinds.answer_text`、`pick.half`、`inner.inner_if`），条件、值、
开关的实现分别落在各自的文件里。开关本身是 `branch.viba` 里的那一步，由 `branch.py` 实现。

    python3 tests/test_interpreter_branch_switch.py

`CASES` 每条说：文件、该跑出什么（value / error / not_implemented / fail）、要看住的副作用调用。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import error_of, message_of, Checks, value_of

from viba import viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                              interpret, viba_data)
from viba.reflect import VibaNode, access as reflect_access
from viba.type import Ok, UnderlyingOpErr, VibaProgramErr

checks = Checks("interpreter_branch_switch")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "branch_switch"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def host_for(calls):
    """The host side: counters, values of every kind, one step with no
    implementation, one that raises. Every name here is a step some case file reached."""
    def get_func(module_path, func_name):
        if func_name in ("tick", "tock"):
            def counted(env):
                calls.append(func_name)
                return len(calls)
            return counted
        if func_name == "asked_twice":
            return lambda env, get_x: get_x(env).value + get_x(env).value
        if module_path == "builtin" and func_name == "echo":
            return lambda env, x: x
        if func_name == "gate":
            return lambda env, mode: mode.value
        if func_name == "not_gate":
            return lambda env, mode: not mode.value
        if func_name == "true_gate":
            return lambda env: True
        if func_name == "false_gate":
            return lambda env: False
        if func_name in ("only_if", "inner_if"):
            # 独立文件里的那一步：它就是 branch 的那个开关
            return branch.get_func(module_path, "echo_or_never")
        if func_name in ("only_if_not", "inner_if_not"):
            return branch.get_func(module_path, "never_or_echo")
        if func_name in ("if_true", "if_false"):
            # 条件写死在那份文件里：这两个只有环境和那个实参
            written = func_name == "if_true"
            switch = branch.get_func(module_path, "echo_or_never")
            return lambda env, get_v: switch(
                env, viba_data(viba_ast.Constant(written)), get_v)
        if func_name == "boolean_switch":
            return branch.get_func(module_path, "echo_or_never")
        if func_name == "always_nine":
            return lambda env: 9
        if func_name == "counted_gate":
            def counted_gate(env):
                calls.append("tick")
                return True
            return counted_gate
        if func_name == "add":
            return lambda env, a, b: a.value + b.value
        if func_name == "half":
            return lambda env, b: 40 + b.value
        if func_name == "answer_zero":
            return lambda env: 0
        if func_name == "answer_empty":
            return lambda env: ""
        if func_name == "answer_true":
            return lambda env: True
        if func_name == "answer_negative":
            return lambda env: -1.5
        if func_name == "answer_sum":
            def answer_sum(env):
                return viba_data(viba_ast.SumChain(
                    [viba_ast.Constant(3), viba_ast.Constant("three")]))
            return answer_sum
        if func_name == "boolean_switch":
            return branch.get_func(module_path, "echo_or_never")
        if func_name == "mode_gate":
            return lambda env, mode: mode.value
        if func_name == "answer_never":
            return lambda env: viba_data(viba_ast.Never())
        if func_name == "answer_int":
            return lambda env: 7
        if func_name == "answer_text":
            return lambda env: "kept"
        if func_name == "answer_flag":
            return lambda env: False
        if func_name == "answer_nothing":
            return lambda env: None
        if func_name == "answer_number":
            return lambda env: 0.5
        if func_name == "answer_product":
            def product(env):
                return viba_data(viba_ast.ProductChain(
                    [viba_ast.Tagged("$n", viba_ast.Constant(1)),
                     viba_ast.Tagged("$s", viba_ast.Constant("one"))]))
            return product
        if func_name == "boom":
            def boom(env):
                raise ZeroDivisionError("boom")
            return boom
        if func_name == "explode":
            def explode(env):
                raise ZeroDivisionError("explode raised")
            return explode
        if func_name == "explode_condition":
            def explode_condition(env):
                raise ZeroDivisionError("explode_condition raised")
            return explode_condition
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(store, calls):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls)),
                       viba_path=str(REPOSITORY_ROOT))


# (文件, 该跑出什么)：
#   ("value", 叶子)     Ok，且叶子是这个值
#   ("error", 片段)     VibaProgramErr，话里含这个片段
#   ("not_implemented", None)  没有实现：宿主没实现那一步
#   ("fail", 片段)      UnderlyingOpErr，话里含这个片段
# 最后一列是要看住的副作用调用；None 表示不看。
CASES_TO_RUN = [
    ("a_half_given_switch_completed_here", "value", 7, []),
    ("a_slot_that_does_not_fit", "error", 'does not fit', []),
    ("a_switch_file_called_for_its_answer", "value", 9, []),
    ("a_value_not_implemented_beside_a_live_condition", "not_implemented", None, []),
    ("an_untaken_side_not_implemented_and_counts", "value", 1, ['tick']),
    ("asked_twice_counts_once", "value", 2, ['tick']),
    ("both_counters_only_one_runs", "value", 1, ['tick']),
    ("both_sides_are_poison_and_untaken", "value", 42, []),
    ("both_sides_read_a_file", "value", 7, []),
    ("closure_of_a_switch", "value", 6, []),
    ("closure_of_a_switch_dropped", "value", 6, []),
    ("condition_counted_once_per_branch", "value", 1, ['tick']),
    ("condition_counter_runs_once", "value", 1, ['tick', 'tick']),
    ("condition_from_a_file", "value", 7, []),
    ("condition_from_a_file_false", "value", 0, []),
    ("condition_from_a_switch_and_a_gate", "value", 7, []),
    ("condition_from_two_files", "value", 'kept', []),
    ("condition_is_a_call", "value", 7, []),
    ("condition_is_a_double_negation", "value", 7, []),
    ("condition_is_a_gate_call", "value", 7, []),
    ("condition_is_a_gate_call_false", "value", 0, []),
    ("condition_is_a_member_call", "value", 9, []),
    ("condition_is_another_files_switch", "value", 7, []),
    ("condition_mode_by_name", "value", 7, []),
    ("condition_order_does_not_matter", "value", 7, []),
    ("condition_two_switches_agree", "value", 7, []),
    ("not_implemented_condition", "not_implemented", None, []),
    ("not_implemented_taken", "not_implemented", None, []),
    ("not_implemented_untaken", "value", 42, []),
    ("drops_never", "value", 0, []),
    ("drops_text", "value", 0, []),
    ("drops_true", "value", 0, []),
    ("drops_zero", "value", 0, []),
    ("fail_condition", "fail", 'explode_condition raised', []),
    ("fail_taken", "fail", 'explode raised', []),
    ("fail_untaken", "value", 42, []),
    ("full_closure_still_runs", "value", 3, []),
    ("guard_sum_of_two", "sum", 2, []),
    ("half_as_a_branch_value", "closure", None, []),
    ("half_completed_by_a_counted_value", "value", 2, ['tick']),
    ("half_completed_from_a_file_value", "value", 47, []),
    ("half_of_a_file_switch", "value", 9, []),
    ("half_reused_twice", "value", 42, []),
    ("half_then_the_rest", "value", 42, []),
    ("half_with_a_call_argument", "value", 41, ['tock']),
    ("if_false_gives", "value", 7, []),
    ("if_true_gives", "value", 7, []),
    ("inner_condition_from_a_file", "value", 9, []),
    ("inner_from_a_name", "value", 9, []),
    ("inner_if_false", "value", 0, []),
    ("inner_if_not_false", "value", 9, []),
    ("inner_if_true", "value", 9, []),
    ("inner_value_from_a_file", "value", 7, []),
    ("keeps_empty_text", "value", '', []),
    ("keeps_flag", "value", False, []),
    ("keeps_negative", "value", -1.5, []),
    ("keeps_never_branch", "never", None, []),
    ("keeps_nothing", "value", None, []),
    ("keeps_number", "value", 0.5, []),
    ("keeps_product", "product", None, []),
    ("keeps_sum_value", "sum_value", 2, []),
    ("keeps_text", "value", 'kept', []),
    ("keeps_true", "value", True, []),
    ("keeps_zero", "value", 0, []),
    ("module_answer_as_a_value", "value", 41, []),
    ("module_with_an_empty_argument", "value", 5, []),
    ("nested_inner_in_the_untaken_side", "value", 42, []),
    ("nested_inner_inside_outer", "value", 9, []),
    ("nested_switch_in_both_branches", "value", 9, []),
    ("only_if_false", "value", 0, []),
    ("only_if_not_false", "value", 7, []),
    ("only_if_not_true", "value", 7, []),
    ("only_if_true", "value", 7, []),
    ("only_the_live_side_counts", "value", 1, ['tick']),
    ("outer_drops_inner_takes", "value", 0, []),
    ("outer_eager_argument_computed", "value", 1, ['tock']),
    ("outer_takes_inner_drops", "value", 42, []),
    ("outer_untaken_eager_argument", "value", 42, []),
    ("poison_argument_untaken", "value", 42, []),
    ("poison_module_untaken", "value", 42, []),
    ("switch_in_a_condition_and_in_a_value", "value", 9, []),
    ("taken_side_runs", "value", 1, ['tick']),
    ("the_counter_runs_where_the_branch_runs", "value", 1, ['tick']),
    ("the_whole_answer_is_a_file_value", "value", 41, []),
    ("three_guards_none_live", "never", None, []),
    ("three_guards_one_live", "value", 3, []),
    ("three_guards_two_live", "sum", 2, []),
    ("three_levels_of_switches", "value", 9, []),
    ("two_halves_completed_apart", "value", 10, []),
    ("two_live_branches_keep_a_sum", "sum", 2, []),
    ("undefined_untaken", "value", 42, []),
    ("undefined_value_taken", "error", 'no definition named', []),
    ("untaken_side_never_runs", "value", 42, []),
    ("value_counter_runs_once", "value", 1, ['tick']),
    ("value_from_a_module_with_args", "value", 41, []),
    ("value_is_a_member_call", "value", 9, []),
    ("value_is_a_plain_name", "value", 7, []),
    ("value_is_a_plain_number", "value", 7, []),
    ("value_is_another_files_switch", "value", 9, []),
    ("value_is_never", "value", 0, []),
    ("value_reads_the_condition_file", "value", True, []),
]


def run(tmp: Path):
    check(len(CASES_TO_RUN) == 101, f"a hundred and one cases: {len(CASES_TO_RUN)}")
    for index, (name, kind, want, calls_wanted) in enumerate(CASES_TO_RUN):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        calls = []
        result = interpret(str(program), environ_for(tmp / f"store-{index}", calls))
        if kind == "value":
            check(isinstance(result, Ok) and value_of(result) == want,
                  f"{name}: expected {want!r}, got {result!r}")
        elif kind == "error":
            check(isinstance(error_of(result), VibaProgramErr) and want in message_of(result),
                  f"{name}: expected an error saying {want!r}, got {result!r}")
        elif kind == "not_implemented":
            checks.not_implemented(result, name)
        elif kind == "fail":
            check(isinstance(error_of(result), UnderlyingOpErr) and want in message_of(result),
                  f"{name}: expected a failure saying {want!r}, got {result!r}")
        elif kind == "sum":
            # 多个非 never 的分支同时活着：结果保留为和值，不擅自选一支。
            data = result.ok_value.data if isinstance(result, Ok) else None
            check(isinstance(data, viba_ast.SumChain) and len(data.elements) == want,
                  f"{name}: expected a sum of {want} live branches, got {result!r}")
        elif kind == "never":
            check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.Never),
                  f"{name}: expected never, got {result!r}")
        elif kind == "product":
            check(isinstance(result, Ok)
                  and isinstance(result.ok_value.data, viba_ast.ProductChain),
                  f"{name}: expected a product, got {result!r}")
        elif kind == "sum_value":
            data = result.ok_value.data if isinstance(result, Ok) else None
            check(isinstance(data, viba_ast.SumChain) and len(data.elements) == want,
                  f"{name}: expected a value that is a sum of {want}, got {result!r}")
        elif kind == "closure":
            check(isinstance(result, Ok)
                  and isinstance(result.ok_value.data, viba_ast.Partial),
                  f"{name}: expected a closure, got {result!r}")
        if calls_wanted is not None:
            check(calls == calls_wanted,
                  f"{name}: expected the side effects {calls_wanted}, got {calls}")

    # 一份闭包在两次运行里都用得上：同一个半成品补两次
    result = interpret(str(CASES / "half_twice_check.viba"),
                       environ_for(tmp / "store-twice", []))
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"a half-given call carried across a run: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-branch-switch-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

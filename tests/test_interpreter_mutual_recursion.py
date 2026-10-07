"""两个函数互调：递归、分支与部分计算。

每个用例都是**两份文件**：`left_<name>.viba` 是跑的那一份，`right_<name>.viba` 是它 import 的
那一份；两边互相引用，绕回去的那条路要么没走、要么走到底。一条用例一个文件，可以直接打开
（`tests/data/mutual_recursion/`）。

分支和部分计算决定那条路走不走：

- 分支值那一格是**函数类型**，写在里面的递归调用只有宿主叫它的时候才算——条件不成立时，绕回去的
  那一步一次都不发作；
- 一个调用先给一半（`right.f << $a 1`）就是一个值：它没进入另一份文件，所以也不发作；
- 条件的那个参数是照旧先算的，所以写在条件里的递归调用一定会发作。

    python3 tests/test_interpreter_mutual_recursion.py

每条说该跑出什么：

    value     $ok，叶子是这个值
    cycle     跨文件绕回去：`left.x -> right.y -> left.x: the run came back to where it started`
    in-file   一份文件里的定义绕回自己
    running   模块调用成环：`module 'left_x' is already running: a module call cycle`
    closure   $ok，给出的是一个还没给环境的调用
    never     $ok，结果是 never
    sum       $ok，结果是几支并起来的和
    product   $ok，结果是积
    name      $ok，结果是写下来的那个名字
    error     $viba_program_err，话里含这个片段
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import answer_of, Checks, is_ok, message_of, stop_tag

from viba import viba_ast
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.reflect import access as reflect_access
from viba.type import Ok, PROGRAM_ERR_TAG

checks = Checks("interpreter_mutual_recursion")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "mutual_recursion"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def host_for():
    """每一步的实现：两个开关的宿主实现、几个别的文件里的步、一个不叫实参的步。"""
    def get_func(module_path, func_name):
        if module_path == "builtin" and func_name == "echo":
            return lambda env, x: x
        if func_name == "f":
            return lambda env, a: a.value + 1
        if func_name == "f2":
            return lambda env, a, b: a.value + b.value
        if func_name == "g":
            return lambda env: True
        if func_name == "g_false":
            return lambda env: False
        if func_name == "ignore":
            # 收到的是函数类型的那个实参：不叫它，另一份文件就不会被进到
            return lambda env, get_x: 7
        if func_name == "take":
            return lambda env, get_x: get_x(env).value
        if func_name == "answer":
            return lambda env: 7
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(store):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for()),
                       viba_path=str(REPOSITORY_ROOT))


# (文件, 该跑出什么)：见上面的说明。
CASES_TO_RUN = [
    ("direct_cycle", "cycle", None),
    ("direct_value", "value", 41),
    ("direct_product", "cycle", None),
    ("direct_member", "cycle", None),
    ("bl_true_keeps", "cycle", None),
    ("bl_false_drops", "value", 42),
    ("bl_true_drops", "value", 42),
    ("bl_false_keeps", "cycle", None),
    ("bl_true_keeps_value", "value", 41),
    ("bl_false_drops_value", "value", 42),
    ("bl_two_recursions_kept", "value", 41),
    ("bl_two_recursions_dropped", "cycle", None),
    ("br_true_keeps", "cycle", None),
    ("br_false_drops", "value", 41),
    ("br_true_drops", "value", 41),
    ("br_false_keeps", "cycle", None),
    ("br_product_keeps", "cycle", None),
    ("br_product_drops", "product", None),
    ("br_two_recursions_kept", "cycle", None),
    ("br_two_recursions_dropped", "cycle", None),
    ("both_keep_true", "cycle", None),
    ("both_left_keeps", "value", 41),
    ("both_right_keeps", "value", 42),
    ("both_drop", "value", 42),
    ("both_keep_swapped", "cycle", None),
    ("both_left_keeps_swapped", "value", 41),
    ("both_right_keeps_swapped", "value", 42),
    ("both_drop_swapped", "value", 42),
    ("slot_value_kept", "cycle", None),
    ("slot_value_dropped", "value", 7),
    ("slot_name_is_a_value", "name", None),
    ("slot_partial_call", "value", 2),
    ("slot_partial_call_dropped", "value", 0),
    ("slot_full_call", "value", 2),
    ("slot_full_call_dropped", "value", 0),
    ("slot_never_side", "cycle", None),
    ("part_is_a_closure", "closure", None),
    ("part_completed", "value", 2),
    ("part_two_arguments_owed", "closure", None),
    ("part_two_arguments_completed", "value", 3),
    ("part_environment_too_early", "error", 'was given'),
    ("part_environment_too_early_one_short", "error", 'was given'),
    ("part_in_a_branch", "value", 2),
    ("part_in_a_branch_dropped", "value", 0),
    ("part_then_environment", "value", 2),
    ("part_with_a_file_value", "value", 8),
    ("cond_true_gate", "value", 7),
    ("cond_false_gate", "value", 0),
    ("cond_both_gates", "value", 0),
    ("cond_reads_back", "cycle", None),
    ("cond_reads_back_false_side", "value", True),
    ("cond_inside_the_kept_side", "value", True),
    ("cond_calls_the_other_file", "value", 2),
    ("cond_dropped_before_the_other_file", "value", 0),
    ("nest_kept_kept", "cycle", None),
    ("nest_kept_dropped", "value", 41),
    ("nest_dropped_kept", "value", 42),
    ("nest_dropped_dropped", "value", 42),
    ("nest_kept_kept_swapped", "value", 41),
    ("nest_kept_dropped_swapped", "cycle", None),
    ("nest_kept_kept_both", "cycle", None),
    ("nest_dropped_both", "value", 42),
    ("three_one_live_cycle", "cycle", None),
    ("three_one_live_value", "value", 2),
    ("three_two_live_sum", "sum", 2),
    ("three_none_live", "never", None),
    ("three_cycle_with_a_live_pair", "cycle", None),
    ("three_cycle_dropped", "value", 3),
    ("three_cycle_in_the_middle", "cycle", None),
    ("three_all_live", "sum", 3),
    ("mod_direct", "running", None),
    ("mod_direct_both_sides", "running", None),
    ("mod_branch_kept", "running", None),
    ("mod_branch_dropped", "value", 42),
    ("mod_call_without_environment", "closure", None),
    ("mod_call_environment_without_arguments", "error", 'was given'),
    ("mod_call_both_and_cycle", "running", None),
    ("mod_value_from_the_other", "value", 41),
    ("lazy_slot_never_called", "value", 7),
    ("lazy_slot_called", "cycle", None),
    ("never_absorbs_first", "never", None),
    ("never_absorbs_second", "cycle", None),
    ("nil_keeps_the_other", "cycle", None),
    ("sum_keeps_the_cycle", "cycle", None),
    ("sum_drops_never_first", "cycle", None),
    ("lazy_switch_never_side", "never", None),
    ("lazy_switch_sum_side", "value", 41),
    ("lazy_two_switches_dropped", "value", 42),
    ("edge_self_definition", "in-file", None),
    ("edge_in_file_pair", "in-file", None),
    ("edge_three_hop", "cycle", None),
    ("edge_three_hop_all", "cycle", None),
    ("edge_through_a_product", "cycle", None),
    ("edge_through_a_sum", "cycle", None),
    ("edge_sum_in_the_other_file", "cycle", None),
    ("edge_function_body_is_a_hint", "value", 2),
    ("edge_two_left_definitions", "value", 5),
    ("edge_left_reads_right_twice", "value", 1),
    ("edge_right_imports_twice", "cycle", None),
    ("edge_cycle_path_is_written", "cycle", None),
]


def _leaf(result):
    """The leaf a run answered, or None when the answer is not a leaf."""
    if not is_ok(result):
        return None
    leaf = reflect_access.leaf(answer_of(result))
    return leaf.ok_value if isinstance(leaf, Ok) else None


def _data(result):
    return answer_of(result).data if is_ok(result) else None


def run(tmp: Path):
    check(len(CASES_TO_RUN) == 100, f"a hundred cases: {len(CASES_TO_RUN)}")
    for index, (name, kind, want) in enumerate(CASES_TO_RUN):
        program = CASES / f"left_{name}.viba"
        helper = CASES / f"right_{name}.viba"
        check(program.is_file() and helper.is_file(),
              f"the case is a pair of files: {program.name} + {helper.name}")
        result = interpret(str(program), environ_for(tmp / f"store-{index}"))
        if kind == "value":
            check(is_ok(result) and _leaf(result) == want,
                  f"{name}: expected {want!r}, got {result!r}")
        elif kind == "cycle":
            check(stop_tag(result) == PROGRAM_ERR_TAG
                  and "came back to where it started" in message_of(result),
                  f"{name}: expected the cross-file cycle report, got {result!r}")
        elif kind == "in-file":
            check(stop_tag(result) == PROGRAM_ERR_TAG
                  and "one file's definitions may not go round" in message_of(result),
                  f"{name}: expected the in-file cycle report, got {result!r}")
        elif kind == "running":
            check(stop_tag(result) == PROGRAM_ERR_TAG
                  and "already running" in message_of(result),
                  f"{name}: expected the module-call cycle report, got {result!r}")
        elif kind == "closure":
            check(isinstance(_data(result), viba_ast.Partial),
                  f"{name}: expected a call still owing its environment, got {result!r}")
        elif kind == "never":
            check(isinstance(_data(result), viba_ast.Never),
                  f"{name}: expected never, got {result!r}")
        elif kind == "sum":
            data = _data(result)
            check(isinstance(data, viba_ast.SumChain) and len(data.elements) == want,
                  f"{name}: expected a sum of {want}, got {result!r}")
        elif kind == "product":
            check(isinstance(_data(result), viba_ast.ProductChain),
                  f"{name}: expected a product, got {result!r}")
        elif kind == "name":
            check(isinstance(_data(result), viba_ast.TypeRef),
                  f"{name}: expected the written name, got {result!r}")
        elif kind == "error":
            check(stop_tag(result) == PROGRAM_ERR_TAG and want in message_of(result),
                  f"{name}: expected an error saying {want!r}, got {result!r}")

    # 绕回去的那条路上,两个文件的名字都在话里
    result = interpret(str(CASES / "left_direct_cycle.viba"),
                       environ_for(tmp / "store-name"))
    check(stop_tag(result) == PROGRAM_ERR_TAG
          and "left_direct_cycle.x" in message_of(result)
          and "right_direct_cycle.y" in message_of(result),
          f"the cycle report names both files: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-mutual-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

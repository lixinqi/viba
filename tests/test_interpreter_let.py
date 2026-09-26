"""`:=`：表达式里的绑定，作用域只在它所在的那个表达式里。

    python3 tests/test_interpreter_let.py

每个 case 是一份可以打开的文件（`tests/data/let/*.viba`），这里只列它该跑出什么。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, value_of, write

from viba import viba_ast
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.reflect import access as reflect_access
from viba.type import NotMyDutyException, Ok, UnderlyingVibaOpFailed, VibaProgramErr

checks = Checks("interpreter_let")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "let"


def host_for(calls):
    """宿主：`tick`/`tock` 记一次副作用，`add`/`ge`/`x_of` 各做一件事。"""
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
        if func_name == "ignore_x":
            # 被标记的函数拿到的是 getter：它不叫，实参就不算
            return lambda get_env, get_x: 7
        if func_name == "take_x":
            return lambda get_env, get_x: get_x().value
        if func_name == "ask_twice":
            def ask_twice(get_env, get_x):
                return get_x().value + get_x().value
            return ask_twice
        if func_name == "x_of":
            def x_of(env, point):
                return reflect_access.leaf(point.by_tag("x")).ok_value
            return x_of
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
#   ("ok", None)      Ok，叶子是什么不管
#   ("len", n)        Ok，答案是元组且长度是 n
#   ("tags", {...})   Ok，答案是可序列化数据，这些 tag 的叶子各是这些值
#   ("error", 片段)   VibaProgramErr，话里含这个片段
#   ("defer", None)   递延
#   ("fail", 片段)    UnderlyingVibaOpFailed，话里含这个片段
# 最后一列是要看住的副作用调用；None 表示不看。
CASES_TO_RUN = [
    ("one_binding", "value", 7, None),
    ("two_bindings", "value", 7, None),
    ("binding_sees_earlier", "value", 7, None),
    ("ten_bindings", "value", 11, None),
    ("binding_shadows_definition", "value", 1, None),
    ("definition_survives_the_block", "value", 41, None),
    ("binding_shadows_import", "value", 5, None),
    ("binding_shadows_environ", "value", 5, None),
    ("binding_invisible_outside", "error", "no definition named 'a'", None),
    ("outer_visible_inside", "value", 3, None),
    ("inner_shadows_outer", "value", 3, None),
    ("same_name_twice", "value", 2, None),
    ("binding_shadows_builtin_name", "value", 7, None),
    ("three_levels_deep", "value", 3, None),
    ("binding_of_material", "value", 1, None),
    ("binding_as_tagged_argument", "value", 7, None),
    ("binding_as_positional_argument", "value", 7, None),
    ("let_as_call_argument", "value", 7, None),
    ("let_result_is_applied", "value", 7, None),
    ("let_applied_inside", "value", 7, None),
    ("two_lets_one_expression", "value", 3, None),
    ("nested_blocks_arithmetic", "value", 6, None),
    ("let_result_is_nil", "value", None, None),
    ("let_result_is_never", "ok", None, None),
    ("let_result_is_any", "ok", None, None),
    ("let_result_is_empty_tuple", "len", 0, None),
    ("let_result_is_a_function", "ok", None, None),
    ("let_in_taken_branch", "value", 1, ["tick"]),
    ("let_in_untaken_branch_poison", "value", 42, []),
    ("let_in_untaken_branch_counts", "value", 42, []),
    ("let_ignored_by_the_host", "value", 7, []),
    ("let_asked_by_the_host", "value", 1, ["tick"]),
    ("let_asked_twice", "value", 2, ["tick"]),
    ("let_with_two_bindings_asked_twice", "value", 6, ["tick", "tock"]),
    ("branches_with_the_same_name", "value", 1, None),
    ("let_in_the_condition", "value", 7, None),
    ("let_in_the_environment_argument", "value", 2, None),
    ("binding_stops_on_deferral", "defer", None, []),
    ("binding_stops_on_failure", "fail", "explode raised", []),
    ("binding_stops_on_error", "error", "no definition named 'nope'", None),
    ("result_stops_on_error", "error", "no definition named 'nope'", None),
    ("unused_binding_is_still_computed", "value", 2, ["tick"]),
    ("binding_value_is_a_code_block", "error", "a code block is documentation", None),
    ("let_binds_a_module_answer", "value", 42, None),
    ("let_result_is_a_module_call", "value", 41, None),
    ("let_binds_the_module_itself", "value", 41, None),
    ("two_lets_two_module_calls", "value", 82, None),
    ("let_inside_a_witness", "error", "belongs in a value", None),
    ("let_in_material_is_refused", "error", "belongs in a value", None),
    ("let_as_the_module_environment", "value", 41, None),
]


def _leaf_of(result):
    return reflect_access.leaf(result.ok_value).ok_value


def run(tmp: Path):
    check(len(CASES_TO_RUN) == 50, f"fifty cases: {len(CASES_TO_RUN)}")
    for index, (name, kind, want, calls_wanted) in enumerate(CASES_TO_RUN):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        calls = []
        result = interpret(str(program), environ_for(calls, tmp / f"store-{index}"))
        if kind == "value":
            check(isinstance(result, Ok) and _leaf_of(result) == want,
                  f"{name}: expected {want!r}, got {result!r}")
        elif kind == "ok":
            check(isinstance(result, Ok), f"{name}: expected Ok, got {result!r}")
        elif kind == "len":
            check(isinstance(result, Ok) and len(result.ok_value) == want,
                  f"{name}: expected a tuple of {want}, got {result!r}")
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
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-let-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

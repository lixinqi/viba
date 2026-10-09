"""`if`：条件挑中的那一支才算，另一支的调用求都不求。

`if` 是内建目录里的一份文件（`viba/if.viba`），按名字取，不用 import。它由那两个开关之和搭成
—— `builtin.echo_or_never` 与 `builtin.never_or_echo`（`viba/builtin.viba` 里 `builtin` 的
两个成员）：不成立的那一支答 `never`，和把它剥掉。两个开关都是内建算子，实现在宿主手里，
`branch.py` 的那两个函数就是它们。

用例在 `tests/data/if/`：每份文件自己给出环境与条件，每一支都是"还欠环境的一次调用"（或者是
`builtin.echo` 包好的一个值，或者一条 `sequential` 链 —— viba 没有 `block`，也没有
`lambda: value`，`sequential` 起的是这两件事的作用）。条件挑中的那一支跑在开关给它的孩子环境里
—— 那份调用拿到的层下的 `echo_or_never` 或 `never_or_echo`；走不到的那一支连它那一整块都不跑。

    python3 tests/test_interpreter_if.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, answer_of, is_ok, message_of, stop_tag, stop_text, value_of

from viba import viba_ast
from viba.interpret import (BUILTIN_DIR, Environment, EnvironmentCompute,
                            EnvironmentStorage, interpret, viba_data)
from viba.partial import parameters_of, product_elements, result_of
from viba.type import BUILTIN_CONCEPT, FAILURE_TAG, PROGRAM_ERR_TAG

checks = Checks("interpreter_if")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "if"
ROOT = Path(__file__).resolve().parent.parent
SWITCHES = ("echo_or_never", "never_or_echo")

# (文件, 该跑出什么, 该看住的副作用调用)：
#   ("value", 叶子)            $ok 那一支，叶子是这个值
#   ("never", None)            $ok，答案是 never
#   ("not_implemented", None)  条件挑中的那一步没有实现
#   ("fail", 片段)             条件挑中的那一步崩了，话里含这个片段
#   ("error", 片段)            $viba_program_err，话里含这个片段
CASES_TO_RUN = [
    ("the_condition_holds", "value", 41, ["tick"]),
    ("the_condition_does_not_hold", "value", 7, ["tock"]),
    ("the_condition_is_computed", "value", 41, ["tick"]),
    ("the_untaken_side_is_never_computed", "value", 41, ["tick"]),
    ("the_untaken_side_has_no_implementation", "value", 41, ["tick"]),
    ("a_value_already_worked_out", "value", 9, []),
    ("the_branch_still_owes_arguments", "value", 41, ["add_to"]),
    ("a_side_is_a_block", "value", 41, ["add_to"]),
    ("a_block_of_two_steps", "value", 42, ["add_to", "add_step"]),
    ("a_block_in_the_side_that_is_not_taken", "value", 7, []),
    ("both_sides_take_a_file", "value", 7, []),
    ("the_other_side_takes_another_file", "value", "kept", []),
    ("a_nested_if", "value", 41, ["tick"]),
    ("a_nested_if_untaken", "value", 7, ["tock"]),
    ("the_environment_may_come_last", "value", 41, ["tick"]),
    ("the_same_name_spelled_as_a_builtin", "value", 41, ["tick"]),
    ("the_taken_side_answers_never", "never", None, []),
    ("the_taken_side_has_no_implementation", "not_implemented", None, []),
    ("the_taken_side_raises", "fail", "explode raised", []),
    ("the_environment_is_a_parameter", "error", "already running a call", []),
]

# (文件, 那一支跑在的环境)：开关给它的孩子，挂在这份调用拿到的那个层下。
WHERE_IT_RAN = [
    ("the_condition_holds", ["root/if/echo_or_never"]),
    ("the_condition_does_not_hold", ["root/if/never_or_echo"]),
]


def host_for(calls, ran_at):
    """宿主：两个开关就是 `branch.py` 的那两个函数，别的名字按用例给。

    `calls` 记下哪一支的步骤跑了；`ran_at` 记下它跑在哪个环境里。剩下的名字里
    `kinds.no_implementation` 一个都没有对应，所以它只能是"没有实现"。
    """
    def get_func(module_path, func_name):
        if func_name in SWITCHES:
            return branch.get_func(module_path, func_name)
        if module_path == "builtin" and func_name == "echo":
            return lambda env, x: x      # a value handed out as a function of the environment
        if func_name in ("tick", "tock"):
            def counted(env):
                calls.append(func_name)
                ran_at.append(env.storage.cur_storage_path)
                return 41 if func_name == "tick" else 7
            return counted
        if func_name == "add_to":
            def add_to(env, x):
                calls.append(func_name)
                return x.value + 1
            return add_to
        if func_name == "add_step":
            def add_step(env, x):
                calls.append(func_name)
                return x.value + 1
            return add_step
        if func_name == "explode":
            def explode(env):
                raise ZeroDivisionError("explode raised")
            return explode
        if func_name == "true_gate":
            return lambda env: True
        if func_name == "false_gate":
            return lambda env: False
        if func_name == "answer_int":
            return lambda env: 7
        if func_name == "answer_text":
            return lambda env: "kept"
        if func_name == "answer_never":
            return lambda env: viba_data(viba_ast.Never())
        return None
    return get_func


def environ_for(store, calls, ran_at):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls, ran_at)),
                       viba_path=str(CASES.parent))


def run(tmp: Path):
    for index, (name, kind, want, calls_wanted) in enumerate(CASES_TO_RUN):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        calls, ran_at = [], []
        result = interpret(str(program), environ_for(tmp / f"store-{index}", calls, ran_at))
        if kind == "value":
            check(is_ok(result) and value_of(result) == want,
                  f"{name}: expected {want!r}, got {result!r}")
        elif kind == "never":
            check(is_ok(result) and isinstance(answer_of(result).data, viba_ast.Never),
                  f"{name}: expected never, got {result!r}")
        elif kind == "not_implemented":
            checks.not_implemented(result, name)
            check(stop_text(result, "$full_qualified_func_name") == "kinds.no_implementation",
                  f"{name}: the stop names the step: {result!r}")
        elif kind == "fail":
            checks.failed(result, want, name)
        elif kind == "error":
            check(stop_tag(result) == PROGRAM_ERR_TAG and want in message_of(result),
                  f"{name}: expected a $viba_program_err saying {want!r}, got {result!r}")
        check(calls == calls_wanted,
              f"{name}: expected the side effects {calls_wanted}, got {calls}")

    for index, (name, want) in enumerate(WHERE_IT_RAN):
        calls, ran_at = [], []
        result = interpret(str(CASES / f"{name}.viba"),
                           environ_for(tmp / f"where-{index}", calls, ran_at))
        check(is_ok(result) and ran_at == want,
              f"{name}: the taken side runs under its switch, at {want}: {ran_at}")

    the_library_says_the_same()


def the_library_says_the_same():
    """内建词汇里的两个开关，与 `branch.viba` 里的那两份：同一个签名。

    `viba/if.viba` 按名字取这两个开关（`viba/builtin.viba` 里 `builtin` 的两个成员），
    所以它是内建目录里的一份文件，不用 import；签名一致才说明这份文件搭的正是那两个开关。
    """
    declared = _switches_in(BUILTIN_DIR / "builtin.viba", in_the_concept=True)
    check(set(declared) == set(SWITCHES),
          f"`builtin` 的两个成员是那两个开关：{sorted(declared)}")
    check((BUILTIN_DIR / "if.viba").is_file(),
          "`viba/if.viba` 是内建目录里的那份文件，所以 `if` 不用 import")
    check(declared == _switches_in(ROOT / "branch.viba"),
          f"两处的签名一致：builtin {declared}，branch {_switches_in(ROOT / 'branch.viba')}")


def _switches_in(path: Path, in_the_concept: bool = False):
    """那份文件里两个开关的签名（结果与各参数），按名字。"""
    tree = viba_ast.parse(path.read_text())
    if not in_the_concept:
        return {node.name: _signature(node.body) for node in tree.body
                if getattr(node, "name", None) in SWITCHES}
    concept = next(node for node in tree.body
                   if getattr(node, "name", None) == BUILTIN_CONCEPT)
    return {factor.tag[1:]: _signature(factor.type)
            for factor in product_elements(concept.body)
            if isinstance(factor, viba_ast.Tagged) and factor.tag[1:] in SWITCHES}


def _signature(chain):
    """一个函数链的签名：结果与每个参数，末尾那句说明不算。"""
    return (viba_ast.unparse_type(result_of(chain)),
            [(tag, viba_ast.unparse_type(source)) for tag, source, _env, _slot
             in parameters_of(chain)])


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-if-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

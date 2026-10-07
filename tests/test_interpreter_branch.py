"""分支代数：积选择一支，和把 never 剥掉。

每条用例是一份可以打开的文件（`tests/data/branch/*.viba`），这里只列它该跑出什么。走哪一支由
`branch.echo_or_never` / `never_or_echo` 决定，比较、门槛和那个值由宿主给，所以同一份文件能在不同
输入下走不同的路。

    python3 tests/test_interpreter_branch.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import error_of, message_of, Checks, value_of

import branch
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import Ok
from viba import viba_ast

checks = Checks("interpreter_branch")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "branch"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def host_for(a_value, threshold):
    """宿主：`foo` 给这个用例的值，`threshold_of` 给门槛，别的都照旧。"""
    def get_func(module_path, func_name):
        if func_name == "foo":
            return lambda env: a_value
        if func_name == "ge":
            return lambda env, x, y: x.value >= y.value
        if func_name == "threshold_of":
            return lambda env: threshold
        if module_path == "builtin" and func_name == "echo":
            return lambda env, x: x
        if func_name in ("pick_high", "pick_nonnegative", "pick_negative"):
            return lambda env: func_name[len("pick_"):]
        return branch.get_func(module_path, func_name)
    return get_func


def _run(tmp: Path, name, a_value=0, threshold=0):
    store = tmp / f"store-{name}-{a_value}-{threshold}"
    store.mkdir(parents=True, exist_ok=True)
    env = Environment(EnvironmentStorage("root", None, str(store)),
                      EnvironmentCompute(host_for(a_value, threshold)),
                      str(REPOSITORY_ROOT))
    return interpret(str(CASES / f"{name}.viba"), env)


# ---- 比较、门槛、选择、汇合：同一个文件跑不同的输入 ----
COMPARISON_CASES = [
    (0, -8, 0),
    (0, -1, 0),
    (0, 0, 0),
    (0, 1, 1),
    (0, 7, 7),
    (5, 0, 0),
    (5, 4, 0),
    (5, 5, 5),
    (5, 6, 6),
    (5, 20, 20),
    (10, 1, 0),
    (10, 9, 0),
    (10, 10, 10),
    (10, 11, 11),
    (10, 99, 99),
]

# ---- 选中的那一支里还有一个开关 ----
NESTED_CASES = [
    (25, 'high'),
    (11, 'high'),
    (10, 'high'),
    (9, 'nonnegative'),
    (1, 'nonnegative'),
    (0, 'nonnegative'),
    (-1, 'negative'),
    (-20, 'negative'),
]

# ---- 两个结果、两种分支顺序，每种值的种类一份文件 ----
LITERAL_CASES = [
    ('literal_true_int', 7),
    ('literal_true_int_reversed', 7),
    ('literal_false_int', 8),
    ('literal_false_int_reversed', 8),
    ('literal_true_text', 'left'),
    ('literal_true_text_reversed', 'left'),
    ('literal_false_text', 'right'),
    ('literal_false_text_reversed', 'right'),
    ('literal_true_flag', True),
    ('literal_true_flag_reversed', True),
    ('literal_false_flag', False),
    ('literal_false_flag_reversed', False),
    ('literal_true_nil', None),
    ('literal_true_nil_reversed', None),
    ('literal_false_nil', None),
    ('literal_false_nil_reversed', None),
]

# (文件, 活下来的是哪几支)
GUARD_CASES = [
    ('guards_0', []),
    ('guards_1', [1]),
    ('guards_2', [2]),
    ('guards_3', [1, 2]),
    ('guards_4', [3]),
    ('guards_5', [1, 3]),
    ('guards_6', [2, 3]),
    ('guards_7', [1, 2, 3]),
]

# (文件, 保不保住那个值)
SELECTOR_CASES = [
    ('selector_echo_or_never_true_int', True),
    ('selector_echo_or_never_true_text', True),
    ('selector_echo_or_never_true_flag', True),
    ('selector_echo_or_never_true_nil', True),
    ('selector_echo_or_never_false_int', False),
    ('selector_echo_or_never_false_text', False),
    ('selector_echo_or_never_false_flag', False),
    ('selector_echo_or_never_false_nil', False),
    ('selector_never_or_echo_true_int', False),
    ('selector_never_or_echo_true_text', False),
    ('selector_never_or_echo_true_flag', False),
    ('selector_never_or_echo_true_nil', False),
    ('selector_never_or_echo_false_int', True),
    ('selector_never_or_echo_false_text', True),
    ('selector_never_or_echo_false_flag', True),
    ('selector_never_or_echo_false_nil', True),
]

# 积的单位元：nil / Object / void / None 都不改变那个值
PRODUCT_IDENTITY_CASES = [
    'product_identity_0',
    'product_identity_1',
    'product_identity_2',
    'product_identity_3',
    'product_identity_4',
    'product_identity_5',
    'product_identity_6',
    'product_identity_7',
    'product_identity_8',
    'product_identity_9',
]

# 积的吸收元：never 让整支消失
PRODUCT_ABSORPTION_CASES = [
    'product_absorption_0',
    'product_absorption_1',
    'product_absorption_2',
    'product_absorption_3',
    'product_absorption_4',
]

# 和的单位元：never 与 Oneof 被剥掉
SUM_IDENTITY_CASES = [
    'sum_identity_0',
    'sum_identity_1',
    'sum_identity_2',
    'sum_identity_3',
    'sum_identity_4',
    'sum_identity_5',
]

# 所有分支都是 never：结果是 never
SUM_NEVER_CASES = [
    'sum_never_0',
    'sum_never_1',
    'sum_never_2',
]

# 多个非 never 分支同时活着：结果是和值，不擅自选一支
SUM_LIVE_CASES = [
    ('sum_live_0', 2),
    ('sum_live_1', 2),
    ('sum_live_2', 2),
    ('sum_live_3', 2),
    ('sum_live_4', 3),
]


def run(tmp: Path):
    for index, (threshold, a_value, want) in enumerate(COMPARISON_CASES):
        result = _run(tmp, "comparison_if_else", a_value, threshold)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"if a={a_value} >= {threshold} answers {want}: {result!r}")

    for a_value, want in NESTED_CASES:
        result = _run(tmp, "nested_classification", a_value)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"nested branch classifies {a_value} as {want}: {result!r}")

    for name, want in LITERAL_CASES:
        result = _run(tmp, name)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"{name} answers {want!r}: {result!r}")

    for name, live in GUARD_CASES:
        result = _run(tmp, name)
        if not live:
            valid = (isinstance(result, Ok)
                     and isinstance(result.ok_value.data, viba_ast.Never))
        elif len(live) == 1:
            valid = isinstance(result, Ok) and value_of(result) == live[0]
        else:
            data = result.ok_value.data if isinstance(result, Ok) else None
            valid = isinstance(data, viba_ast.SumChain) and len(data.elements) == len(live)
        check(valid, f"{name} preserves branches {live}: {result!r}")

    for name, keeps in SELECTOR_CASES:
        result = _run(tmp, name)
        data = result.ok_value.data if isinstance(result, Ok) else None
        if keeps:
            # 保住了：拿走的那一支就是那个值（`nil` 也算值，只有 `never` 不算）
            check(isinstance(result, Ok) and not isinstance(data, viba_ast.Never),
                  f"{name} keeps its value: {result!r}")
        else:
            check(isinstance(result, Ok) and isinstance(data, viba_ast.Never),
                  f"{name} eliminates its branch: {result!r}")

    for name in PRODUCT_IDENTITY_CASES:
        result = _run(tmp, name)
        check(isinstance(result, Ok) and value_of(result) == 7,
              f"{name} is 7: {result!r}")

    for name in PRODUCT_ABSORPTION_CASES:
        result = _run(tmp, name)
        check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.Never),
              f"{name} is never: {result!r}")

    for name in SUM_IDENTITY_CASES:
        result = _run(tmp, name)
        check(isinstance(result, Ok) and value_of(result) == 7,
              f"{name} is 7: {result!r}")

    for name in SUM_NEVER_CASES:
        result = _run(tmp, name)
        check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.Never),
              f"{name} is never: {result!r}")

    for name, count in SUM_LIVE_CASES:
        result = _run(tmp, name)
        data = result.ok_value.data if isinstance(result, Ok) else None
        check(isinstance(data, viba_ast.SumChain) and len(data.elements) == count,
              f"{name} keeps {count} live branches: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-branch-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

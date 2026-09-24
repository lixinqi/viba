"""`__args__`：模块的实参是一份积类型，调用时必须给全。

    python3 tests/test_interpreter_module_args.py

每个 case 是一份可以打开的文件（`tests/data/module_args/*.viba`），这里只列它该跑出什么。
`tests/data/module_args/` 里其余的文件不是 case，是被 import 的模块（`square_sum.viba` 就是
用户给的那份）：case 永远是调用方。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, value_of

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.reflect import access as reflect_access
from viba.type import NotMyDutyException, Ok, VibaProgramErr

checks = Checks("interpreter_module_args")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "module_args"


def host_for(calls):
    """宿主：四则、读积的成员、答 7。"""
    def get_func(module_path, func_name):
        if func_name == "add":
            return lambda env, a, b: a.value + b.value
        if func_name == "mul":
            return lambda env, a, b: a.value * b.value
        if func_name == "twice":
            return lambda env, n: n.value * 2
        if func_name == "leaf":
            return lambda env: 7
        if func_name == "sum_of":
            # 整份实参积交到宿主手里，宿主按 tag 读它的两个成员
            return lambda env, pair: (reflect_access.leaf(pair.by_tag("a")).ok_value
                                      + reflect_access.leaf(pair.by_tag("b")).ok_value)
        if func_name == "x_of":
            return lambda env, point: reflect_access.leaf(point.by_tag("x")).ok_value
        return None
    return get_func


def environ_for(calls, store):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls)),
                       viba_path=str(CASES))


# (文件, 该跑出什么)：
#   ("value", 叶子)   Ok，且叶子是这个值
#   ("error", 片段)   VibaProgramErr，话里含这个片段
CASES_TO_RUN = [
    ("calls_positionally", "value", 25, None),
    ("calls_by_tag", "value", 25, None),
    ("too_few", "error", "$b missing", None),
    ("too_many", "error", "takes no more arguments", None),
    ("unknown_tag", "error", "takes no $c argument", None),
    ("tag_twice", "error", "was given $a twice", None),
    ("no_environment", "error", "needs an Environment", None),
    ("bad_args_declared", "error", "__args__ is not a product", None),
    ("asked_without_the_environment", "error", "still waiting for arguments", None),
    ("partial_module_value", "error", "$a, $b missing", None),
    ("args_as_a_value", "value", 7, None),
    ("members_by_tag_and_directly", "value", 7, None),
    ("member_missing", "error", "no member tagged", None),
    ("nested_product_argument", "value", 2, None),
    ("no_args_called", "value", 7, None),
    ("no_args_without_the_slot", "error", "an empty one is written ()", None),
    ("empty_args_called", "value", 7, None),
    ("empty_args_without_the_slot", "error", "an empty one is written ()", None),
    ("module_calls_module_with_args", "value", 14, None),
    ("args_inside_a_binding", "value", 17, None),
]


def run(tmp: Path):
    check(len(CASES_TO_RUN) == 20, f"twenty cases: {len(CASES_TO_RUN)}")
    for index, (name, kind, want, calls_wanted) in enumerate(CASES_TO_RUN):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        calls = []
        result = interpret(str(program), environ_for(calls, tmp / f"store-{index}"))
        if kind == "value":
            check(isinstance(result, Ok) and value_of(result) == want,
                  f"{name}: expected {want!r}, got {result!r}")
        elif kind == "error":
            check(isinstance(result, VibaProgramErr) and want in result.err_msg,
                  f"{name}: expected an error saying {want!r}, got {result!r}")
        if calls_wanted is not None:
            check(calls == calls_wanted,
                  f"{name}: expected the calls {calls_wanted}, got {calls}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-module-args-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

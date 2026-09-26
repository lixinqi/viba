"""一份文件跑不出递归：文件里的定义不许绕回自己。

函数递归执行要"再一次进到同一个定义"，而一份文件自己的定义图是无环的——所以单独给一份文件，
它跑不出递归。跨文件不设这条限制（两份文件可以互相调用），真绕回去也只是报出来，不是崩。

    python3 tests/test_interpreter_recursion.py

每个 case 是一份可以打开的文件（`tests/data/recursion/*.viba`）；`right.viba`、`cycle_right.viba`
这些不是 case，是被 import 的另一份文件。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, value_of

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import Ok, VibaProgramErr

checks = Checks("interpreter_recursion")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "recursion"


def get_func(module_path, func_name):
    if func_name == "leaf":
        return lambda env: 7
    if func_name == "use":
        return lambda env, x: x.value
    if func_name == "ignore":
        # 被标记的函数拿到的是 getter：它不叫，实参就不算
        return lambda get_env, get_x: 7
    return None


def environ_for(store):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(get_func),
                       viba_path=str(CASES))


# (文件, 该跑出什么)：叶子值，或者错误话里的片段。
CASES_TO_RUN = [
    # 一份文件里的定义绕回自己：报出绕的路径，说清这条规矩
    ("self_definition", "A -> A"),
    ("mutual_pair", "A -> B -> A"),
    ("three_cycle", "A -> B -> C -> A"),
    ("cycle_through_a_binding", "A -> B -> A"),
    ("self_call", "A -> A"),
    # 没真跑起来的递归不算：按需的实参没人叫、递归类型只是类型
    ("lazy_self_terminates", 7),
    ("type_recursion_runs", 7),
    ("tagged_type_piece", 7),
    # 跨文件：设计上不限制，跑起来绕回去就报
    ("self_import", "already running"),
    ("two_files_import_each_other", 41),
    ("cycle_left", "cycle_left.x -> cycle_right.y -> cycle_left.x"),
    ("call_cycle_left", "already running"),
]


def run(tmp: Path):
    for index, (name, want) in enumerate(CASES_TO_RUN):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        result = interpret(str(program), environ_for(tmp / f"store-{index}"))
        if isinstance(want, int):
            check(isinstance(result, Ok) and value_of(result) == want,
                  f"{name}: expected {want!r}, got {result!r}")
        else:
            check(isinstance(result, VibaProgramErr) and want in result.err_msg,
                  f"{name}: expected an error saying {want!r}, got {result!r}")

    # 这条规矩本身也写在话里
    result = interpret(str(CASES / "self_definition.viba"), environ_for(tmp / "store-rule"))
    check(isinstance(result, VibaProgramErr)
          and "one file's definitions may not go round" in result.err_msg,
          f"the message states the rule: {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-recursion-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

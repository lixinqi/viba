"""简版 Y 组合子（int -> int）：现有的解释器撑不撑得住。

用例在 `tests/data/y_combinator/`：

    parts.viba             宿主实现的几步：lt / sub / add
    fib_module.viba        F：一步 fib，$fib 是"下一层怎么算"
    int_to_int_y_helper.viba  自应用那一步：helper(f, y, n) = f(y(f, y), n)
    int_to_int_y.viba      Y：把 helper 交给自己
    main.viba              Y F 10，答 55
    main_at_1.viba         Y F 1，答 1（base case 在顶层就走通）
    main_at_5.viba         Y F 5，答 5

三份 `main_*` 是**真正要跑通的**：文件就是函数（`__def__` 进、`__ret__` 出），递归靠
"欠着实参的调用就是值"（Y 的 eta 展开）和"一次调用的身份是它的 storage 路径"。

另外几条是记录，不是目标：一份**要别的文件把实参给它的模块不能当主文件跑**（宿主只给环境，
没人写下 `$f`、`$n`），所以 `int_to_int_y.viba` / `fib_module.viba` 单独跑只会报那句话；
`parts.viba` 只有设计、没有 `__ret__`，本来就不是程序。

    python3 tests/test_interpreter_y_combinator.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, value_of

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import Ok, UnderlyingVibaOpFailed, VibaProgramErr

checks = Checks("interpreter_y_combinator")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "y_combinator"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent



def host_for():
    """每一步的实现：比较、减、加，以及 `builtin.echo`。"""
    def get_func(module_path, func_name):
        if func_name == "builtin.echo":
            return lambda env, x: x
        if func_name == "lt":
            return lambda env, x, y: x.value < y.value
        if func_name == "sub":
            return lambda env, x, y: x.value - y.value
        if func_name == "add":
            return lambda env, x, y: x.value + y.value
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(store):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for()),
                       viba_path=f"{CASES}:{REPOSITORY_ROOT}")


# (文件, 该答什么)
ANSWERS = [
    ("main", 55),
    ("main_at_1", 1),
    ("main_at_5", 5),
]

# (文件, 停在哪种结果, 话里的片段)：记录，不是目标 —— 见上面那段说明
RECORDED = [
    ("int_to_int_y", "error", "has no member tagged '$f'"),
    ("int_to_int_y_helper", "error", "has no member tagged '$f'"),
    ("fib_module", "error", "has no member tagged '$n'"),
    ("parts", "error", "has no __ret__"),
]


def run(tmp: Path):
    for index, (name, want) in enumerate(ANSWERS):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        result = interpret(str(program), environ_for(tmp / f"store-{index}"))
        check(isinstance(result, Ok) and value_of(result) == want,
              f"{name}: Y F that many answers {want}, got {result!r}")

    for index, (name, kind, want) in enumerate(RECORDED):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        result = interpret(str(program), environ_for(tmp / f"record-{index}"))
        if kind == "error":
            check(isinstance(result, VibaProgramErr) and want in result.err_msg,
                  f"{name}: recorded stop {want!r}, got {result!r}")
        elif kind == "failed":
            check(isinstance(result, UnderlyingVibaOpFailed) and want in result.msg,
                  f"{name}: recorded stop {want!r}, got {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-y-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

"""内建目录里的 Y（`viba/Y/`）：20 个递归函数各自跑出该答的值。

用例在 `tests/data/y_functions/`：

    primitives.viba        宿主那几步：lt / eq / add / sub / mul / div / rem / add_f
    steps/<名字>.viba      一步 f：头一个参数（按这份文件的名字叫它，比如 gcd.viba 里是 `$gcd`）
                          是 Y 交回来的"下一层怎么算"，剩下的就是这一层的实参
    main/<名字>.viba       用 Y 跑那一步：`Y[step] << args.env << <实参>`

每个 `main/<名字>.viba` 就是一份可以打开、可以直接跑的程序（文件就是函数）：`__ret__` 里那一次
调用就是"Y 跑这一步"。`steps/<名字>.viba` 是那一步本身 —— 它只管自己这一层，往下几层交给
自己那个参数（`steps/gcd.viba` 里它叫 `$gcd`，就是它自己的名字），所以一份文件里的定义不必
绕回自己。

Y 与 y_helper 住在包的内建目录里（`viba/Y/`、`viba/y_helper/`，各自是一个泛型目录），搜索路径的
最后一站就是那个目录，所以写 `import Y` 就能拿到它，任何 `viba_path` 都不用再写上包的位置。
`import steps.<名字> as step` 的别名与 Y 那一位形参的名字不同，所以这里也顺带钉住"决断读的是
调用方写下的那一份实参，不是形参那个名字"（写的是 `Y[step]`，Y 里那一位叫 `F`）。

`main/gcd_zero.viba` 是从基例那一侧进来的那一份（b 是 0）：`steps/gcd.viba` 把
`rest`、`deeper` 写在分支外面，靠的是定义"读到时才算"，这一份把这个"读到"钉住。

    python3 tests/test_interpreter_y_functions.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, value_of

from viba.interpret import (BUILTIN_DIR, Environment, EnvironmentCompute,
                            EnvironmentStorage, interpret)
from viba.type import Ok

checks = Checks("interpreter_y_functions")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "y_functions"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# (名字, Y 跑这一步该答什么)：实参写在 `main/<名字>.viba` 里，答案和它成对
ANSWERS = [
    ("factorial", 120),          # 5!
    ("fib", 55),                 # 斐波那契第 10 个
    ("triangle", 55),            # 1 加到 10
    ("power", 1024),             # 2 的 10 次方
    ("gcd", 6),                  # 48 和 18 的最大公约数
    ("hanoi", 1023),             # 10 层汉诺塔要搬 2 的 10 次方减 1 步
    ("squares", 385),            # 1 到 10 的平方和
    ("catalan", 42),             # 第 5 个卡特兰数
    ("digit_sum", 35),           # 98765 各位相加
    ("mul_by_adding", 42),       # 7 乘 6，用加法做
    ("is_even", True),           # 10 是偶数（答的是 bool）
    ("factorial_acc", 120),      # 5! 用累加器攒
    ("fib_acc", 55),             # 斐波那契用两个累加器往前推
    ("float_sum", 4.0),          # 4 个 1.0 的和（答的是 float）
    ("reverse_number", 54321),   # 12345 各位倒过来
    ("bin_digits", 10),          # 1000 写成二进制要 10 位
    ("lucas", 123),              # 卢卡斯数第 10 个（一层里套一层开关）
    ("binomial", 252),           # 从 10 个里取 5 个
    ("collatz", 8),              # 6 走到 1 要 8 步
    ("ackermann", 9),            # 阿克曼 (2, 3)
]

# (名字, 该答什么)：从基例那一侧进来。`steps/gcd.viba` 把 `rest`（a 除以 b 的余数）和
# `deeper`（再往下的一层）写在分支外面 —— 定义是读到时才算的，b 是 0 时那一侧一个字都不读，
# 所以这里既不该出现除以 0，也不该再往下调一层。
BASE_ENTRY = [
    ("gcd_zero", 7),             # gcd(7, 0) = 7
]


def host_for():
    """每一步的实现：比较、四则、取余、浮点加，以及 `builtin.echo`。"""
    def get_func(module_path, func_name):
        if func_name == "builtin.echo":
            return lambda env, x: x
        if func_name == "lt":
            return lambda env, x, y: x.value < y.value
        if func_name == "eq":
            return lambda env, x, y: x.value == y.value
        if func_name == "add":
            return lambda env, x, y: x.value + y.value
        if func_name == "sub":
            return lambda env, x, y: x.value - y.value
        if func_name == "mul":
            return lambda env, x, y: x.value * y.value
        if func_name == "div":
            return lambda env, x, y: x.value // y.value
        if func_name == "rem":
            return lambda env, x, y: x.value % y.value
        if func_name == "add_f":
            return lambda env, x, y: x.value + y.value
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(store):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for()),
                       viba_path=f"{CASES}:{REPOSITORY_ROOT}")


def run(tmp: Path):
    # Y 与 y_helper 是包的一部分，不是用例文件：两个泛型目录都在内建目录里
    for name in ("Y", "y_helper"):
        check((BUILTIN_DIR / name / "__generic__.viba").is_file()
              and (BUILTIN_DIR / name / "100.viba").is_file(),
              f"the builtin directory holds the generic {name}")

    for index, (name, want) in enumerate(ANSWERS):
        step = CASES / "steps" / f"{name}.viba"
        program = CASES / "main" / f"{name}.viba"
        check(step.is_file(), f"the step is a file: {step.name}")
        check(program.is_file(), f"the program is a file: {program.name}")
        result = interpret(str(program), environ_for(tmp / f"store-{index}"))
        check(isinstance(result, Ok) and value_of(result) == want,
              f"{name}: Y on that step answers {want!r}, got {result!r}")

    for index, (name, want) in enumerate(BASE_ENTRY):
        program = CASES / "main" / f"{name}.viba"
        check(program.is_file(), f"the program is a file: {program.name}")
        result = interpret(str(program), environ_for(tmp / f"base-{index}"))
        check(isinstance(result, Ok) and value_of(result) == want,
              f"{name}: entered at the base case it answers {want!r} without reading "
              f"the recursion, got {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-y-functions-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

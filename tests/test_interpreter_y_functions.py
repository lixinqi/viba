"""内建目录里的 Y（`viba/Y.viba`）：20 个递归函数各自跑出该给出的值。

用例在 `tests/data/y_functions/`：

    steps/<名字>.viba      一步 f：头一个参数（按这份文件的名字叫它，比如 gcd.viba 里是 `$gcd`）
                          是 Y 交回来的"下一层怎么算"，剩下的就是这一层的实参
    main/<名字>.viba       用 Y 跑那一步：`Y << step << ($sub_env << args.env << "Y") << $a 7 << $b 0`

每一步那几件小事（比较、加减乘除、取余、浮点加）是内建算子：`lt` / `eq` / `add` / `sub` /
`mul` / `div` / `rem` / `add_f`，写在内建目录的 `viba/builtin.viba` 里，任何模块都看得到，
所以这些 step 一个 import 都不用写（`tests/test_interpreter_builtin.py` 是这些算子自己的
那份用例）。宿主拿到的名字是 `builtin.lt` 这样带前缀的那个。

每个 `main/<名字>.viba` 就是一份可以打开、可以直接跑的程序（文件就是函数）：`__impl__` 里那一次
调用就是"Y 跑这一步"。`steps/<名字>.viba` 是那一步本身 —— 它只管自己这一层，往下几层交给
自己那个参数（`steps/gcd.viba` 里它叫 `$gcd`，就是它自己的名字），所以一份文件里的定义不必
绕回自己。往下那几层是一次新的调用，所以头一个参数的类型写的是 `Any`：交回来的那个函数是什么
类型，只有给它的那一方（`y_helper`）说得出来，这一步只知道自己会拿环境与这一层的实参去叫它。

Y 与 y_helper 住在包的内建目录里（`viba/Y.viba` 与 `viba/y_helper.viba`，各自是一个模块），
搜索路径的最后一站就是那个目录，所以写 `import Y` 就能拿到它，任何 `viba_path` 都不用再写上包
的位置。**两个都写 `$env Env`**：给环境就是执行，所以谁调用它们，谁就写下这一层的数据路径 ——
`main/<名字>.viba` 写 `args.env.sub_env << args.env << "Y"`，`steps/<名字>.viba` 往下调时写
`$sub_env << args.env << "low"` 那样的名字 —— 这一层只是名字的出处，`y_helper` 用
`convert_sub_to_sibling` 把它的数据路径压成 workspace 旁边的一个名字（`{workspace 的路径}_{sha1(...)}`），
所以往下多少层，数据路径都只有一个哈希那么长（`viba-interpreter.md` 的「链上的成员」）。这一步写成什么样都不影响：`Y << step` 把 step 本身
当一个值收下 —— 不是泛型，也不按参数个数分文件；这一层的实参接着 `<<` 写在后面
（`Y << step << ($sub_env << args.env << "Y") << $a 7 << $b 0`），`Y` 那个写 `$args ...` 的参数
把它们收成一份积，`apply` 按积里有几个成员选 `apply_impl` 那一支。

`main/gcd_zero.viba` 是从基例那一侧进来的那一份（b 是 0）：`steps/gcd.viba` 把
`rest`、`deeper` 写在分支外面，靠的是定义按需求值（call-by-need，用到才算、只算一次），这一份
把"那一侧不用它们"钉住。

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

# (名字, Y 跑这一步该给出什么)：实参写在 `main/<名字>.viba` 里，结果和它成对
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
    ("is_even", True),           # 10 是偶数（给出的是 bool）
    ("factorial_acc", 120),      # 5! 用累加器攒
    ("fib_acc", 55),             # 斐波那契用两个累加器往前推
    ("float_sum", 4.0),          # 4 个 1.0 的和（给出的是 float）
    ("reverse_number", 54321),   # 12345 各位倒过来
    ("bin_digits", 10),          # 1000 写成二进制要 10 位
    ("lucas", 123),              # 卢卡斯数第 10 个（一层里套一层开关）
    ("binomial", 252),           # 从 10 个里取 5 个
    ("collatz", 8),              # 6 走到 1 要 8 步
    ("ackermann", 9),            # 阿克曼 (2, 3)
]

# (名字, 该给出什么)：从基例那一侧进来。`steps/gcd.viba` 把 `rest`（a 除以 b 的余数）和
# `deeper`（再往下的一层）写在分支外面 —— 定义按需求值（call-by-need），b 是 0 时那一侧一个都
# 用不到，所以这里既不该出现除以 0，也不该再往下调一层。
BASE_ENTRY = [
    ("gcd_zero", 7),             # gcd(7, 0) = 7
]


def host_for():
    """每一步的实现：内建的那几个算子（`builtin.lt` 这样），以及 `builtin.echo`。"""
    def get_func(module_path, func_name):
        if func_name == "builtin.echo":
            return lambda env, x: x
        if func_name == "builtin.lt":
            return lambda env, x, y: x.value < y.value
        if func_name == "builtin.eq":
            return lambda env, x, y: x.value == y.value
        if func_name == "builtin.add":
            return lambda env, x, y: x.value + y.value
        if func_name == "builtin.sub":
            return lambda env, x, y: x.value - y.value
        if func_name == "builtin.mul":
            return lambda env, x, y: x.value * y.value
        if func_name == "builtin.div":
            return lambda env, x, y: x.value // y.value
        if func_name == "builtin.rem":
            return lambda env, x, y: x.value % y.value
        if func_name == "builtin.add_f":
            return lambda env, x, y: x.value + y.value
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(store):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for()),
                       viba_path=f"{CASES}:{REPOSITORY_ROOT}")


def run(tmp: Path):
    # Y and y_helper are part of the package, not case files: both are modules in
    # the builtin directory (`viba/Y.viba`, `viba/y_helper.viba`).
    for name in ("Y", "y_helper"):
        check((BUILTIN_DIR / f"{name}.viba").is_file(),
              f"the builtin directory holds {name}.viba")

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

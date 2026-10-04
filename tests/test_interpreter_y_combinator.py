"""简版 Y 组合子（int -> int）：现有的解释器撑不撑得住。

Y 与 y_helper 是**内建**的：`viba/Y/` 与 `viba/y_helper/` 就在内建词汇那一份旁边，是搜索路径的
最后一站，所以任何模块写 `import ycombinator` 就拿到 `Y`（`viba-interpreter.md`）。用例在
`tests/data/y_combinator/`：

    parts.viba             宿主实现的几步：lt / sub / add
    fib_module.viba        F：一步 fib，`$fib` 是"下一层怎么算"（一个参数）
    add2_module.viba       另一份 F：`f(n, m)`（两个参数）
    add3_module.viba       再一份 F：`f(n, m, k)`（三个参数）
    main.viba              Y F 10，答 55
    main_at_1.viba         Y F 1，答 1（base case 在顶层就走通）
    main_at_5.viba         Y F 5，答 5
    main_in_a_sub_env.viba Y F 10，调用方自己写一层子环境，答 55
    main_two_args.viba     Y F 3 4，答 7（`F` 两个参数 → y_helper/200.viba）
    main_three_args.viba   Y F 3 4 5，答 12（`F` 三个参数 → y_helper/300.viba）
    helper_by_hand.viba    不用 Y，直接把 helper 用起来，答 55

内建的那两份（`viba/y_impl/100.viba`、`viba/y_helper/` 下 1 到 16 每个长度一份）也在这里一起看。

`main_*` 与 `helper_by_hand` 是**真正要跑通的**：文件就是函数（`__decl__` 进、`__impl__` 出），
递归靠"欠着实参的调用就是值"（Y 的 eta 展开）和"一次调用的身份是它的 storage 路径"。
`ycombinator.Y[F]` 只用一个参数（`F` 本身）：`F` 的 `__decl__` 怎么写、helper 那一位怎么接，都由
`y_helper` 的 `pattern` 行从实参里读出来 —— 所以参数多的 `F` 落到另一份文件上。`y_helper`
里 1 到 16 每个长度一个文件，那 16 个长度在最后那一遍里**各跑一次**：step 模块由这一遍当场
写出来（那里只有参数个数要紧），跑通了才说明每个长度都接得上。

另外几条是记录，不是目标：一份**要别的文件把实参给它的模块不能当主文件跑**（宿主只给环境，
没人写下 `$f`、`$n`），所以 `fib_module.viba` / `add2_module.viba` / `add3_module.viba` 与
`viba/y_helper/` 下那些 pattern 文件单独跑只会报那句话；`parts.viba` 与 `viba/y_impl/100.viba`
只有设计、没有 `__impl__`，本来就不是程序。

    python3 tests/test_interpreter_y_combinator.py
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
from viba.type import Ok, UnderlyingVibaOpFailed, VibaProgramErr

checks = Checks("interpreter_y_combinator")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "y_combinator"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# 一个长度一个文件：第 N 份文件的签名是 3 + N 个位置
LENGTHS = list(range(1, 17))


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


def environ_for(store, also=()):
    """This host: files come from the case directory, the repository (for
    `branch.viba`), and anything else a check wrote for itself."""
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for()),
                       viba_path=":".join([*(str(path) for path in also),
                                           str(CASES), str(REPOSITORY_ROOT)]))


# (文件, 该答什么)
ANSWERS = [
    ("main", 55),
    ("main_at_1", 1),
    ("main_at_5", 5),
    ("main_in_a_sub_env", 55),
    ("main_two_args", 7),
    ("main_three_args", 12),
    ("helper_by_hand", 55),
]

# (文件, 停在哪种结果, 话里的片段)：记录，不是目标 —— 见上面那段说明
RECORDED = [
    (BUILTIN_DIR / "y_impl" / "100.viba", "error", "has no __impl__"),
    (BUILTIN_DIR / "y_helper" / "100.viba", "error", "has no member tagged '$f'"),
    (BUILTIN_DIR / "y_helper" / "1600.viba", "error", "has no member tagged '$f'"),
    (CASES / "fib_module.viba", "error", "has no member tagged '$n'"),
    (CASES / "add2_module.viba", "error", "has no member tagged '$n'"),
    (CASES / "add3_module.viba", "error", "has no member tagged '$n'"),
    (CASES / "parts.viba", "error", "has no __impl__"),
]


def _a_step_of(length: int) -> str:
    """A step function of that many parameters, written here.

    Only the parameter count matters to this check: the length of the signature
    is what decides which `y_helper` file answers, and the answer is the first
    parameter counted down to zero — so the run says which length it started
    from.
    """
    names = [f"n{index}" for index in range(length)]
    return "\n".join([
        f"# written by the test: a step of {length} parameters",
        "__decl__ =",
        "    Any",
        "  <- $env Env",
        f"  <- $self ({' <- '.join(['int'] * (length + 1))})",
    ] + [f"  <- ${name} int" for name in names] + [
        "",
        "args = __get_args__ << __decl__",
        "",
        "import branch",
        "import parts as parts",
        "",
        "small = parts.lt << $env args.env << $x args.n0 << $y 1",
        "below = parts.sub << $env args.env << $x args.n0 << $y 1",
        "low = " + " << ".join(['args.self', '($sub_env << args.env << "low")',
                                 '$n0 below']
                                + [f"${name} args.{name}" for name in names[1:]]),
        "__impl__ =",
        "  Oneof",
        "  | (branch.echo_or_never << $env args.env << $cond small",
        "      << $get_v (builtin.echo << $x 0))",
        "  | (branch.never_or_echo << $env args.env << $cond small",
        "      << $get_v (parts.add << $env args.env << $x low << $y 1))",
    ]) + "\n"


def _a_main_of(length: int) -> str:
    """`Y F <length> 0 … 0`: the step counts the first parameter down to zero."""
    given = " << ".join([f"$n0 {length}"]
                        + [f"$n{index} 0" for index in range(1, length)])
    return "\n".join([
        "__decl__ = int <- $env Env",
        "args = __get_args__ << __decl__",
        "import step as F",
        "import ycombinator",
        f"__impl__ = ycombinator.Y[F] << args.env << {given}",
    ]) + "\n"


def _every_length_runs(tmp: Path):
    """1 到 16 每个长度都跑一次：那 16 份模式文件没有一份是没人走过的。

    `y_helper` 一个长度一个文件，落到哪一份上看 `F` 的参数列表有多长；这一遍为每个长度当场
    写一份 step 模块（只有参数个数要紧），跑出来的答案就是那个长度。
    """
    orders = sorted(int(path.stem) for path in (BUILTIN_DIR / "y_helper").glob("*.viba")
                    if path.stem != "__generic__")
    check(orders == [100 * length for length in LENGTHS],
          f"one file per length, {LENGTHS[0]} through {LENGTHS[-1]}: {orders}")
    for length in LENGTHS:
        where = tmp / f"length-{length}"
        where.mkdir()
        (where / "step.viba").write_text(_a_step_of(length))
        (where / "main.viba").write_text(_a_main_of(length))
        result = interpret(str(where / "main.viba"),
                           environ_for(where / "store", also=[where]))
        check(isinstance(result, Ok) and value_of(result) == length,
              f"a step of {length} parameters lands on y_helper/{length * 100}.viba "
              f"and runs, got {result!r}")


def run(tmp: Path):
    _every_length_runs(tmp)

    for index, (name, want) in enumerate(ANSWERS):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        result = interpret(str(program), environ_for(tmp / f"store-{index}"))
        check(isinstance(result, Ok) and value_of(result) == want,
              f"{name}: Y F that many answers {want}, got {result!r}")

    for index, (program, kind, want) in enumerate(RECORDED):
        check(program.is_file(), f"the case is a file: {program.name}")
        result = interpret(str(program), environ_for(tmp / f"record-{index}"))
        label = program.name
        if kind == "error":
            check(isinstance(result, VibaProgramErr) and want in result.err_msg,
                  f"{label}: recorded stop {want!r}, got {result!r}")
        elif kind == "failed":
            check(isinstance(result, UnderlyingVibaOpFailed) and want in result.msg,
                  f"{label}: recorded stop {want!r}, got {result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-y-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

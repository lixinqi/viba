"""内建目录里的 sequential：一串步骤严格按书写次序跑完，最后那一步的答案就是结果。

    sequential << $x (…) << $y (…) << (…) << env

每一步都是一次调用，写在 `<<` 后面；除最后那一步之外，每一步前面都要写一个 tag（`$x (…)`），
它的答案记在那个 tag 名下，后面的步骤用 `($var "x")` 读得到。**最后那个参数不写 tag**：它没有名字，
它的答案就是整条链的答案 —— 想返回前面某一步的值，就把那个名字读出来交给内建的 `echo`
（`echo << $x V` 原样答 V）：`<< (echo << $x ($var "x"))`。步骤的实参里也可以写变量引用
（`$a ($var "x")`），它们在调用跑起来之前先换成值。

`sequential_impl` 按步骤数分文件（2..64 步各一份；一步没有 tag 钉住个数，按写下来的调用实参个数
各一份），`sequential_step` 按一步那次调用的实参个数分文件（1..16），`sequential_arg`
按实参的写法分文件。每一个调用都跑在自己的地址上（`step0`、`step1`、…，最后一步是 `last`，
实参是 `arg0`、`arg1`、…）：一条 storage 路径只答一次调用。

文件名前面写着这个文件读几份（`sequential_impl/2_200.viba`、`sequential_step/2_300.viba`）：
决断先数一遍这次应用摆出来几份，只读那个份数对得上的文件，所以一条链只读它自己的那几份，
79 份模式文件不必翻一遍（`viba-pattern.md` 第 4 节）。

用例在 `tests/data/sequential/`：

    one_step.viba                 一步不写 tag，答 3
    two_steps.viba                两步：x 给最后一步读两次，答 9
    three_steps.viba              三步一条链，答 8
    a_var_step.viba               最后一步用 `echo` 把 `x` 交出去，答 3
    a_far_variable.viba           四步：最后一步读第一步的变量，答 6（隔层的成员也取得出来）
    the_chain_is_a_closure.viba   链先不给环境（是个闭包），`proc << args.env` 才跑，答 3
    order.viba                    三步之间没有依赖，宿主被问的次序仍是书写次序
    builtin_prefix.viba           `builtin.sequential` 叫的是同一个模块，答 3
    the_environment_first.viba    环境给在第一位也认，答 3
    a_module_in_a_step.viba       步骤是一次模块调用（helper 跑在步骤自己的那一层上），答 7
    a_missing_variable.viba       读一个没人写过的名字
    seventeen_steps.viba          17 步一条链（超过 16 也接），答 17
    sixty_four_steps.viba         64 步（计数文件的上界），答 64
    a_tagged_last_step.viba       最后一个参数写了 tag：没有名字可记，没人接
    a_step_that_is_no_call.viba   最后那个参数不是一次调用
    typed_slots.viba              变量引用落在 `$a int` 上：设计时读不出它的值，当场拒
    sixty_five_steps.viba         65 步：2..64 各一份计数文件，没有文件接

除这些用例之外，`sequential_impl` 的 **1..64 步槽位**各跑一遍：各造一条链（每一步把前一步的
答案加一，最后一步不写 tag），答步数本身。这些链在跑的时候写进临时目录，不在 `tests/data/` 里。

    python3 tests/test_interpreter_sequential.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, value_of

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret)
from viba.type import Ok, VibaProgramErr

checks = Checks("interpreter_sequential")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "sequential"

# (用例文件, 该答多少)
ANSWERS = [("one_step", 3), ("two_steps", 9), ("three_steps", 8),
           ("a_var_step", 3), ("a_far_variable", 6), ("order", 0),
           ("seventeen_steps", 17), ("sixty_four_steps", 64),
           ("the_chain_is_a_closure", 3), ("builtin_prefix", 3),
           ("the_environment_first", 3), ("a_module_in_a_step", 7)]

# 严格次序的证据：`order.viba` 的三步互不依赖，宿主被问的次序仍是书写次序。
ORDER = ["first", "second", "third"]

# `sequential_impl` 的步数槽位：1..64 各跑一条，链长就是槽位。每一步都要读一遍自己那份模式文件，
# 一份一代价不到一毫秒（文件名上写着份数，决断不必翻别的文件）。
SLOTS = 64

# (用例文件, 话里的片段)：设计层与决定层的拒绝
RECORDED = [("a_missing_variable", "no member tagged '$nope'"),
            ("a_tagged_last_step", "no pattern of 'sequential_impl' matches"),
            ("a_step_that_is_no_call", "no pattern of 'sequential_step' matches"),
            ("typed_slots", 'does not fit $a int'),
            ("sixty_five_steps", "no pattern of 'sequential_impl' matches")]


def host_for(record):
    """宿主：算 add / mul，原样答回实参的 echo，以及记一笔的 note。"""
    def get_func(module_path, func_name):
        if func_name in ("add", "builtin.add"):
            return lambda env, a, b: a.value + b.value
        if func_name == "mul":
            return lambda env, a, b: a.value * b.value
        if func_name in ("echo", "builtin.echo"):
            return lambda env, x: x
        if func_name == "note":
            def note(env, v):
                record.append(v.value)
                return 0
            return note
        return None
    return get_func


def environ_for(store, record):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(record)),
                       viba_path=str(CASES))


def run(tmp: Path):
    _the_steps_in_order(tmp)
    _every_slot(tmp)
    _errors_are_named(tmp)


def _the_steps_in_order(tmp: Path):
    """写下来的步骤一步步跑完：每个用例自己一个 store。"""
    for name, want in ANSWERS:
        record = []
        result = interpret(str(CASES / f"{name}.viba"),
                           environ_for(tmp / name, record))
        check(isinstance(result, Ok) and value_of(result) == want,
              f"{name} 答 {want}：{result!r}")
    record = []
    order = interpret(str(CASES / "order.viba"), environ_for(tmp / "order", record))
    check(isinstance(order, Ok) and record == ORDER,
          f"三步没有依赖，宿主被问的次序是书写次序 {ORDER}：{record!r}")


def _a_chain_of(n: int) -> str:
    """n 步的链：每一步把前一步的答案加一，最后一步不写 tag，答 n。"""
    lines = ["add =",
             "    int",
             "  <- $env Env",
             "  <- $x Any",
             "  <- $y Any",
             "  <- { 两个整数相加 }",
             "",
             "__decl__ =",
             "    int",
             "  <- $env Env",
             "",
             "args = __get_args__ << __decl__",
             "",
             "__impl__ =",
             "    sequential"]
    if n == 1:
        lines.append("  << (add << $x 0 << $y 1)")
    else:
        lines.append("  << $s0 (add << $x 0 << $y 1)")
        for i in range(1, n - 1):
            lines.append(f'  << $s{i} (add << $x ($var "s{i - 1}") << $y 1)')
        lines.append(f'  << (add << $x ($var "s{n - 2}") << $y 1)')
    lines.append("  << args.env")
    return "\n".join(lines) + "\n"


def _every_slot(tmp: Path):
    """`sequential_impl` 的 1..64 步槽位各跑一遍：各造一条链，答步数本身。

    造出来的链写进临时目录，不进 `tests/data/`。每一步都是一次调用（除最后一步外都带 tag），
    所以这一轮同时压到一步那些文件（一步的调用有几个实参就读几份的那几份）与 2..64 步各一份。
    """
    cases = tmp / "slots"
    cases.mkdir(parents=True, exist_ok=True)
    for n in range(1, SLOTS + 1):
        path = cases / f"{n}_steps.viba"
        path.write_text(_a_chain_of(n))
        result = interpret(str(path), environ_for(tmp / f"slot-{n}", []))
        check(isinstance(result, Ok) and value_of(result) == n,
              f"{n} 步的链答 {n}：{result!r}")


def _errors_are_named(tmp: Path):
    """写错了的那几种：话里点名是哪一处。"""
    for name, want in RECORDED:
        result = interpret(str(CASES / f"{name}.viba"),
                           environ_for(tmp / f"error-{name}", []))
        check(isinstance(result, VibaProgramErr) and want in result.err_msg,
              f"{name}: recorded stop {want!r}，{result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-sequential-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

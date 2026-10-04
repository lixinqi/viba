"""apply：给一个函数与一份实参积，答出那个函数答的东西。

    python3 tests/test_interpreter_apply.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host, value_of

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import Ok

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "tests" / "data" / "apply"

checks = Checks("interpreter_apply")
check = checks.check

# (用例文件, 该答多少)
# `run_first` 那一份把环境给在第一位：给在哪一位都一样；`direct`、`apply_once`、
# `apply_twice`、`apply_thrice` 是一个函数被 `apply` 套 0 层到 3 层，环境一层层往后缀；
# `by_tag` 的积故意写成 ($b 2 * $a 1)，答 12 才说明成员是按 tag 落位的（按位置会给 21）。
ANSWERS = [("run", 3), ("run_first", 3), ("run_three", 6),
           ("direct", 3), ("apply_once", 3), ("apply_twice", 3), ("apply_thrice", 3),
           ("by_tag", 12), ("capture", 3)]


def environ_for(store):
    def get_func(module_path, func_name):
        if func_name == "add":
            return lambda env, a, b: a.value + b.value
        if func_name == "add3":
            return lambda env, a, b, c: a.value + b.value + c.value
        if func_name == "arrange":
            # 两个实参分得出先后：按位置的放置会把 12 答成 21。
            return lambda env, a, b: a.value * 10 + b.value
        return Host().get_func(module_path, func_name)
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(get_func), viba_path=str(CASES.parent))


def run(tmp: Path):
    for name, want in ANSWERS:
        result = interpret(str(CASES / f"{name}.viba"), environ_for(tmp))
        check(isinstance(result, Ok) and value_of(result) == want,
              f"{name} 答 {want}：{result!r}")
    impl_files()


def impl_files():
    """`apply_impl` 的每一支：文件号就是实参个数，`pattern` 认几个成员，就给函数几个实参。

    这些文件是同一份东西按个数摆开的，没有 4 个以上实参的用例会选到它们，所以这里
    直接数一遍：文件号、`pattern` 的成员数、`$__getattr__` 的个数三者一致。
    """
    for count in range(1, 17):
        text = (ROOT / "viba" / "apply_impl" / f"{count * 100}.viba").read_text()
        members = text.count("tagged[")
        given = text.count("$__getattr__")
        check(members == count and given == count,
              f"apply_impl/{count * 100}.viba: {count} 个实参，"
              f"pattern 了 {members} 个成员，给了 {given} 个实参")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-apply-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

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

CASES = Path(__file__).resolve().parent / "data" / "apply"

checks = Checks("interpreter_apply")
check = checks.check

# (用例文件, 该答多少)
ANSWERS = [("run", 3), ("run_three", 6)]


def environ_for(store):
    def get_func(module_path, func_name):
        if func_name == "add":
            return lambda env, a, b: a.value + b.value
        if func_name == "add3":
            return lambda env, a, b, c: a.value + b.value + c.value
        return Host().get_func(module_path, func_name)
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(get_func), viba_path=str(CASES.parent))


def run(tmp: Path):
    for name, want in ANSWERS:
        result = interpret(str(CASES / f"{name}.viba"), environ_for(tmp))
        check(isinstance(result, Ok) and value_of(result) == want,
              f"{name} 答 {want}：{result!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-apply-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

"""点分名字的定义：`a.b = A` 就是 `a = $b A`。

`a.b = A` 与 `a.c = C` 一起就是 `a = $b A * $c C`，按书写顺序合成；每一级前缀都是一个概念，
叶子挂在最后一段上，所以 `a.b.c = T` 是 `a = $b ($c T)`。短的赢：写了 `a = …`，那一棵子树不再
展开；写了 `a.b = …` 又写了 `a.b.c = …`，`a.b` 那一份赢。

    python3 tests/test_dotted_definitions.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host, value_of

from viba import viba_ast
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import AstNodeType, Ok, VibaProgramErr, custom_module
from viba.is_sub_type import is_sub_type

CASES = Path(__file__).resolve().parent / "data" / "dotted"

checks = Checks("dotted_definitions")
check = checks.check
labelled = checks.labelled


def one_line(text: str) -> str:
    return " ".join(text.split())


def entry(text: str, index: int = -1):
    tree = viba_ast.parse(text)
    defs = [n for n in tree.body if not isinstance(n, viba_ast.Import)]
    node = defs[index]
    body = node.body if isinstance(node, viba_ast.TypeDefinition) else node
    return AstNodeType(body, custom_module(text))


def run(tmp: Path):
    _written_out()
    _read_as_a_type()
    _read_as_a_step(tmp)
    _the_later_definition_wins()
    _refused()


def _written_out():
    """展开成什么：一个叶子、合并、嵌套、短的那个赢。"""
    table = [
        ("a.b = int", "a = $b int"),
        ("a.b = A\na.c = C", "a = $b A * $c C"),
        ("a.c = C\na.b = A", "a = $c C * $b A"),
        ("a.b.c = T", "a = $b($c T)"),
        ("a = X\na.b = Y", "a = X"),
        ("a.b = X\na.b.c = Y", "a = $b X"),
        ("a.b = X\na.b = Y", "a = $b Y"),
    ]
    for source, want in table:
        got = one_line(viba_ast.unparse(viba_ast.parse(source)))
        check(got == want, f"{source!r} 写成 {want!r}，得到 {got!r}")


def _read_as_a_type():
    """父概念与叶子都能当类型读：`a.b` 是 `$b` 的声明类型，`a` 是那个积。"""
    group = "a.b = int\na.c = str\n"
    check(is_sub_type(entry(group + "X = a.b\n"), entry("X = int\n")),
          "叶子的名字读出来是它的声明类型")
    check(is_sub_type(entry(group + "X = int\n"), entry("X = a.b\n")),
          "反过来也成立")
    check(is_sub_type(entry(group + "X = a\n"), entry("X = $b int * $c str\n")),
          "父概念就是孩子们组成的那个积")
    check(is_sub_type(entry(group + "X = $b int * $c str\n"), entry("X = a\n")),
          "反过来也成立")


def _read_as_a_step(tmp: Path):
    """叶子当一步调用：名字仍然是写下来的整串，宿主按整串找实现。"""
    def get_func(path, func_name):
        if func_name == "demo.math.add":
            return lambda env, a, b: a.value + b.value
        if func_name == "demo.math.mul":
            return lambda env, a, b: a.value * b.value
        return Host().get_func(path, func_name)

    env = Environment(EnvironmentStorage("root", None, None), EnvironmentCompute(get_func))
    result = interpret(str(CASES / "apis.viba"), env)
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"调的是 demo.math.add，宿主按整串找到实现：{result!r}")

    # 嵌套的那种也能读：`a.b.c` 是 `$c` 的声明类型
    check(is_sub_type(entry("a.b.c = int\nX = a.b.c\n"), entry("X = int\n")),
          f"a.b.c 读出来是 int")


def _the_later_definition_wins():
    """不带点的重名也以后一个为准：解析、判定、描述符三处读的是同一份。"""
    twice = "A = int\nA = str\n"
    check(is_sub_type(entry(twice + "X = A\n"), entry("X = str\n")),
          "重名定义以后一个为准")
    earlier = is_sub_type(entry(twice + "X = A\n"), entry("X = int\n"))
    check(isinstance(earlier, Ok) and earlier.ok_value is False,
          f"先写的那一份不再算数：{earlier!r}")


def _refused():
    """后写的覆盖先写的；带形参的点分名编不过。"""
    twice = "a.b = int\na.b = str\n"
    check(is_sub_type(entry(twice + "X = a.b\n"), entry("X = str\n")),
          "同一个叶子写两次，以后一个为准")
    earlier = is_sub_type(entry(twice + "X = a.b\n"), entry("X = int\n"))
    check(isinstance(earlier, Ok) and earlier.ok_value is False,
          f"先写的那一份不再算数：{earlier!r}")

    try:
        viba_ast.parse("a.f[T] = T\n")
        check(False, "带形参的点分名字编不过")
    except SyntaxError as exc:
        check("generic" in str(exc) or "形参" in str(exc),
              f"带形参的点分名字编不过：{exc}")

    gone = "a = X\na.b = Y\nX = int\n"
    check(isinstance(is_sub_type(entry(gone + "Y = a.b\n"), entry("Y = int\n")),
                     VibaProgramErr),
          "显式写了 a = X 之后，a.b 不再存在")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-dotted-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

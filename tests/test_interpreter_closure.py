"""闭包：没给环境的调用就是一个值，可序列化、可存、可传。

环境是执行那一步，也只有它是：给了环境就是执行（实参必须齐），没给环境就是闭包——写下来的函数名
加上已经算好的实参。这份材料里没有环境，所以它跨得过运行边界。

    python3 tests/test_interpreter_closure.py

每个 case 是一份可以打开的文件（`tests/data/closure/*.viba`）；`square_sum.viba` 是被 import 的
那个模块，不是 case。
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, value_of

from viba.reflect import access as reflect_access

from viba import serialize, viba_ast
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import Ok, VibaProgramErr

checks = Checks("interpreter_closure")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "closure"


def host_for(calls):
    """宿主：四则、读积的那个成员、答 7、把拿到的闭包原样交回来。"""
    def get_func(module_path, func_name):
        if func_name == "add":
            return lambda env, a, b: a.value + b.value
        if func_name == "mul":
            return lambda env, a, b: a.value * b.value
        if func_name == "inc":
            return lambda env, x: x.value + 1
        if func_name == "leaf":
            return lambda env: 7
        if func_name == "x_of":
            return lambda env, point: reflect_access.leaf(point.by_tag("x")).ok_value
        if func_name == "ge":
            return lambda env, x, y: x.value >= y.value
        if func_name == "show":
            return lambda env, f: f          # 宿主只把它当数据，原样递回来
        if func_name == "tick":
            def counted(env):
                calls.append("tick")
                return len(calls)
            return counted
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(store, calls=()):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls)),
                       viba_path=str(CASES.parents[2]))


# (文件, 该跑出什么)：
#   ("value", 叶子)     Ok，且叶子是这个值
#   ("closure", 写法)   Ok，答的是一个闭包，写出来是这个样子
#   ("error", 片段)     VibaProgramErr，话里含这个片段
#   ("material", None)  Ok，是一个闭包装在材料里的值（后面单独看）
# 最后一列是要看住的副作用调用；None 表示不看。
CASES_TO_RUN = [
    # 闭包是什么、能拿它做什么
    ("closure_from_definition", "value", 42, None),
    ("closure_two_stages", "value", 3, None),
    ("closure_tag_order", "value", 3, None),
    ("closure_reused_twice", "value", 83, None),
    ("closure_aliased", "value", 3, None),
    ("stored_then_run", "value", 42, None),
    ("closure_as_answer", "closure", "add << $a 1 << $b 2", None),
    ("closure_argument_computed_once", "value", 3, ["tick"]),
    # 闭包里装的东西：别的调用、积、另一个闭包
    ("closure_of_a_member_function", "value", 42, None),
    ("closure_with_module_call_argument", "value", 29, None),
    ("closure_with_product_argument", "value", 2, None),
    ("closure_with_a_closure_argument", "closure", "add << $a 40", None),
    ("closure_in_material", "material", None, None),
    # 模块闭包：同型，四种给法
    ("module_closure", "closure", "square_sum << $a 3 << $b 4", None),
    # 按需那一格留着不给，别的先给：这正是标记挪到实参上换来的
    ("partial_across_a_by_need_slot", "value", 1, ["tick"]),
    ("module_closure_by_tag", "value", 25, None),
    ("module_closure_bare_name", "value", 25, None),
    ("module_closure_empty_product", "value", 7, None),
    # 两种不许存下来的
    ("half_with_environment", "error", "was given 2 of its 3 arguments", None),
    ("closure_holding_environment", "error", "a closure holds material only", None),
    ("by_need_argument_is_not_stored", "error",
     "computed only when it is wanted, so it cannot be stored", None),
]


def written(value) -> str:
    """One piece of material as written, layout flattened away."""
    answer = serialize.serialize("piece", value)
    if not isinstance(answer, Ok):
        return repr(value)
    return " ".join(answer.ok_value.split("=", 1)[1].split())


def one_line(node) -> str:
    """The same, straight off the syntax tree."""
    return " ".join(viba_ast.unparse_type(node).split())


def run(tmp: Path):
    check(len(CASES_TO_RUN) == 21, f"twenty-one cases: {len(CASES_TO_RUN)}")
    for index, (name, kind, want, calls_wanted) in enumerate(CASES_TO_RUN):
        program = CASES / f"{name}.viba"
        check(program.is_file(), f"the case is a file: {program.name}")
        calls = []
        result = interpret(str(program), environ_for(tmp / f"store-{index}", calls))
        if kind == "value":
            check(isinstance(result, Ok) and value_of(result) == want,
                  f"{name}: expected {want!r}, got {result!r}")
        elif kind == "material":
            check(isinstance(result, Ok),
                  f"{name}: expected the material, got {result!r}")
        elif kind == "closure":
            check(isinstance(result, Ok) and isinstance(result.ok_value.data, viba_ast.Partial),
                  f"{name}: expected a closure, got {result!r}")
            if isinstance(result, Ok):
                check(written(result.ok_value) == want,
                      f"{name}: the closure is written {want!r}, got {written(result.ok_value)!r}")
        elif kind == "error":
            check(isinstance(result, VibaProgramErr) and want in result.err_msg,
                  f"{name}: expected an error saying {want!r}, got {result!r}")
        if calls_wanted is not None:
            check(calls == calls_wanted,
                  f"{name}: expected the calls {calls_wanted}, got {calls}")

    # 存下来的闭包：写出来就是那条链，读回来还是同一个闭包
    source = (CASES / "closure_as_answer.viba").read_text()
    body = viba_ast.parse(source).body[-2].body
    stored = one_line(body)
    check(stored == "add << $a 1 << $b 2",
          f"a stored closure is written back as its chain: {stored!r}")
    again = viba_ast.parse(f"again = {stored}\n").body[0].body
    check(one_line(again) == stored,
          f"and reading it back gives the same closure: {one_line(again)!r}")

    # 材料里装一个闭包：装的是数据，不会被执行（那一格是 `$f`，不是一次调用）
    material = interpret(str(CASES / "closure_in_material.viba"), environ_for(tmp / "store-mat"))
    check(isinstance(material, Ok), f"a closure inside material stays material: {material!r}")
    if isinstance(material, Ok):
        inner = material.ok_value.by_tag("f")
        check(isinstance(inner.data, viba_ast.Partial),
              f"and what is inside is the written call: {inner.data!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-closure-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

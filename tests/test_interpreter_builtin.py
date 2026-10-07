"""内建目录里的算子（`viba/builtin.viba` 的 `builtin`）：一个算子一份用例，两种写法都行。

用例在 `tests/data/builtin/`：

    <算子>.viba                 一个算子一份：`add.viba` 跑 `add`，该给出什么写在下面这张表里
    both_spellings.viba         `add` 与 `builtin.add` 是同一次调用
    shadowed_by_the_module.viba 模块自己写一个 `add`，它赢过内建那个
    runs_where_it_is_given.viba 内建算子跑在给它的那个环境里
    the_wrong_type.viba         参数类型不对，拒绝

`OPERATORS` 就是这张表：每个算子在宿主那一步怎么算，以及它的用例该给出什么。实参写死在用例
文件里，结果写在这里 —— 一个算子一份，所以每个算子的结果都单独被钉住。一个算子的两个方向都
要钉住时（`lt` 那样），用例给出的是一个串：`"truefalse"`。

宿主只实现内建算子：`get_func` 拿到的名字是 `builtin.<算子>`，别的名字一律不认（那份自己写了
`add` 的用例除外，那里多一个本模块的 `add`）。

    python3 tests/test_interpreter_builtin.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import module_of, Checks, is_ok, value_of

from viba import viba_ast
from viba.interpret import (BUILTIN_DIR, Environment, EnvironmentCompute,
                            EnvironmentStorage, interpret)
from viba.partial import product_elements
from viba.reflect import VObject
from viba.type import BUILTIN_CONCEPT

checks = Checks("interpreter_builtin")
check = checks.check
labelled = checks.labelled

CASES = Path(__file__).resolve().parent / "data" / "builtin"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# 每个算子：怎么算它，用例该给出什么。次序与 `viba/builtin.viba` 里写的次序一致。
OPERATORS = {
    "add": (lambda x, y: x + y, 7),
    "sub": (lambda x, y: x - y, 7),
    "mul": (lambda x, y: x * y, 42),
    "div": (lambda x, y: x // y, 3),
    "rem": (lambda x, y: x % y, 2),
    "neg": (lambda x: -x, -4),
    "abs": (lambda x: abs(x), 4),
    "min": (lambda x, y: min(x, y), 3),
    "max": (lambda x, y: max(x, y), 4),
    "pow": (lambda x, y: x ** y, 1024),

    "add_f": (lambda x, y: x + y, 4.0),
    "sub_f": (lambda x, y: x - y, 5.0),
    "mul_f": (lambda x, y: x * y, 10.0),
    "div_f": (lambda x, y: x / y, 1.25),
    "rem_f": (lambda x, y: x % y, 1.5),
    "neg_f": (lambda x: -x, -1.5),
    "abs_f": (lambda x: abs(x), 1.5),
    "min_f": (lambda x, y: min(x, y), 1.5),
    "max_f": (lambda x, y: max(x, y), 2.5),
    "pow_f": (lambda x, y: x ** y, 8.0),

    # 比较：两个方向都写在一个用例里，给出的是 "truefalse"
    "lt": (lambda x, y: x < y, "truefalse"),
    "le": (lambda x, y: x <= y, "truefalse"),
    "gt": (lambda x, y: x > y, "truefalse"),
    "ge": (lambda x, y: x >= y, "truefalse"),
    "eq": (lambda x, y: x == y, "truefalse"),
    "ne": (lambda x, y: x != y, "truefalse"),

    "lt_f": (lambda x, y: x < y, "truefalse"),
    "le_f": (lambda x, y: x <= y, "truefalse"),
    "gt_f": (lambda x, y: x > y, "truefalse"),
    "ge_f": (lambda x, y: x >= y, "truefalse"),
    "eq_f": (lambda x, y: x == y, "truefalse"),
    "ne_f": (lambda x, y: x != y, "truefalse"),

    "concat": (lambda x, y: x + y, "viba"),
    "len_str": (lambda x: len(x), 4),
    "upper_str": (lambda x: x.upper(), "VIBA"),
    "lower_str": (lambda x: x.lower(), "viba"),
    "eq_str": (lambda x, y: x == y, "truefalse"),
    "ne_str": (lambda x, y: x != y, "truefalse"),
    "lt_str": (lambda x, y: x < y, "truefalse"),
    "le_str": (lambda x, y: x <= y, "truefalse"),
    "gt_str": (lambda x, y: x > y, "truefalse"),
    "ge_str": (lambda x, y: x >= y, "truefalse"),

    "and": (lambda x, y: x and y, "truefalsefalse"),
    "or": (lambda x, y: x or y, "falsetruetrue"),
    "not": (lambda x: not x, "falsetrue"),
    "xor": (lambda x, y: x != y, "falsetruefalse"),
    "eq_bool": (lambda x, y: x == y, "truefalse"),
    "ne_bool": (lambda x, y: x != y, "truefalse"),

    "int_to_float": (lambda x: float(x), 7.0),
    "float_to_int": (lambda x: int(x), "2|-2"),
    "int_to_str": (lambda x: str(x), "123"),
    "float_to_str": (lambda x: str(x), "1.5|2.0"),
    "bool_to_str": (lambda x: "true" if x else "false", "true|false"),
    "str_to_int": (lambda x: int(x), 42),
    "str_to_float": (lambda x: float(x), 2.5),
}


def host_for(seen=None, extra=None, ran_at=None):
    """宿主：内建算子按 `OPERATORS` 实现，另加 `extra` 里那几样本模块自己的定义。

    `seen` 记下 `get_func` 收到的（模块路径，名字）；`ran_at` 记下一步跑在哪个环境里
    —— 那是环境带的数据路径，`get_func` 不再收到它（`viba-interpreter.md`）。
    """
    def get_func(module_path, func_name):
        if seen is not None:
            seen.append((module_path, func_name))
        if module_path == "builtin":
            entry = OPERATORS.get(func_name)
            step = entry[0] if entry is not None else None
        else:
            step = (extra or {}).get(func_name)
        if step is None:
            return None

        def ran(env, *args):
            if ran_at is not None:
                ran_at.append(env.storage.cur_storage_path)
            return step(*[_leaf(one) for one in args])
        return ran
    return get_func


def _leaf(one):
    """宿主拿到的一个实参：viba 数据读它的叶子，别的（环境）原样。"""
    return one.value if isinstance(one, VObject) else one


def environ_for(get_func, store):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(get_func),
                       viba_path=f"{CASES}:{REPOSITORY_ROOT}")


def run(tmp: Path):
    _every_operator(tmp)
    _the_two_spellings_are_one_call(tmp)
    _a_module_may_shadow_a_builtin(tmp)
    _it_runs_where_it_is_given(tmp)
    _the_wrong_type_is_refused(tmp)
    _the_library_and_the_table_agree()


def _every_operator(tmp: Path):
    """每个算子一份用例：它那一步跑到了，结果就该是那个值。"""
    for index, (name, (_step, want)) in enumerate(OPERATORS.items()):
        seen = []
        result = interpret(str(CASES / f"{name}.viba"),
                           environ_for(host_for(seen), tmp / f"operator-{index}"))
        check(is_ok(result) and value_of(result) == want,
              f"{name}: answers {want!r}, got {result!r}")
        asked = set(seen)
        check(("builtin", name) in asked,
              f"{name}: its own step is the one that ran: {sorted(asked)}")


def _the_two_spellings_are_one_call(tmp: Path):
    """`add` 与 `builtin.add` 是同一次调用：宿主两次拿到的名字一样。"""
    seen = []
    result = interpret(str(CASES / "both_spellings.viba"),
                       environ_for(host_for(seen), tmp / "spellings"))
    check(is_ok(result) and value_of(result) is True,
          f"the two spellings answer the same value: {result!r}")
    check(seen.count(("builtin", "add")) == 2 and
          set(seen) == {("builtin", "add"), ("builtin", "eq")},
          f"both spellings land on the same member: {seen}")


def _a_module_may_shadow_a_builtin(tmp: Path):
    """模块自己写的 `add` 赢；`builtin.add` 仍然拿到内建那个。"""
    seen = []
    local = {"add": lambda x, y: x * y}
    result = interpret(str(CASES / "shadowed_by_the_module.viba"),
                       environ_for(host_for(seen, local), tmp / "shadowed"))
    check(is_ok(result) and value_of(result) == 13,
          f"its own add runs first (3 times 4), the builtin one adds 1: {result!r}")
    check({("builtin", "add"),
           (module_of(CASES / "shadowed_by_the_module"), "add")} <= set(seen),
          f"both names reached the host: {seen}")


def _it_runs_where_it_is_given(tmp: Path):
    """内建算子跑在给它的环境里，不在调用方那条路径上。"""
    seen, ran_at = [], []
    result = interpret(str(CASES / "runs_where_it_is_given.viba"),
                       environ_for(host_for(seen, ran_at=ran_at), tmp / "where"))
    check(is_ok(result) and value_of(result) == 3,
          f"the operator answers 3: {result!r}")
    check(ran_at == ["root/operators"],
          f"it ran at the environment it was given, which is what carries the data path: "
          f"{ran_at}")


def _the_wrong_type_is_refused(tmp: Path):
    """参数类型不对：拒绝，而不是把字符串交给宿主。"""
    result = interpret(str(CASES / "the_wrong_type.viba"),
                       environ_for(host_for([]), tmp / "wrong"))
    labelled(result, "does not fit",
             "a str given to an int parameter is refused")


def _the_library_and_the_table_agree():
    """`viba/builtin.viba` 里 `builtin` 的成员，与上面这张表一一对应。"""
    declared = _declared_operators()
    check(set(declared) == set(OPERATORS) | {"echo"},
          f"the library's members are the operators in the table, plus echo: "
          f"{sorted(set(declared) - set(OPERATORS) - {'echo'})} extra, "
          f"{sorted(set(OPERATORS) - set(declared))} missing")


def _declared_operators():
    """`viba/builtin.viba` 里 `builtin` 的成员名，按写的次序。"""
    tree = viba_ast.parse((BUILTIN_DIR / "builtin.viba").read_text())
    concept = next(node for node in tree.body
                   if getattr(node, "name", None) == BUILTIN_CONCEPT)
    return [factor.tag[1:] for factor in product_elements(concept.body)
            if isinstance(factor, viba_ast.Tagged)]


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-builtin-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

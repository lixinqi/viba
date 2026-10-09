"""内建目录里的算子（`viba/builtin.viba` 的 `builtin`）：一个算子一份用例，两种源码形式都行。

用例在 `tests/data/builtin/`：

    <算子>.viba                 一个算子一份：`add.viba` 跑 `add`，该给出什么列在下面这张表里
    both_spellings.viba         `add` 与 `builtin.add` 是同一次调用
    shadowed_by_the_module.viba 模块自己给一个 `add`，它赢过内建那个
    runs_where_it_is_given.viba 内建算子跑在给它的那个环境里
    the_wrong_type.viba         参数类型不对，拒绝
    find_not_found.viba         找不到是 `nil`：一个值，不是停
    REFUSAL_CASES 里那几个       给出的东西没有答案：`substr` / `char_at` 越界、空分隔符、
                                空 `$old`、负次数，都由实现停下

`OPERATORS` 就是这张表：每个算子在宿主那一步怎么算，以及它的用例该给出什么。实参固定落在用例
文件里，结果列在这里 —— 一个算子一份，所以每个算子的结果都单独被钉住。一个算子的两个方向都
要钉住时（`lt` 那样），用例给出的是一个串：`"truefalse"`。

宿主只实现内建算子：`get_func` 拿到的名字是 `builtin.<算子>`，别的名字一律不认（那份自己给出
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
                            EnvironmentStorage, interpret, viba_data)
from viba.partial import product_elements
from viba.reflect import VObject, access as reflect_access
from viba.type import BUILTIN_CONCEPT, Ok

checks = Checks("interpreter_builtin")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "builtin"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def _leaf(one):
    """宿主拿到的一个实参：viba 数据取它的叶子，别的（环境）原样。

    没有叶子的那一类（一个容器）原样交出去：实现按 `VObject` 的访问器取它的成员
    （`tests/test_interpreter_containers.py` 里宿主拿到列表也是这样）。
    """
    if not isinstance(one, VObject):
        return one
    leaf = reflect_access.leaf(one)
    return leaf.ok_value if isinstance(leaf, Ok) else one


def _refused(why: str):
    """一步拒绝了它拿到的东西：实现在这里停（`$underlying_viba_op_err`）。

    宿主没有别的说法：它答的就是值，要说不，只能在这里抛出。
    """
    raise ValueError(why)


def _split(x, sep):
    """`split`：切出来的几段，作为一个 `ListLiteral` 交给这次运行。

    宿主答不了一个 Python 列表（它没有叶子），能答的是一个 `VObject`、一块
    语法树、一个标量或者 None，所以这里按语法树给。
    """
    if not sep:
        _refused("split takes a separator, and this one is empty")
    return viba_data(viba_ast.TypeApp(
        "ListLiteral", [viba_ast.Constant(one) for one in x.split(sep)]))


# 表里没有的那几个成员：`echo` 把实参原样交回去，两个开关的 `$get_v` 是一个函数型的槽
# （那一支的调用，由开关决定算不算）。它们的用例在别处（`tests/data/echo/`、
# `tests/data/if/`），这张表按值算不了它们。
NOT_OPERATORS = {"echo", "echo_or_never", "never_or_echo"}

# 每个算子：怎么算它，用例该给出什么。次序与 `viba/builtin.viba` 里的次序一致。
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

    # 比较：两个方向都放在一个用例里，给出的是 "truefalse"
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

    # 一段一段地处理 str：切、取、找、接、换、修。越界、空分隔符、空 old、负次数
    # 都在这里停下（`_refused`），各自的用例见下面那组 `REFUSAL_CASES`。
    "substr": (lambda x, start, end: x[start:end]
               if 0 <= start <= end <= len(x)
               else _refused("the range is not inside the string"), "ib"),
    "char_at": (lambda x, at: x[at]
                if 0 <= at < len(x)
                else _refused("there is no character there"), "i"),
    "find": (lambda x, needle: x.find(needle) if x.find(needle) >= 0 else None, 2),
    "contains": (lambda x, needle: needle in x, True),
    "starts_with": (lambda x, prefix: x.startswith(prefix), True),
    "ends_with": (lambda x, suffix: x.endswith(suffix), True),
    "split": (_split, "b"),
    "join": (lambda parts, sep: sep.join(one.value for one in parts), "a-b-c"),
    "replace": (lambda x, old, new: x.replace(old, new) if old
                else _refused("replace takes something to replace, and this is empty"), "voba"),
    "trim": (lambda x: x.strip(), "viba"),
    "repeat": (lambda x, times: x * times if times >= 0
               else _refused("repeat takes a count that is not negative"), "ababab"),

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


def environ_for(get_func, store):
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(get_func),
                       viba_path=f"{CASES}:{REPOSITORY_ROOT}")


def run(tmp: Path):
    _every_operator(tmp)
    _the_two_spellings_are_one_call(tmp)
    _a_module_may_shadow_a_builtin(tmp)
    _it_runs_where_it_is_given(tmp)
    _the_wrong_type_reaches_the_step(tmp)
    _what_the_string_operators_refuse(tmp)
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
    """模块自己给出的 `add` 赢；`builtin.add` 仍然拿到内建那个。"""
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


def _the_wrong_type_reaches_the_step(tmp: Path):
    """参数类型不对：值层不拦，那个字符串交到这一步的实现手里，崩在那里。"""
    result = interpret(str(CASES / "the_wrong_type.viba"),
                       environ_for(host_for([]), tmp / "wrong"))
    checks.failed(result, "raised",
                  "a str given to an int parameter breaks the step it was given to")


# 找不着、越界、空的分隔符或 old、负的重复次数：这几个用例给出的东西没有答案。
# (用例文件, 停法那句话里该有什么, 说明)
REFUSAL_CASES = [
    ("substr_out_of_range", "raised", "substr: the end is past the string"),
    ("char_at_out_of_range", "raised", "char_at: there is no character there"),
    ("split_empty_separator", "raised", "split: an empty separator cuts nothing"),
    ("replace_empty_old", "raised", "replace: nothing to replace"),
    ("repeat_negative", "raised", "repeat: a negative count"),
]


def _what_the_string_operators_refuse(tmp: Path):
    """`find` 找不到给 nil（是一个值，不是停）；越界、空分隔符、空 old、负次数都停下。"""
    missing = interpret(str(CASES / "find_not_found.viba"),
                        environ_for(host_for([]), tmp / "find-missing"))
    check(is_ok(missing) and value_of(missing) is None,
          f"find: a needle that is not there answers nil, got {missing!r}")

    for name, want, label in REFUSAL_CASES:
        result = interpret(str(CASES / f"{name}.viba"),
                           environ_for(host_for([]), tmp / name))
        checks.failed(result, want, label)


def _the_library_and_the_table_agree():
    """`viba/builtin.viba` 里 `builtin` 的成员，与上面这张表一一对应。"""
    declared = _declared_operators()
    check(set(declared) == set(OPERATORS) | NOT_OPERATORS,
          f"the library's members are the operators in the table, plus "
          f"{sorted(NOT_OPERATORS - {'echo'})}: "
          f"{sorted(set(declared) - set(OPERATORS) - NOT_OPERATORS)} extra, "
          f"{sorted(set(OPERATORS) - set(declared))} missing")


def _declared_operators():
    """`viba/builtin.viba` 里 `builtin` 的成员名，按源码里的次序。"""
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

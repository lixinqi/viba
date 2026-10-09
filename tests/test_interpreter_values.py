"""值：宿主给出什么、__impl__ 能是哪些源码形式、一个名字算几次。

不纯的东西从这里出去（宿主函数），回来的必须是叶子或可序列化数据；函数、模块、泛型应用都不是值。
每条用例是一份可以打开的文件（`tests/data/values/*.viba`），这里只列它该跑出什么。

    python3 tests/test_interpreter_values.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import (branch_of, full_name_of, message_of, module_of,
                                  module_path_of, stop_node, stop_tag, stop_text,
                                  Checks, Host, is_ok, value_of)

from viba import viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret)
from viba.reflect import access as reflect_access
from viba.type import (ERR_TAG, FAILURE_TAG, NOT_IMPLEMENTED_TAG, OK_TAG,
                       PROGRAM_ERR_TAG)

checks = Checks("interpreter_values")
check = checks.check
labelled = checks.labelled

CASES = Path(__file__).resolve().parent / "data" / "values"


def _case(name: str) -> str:
    return str(CASES / f"{name}.viba")


def run(tmp: Path):
    _the_branches(tmp)
    _host_answers(tmp)
    _given_as_ret(tmp)
    _given(tmp)
    _names_and_repeats(tmp)
    _crossing_the_host_boundary(tmp)


# (文件, 宿主给出什么, 说明)
HOST_ANSWER_CASES = [
    ("leaf", 7, "an int"),
    ("text", "hi", "a str"),
    ("flag", True, "a bool"),
    ("ratio", 0.5, "a float"),
    ("nothing", None, "None lands as nil"),
]

# 假值但不是 nil：0、空串、false 都是叶子
FALSY_CASES = [
    ("falsy_zero", 0, "0"),
    ("falsy_empty", "", "an empty str"),
    ("falsy_falsey", False, "false"),
]

# 给不了叶子的容器
NO_LEAF_CASES = [
    ("answer_a_list", "a list"),
    ("answer_a_tuple", "a tuple"),
    ("answer_a_dict", "a dict"),
]


def _the_branches(tmp: Path):
    """结果就是那一支数据：跑出值的落在 `$ok` 上，`$err` 那一支也在设计里。"""
    host = Host()
    result = interpret(_case("leaf"), host.environ())
    check(branch_of(result) == OK_TAG,
          f"a run that answered is on the $ok branch: {result!r}")
    check([tag for tag, _step, _given in reflect_access.member_steps(result)]
          == [OK_TAG, ERR_TAG],
          "the design carries both branches, the value one first")
    check(FAILURE_TAG == "$underlying_viba_op_err"
          and NOT_IMPLEMENTED_TAG == "$not_implemented_err"
          and PROGRAM_ERR_TAG == "$viba_program_err",
          f"the tags a stop answers under: {FAILURE_TAG!r}, "
          f"{NOT_IMPLEMENTED_TAG!r}, {PROGRAM_ERR_TAG!r}")


def _host_answers(tmp: Path):
    """宿主返回什么，__impl__ 就是什么；宿主出错就是 VibaProgramErr，不是崩。"""
    host = Host()
    environ = host.environ()
    for name, want, label in HOST_ANSWER_CASES:
        result = interpret(_case(name), environ)
        check(is_ok(result) and value_of(result) == want,
              f"a host function returning {label}: {result!r}")

    for name, want, label in FALSY_CASES:
        result = interpret(_case(name), environ)
        check(is_ok(result) and value_of(result) == want,
              f"a host answer of {label} is a leaf, not nil: {result!r}")

    for name, label in NO_LEAF_CASES:
        checks.failed(interpret(_case(name), environ), "no leaf",
                      f"a host answer that is {label}")

    boom = _case("boom")
    exploded = interpret(boom, environ)
    checks.failed(exploded, "ZeroDivision", "a host function that raises")
    check(stop_text(exploded, "$module_path") == module_path_of(boom)
          and stop_text(exploded, "$full_qualified_func_name") ==
          full_name_of(boom, "explode")
          and message_of(exploded).startswith("raised"),
          f"and the failure names the step and why: {exploded!r}")
    checks.failed(interpret(boom, Host(get_func_raises=True).environ()), "raised",
                  "a get_func that raises")

    stopped = interpret(_case("no_impl"), environ)
    check(stop_tag(stopped) == NOT_IMPLEMENTED_TAG
          and message_of(stopped) == "no implementation",
          "get_func says None: the run stops with no implementation, not a VibaProgramErr")
    check(stop_tag(stopped) != PROGRAM_ERR_TAG,
          "and that stop is not a VibaProgramErr: nothing broke, one step has no implementation")
    check(stop_text(stopped, "$module_path") == module_path_of(_case("no_impl"))
          and stop_text(stopped, "$full_qualified_func_name") ==
          full_name_of(_case("no_impl"), "ghost")
          and message_of(stopped) == "no implementation",
          f"the stop names the step and why: {stopped!r}")
    call = stop_node(stopped, "$call").data
    check(isinstance(call, viba_ast.Partial)
          and isinstance(call.function, viba_ast.TypeRef)
          and call.function.name == "__dyn_call__"
          and isinstance(call.argument, viba_ast.Constant)
          and call.argument.value == full_name_of(_case("no_impl"), "ghost"),
          f"a call with no argument of its own is the name and then nothing: "
          f"{stop_node(stopped, '$call')!r}")

    result = interpret(_case("echo"), environ)
    check(is_ok(result) and value_of(result) == 42,
          f"a host function echoing its argument: {result!r}")

    checks.failed(interpret(_case("arity"), environ), "raised",
                  "a host function of the wrong arity")

    result = interpret(_case("made"), environ)
    check(is_ok(result) and value_of(result) == 11,
          f"a node the host built itself: {result!r}")

    checks.failed(interpret(_case("afunc"), environ), "no leaf",
                  "a host answer that is a function")


# (文件, 该跑出什么)
LITERAL_CASES = [("lit_42", 42, "a literal int"), ("lit_hi", "hi", "a literal str"),
                 ("lit_true", True, "a literal bool")]
UNIT_CASES = [("unit_nil", "nil"), ("unit_never", "never"), ("unit_Any", "Any")]
# (文件, 说法, 停下来的那句话里有什么)
NOT_A_VALUE_CASES = [("application_in_source", "a generic application",
                      "no definition named 'list'"),
                     ("exponent_in_source", "an exponent", "cannot compute")]


def _given_as_ret(tmp: Path):
    """__impl__ 的源码形式是字面量、单位、和/积、名字：哪些是值。"""
    host = Host()
    environ = host.environ()

    for name, want, label in LITERAL_CASES:
        result = interpret(_case(name), environ)
        check(is_ok(result) and value_of(result) == want,
              f"__impl__ given as {label}: {result!r}")
    for name, label in UNIT_CASES:
        result = interpret(_case(name), environ)
        check(is_ok(result), f"__impl__ given as {label}: {result!r}")
    for name, label, want in NOT_A_VALUE_CASES:
        # `list[int]` 出现在值的位置上时，没有哪个泛型回答 `list`，于是按简称去判这个名字 ——
        # 判不成，报的就是这个名字（`_shorthand_chain`）。类型名不是一个值。
        labelled(interpret(_case(name), environ), want,
                 f"__impl__ given as {label} -> VibaProgramErr")

    # 源码里的和值也是 viba 数据：没给出的那几支留下的都是 never
    result = interpret(_case("sum_in_source"), environ)
    check(is_ok(result),
          f"__impl__ given as a sum is viba data, not an error: {result!r}")

    check(is_ok(interpret(_case("module_value"), environ)),
          "__impl__ given as a module is the closure it stands for")

    labelled(interpret(_case("builtin_value"), environ), "no definition named",
             "a builtin type name used as a value -> VibaProgramErr")


# (文件, 有几个成员)
DATA_CASES = [("tuple_in_source", 2, "a tuple of two"),
              ("empty_tuple_in_source", 0, "the empty tuple")]


def _given(tmp: Path):
    """出现在值位置上的数据就是可序列化数据：元组、tag、积。"""
    host = Host()
    environ = host.environ()
    for name, want, label in DATA_CASES:
        result = interpret(_case(name), environ)
        check(is_ok(result) and len(result.by_tag(OK_TAG)) == want,
              f"__impl__ given as {label} is viba data: {result!r}")

    # 一份按 tag 与积给出的可序列化数据，源码形式和它的类型源码形式一样
    result = interpret(_case("tags_and_product"), environ)
    check(is_ok(result), f"data given as tags and a product: {result!r}")
    if is_ok(result):
        node = result.by_tag(OK_TAG)
        check(reflect_access.leaf(node.by_tag("victim").by_tag("x")).ok_value == 0 and
              reflect_access.leaf(node.by_tag("suspect").by_tag("y")).ok_value == 4,
              f"and its members come back: {node!r}")


def _names_and_repeats(tmp: Path):
    """名字与次序：定义出现在用之后也算、一个定义只算一次、同一个名字不许给两次。"""
    host = Host()
    environ = host.environ()

    labelled(interpret(_case("twice_defined"), environ), "is defined twice",
             "one name defined twice in one module -> VibaProgramErr")

    result = interpret(_case("defined_after"), environ)
    check(is_ok(result) and value_of(result) == 7,
          f"a definition in the source after its use: {result!r}")

    # `__impl__` first, everything it uses after it: the same answer either way
    result = interpret(_case("impl_given_first"), environ)
    check(is_ok(result) and value_of(result) == 3,
          f"__impl__ in the source before everything it uses: {result!r}")

    result = interpret(_case("nested"), environ)
    check(is_ok(result) and value_of(result) == 3 and
          (module_of(_case("nested")), "unused") not in host.calls,
          f"a definition nobody asks for is never run: {result!r}")

    # 一个定义算一次：两处用它，宿主只被叫一次
    host.calls.clear()
    result = interpret(_case("memo"), environ)
    check(is_ok(result) and value_of(result) == 14,
          f"a definition used twice: {result!r}")
    check(host.calls.count((module_of(_case("memo")), "leaf")) == 1,
          f"a definition is computed once: {host.calls}")


def _crossing_the_host_boundary(tmp: Path):
    """宿主手里拿到 viba 函数：只是数据——拿着、存着、原样递回来，调不动。"""
    host = Host()
    environ = host.environ()

    higher = _case("higher")
    original = host.get_func

    def hand_back(path, func_name):
        if func_name == "show":
            return lambda env, f: f            # 拿到的就是那个节点，原样交回来
        return original(path, func_name)

    host.get_func = hand_back
    host2 = Environment(EnvironmentStorage("root"), EnvironmentCompute(host.get_func))
    result = interpret(higher, host2)
    check(is_ok(result)
          and isinstance(result.by_tag(OK_TAG).data, viba_ast.TypeRef),
          f"a viba function handed to the host is viba data (the name it carries): {result!r}")
    check(result.by_tag(OK_TAG).data.name == "inc",
          f"and what it holds is that name: {result.by_tag(OK_TAG).data!r}")
    host.get_func = original

    # 宿主自己说：get_func 交回那份"没有实现"的数据，等于这次调用没有实现
    host.knobs["refuse"] = ("twice",)
    checks.not_implemented(interpret(higher, environ),
                             "a get_func that hands back the failure")
    host.knobs.pop("refuse")

    # 函数类型的实参交给宿主的是"它代表的那次调用"：宿主可以带上自己的实参叫它
    # （wrapper 就是这么转发的），多喂的那一个由 viba 这边说清楚 —— 这次调用不
    # 再有槽位收它，所以是程序错，而不是把 Python 的 TypeError 报成实现坏了。
    checks.labelled(interpret(_case("overfeed"), environ), "takes no more arguments",
                    "a host that gives the viba function too many arguments")

    checks.failed(interpret(_case("fed"), environ), "raised",
                  "a host handing a viba function something with no leaf")

    # 那个参数声明的是 int：值层不拦，函数照样交到宿主手里 —— 和声明 Any 时一样，拿到的是
    # 它代表的那次调用（声明是设计那一层的事，值层不拿它核实参）
    result = interpret(_case("handed"), environ)
    check(is_ok(result)
          and isinstance(result.by_tag(OK_TAG).data, viba_ast.TypeRef),
          f"a viba function handed where an int is declared comes back as data too: "
          f"{result!r}")

    # 那个参数声明的是 Any：同一件事
    result = interpret(_case("handed_any"), environ)
    check(is_ok(result)
          and isinstance(result.by_tag(OK_TAG).data, viba_ast.TypeRef),
          f"a viba function handed where Any is declared comes back as data: {result!r}")

    # 宿主里面再跑一次 interpret：两个 run 互不干扰
    host.knobs["inner_file"] = _case("inner_module")
    result = interpret(_case("outer_run"), environ)
    check(is_ok(result) and value_of(result) == 7,
          f"a host function that runs interpret itself: {result!r}")
    host.knobs.pop("inner_file")

    # 宿主抛的不是 Exception：interpret 不吞
    try:
        interpret(_case("interrupt"), environ)
        check(False, "a host raising KeyboardInterrupt is not swallowed")
    except KeyboardInterrupt:
        check(True, "a host raising KeyboardInterrupt is not swallowed")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-values-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

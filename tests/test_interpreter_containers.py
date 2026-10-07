"""容器：`ListLiteral` / `SetLiteral` / `DictLiteral` 算成值，`$__getitem__` 把元素读出来。

三个字面量构造器写在**实例**那一侧（`viba-style.md` 第 8 节）：`ListLiteral[1, 2]` 是
`list[int]` 的居民，和 `$x 1 * $y 2` 一样是就地写下来的可序列化数据，所以一次运行能把它算成
值，值能数、能寻址、能写回去。寻址在语言这一侧是 `$__getitem__`：位置给 int，键给 str
（`$__getattr__` 是按名字取成员的那一个，这是按地址取元素的那一个）。

    python3 tests/test_interpreter_containers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host, error_of, value_of

from viba import serialize, viba_ast
from viba.interpret import Environment, EnvironmentCompute, interpret
from viba.reflect import access as reflect_access
from viba.type import Ok, VibaProgramErr

CASES = Path(__file__).resolve().parent / "data" / "containers"

checks = Checks("interpreter_containers")
check = checks.check


def _run(name: str, env=None):
    return interpret(str(CASES / f"{name}.viba"), env or Host().environ())


def _data(result):
    """What the run answered, as it is written."""
    node = getattr(result, "ok_value", None)
    return None if node is None else viba_ast.unparse_type(node.data).replace("\n", " ")


def _leaf(node):
    """The leaf one node carries; the node itself when it has none."""
    got = reflect_access.leaf(node)
    return got.ok_value if isinstance(got, Ok) else got


def _written(result):
    """The same piece written back out as viba source."""
    written = serialize.serialize("x", result.ok_value)
    return written.ok_value if isinstance(written, Ok) else written


def run():
    _a_literal_is_a_value()
    _a_member_is_a_container()
    _what_getitem_reads()
    _what_getitem_refuses()
    _what_in_answers()
    _the_index_shorthand()
    _a_container_travels()


def _a_literal_is_a_value():
    """三个字面量都算成值：能数、能寻址、写得回去。"""
    got = _run("list")
    check(isinstance(got, Ok) and _data(got) == "ListLiteral[1, 2]",
          f"a list literal is the list it writes: {got!r}")
    check(got.ok_value.is_list and len(got.ok_value) == 2,
          f"and it is a list of two: {got!r}")
    check(_leaf(got.ok_value.at_index(1)) == 2,
          f"whose second element is the second literal")
    check(_written(got) == 'x =\n  ListLiteral[1, 2]\n',
          f"and it is written back the way it was written: {_written(got)!r}")

    empty = _run("empty_list")
    check(isinstance(empty, Ok) and empty.ok_value.is_list
          and len(empty.ok_value) == 0 and _written(empty) == "x =\n  ListLiteral[]\n",
          f"the empty list is a list of nothing: {empty!r}")

    a_set = _run("set")
    check(isinstance(a_set, Ok) and a_set.ok_value.is_set
          and _leaf(a_set.ok_value.at_index(1)) == "b",
          f"a set literal is the set it writes: {a_set!r}")

    table = _run("dict")
    check(isinstance(table, Ok) and table.ok_value.is_dict
          and table.ok_value.keys() == ["k", "m"]
          and _leaf(table.ok_value.at_key("m")) == 2,
          f"a dict literal is read by key: {table!r}")
    check(_written(table) == 'x =\n  DictLiteral[("k", 1), ("m", 2)]\n',
          f"and written back with its pairs: {_written(table)!r}")

    nothing = _run("empty_dict")
    check(isinstance(nothing, Ok) and nothing.ok_value.is_dict
          and nothing.ok_value.keys() == [],
          f"the empty dict has no keys: {nothing!r}")

    deep = _run("nested")
    inner = deep.ok_value.at_index(1)
    check(isinstance(deep, Ok) and len(deep.ok_value) == 2
          and _leaf(inner.at_index(1)) == 3,
          f"a container holds containers: {deep!r}")
    check(_written(deep) == 'x =\n  ListLiteral[ListLiteral[1], ListLiteral[2, 3]]\n',
          f"and the nesting is written back whole: {_written(deep)!r}")

    lists = _run("nested_dict")
    check(isinstance(lists, Ok) and _leaf(lists.ok_value.at_key("b").at_index(0)) == 3,
          f"a dict of lists reads all the way down: {lists!r}")

    mixed = _run("mixed")
    check(_leaf(mixed.ok_value.at_index(0)) == 1
          and _leaf(mixed.ok_value.at_index(1)) == "a",
          f"a list holds what it was written with, member by member: {mixed!r}")


def _a_member_is_a_container():
    """积里那一格也是容器：数得出、寻得到、写得回去。"""
    got = _run("box")
    check(isinstance(got, Ok) and _leaf(got.ok_value.by_tag("$y")) == 3,
          f"the product is the product it writes: {got!r}")
    xs = got.ok_value.get_xs()
    check(xs.is_list and len(xs) == 2 and _leaf(xs.at_index(0)) == 1,
          f"and the list inside it is a list: {xs!r}")
    check(_written(got) == "x =\n  $xs ListLiteral[1, 2]\n  * $y 3\n",
          f"written back, the member keeps its own spelling: {_written(got)!r}")


def _what_getitem_reads():
    """`$__getitem__`：位置给 int，键给 str，读出来就是那个元素。"""
    by_index = _run("get_by_index")
    check(value_of(by_index) == 20, f"an element by position: {by_index!r}")

    by_key = _run("get_by_key")
    check(value_of(by_key) == 8, f"a value by key: {by_key!r}")

    nested = _run("get_nested")
    check(value_of(nested) == 3, f"an address into an element: {nested!r}")

    a_tuple = _run("get_tuple")
    check(value_of(a_tuple) == 20, f"a tuple is read by position too: {a_tuple!r}")

    later = _run("get_partial")
    check(value_of(later) == 20,
          f"an address given after the fact still reads the element: {later!r}")


def _what_getitem_refuses():
    """取不到的时候说清楚：没有那个位置、没有那个键、地址不对、那不是容器。"""
    for name, want in (("get_out_of_range", "no such address"),
                       ("get_missing_key", "no such address"),
                       ("get_index_on_dict", "no such address"),
                       ("get_key_on_list", "no such address"),
                       ("get_bad_address", "an element is read by position (an int)"),
                       ("get_no_address", "an element is read by an address"),
                       ("get_non_container", "no such address")):
        got = _run(name)
        error = error_of(got)
        check(isinstance(error, VibaProgramErr) and want in error.msg,
              f"{name}: expected a program error saying {want!r}, got {got!r}")


def _what_in_answers():
    """`$__in__`：元素在不在（list / set / tuple），键在不在（dict）。"""
    for name, want in (("in_list", True), ("in_list_no", False), ("in_set", True),
                       ("in_tuple", True), ("in_dict_key", True),
                       ("in_dict_value", False), ("in_nested", True),
                       ("in_partial", True)):
        got = _run(name)
        check(isinstance(got, Ok) and value_of(got) is want,
              f"{name}: expected {want}, got {got!r}")

    for name, want in (("in_not_container", "asks about a list, a set, a tuple or a dict"),
                       ("in_dict_bad_key", "a dict holds keys"),
                       ("in_no_piece", "asked about a piece, and none was given")):
        got = _run(name)
        error = error_of(got)
        check(isinstance(error, VibaProgramErr) and want in error.msg,
              f"{name}: expected a program error saying {want!r}, got {got!r}")


def _the_index_shorthand():
    """`xs[i]` / `table[key]` 是 `$__getitem__ << xs << i` 的简写。"""
    for name, want in (("index_list", 20), ("index_dict", 8), ("index_member", 20),
                       ("index_element", 2)):
        got = _run(name)
        check(isinstance(got, Ok) and value_of(got) == want,
              f"{name}: expected {want}, got {got!r}")

    out_of_range = _run("index_out_of_range")
    error = error_of(out_of_range)
    check(isinstance(error, VibaProgramErr) and "no such address" in error.msg,
          f"index_out_of_range: expected a program error, got {out_of_range!r}")

    undefined = _run("index_undefined")
    error = error_of(undefined)
    check(isinstance(error, VibaProgramErr) and "no definition named 'Nope'" in error.msg,
          f"a name no generic answers to is read as a value: {undefined!r}")


def _a_container_travels():
    """容器当实参交给宿主：宿主拿到的是那个列表本身。"""
    def get_func(module_path, func_name):
        if func_name == "head":
            def head(env, xs):
                return xs.at_index(0)
            return head
        return None

    env = Environment(Host().environ().storage, EnvironmentCompute(get_func))
    got = _run("argument", env)
    check(value_of(got) == 7,
          f"a list written as an argument reaches the host as a list: {got!r}")


if __name__ == "__main__":
    run()
    sys.exit(checks.report())

"""容器：`ListLiteral` / `SetLiteral` / `DictLiteral` 算成值，`$get_item` 取元素、
`$in` 问在不在、`$len` 数一数、`$keys` 取字典的键。

三个字面量构造器站在**实例**那一侧（`viba-style.md` 第 8 节）：`ListLiteral[1, 2]` 是
`list[int]` 的居民，和 `$x 1 * $y 2` 一样是就地给出的可序列化数据，所以一次运行能把它算成
值，值能数、能寻址、能序列化成源码。寻址在语言这一侧是 `$get_item`：位置给 int，键给 str
（`$get_attr` 是按名字取成员的那一个，这是按地址取元素的那一个）。

    python3 tests/test_interpreter_containers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import answer_of, Checks, Host, is_ok, value_of

from viba import serialize, viba_ast
from viba.interpret import Environment, EnvironmentCompute, interpret
from viba.reflect import access as reflect_access
from viba.type import Ok

CASES = Path(__file__).resolve().parent / "data" / "containers"

checks = Checks("interpreter_containers")
check = checks.check


def _run(name: str, env=None):
    return interpret(str(CASES / f"{name}.viba"), env or Host().environ())


def _data(result):
    """What the run answered, in the source; None when it stopped."""
    if not is_ok(result):
        return None
    return viba_ast.unparse_type(answer_of(result).data).replace("\n", " ")


def _leaf(node):
    """The leaf one node carries; the node itself when it has none."""
    got = reflect_access.leaf(node)
    return got.ok_value if isinstance(got, Ok) else got


def _serialized(result):
    """The same piece serialized back out as viba source."""
    serialized = serialize.serialize("x", answer_of(result))
    return serialized.ok_value if isinstance(serialized, Ok) else serialized


def run():
    _a_literal_is_a_value()
    _a_member_is_a_container()
    _what_getitem_takes()
    _what_getitem_refuses()
    _what_in_answers()
    _what_len_counts()
    _what_keys_gives()
    _a_chain_is_a_piece()
    _a_member_call_owes_an_environment()
    _the_index_shorthand()
    _a_container_travels()


def _a_literal_is_a_value():
    """三个字面量都算成值：能数、能寻址、能序列化成源码。"""
    got = _run("list")
    check(is_ok(got) and _data(got) == "ListLiteral[1, 2]",
          f"a list literal is the list it gives: {got!r}")
    check(is_ok(got) and answer_of(got).is_list and len(answer_of(got)) == 2,
          f"and it is a list of two: {got!r}")
    check(_leaf(answer_of(got).at_index(1)) == 2,
          f"whose second element is the second literal")
    check(_serialized(got) == 'x =\n  ListLiteral[1, 2]\n',
          f"and it is serialized the way it was given: {_serialized(got)!r}")

    empty = _run("empty_list")
    check(is_ok(empty) and answer_of(empty).is_list
          and len(answer_of(empty)) == 0
          and _serialized(empty) == "x =\n  ListLiteral[]\n",
          f"the empty list is a list of nothing: {empty!r}")

    a_set = _run("set")
    check(is_ok(a_set) and answer_of(a_set).is_set
          and _leaf(answer_of(a_set).at_index(1)) == "b",
          f"a set literal is the set it gives: {a_set!r}")

    table = _run("dict")
    check(is_ok(table) and answer_of(table).is_dict
          and answer_of(table).keys() == ["k", "m"]
          and _leaf(answer_of(table).at_key("m")) == 2,
          f"a dict literal is taken by key: {table!r}")
    check(_serialized(table) == 'x =\n  DictLiteral[("k", 1), ("m", 2)]\n',
          f"and serialized with its pairs: {_serialized(table)!r}")

    nothing = _run("empty_dict")
    check(is_ok(nothing) and answer_of(nothing).is_dict
          and answer_of(nothing).keys() == [],
          f"the empty dict has no keys: {nothing!r}")

    deep = _run("nested")
    inner = answer_of(deep).at_index(1)
    check(is_ok(deep) and len(answer_of(deep)) == 2
          and _leaf(inner.at_index(1)) == 3,
          f"a container holds containers: {deep!r}")
    check(_serialized(deep) == 'x =\n  ListLiteral[ListLiteral[1], ListLiteral[2, 3]]\n',
          f"and the nesting is serialized member by member: {_serialized(deep)!r}")

    lists = _run("nested_dict")
    check(is_ok(lists)
          and _leaf(answer_of(lists).at_key("b").at_index(0)) == 3,
          f"a dict of lists is taken from all the way down: {lists!r}")

    mixed = _run("mixed")
    check(_leaf(answer_of(mixed).at_index(0)) == 1
          and _leaf(answer_of(mixed).at_index(1)) == "a",
          f"a list holds what it was given, member by member: {mixed!r}")


def _a_member_is_a_container():
    """积里那一格也是容器：数得出、寻得到、能序列化成源码。"""
    got = _run("box")
    check(is_ok(got) and _leaf(answer_of(got).by_tag("$y")) == 3,
          f"the product is the product it gives: {got!r}")
    xs = answer_of(got).get_xs()
    check(xs.is_list and len(xs) == 2 and _leaf(xs.at_index(0)) == 1,
          f"and the list inside it is a list: {xs!r}")
    check(_serialized(got) == "x =\n  Object\n  * $xs ListLiteral[1, 2]\n  * $y 3\n",
          f"serialized back, the member keeps its own spelling: {_serialized(got)!r}")


def _what_getitem_takes():
    """`$get_item`：位置给 int，键给 str，取出来就是那个元素。"""
    by_index = _run("get_by_index")
    check(value_of(by_index) == 20, f"an element by position: {by_index!r}")

    by_key = _run("get_by_key")
    check(value_of(by_key) == 8, f"a value by key: {by_key!r}")

    nested = _run("get_nested")
    check(value_of(nested) == 3, f"an address into an element: {nested!r}")

    a_tuple = _run("get_tuple")
    check(value_of(a_tuple) == 20, f"a tuple is taken by position too: {a_tuple!r}")

    later = _run("get_partial")
    check(value_of(later) == 20,
          f"an address given after the fact still takes the element: {later!r}")


def _what_getitem_refuses():
    """取不到的时候说清楚：没有那个位置、没有那个键、地址不对、那不是容器。"""
    for name, want in (("get_out_of_range", "no such address"),
                       ("get_missing_key", "no such address"),
                       ("get_index_on_dict", "no such address"),
                       ("get_key_on_list", "no such address"),
                       ("get_bad_address", "an element is taken by position (an int)"),
                       ("get_no_address", "an element is taken by an address"),
                       ("get_non_container", "no such address")):
        got = _run(name)
        checks.labelled(got, want,
                        f"{name}: expected a program error saying {want!r}")


def _what_in_answers():
    """`$in`：元素在不在（list / set / tuple），键在不在（dict）。"""
    for name, want in (("in_list", True), ("in_list_no", False), ("in_set", True),
                       ("in_tuple", True), ("in_dict_key", True),
                       ("in_dict_value", False), ("in_nested", True),
                       ("in_partial", True)):
        got = _run(name)
        check(is_ok(got) and value_of(got) is want,
              f"{name}: expected {want}, got {got!r}")

    for name, want in (("in_not_container", "asks about a list, a set, a tuple or a dict"),
                       ("in_dict_bad_key", "a dict holds keys"),
                       ("in_no_piece", "asked about a piece, and none was given")):
        got = _run(name)
        checks.labelled(got, want,
                        f"{name}: expected a program error saying {want!r}")


def _what_len_counts():
    """`$len`：list / set / dict / tuple 都数得出，dict 数的是键；别的不是容器。"""
    for name, want in (("len_list", 3), ("len_set", 2), ("len_dict", 2),
                       ("len_tuple", 3), ("len_empty_list", 0),
                       ("len_empty_dict", 0)):
        got = _run(name)
        check(is_ok(got) and value_of(got) == want,
              f"{name}: expected {want}, got {got!r}")

    for name in ("len_not_container", "len_a_str"):
        got = _run(name)
        checks.labelled(got, "counts a list, a set, a tuple or a dict",
                        f"{name}: expected a program error saying that")


def _what_keys_gives():
    """`$keys`：dict 的键，顺序按实现给的；拿到的是一份能寻址的列表。"""
    got = _run("keys_dict")
    check(is_ok(got) and value_of(got) == "m",
          f"keys_dict: the second key of two: {got!r}")

    empty = _run("keys_empty")
    check(is_ok(empty) and value_of(empty) == 0,
          f"keys_empty: an empty dict hands over no keys to count: {empty!r}")

    for name, want in (("keys_not_dict", "gives the keys of a dict"),
                       ("keys_not_literal", "not a literal")):
        got = _run(name)
        checks.labelled(got, want, f"{name}: expected a program error saying {want!r}")


def _a_chain_is_a_piece():
    """成员链当成积或字面量的一份给出时，留着的是那段源码，而设计层描述得出它。

    字面量里的成员是**源码里怎么写的就怎么留着**（跟积的成员一样），所以
    `ListLiteral[$len << xs, 3]` 里那一份是 `$len << xs` 这条链本身，不是 2。
    这样的文件以前整个取不动（报 "…. is no value to take the member '$len' from"）：
    设计层把 `$len` 当成 `xs` 的成员去找。现在它按这个成员自己答的类型描述那条链 ——
    `$len` 是 `int`、`$in` 是 `bool`、`$keys` 是 `list[str]`、`$get_item` 是 `Any`。
    """
    got = _run("chain_in_a_literal")
    check(is_ok(got)
          and _data(got) == "ListLiteral[$len << xs, $in << xs << 20, $get_item << xs << 1]",
          f"a list whose pieces are member chains: {got!r}")

    table = _run("chain_in_a_dict_value")
    check(is_ok(table) and _data(table) == 'DictLiteral[("keys", $keys << table)]',
          f"a dict value that is a member chain: {table!r}")

    field = _run("a_value_member_in_a_literal")
    check(is_ok(field) and _data(field) == "ListLiteral[$y << box, 7]",
          f"a member that is a value, as a piece: {field!r}")

    declared = _run("chain_in_a_declared_type")
    check(is_ok(declared) and _data(declared) == "ListLiteral[1, 2]",
          f"`__decl__` carrying a member chain: {declared!r}")


def _a_member_call_owes_an_environment():
    """这五个成员也收环境：没给环境时它是一条闭包，给了才算。

    环境是每个跑起来的调用的第一条参数，而且总是要：不给就留着不动。这五个自己不用它（答案只从
    它们自己的实参算出来），所以 `$env nil` 就是"现在跑"——`$len << $env nil << xs` 照样答得出个数；
    解释器自己拼出来的链（方括号简式）就是这么拼的。给在哪儿都认：按 `$env` 这个 tag 给，或者那一份
    本身就是环境（`args.env`）；末尾给（`apply` 与 `sequential` 就是那样把环境接上的）也认。没给环境
    时链留着不动：那是一份还欠着环境的调用，也就是 `sequential` 那样场景里能当一步的东西。不是环境的
    值给在那一位上不是"不用环境"，是错的。
    """
    closure = _run("a_member_call_is_a_closure")
    check(is_ok(closure) and _data(closure) == "$len << ListLiteral[10, 20]",
          f"a member call with no environment is a closure: {closure!r}")

    last = _run("a_member_call_env_last")
    check(is_ok(last) and value_of(last) == 2,
          f"the environment may be given last: {last!r}")

    own = _run("a_member_call_in_an_environment")
    check(is_ok(own) and value_of(own) == 20,
          f"the environment may be the run's own: {own!r}")

    wrong = _run("a_member_call_wrong_env")
    checks.labelled(wrong, "was not given an Environment",
                    "a member call's environment must be an environment")


def _the_index_shorthand():
    """`xs[i]` / `table[key]` 是 `$get_item << xs << i` 的简略形式。"""
    for name, want in (("index_list", 20), ("index_dict", 8), ("index_member", 20),
                       ("index_element", 2)):
        got = _run(name)
        check(is_ok(got) and value_of(got) == want,
              f"{name}: expected {want}, got {got!r}")

    out_of_range = _run("index_out_of_range")
    checks.labelled(out_of_range, "no such address",
                    "index_out_of_range: expected a program error")

    undefined = _run("index_undefined")
    checks.labelled(undefined, "no definition named 'Nope'",
                    "a name no generic answers to counts as a value")


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
          f"a list given as an argument reaches the host as a list: {got!r}")


if __name__ == "__main__":
    run()
    sys.exit(checks.report())

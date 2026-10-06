"""`$tag << X << …`：从第一个参数身上取成员，再往后给。

`$sub_env << args.env << "child"` 就是 `args.env.sub_env << args.env << "child"`，`$tmp_env << args.env`
就是 `args.env.tmp_env << args.env`。tag 不是值（`method = $sub_env` 编不过），只有把它写在链头、后面
跟着第一个参数时才有意义；第一个参数既是被取成员的那个值，也是交给成员的第一个实参。

    python3 tests/test_interpreter_member.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import error_of, message_of, Checks, Host, value_of

from viba import viba_ast
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.is_sub_type import is_sub_type
from viba.type import AstNodeType, Ok, custom_module

CASES = Path(__file__).resolve().parent / "data" / "member"

checks = Checks("interpreter_member")
check = checks.check
labelled = checks.labelled


def environ_for():
    return Host().environ()


def _path_of(result) -> str:
    value = result.ok_value
    return getattr(getattr(value, "storage", None), "cur_storage_path", repr(value))


def run(tmp: Path):
    _the_same_child(tmp)
    _the_same_temporary_child(tmp)
    _the_first_argument_is_the_receiver(tmp)
    _a_member_that_takes_its_owner()
    _a_member_read_by_a_name()
    _the_tag_is_not_a_value()
    _what_is_not_there_is_reported(tmp)


def _the_same_child(tmp: Path):
    """`$sub_env << args.env << "child"` 和 `args.env.sub_env << args.env << "child"` 是同一次调用。"""
    by_tag = interpret(str(CASES / "child_by_tag.viba"), environ_for())
    by_dot = interpret(str(CASES / "child_by_dot.viba"), environ_for())
    check(isinstance(by_tag, Ok) and isinstance(by_tag.ok_value, Environment),
          f"taking the member by tag answers an environment: {by_tag!r}")
    check(_path_of(by_tag) == "root/child",
          f"and the name is the one written: {_path_of(by_tag)!r}")
    check(_path_of(by_tag) == _path_of(by_dot),
          f"the dotted spelling names the same child: {_path_of(by_dot)!r}")


def _the_same_temporary_child(tmp: Path):
    """空实参不用写：`$tmp_env << environ` 就是执行。"""
    result = interpret(str(CASES / "temporary_by_tag.viba"), environ_for())
    check(isinstance(result, Ok) and isinstance(result.ok_value, Environment),
          f"a member that takes no argument runs when the member is taken: {result!r}")
    check(_path_of(result).startswith("root/tmp_"),
          f"and it is a temporary child: {_path_of(result)!r}")


def _the_first_argument_is_the_receiver(tmp: Path):
    """第一个参数是取成员的那个值：它就是最前面写的那个。"""
    nested = interpret(str(CASES / "nested_by_tag.viba"), environ_for())
    check(isinstance(nested, Ok) and _path_of(nested) == "root/a/b",
          f"a child of a child: {nested!r}")

    by_tag = interpret(str(CASES / "argument_written_by_tag.viba"), environ_for())
    check(isinstance(by_tag, Ok) and _path_of(by_tag) == "root/kid",
          f"the first argument may be written by tag: {by_tag!r}")


def _a_member_that_takes_its_owner():
    """成员要 owner 时，两种写法是同一次调用：第一个参数也交给成员。"""
    def get_func(path, func_name):
        if func_name == "inc":
            # 收 owner、环境、x 三样
            return lambda box, environ, x: 2
        return Host().get_func(path, func_name)

    host = Environment(EnvironmentStorage("root", None, None), EnvironmentCompute(get_func))
    by_tag = interpret(str(CASES / "owner_member_by_tag.viba"), host)
    by_dot = interpret(str(CASES / "owner_member_by_dot.viba"), host)
    check(isinstance(by_tag, Ok) and value_of(by_tag) == 2,
          f"a member that takes its owner: {by_tag!r}")
    check(isinstance(by_dot, Ok) and value_of(by_dot) == 2,
          f"and the dotted spelling is the same call: {by_dot!r}")


def _judge(source: str, sub: str, sup: str):
    """`sub <: sup` read in a module built from `source`."""
    module = custom_module(source)
    got = is_sub_type(
        AstNodeType(viba_ast.parse(f"__x__ = {sub}").body[0].body, module),
        AstNodeType(viba_ast.parse(f"__x__ = {sup}").body[0].body, module))
    return got.ok_value if isinstance(got, Ok) else got


def _a_member_read_by_a_name():
    """名字写在字符串里时，成员就按那个名字取：`tagged["f"]` 与 `$__getattr__`。

    `tagged["f"] << box << …` 是 `$f << box << …`；`$__getattr__ << box << name << …`
    是 `box.f << …`，区别只在于名字是一份可以算出来的值。
    """
    def get_func(path, func_name):
        if func_name == "inc":
            # 成员要 owner、环境、x 三样（`$f << box << args.env << 1`）。
            return lambda box, environ, x: 2
        return Host().get_func(path, func_name)

    host = Environment(EnvironmentStorage("root", None, None), EnvironmentCompute(get_func))
    by_tag = interpret(str(CASES / "tagged_member.viba"), host)
    check(isinstance(by_tag, Ok) and value_of(by_tag) == 2,
          f"a tag written as a symbol takes the member it names: {by_tag!r}")
    by_name = interpret(str(CASES / "getattr_member.viba"), host)
    check(isinstance(by_name, Ok) and value_of(by_name) == 2,
          f"a member read by the name a value spells: {by_name!r}")

    source = ("Args = Object * $a int * $b str\n"
              'Picked = $__getattr__ << Args << "a"\n'
              'Named = "a"\n'
              "By_name = $__getattr__ << Args << Named\n"
              "Dynamic = $__getattr__ << Args << SomethingElse\n")
    check(_judge(source, "Picked", "int") is True,
          "a written name picks the member it tags")
    check(_judge(source, "Picked", "str") is False,
          "and not another member")
    check(_judge(source, "By_name", "int") is True,
          "a name that stands for a string picks it too")
    check(_judge(source, "Dynamic", "Any") is True
          and _judge(source, "Dynamic", "int") is False,
          "a name no design can read there is Any")


def _the_tag_is_not_a_value():
    """单独一个 `$tag` 编不过：它是链头的一种写法，不是值。"""
    for source in ("method = $sub_env\n", "X = $sub_env\n", "X = $sub_env | int\n"):
        try:
            viba_ast.parse(source)
            check(False, f"a bare tag is not a value: {source!r} parsed")
        except SyntaxError:
            check(True, f"a bare tag is not a value: {source!r}")


def _what_is_not_there_is_reported(tmp: Path):
    """成员不在、第一个参数不是那个值，都当场说清楚。"""
    missing = interpret(str(CASES / "no_such_member.viba"), environ_for())
    labelled(missing, "the environment has no 'nope'",
             "a tag the environment does not hang anything on")

    wrong = interpret(str(CASES / "first_argument_is_not_the_value.viba"), environ_for())
    labelled(wrong, "no member tagged '$sub_env' to take from it",
             "the first argument is the value the member is taken from")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-member-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

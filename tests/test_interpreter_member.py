"""`$tag << X << …`：从第一个参数身上取成员，再往后给。

`$sub_env << args.env << "child"` 就是 `args.env.sub_env << args.env << "child"`，`$tmp_env << args.env`
就是 `args.env.tmp_env << args.env`。tag 不是值（`method = $sub_env` 编不过），只有把它放在链头、后面
跟着第一个参数时才有意义；第一个参数既是被取成员的那个值，也是交给成员的第一个实参。

    python3 tests/test_interpreter_member.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import (Checks, Host, is_ok, message_of, stop_tag,
                                  value_of)

from viba import viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            exec, interpret)
from viba.is_sub_type import is_sub_type
from viba.type import AstNodeType, Ok, PROGRAM_ERR_TAG, custom_module

CASES = Path(__file__).resolve().parent / "data" / "member"

checks = Checks("interpreter_member")
check = checks.check
labelled = checks.labelled


def environ_for():
    """The shared host, plus the one step that says where an environment lives.

    A run answers viba data only, so the case files hand the environment a member
    answered to this step; its storage path is then the answer, and the member is
    still what named the child.
    """
    host = Host()

    def get_func(module_path, func_name):
        if func_name == "path_of":
            return lambda env, where: where.storage.cur_storage_path
        return host.get_func(module_path, func_name)

    return Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))


def _path_of(result) -> str:
    """Where the environment the run named lives: the path is the answer itself."""
    return value_of(result)


def run(tmp: Path):
    _the_same_child(tmp)
    _the_same_temporary_child(tmp)
    _the_first_argument_is_the_receiver(tmp)
    _a_member_that_takes_its_owner()
    _a_member_taken_by_a_name()
    _the_tag_is_not_a_value()
    _answering_the_environment_is_no_answer(tmp)
    _what_is_not_there_is_reported(tmp)


def _the_same_child(tmp: Path):
    """`$sub_env << args.env << "child"` 和 `args.env.sub_env << args.env << "child"` 是同一次调用。"""
    by_tag = interpret(str(CASES / "child_by_tag.viba"), environ_for())
    by_dot = interpret(str(CASES / "child_by_dot.viba"), environ_for())
    check(is_ok(by_tag) and isinstance(value_of(by_tag), str),
          f"taking the member by tag answers where that child lives: {by_tag!r}")
    check(_path_of(by_tag) == "root/child",
          f"and the name is the one given: {_path_of(by_tag)!r}")
    check(_path_of(by_tag) == _path_of(by_dot),
          f"the dotted spelling names the same child: {_path_of(by_dot)!r}")


def _the_same_temporary_child(tmp: Path):
    """空实参不用给：`$tmp_env << environ` 就是执行。"""
    result = interpret(str(CASES / "temporary_by_tag.viba"), environ_for())
    check(is_ok(result) and isinstance(value_of(result), str),
          f"a member that takes no argument runs when the member is taken: {result!r}")
    check(_path_of(result).startswith("root/tmp_"),
          f"and it is a temporary child: {_path_of(result)!r}")


def _the_first_argument_is_the_receiver(tmp: Path):
    """第一个参数是取成员的那个值：它就是最前面给出的那个。"""
    nested = interpret(str(CASES / "nested_by_tag.viba"), environ_for())
    check(is_ok(nested) and _path_of(nested) == "root/a/b",
          f"a child of a child: {nested!r}")

    by_tag = interpret(str(CASES / "argument_given_by_tag.viba"), environ_for())
    check(is_ok(by_tag) and _path_of(by_tag) == "root/kid",
          f"the first argument may be given by tag: {by_tag!r}")


def _a_member_that_takes_its_owner():
    """成员要 owner 时，两种源码形式是同一次调用：第一个参数也交给成员。"""
    def get_func(path, func_name):
        if func_name == "inc":
            # 收 owner、环境、x 三样
            return lambda box, environ, x: 2
        return Host().get_func(path, func_name)

    host = Environment(EnvironmentStorage("root", None, None), EnvironmentCompute(get_func))
    by_tag = interpret(str(CASES / "owner_member_by_tag.viba"), host)
    by_dot = interpret(str(CASES / "owner_member_by_dot.viba"), host)
    check(is_ok(by_tag) and value_of(by_tag) == 2,
          f"a member that takes its owner: {by_tag!r}")
    check(is_ok(by_dot) and value_of(by_dot) == 2,
          f"and the dotted spelling is the same call: {by_dot!r}")


def _judge(source: str, sub: str, sup: str):
    """`sub <: sup` judged in a module built from `source`."""
    module = custom_module(source)
    got = is_sub_type(
        AstNodeType(viba_ast.parse(f"__x__ = {sub}").body[0].body, module),
        AstNodeType(viba_ast.parse(f"__x__ = {sup}").body[0].body, module))
    return got.ok_value if isinstance(got, Ok) else got


def _a_member_taken_by_a_name():
    """名字出现在字符串里时，成员就按那个名字取：`tagged["f"]` 与 `$get_attr`。

    `tagged["f"] << box << …` 是 `$f << box << …`；`$get_attr << box << name << …`
    是 `box.f << …`，区别只在于名字是一份可以算出来的值。
    """
    def get_func(path, func_name):
        if func_name == "inc":
            # 成员要 owner、环境、x 三样（`$f << box << args.env << 1`）。
            return lambda box, environ, x: 2
        return Host().get_func(path, func_name)

    host = Environment(EnvironmentStorage("root", None, None), EnvironmentCompute(get_func))
    by_tag = interpret(str(CASES / "tagged_member.viba"), host)
    check(is_ok(by_tag) and value_of(by_tag) == 2,
          f"a tag given as a symbol takes the member it names: {by_tag!r}")
    by_name = interpret(str(CASES / "getattr_member.viba"), host)
    check(is_ok(by_name) and value_of(by_name) == 2,
          f"a member taken by the name a value spells: {by_name!r}")

    source = ("Args = Object * $a int * $b str\n"
              'Picked = $get_attr << Args << "a"\n'
              'Named = "a"\n'
              "By_name = $get_attr << Args << Named\n"
              "Dynamic = $get_attr << Args << SomethingElse\n")
    check(_judge(source, "Picked", "int") is True,
          "a name in the source picks the member it tags")
    check(_judge(source, "Picked", "str") is False,
          "and not another member")
    check(_judge(source, "By_name", "int") is True,
          "a name that stands for a string picks it too")
    check(_judge(source, "Dynamic", "Any") is True
          and _judge(source, "Dynamic", "int") is False,
          "a name no design can judge there counts as Any")


def _the_tag_is_not_a_value():
    """单独一个 `$tag` 编不过：它是链头的一种源码形式，不是值。"""
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


# 环境不是可序列化数据：把它当答案序列化成源码就不是值，整次运行停在程序错上。
ANSWERS_THE_ENVIRONMENT = """__decl__ =
    Any
  <- $env Env

args = __get_args__ << __decl__

__impl__ = $sub_env << args.env << "child"
"""


def _answering_the_environment_is_no_answer(tmp: Path):
    """环境只是这次调用的规则，不是值：答出环境就是 `$viba_program_err`。"""
    stopped = exec(ANSWERS_THE_ENVIRONMENT, environ_for())
    check(stop_tag(stopped) == PROGRAM_ERR_TAG
          and "the run answered the environment" in message_of(stopped),
          f"a run that answers the environment stops: {stopped!r}")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-member-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

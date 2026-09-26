"""`$tag << X << …`：从第一个参数身上取成员，再往后给。

`$sub_env << environ << "child"` 就是 `environ.sub_env << environ << "child"`，`$tmp_sub_env << environ`
就是 `environ.tmp_sub_env << environ`。tag 不是值（`method = $sub_env` 编不过），只有把它写在链头、后面
跟着第一个参数时才有意义；第一个参数既是被取成员的那个值，也是交给成员的第一个实参。

    python3 tests/test_interpreter_member.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host

from viba import viba_ast
from viba.interpret import Environment, interpret
from viba.type import Ok

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
    _the_tag_is_not_a_value()
    _what_is_not_there_is_reported(tmp)


def _the_same_child(tmp: Path):
    """`$sub_env << environ << "child"` 和 `environ.sub_env << environ << "child"` 是同一次调用。"""
    by_tag = interpret(str(CASES / "child_by_tag.viba"), environ_for())
    by_dot = interpret(str(CASES / "child_by_dot.viba"), environ_for())
    check(isinstance(by_tag, Ok) and isinstance(by_tag.ok_value, Environment),
          f"taking the member by tag answers an environment: {by_tag!r}")
    check(_path_of(by_tag) == "root/child",
          f"and the name is the one written: {_path_of(by_tag)!r}")
    check(_path_of(by_tag) == _path_of(by_dot),
          f"the dotted spelling names the same child: {_path_of(by_dot)!r}")


def _the_same_temporary_child(tmp: Path):
    """空实参不用写：`$tmp_sub_env << environ` 就是执行。"""
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

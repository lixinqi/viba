"""`Optional[T]`: 内建的那一个，就是 `T | nil`。

`Optional` 是 `viba/builtin.viba` 里的一条泛型定义，所以每个模块都看得见（最低优先级），
不用 import；它跟别的定义一样展开成自己的体，判定、JSON Schema、反射走的都是同一条路。
一个模块自己给出 `Optional`，裸名 `Optional` 就是它自己的那一个。

    python3 tests/test_optional.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import viba_ast
from viba.convert_to_json_schema import convert_to_json_schema
from viba.is_sub_type import is_sub_type
from viba.reflect import VibaData, access, by_tag
from viba.type import Ok, custom_module, entry_type
from viba.viba_type_descriptor import (empty_pool, parse_viba_file, pool_add_file,
                                       pool_find_definition)

PASS = FAIL = 0


def check(ok: bool, label: str) -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def _both(a, b, label: str) -> None:
    """The two types are one: each falls into the other."""
    first = is_sub_type(a, b)
    second = is_sub_type(b, a)
    check(isinstance(first, Ok) and first.ok_value
          and isinstance(second, Ok) and second.ok_value,
          f"{label}: {first!r} / {second!r}")


def _fits(sub, sup, label: str, want: bool = True, module=None) -> None:
    got = is_sub_type(entry_type(sub, module), entry_type(sup, module))
    check(isinstance(got, Ok) and got.ok_value is want,
          f"{label}: {sub} <: {sup} should be {want}, got {got!r}")


def run_judgment() -> None:
    _both(entry_type("Optional[int]"), entry_type("int | nil"),
          "Optional[int] is int | nil")
    _fits("int", "Optional[int]", "a present value")
    _fits("nil", "Optional[int]", "the absent value")
    _fits("str", "Optional[int]", "a value of another type", want=False)
    _fits("Optional[int]", "int", "Optional[int] is wider than int", want=False)
    _fits("Optional[Optional[int]]", "int | nil", "a nested Optional")
    _fits("Optional[int]", "Optional[str]", "two actuals differ", want=False)


def run_own_name() -> None:
    """一个模块自己的 `Optional` 赢过内建的这一个。"""
    module = custom_module("Optional[T] = $mine T\nX = Optional[int]\n")
    _both(entry_type("X", module), entry_type("$mine int"), "the module's own Optional")
    _fits("nil", "X", "and the builtin nil branch is not reached", want=False,
          module=module)


def run_json_schema() -> None:
    """`Optional[str]` 读回来是一个可为 null 的字符串。"""
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "main.viba").write_text("X = $maybe Optional[str]\n")
        answer = convert_to_json_schema([directory], ("module_name", "main"))
    if not isinstance(answer, Ok):
        check(False, f"Optional converts to a schema: {answer!r}")
        return
    document = json.loads(answer.ok_value)
    member = document["$defs"]["main.X"]["properties"]["maybe"]
    key = member["$ref"].split("/")[-1]
    check(document["$defs"][key] == {"oneOf": [{"type": "string"}, {"type": "null"}]},
          f"a nullable string: {document['$defs'][key]!r}")


def run_reflection() -> None:
    """可序列化数据里，`Optional[str]` 这一位收得下 nil，也收得下字符串。"""
    source = "X = $maybe Optional[str]\n"
    pool = empty_pool()
    parsed = parse_viba_file(pool, source, "main.viba", "main")
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    definition = pool_find_definition(pool, "main.X").ok_value
    for label, value, want in (("absent", viba_ast.Nil(), None),
                               ("present", viba_ast.Constant("s"), "s")):
        data = viba_ast.ProductChain([viba_ast.Tagged("$maybe", value)])
        root = access.root(definition, VibaData(data))
        member = access.get(root.ok_value, by_tag("$maybe")) \
            if isinstance(root, Ok) else root
        if not isinstance(root, Ok) or not isinstance(member, Ok):
            check(False, f"{label}: {root!r} / {member!r}")
            continue
        leaf = access.leaf(member.ok_value)
        check(isinstance(leaf, Ok) and leaf.ok_value == want,
              f"{label}: {leaf!r}, wanted {want!r}")


def run() -> None:
    run_judgment()
    run_own_name()
    run_json_schema()
    run_reflection()
    print(f"optional: {PASS} passed, {FAIL} failed")


if __name__ == "__main__":
    run()
    sys.exit(1 if FAIL else 0)

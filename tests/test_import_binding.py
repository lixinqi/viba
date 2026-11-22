"""`__import__`：运行时取一个模块。

`d = __import__ << args.env << "a.b.c"` 在跑的时候取模块：环境决定在哪儿找（`nil`
用这次运行自己的搜索路径），名字是 `str`，可以是数据。取到之后三种用法都要成立：
`d.X` 是那个模块的 `X`，`d << env` 是跑那个模块，`d[T]` 是泛型应用 ——
`import a.b.c as d` 仍是读文件时的绑定，类型也按它取。

    python3 tests/test_import_binding.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host, is_ok, message_of, value_of

from viba.wiki.convert_to_json_schema import convert_to_json_schema
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret)
from viba.is_complete import is_complete
from viba.type import Ok

CASES = Path(__file__).resolve().parent / "data" / "import_binding"

checks = Checks("import_binding")
check = checks.check


def environ(store, viba_path=None):
    def get_func(module_path, func_name):
        if func_name == "Target":
            return lambda env: 7
        if func_name == "Add":
            return lambda env, a, b: a.value + b.value
        if func_name == "Which":
            return lambda env: "target"
        if func_name == "choice":
            return lambda env: 9
        return Host().get_func(module_path, func_name)

    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(get_func),
                       viba_path=str(viba_path or CASES))


def run(tmp: Path) -> None:
    # The module itself: a literal name, `nil`, a name that is data, and the
    # environment's path deciding where it is looked up.
    for name in ("entry.viba", "entry_nil.viba"):
        result = interpret(str(CASES / name), environ(tmp))
        check(is_ok(result) and value_of(result) == 7,
              f"{name}: the module is taken while the run is going: {result!r}")
    dynamic = interpret(str(CASES / "entry_dynamic.viba"), environ(tmp))
    check(is_ok(dynamic) and value_of(dynamic) == 7,
          f"the module name may be data: {dynamic!r}")
    for directory, answer in (("one", 8), ("two", 11)):
        result = interpret(str(CASES / "entry_far.viba"),
                           environ(tmp, CASES / directory))
        check(is_ok(result) and value_of(result) == answer,
              f"the environment says where to look ({directory}): {result!r}")

    # `a.b`: a member of the module — called, taken as a value, or taken as a
    # closure and called later; a member it has not is reported.
    for name, answer, what in (
            ("entry_member.viba", 7, "a value of the module"),
            ("entry_member_closure.viba", 7,
             "a step taken as a value, then called"),
            ("entry_member_args.viba", 7, "a step called with arguments")):
        result = interpret(str(CASES / name), environ(tmp))
        check(is_ok(result) and value_of(result) == answer,
              f"{name}: {what}: {result!r}")
    missing = message_of(interpret(str(CASES / "entry_member_missing.viba"),
                                   environ(tmp))) or ""
    check("Nope" in missing,
          f"a member the module has not is reported: {missing!r}")

    # `a << b`: the module called, and with the arguments it is given; a generic
    # is no module to call.
    for name, answer, what in (
            ("entry_call.viba", 7, "the module runs"),
            ("entry_call_args.viba", 5, "the module takes its arguments")):
        result = interpret(str(CASES / name), environ(tmp))
        check(is_ok(result) and value_of(result) == answer,
              f"{name}: {what}: {result!r}")
    not_a_module = message_of(interpret(str(CASES / "entry_call_generic.viba"),
                                        environ(tmp))) or ""
    check("generic" in not_a_module and "application" in not_a_module,
          f"a generic is no module to call: {not_a_module!r}")

    # `a[b]`: the application as a value, a member of the chosen file, that
    # file's step, the file's module semantics, and an alias of the module.
    for name, answer, what in (
            ("entry_generic.viba", 7, "the chosen file's `__decl__`"),
            ("entry_generic_member.viba", 8, "a member of the chosen file"),
            ("entry_generic_call.viba", 9, "the chosen file's step"),
            ("entry_generic_member_call.viba", 9, "a step of the chosen file"),
            ("entry_generic_module.viba", 7, "a chosen file that is a module"),
            ("entry_generic_alias.viba", 7, "an alias of the module")):
        result = interpret(str(CASES / name), environ(tmp))
        check(is_ok(result) and value_of(result) == answer,
              f"{name}: {what}: {result!r}")
    no_pattern = message_of(interpret(str(CASES / "entry_generic_fail.viba"),
                                      environ(tmp))) or ""
    check("pattern" in no_pattern and "float" in no_pattern,
          f"a pattern no file answers is reported: {no_pattern!r}")
    bare = message_of(interpret(str(CASES / "entry_generic_bare.viba"),
                                environ(tmp))) or ""
    check("generic" in bare and "application" in bare,
          f"a generic is no name space: {bare!r}")

    # `import a.b as d` is still the binding it was: a type comes through it at
    # build time, which a run-time `__import__` cannot give (its answer is `Any`).
    for name, answer in (("entry_import.viba", 7), ("entry_call_import.viba", 7),
                         ("entry_generic_import.viba", 7)):
        result = interpret(str(CASES / name), environ(tmp))
        check(is_ok(result) and value_of(result) == answer,
              f"{name}: the `import` spelling gives the same answer: {result!r}")
    answer = convert_to_json_schema([str(CASES)],
                                    ("type_names", ["entry_import.X"]))
    check(isinstance(answer, Ok), f"the imported type has a schema: {answer!r}")
    if isinstance(answer, Ok):
        schema = json.loads(answer.ok_value)
        check(schema["$defs"]["target.TargetType"]["properties"]
              == {"x": {"type": "integer"}},
              f"the schema is the target's type: {schema!r}")

    bad = (CASES / "bad.viba").read_text()
    check(not is_complete(bad, viba_paths=[str(CASES)]),
          "a module taken at run time is no design type")
    check("nope" in (message_of(interpret(str(CASES / "bad.viba"),
                                          environ(tmp))) or ""),
          "a missing module is reported by name")
    check("str" in (message_of(interpret(str(CASES / "entry_bad_name.viba"),
                                         environ(tmp))) or ""),
          "a module name that is no string is reported")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="viba-import-binding-") as scratch:
        run(Path(scratch))
    return checks.report()


if __name__ == "__main__":
    sys.exit(main())

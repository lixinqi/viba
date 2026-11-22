"""__import__：`d = __import__ << "a.b.c"`读成 `import a.b.c as d`。

`d.TargetType` 是 `a.b.c.TargetType`，`d` 本身是那个模块；`import a.b.c as d`
写出来的同一个绑定也一样。

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


def environ(store):
    def get_func(module_path, func_name):
        if func_name == "Target":
            return lambda env: 7
        return Host().get_func(module_path, func_name)

    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(get_func), viba_path=str(CASES))


def run(tmp: Path) -> None:
    for name in ("entry.viba", "entry_import.viba"):
        result = interpret(str(CASES / name), environ(tmp))
        check(is_ok(result) and value_of(result) == 7,
              f"{name}: d.Target is the target module's step: {result!r}")

    answer = convert_to_json_schema([str(CASES)], ("type_names", ["entry.X"]))
    check(isinstance(answer, Ok), f"the imported type has a schema: {answer!r}")
    if isinstance(answer, Ok):
        schema = json.loads(answer.ok_value)
        check(schema["$defs"]["target.TargetType"]["properties"]
              == {"x": {"type": "integer"}},
              f"the schema is the target's type: {schema!r}")

    bad = (CASES / "bad.viba").read_text()
    check(not is_complete(bad, viba_paths=[str(CASES)]),
          "a missing module is not complete")
    check("nope" in (message_of(interpret(str(CASES / "bad.viba"),
                                          environ(tmp))) or ""),
          "a missing module is reported by name")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="viba-import-binding-") as scratch:
        run(Path(scratch))
    return checks.report()


if __name__ == "__main__":
    sys.exit(main())

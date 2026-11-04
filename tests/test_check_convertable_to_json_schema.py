"""Tests for check_convertable_to_json_schema: can this type be written as a JSON Schema?

Every case is one module of data/check_convertable_to_json_schema/kb:

    legal.viba      sums that read back: tags, kinds, nested sums, literals
    illegal.viba    sums that do not: two ints, int|float, 7|int, "x"|"x",
                    a tuple against a list (both arrays),
                    two products
    no_value.viba   no JSON value at all: Any, a function, ...
    products.viba   one tag twice once the inline slot is spread
    cycle.viba      an inline chain that comes back to itself
    use.viba        a sum that flattens through an import
    generics.viba   a generic applied, a parameter left open, Either
    uses_elem.viba  a generic written as one Markdown document (elem.md)
    empty.viba      a module with no definitions

    python3 tests/test_check_convertable_to_json_schema.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.check_convertable_to_json_schema import check_convertable_to_json_schema
from viba.type import Ok, Err

DATA = Path(__file__).resolve().parent / "data" / "check_convertable_to_json_schema" / "kb"

PASS = 0
FAIL = 0


def check(name: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {name}\n  expected: {want!r}\n  got:      {got!r}")


def judge(entry, strict=None):
    return check_convertable_to_json_schema([str(DATA)], entry, strict=strict)


def types(*names):
    return ("type_names", list(names))


def reads(name: str) -> None:
    got = judge(types(name))
    check(f"{name} reads back", isinstance(got, Ok) and got.ok_value is None, True)


def refused(name: str, said: str, entry=None) -> None:
    got = judge(entry if entry is not None else types(name))
    if not isinstance(got, Err):
        check(f"{name} is refused", got, f"Err naming {said!r}")
        return
    check(f"{name} is refused for {said!r}", said in got.msg, True)


def run_reading() -> None:
    """Sums a JSON value can land on."""
    for name in ("legal.A", "legal.B", "legal.C", "legal.D", "legal.E",
                 "illegal.B", "illegal.C", "use.Use"):
        reads(name)


def run_refused() -> None:
    """Sums a JSON value cannot land on."""
    refused("illegal.A", "B and C")
    refused("illegal.A", "read as a number")
    refused("illegal.N", "int and float")
    refused("illegal.L", "7 and int")
    refused("illegal.Same", '"x" and "x"')
    refused("illegal.Objects", "read as an object")
    refused("illegal.Tupled", "read as an array")
    refused("use.Repeated", "int and int")
    refused("illegal.IntKeys", "a JSON object's keys are strings")


def run_no_value() -> None:
    """A type that is no JSON value is not read back at all."""
    refused("no_value.X", "Any is no JSON value")
    refused("no_value.F", "a function is no JSON value")
    refused("no_value.E", "... is no JSON value")
    refused("generics.Open", "Any is no JSON value")
    refused("generics.Box", "the generic parameter 'T' has no argument")


def run_products() -> None:
    """The product side is check_tag_and_inline's."""
    refused("products.Child", "the tag $a appears twice in one product")
    refused("cycle.Loop", "the inline chain comes back to 'Loop'")


def run_generics() -> None:
    reads("generics.Filled")
    reads("generics.Choice")
    refused("generics.Numeric", "read as a number")
    # `elem` is a generic written as one Markdown document (elem.md)
    reads("uses_elem.Item")
    reads("uses_elem.Pair")


def run_entries() -> None:
    legal = judge(("module_name", "legal"))
    check("module legal", isinstance(legal, Ok) and legal.ok_value is None, True)
    refused("module illegal", "read as a number",
            entry=("module_name", "illegal"))
    refused("module empty", "defines no type", entry=("module_name", "empty"))
    refused("module missing", "cannot load the module 'missing'",
            entry=("module_name", "missing"))
    refused("module products", "the tag $a appears twice in one product",
            entry=("module_name", "products"))

    bad = judge(types("legal.A", "illegal.N"))
    check("two names: the bad one decides",
          isinstance(bad, Err) and "int and float" in bad.msg, True)
    both = judge(types("legal.A", "legal.B"))
    check("two names, both good",
          isinstance(both, Ok) and both.ok_value is None, True)
    refused("legal.Nope", "no type 'legal.Nope'")
    refused("A", "is no type name")


def run_strict() -> None:
    """A program definition: passed over by default, refused under strict."""
    loose = judge(("module_name", "program"))
    check("a program module is judged by its types", isinstance(loose, Ok), True)
    waited = judge(("module_name", "program"), strict=True)
    check("strict refuses the function",
          isinstance(waited, Err) and "a function is no JSON value" in waited.msg,
          True)
    check("a named type is judged either way",
          isinstance(judge(("type_names", ["program.Report"]), strict=True), Ok),
          True)
    only = judge(("module_name", "functions"))
    check("a module of functions holds no type",
          isinstance(only, Err) and "the entry holds no type" in only.msg, True)
    strict_only = judge(("module_name", "functions"), strict=True)
    check("strict refuses it as a function",
          isinstance(strict_only, Err)
          and "a function is no JSON value" in strict_only.msg, True)
    check("strict is a bool or nil",
          isinstance(judge(("module_name", "program"), strict="yes"), Err),
          True)


def run_units() -> None:
    """The units a design writes: documentation slots, and a block sum's head."""
    for name in ("units.X", "units.Y", "units.Z", "units.Maybe", "units.Level",
                 "units.Docs", "units.Hinted", "units.Comment",
                 "units.Asserted", "units.Appendixed", "units.UnitOption"):
        check(f"{name} is accepted",
              isinstance(judge(("type_names", [name])), Ok), True)


def run_arguments() -> None:
    refused("entry is not a tuple", 'entry is ("module_name", name) or ("type_names", [names])',
            entry="legal")
    refused("entry has no such branch",
            "'types' is no branch of entry: module_name or type_names",
            entry=("types", ["legal.A"]))
    refused("module_name is a string", "module_name is a string, not int",
            entry=("module_name", 1))
    refused("type_names is a list", "type_names is a list of names, not one name",
            entry=("type_names", "legal.A"))
    got = check_convertable_to_json_schema(str(DATA), ("module_name", "legal"))
    check("viba_path is a list",
          isinstance(got, Err)
          and "viba_path is a list of directories, not one directory" in got.msg, True)
    got = check_convertable_to_json_schema([1], ("module_name", "legal"))
    check("viba_path holds directories",
          isinstance(got, Err)
          and "viba_path holds a directory, not int" in got.msg, True)


def main() -> int:
    run_reading()
    run_refused()
    run_no_value()
    run_products()
    run_generics()
    run_entries()
    run_strict()
    run_units()
    run_arguments()
    print(f"check_convertable_to_json_schema: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

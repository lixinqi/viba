"""Tests for convert_to_json_schema: the JSON Schema a type answers to.

The cases are the modules of data/check_convertable_to_json_schema — the same types the
judgment is tested on, because a type with no schema and a type that cannot go
to JSON are the same types.

    python3 tests/test_convert_to_json_schema.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.convert_to_json_schema import convert_to_json_schema
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


def convert(entry, strict=None):
    return convert_to_json_schema([str(DATA)], entry, strict=strict)


def schema(name: str) -> dict:
    """The document for one type; an empty document when it did not convert."""
    answer = convert(("type_names", [name]))
    if not isinstance(answer, Ok):
        check(f"{name} converts", answer.msg, "a schema")
        return {}
    check(f"{name} ends with a newline", answer.ok_value.endswith("\n"), True)
    return json.loads(answer.ok_value)


def definition(doc: dict, name: str) -> dict:
    return doc.get("$defs", {}).get(name, {})


def refs(node):
    """Every `$ref` string in a document, wherever it sits."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            else:
                yield from refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from refs(item)


def refs_resolve(doc: dict) -> bool:
    names = set(doc.get("$defs", {}))
    return all(ref.startswith("#/$defs/") and ref[len("#/$defs/"):] in names
               for ref in refs(doc))


def tagged(key: str, body: dict) -> dict:
    return {"type": "object", "properties": {key: body}, "required": [key],
            "additionalProperties": False}


def run_sums() -> None:
    doc = schema("legal.A")
    check("a tagged sum's root", doc.get("$ref"), "#/$defs/legal.A")
    check("a tagged sum has one entry per option",
          len(definition(doc, "legal.A").get("oneOf", [])), 2)
    check("a tagged sum's options", definition(doc, "legal.A")["oneOf"],
          [tagged("a", {"type": "integer"}), tagged("b", {"type": "integer"})])
    check("a tagged sum's refs resolve", refs_resolve(doc), True)

    check("a sum of kinds",
          definition(schema("legal.B"), "legal.B")["oneOf"],
          [{"type": "integer"}, {"type": "string"}])

    kinds = [one.get("type") or "tagged"
             for one in definition(schema("legal.C"), "legal.C")["oneOf"]]
    check("an array, an object and null", kinds, ["object", "object", "null"])

    check("a nested sum is flattened, not nested",
          len(definition(schema("legal.D"), "legal.D")["oneOf"]), 3)

    check("two literals", definition(schema("legal.E"), "legal.E")["oneOf"],
          [{"const": "a"}, {"const": "b"}])

    check("a null option", definition(schema("objects.Maybe"), "objects.Maybe")["oneOf"],
          [{"type": "integer"}, {"type": "null"}])


def run_units() -> None:
    """The language's units: a documentation slot is no member, no option either."""
    doc = schema("units.X")
    check("a tagged member that is one value", definition(doc, "units.X"),
          {"type": "object", "properties": {"a": {"type": "string"}},
           "required": ["a"], "additionalProperties": False})
    doc = schema("units.Y")
    check("a Hint slot is no member", definition(doc, "units.Y"),
          {"type": "object", "properties": {"a": {"type": "string"}},
           "required": ["a"], "additionalProperties": False})
    doc = schema("units.Z")
    check("a block sum keeps its options",
          len(definition(doc, "units.Z").get("oneOf", [])), 2)
    check("never is no option",
          False in definition(doc, "units.Z").get("oneOf", []), False)
    doc = schema("units.Maybe")
    check("nil is an option", definition(doc, "units.Maybe").get("oneOf"),
          [{"type": "integer"}, {"type": "null"}])
    doc = schema("units.Level")
    check("a documented constant is that constant",
          definition(doc, "units.Level").get("oneOf"),
          [{"const": "S0"}, {"const": "S1"}])
    doc = schema("units.Docs")
    check("a bare hint is no member", definition(doc, "units.Docs"),
          {"type": "object",
           "properties": {"runs": {"type": "array", "items": {"type": "string"}}},
           "required": ["runs"], "additionalProperties": False})
    check("Hint writes the same schema",
          definition(schema("units.Hinted"), "units.Hinted"),
          definition(doc, "units.Docs"))
    doc = schema("units.Comment")
    check("a hint alone is the unit", definition(doc, "units.Comment"),
          {"type": "null"})
    for name in ("units.Asserted", "units.Appendixed"):
        check(f"{name} keeps its member", definition(schema(name), name),
              {"type": "object", "properties": {"a": {"type": "integer"}},
               "required": ["a"], "additionalProperties": False})
    doc = schema("units.UnitOption")
    check("a unit option is null", definition(doc, "units.UnitOption").get("oneOf"),
          [{"type": "integer"}, {"type": "null"}])


def run_generics() -> None:
    """A generic written as one Markdown document answers what its directory would."""
    doc = schema("uses_elem.Item")
    check("a list's element is the type it stands for",
          definition(doc, "uses_elem.Item"),
          {"$ref": "#/$defs/elem.100.type@3a92772d"})
    check("every ref in the document resolves", refs_resolve(doc), True)
    doc = schema("uses_elem.Pair")
    check("a dict's keys and values are a tuple",
          doc["$defs"].get("elem.200.type@7a123e56"),
          {"type": "array", "prefixItems": [{"type": "string"},
                                            {"type": "integer"}],
           "items": False, "minItems": 2})


def run_products() -> None:
    point = definition(schema("objects.Point"), "objects.Point")
    check("a product is an object", point.get("type"), "object")
    check("a product's properties", point.get("properties"),
          {"x": {"type": "number"}, "y": {"type": "number"}})
    check("a product's keys are all required", point.get("required"), ["x", "y"])
    check("a product takes no other key", point.get("additionalProperties"), False)

    inline = definition(schema("objects.Inline"), "objects.Inline")
    check("an inline slot is spread", list(inline.get("properties", {})),
          ["x", "y", "label"])
    check("an inline slot's keys are required too", inline.get("required"),
          ["x", "y", "label"])

    pair = definition(schema("objects.Pair"), "objects.Pair")
    check("a tuple is an array of fixed length",
          [pair.get("type"), len(pair.get("prefixItems", [])),
           pair.get("items"), pair.get("minItems")],
          ["array", 2, False, 2])

    check("a bare tag is an object", definition(schema("objects.Tagged"),
                                                "objects.Tagged"),
          tagged("a", {"type": "integer"}))


def run_references() -> None:
    node = schema("objects.Node")
    check("a recursive type refs its own definition",
          {"$ref": "#/$defs/objects.Node"} in node["$defs"]["objects.Node"]
          ["properties"]["next"]["oneOf"], True)
    check("a recursive type resolves", refs_resolve(node), True)

    choice = schema("generics.Choice")
    keys = [key for key in choice.get("$defs", {})
            if key.startswith("EitherImpl.200.type@")]
    check("a decided generic's answer is one definition", len(keys), 1)
    if keys:
        check("a decided generic's answer is the sum",
              choice["$defs"][keys[0]], {"oneOf": [{"type": "integer"},
                                                   {"type": "string"}]})
        check("the entry refs it", choice.get("$ref"), "#/$defs/generics.Choice")
        check("the entry's definition refs the answer",
              choice["$defs"]["generics.Choice"], {"$ref": "#/$defs/" + keys[0]})

    filled = schema("generics.Filled")
    box = [key for key in filled.get("$defs", {}) if key.startswith("generics.Box@")]
    check("a generic application is one definition", len(box), 1)
    check("a generic application's arguments reach the properties",
          filled["$defs"][box[0]]["properties"], {"some": {"type": "integer"}})


def run_entries() -> None:
    whole = convert(("module_name", "legal"))
    if not isinstance(whole, Ok):
        check("a module converts", whole.msg, "a schema")
        return
    doc = json.loads(whole.ok_value)
    check("a module's root is its first definition", doc.get("$ref"),
          "#/$defs/legal.A")
    check("a module's definitions are all there",
          sorted(name for name in doc["$defs"] if "." not in name.split("@")[0]
                 or name.startswith("legal.")),
          ["legal.A", "legal.B", "legal.C", "legal.D", "legal.E"])
    check("a module's refs resolve", refs_resolve(doc), True)

    both = convert(("type_names", ["legal.A", "objects.Point"]))
    if isinstance(both, Ok):
        doc = json.loads(both.ok_value)
        check("two types: the root is the first", doc.get("$ref"), "#/$defs/legal.A")
        check("two types: both are defined",
              {"legal.A", "objects.Point"} <= set(doc["$defs"]), True)
    else:
        check("two types convert", both.msg, "a schema")


def run_errors() -> None:
    answer = convert(("type_names", ["illegal.N"]))
    check("a type that cannot go to JSON has no schema",
          isinstance(answer, Err)
          and "int and float" in answer.msg, True)
    answer = convert(("module_name", "nope"))
    check("a module that is not there",
          isinstance(answer, Err)
          and "cannot load the module" in answer.msg, True)
    answer = convert(("module_name", "empty"))
    check("a module with no definitions",
          isinstance(answer, Err)
          and "defines no type" in answer.msg, True)
    answer = convert(("type_names", ["legal.Nope"]))
    check("a type that is not there",
          isinstance(answer, Err) and "no type" in answer.msg, True)
    answer = convert("legal")
    check("an entry that is no branch",
          isinstance(answer, Err), True)


def run_strict() -> None:
    """What a program definition does: passed over, or refused under strict."""
    loose = convert(("module_name", "program"))
    if not isinstance(loose, Ok):
        check("a program module converts", loose.msg, "a schema")
    else:
        doc = json.loads(loose.ok_value)
        check("a program module keeps its type", "program.Report" in doc["$defs"], True)
        check("and leaves its program out",
              [name for name in doc["$defs"] if "__" in name], [])
        check("a program module's refs resolve", refs_resolve(doc), True)

    waited = convert(("module_name", "program"), strict=True)
    check("strict refuses the program definition",
          isinstance(waited, Err) and "a function is no JSON value" in waited.msg,
          True)

    by_type = convert(("type_names", ["program.Report"]), strict=True)
    check("strict takes a type that is data",
          isinstance(by_type, Ok), True)

    only_functions = convert(("module_name", "functions"))
    check("a module of functions holds no type",
          isinstance(only_functions, Err)
          and "the entry holds no type" in only_functions.msg, True)
    strict_functions = convert(("module_name", "functions"), strict=True)
    check("strict refuses it as a function",
          isinstance(strict_functions, Err)
          and "a function is no JSON value" in strict_functions.msg, True)

    wrong = convert(("module_name", "program"), strict="yes")
    check("strict is a bool or nil",
          isinstance(wrong, Err)
          and "strict is a bool or nil, not str" in wrong.msg, True)
    false_too = convert(("module_name", "program"), strict=False)
    check("False is the default", isinstance(false_too, Ok), True)


def main() -> int:
    run_sums()
    run_generics()
    run_units()
    run_products()
    run_references()
    run_entries()
    run_strict()
    run_errors()
    print(f"convert_to_json_schema: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

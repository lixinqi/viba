"""viba.convert_to_json_schema — the JSON Schema a design's types answer to.

    from viba.convert_to_json_schema import convert_to_json_schema

    answer = convert_to_json_schema(["kb"], ("type_names", ["report.Report"]))
    answer.ok_value        # the schema, as JSON text

The entry is the same one `check_json_serdeable` takes: a module name, or a
group of type names. The judgment runs first — a design that cannot go to JSON
and come back has no schema worth writing — and its error is the answer.

What comes out is one JSON Schema document (2020-12):

    {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "$ref": "#/$defs/report.Report",
      "$defs": {
        "report.Report": { ... },
        "item.Item": { ... }
      }
    }

Every definition the entry reaches gets a `$defs` entry, keyed by its whole
name, and is referenced by `$ref` — that is what keeps a recursive type
finite. The root `$ref` is the first entry (for a module, its first
definition); the rest are there under their names.

How a type is written:

- `int` / `float` / `str` / `bool` → `{"type": "integer" | "number" | "string" |
  "boolean"}`; `nil` / `Object` → `{"type": "null"}`; a literal → `{"const": …}`;
  `never` → `false` (no value at all).
- `list[T]` → `{"type": "array", "items": …}`; `set[T]` the same with
  `"uniqueItems": true`; `dict[str, T]` → `{"type": "object",
  "additionalProperties": …}` (a JSON object's keys are strings).
- a product → `{"type": "object", "properties": {…}, "required": […],
  "additionalProperties": false}`; a member without a tag is spread into the
  product it stands in (an inline slot), and a positional member is refused
  (it has no key).
- a tuple → `{"type": "array", "prefixItems": […], "items": false,
  "minItems": n}`.
- a sum → `{"oneOf": […]}`, one entry per option.
- a tagged option (or a bare `$tag T`) → `{"type": "object",
  "properties": {"tag": …}, "required": ["tag"], "additionalProperties":
  false}`, the tag's `$` left off.
- a name that stands for a definition → `{"$ref": "#/$defs/<name>"}`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional

from viba import viba_ast
from viba.check_json_serdeable import (CheckErr, argument_problem,
                                        check_json_serdeable, entry_definitions)
from viba.design_walk import (
    chain_elements,
    is_definition,
    module_label,
    split_type_name,
    with_params,
)
from viba.interpret import _Runner
from viba.pattern import decide, generic_named
from viba.type import (
    AstNodeType,
    CustomModuleType,
    Ok,
    Result,
    Err,
    module_get_type,
)
from viba.viba_ast.tagged import TAGGED_NAME

__all__ = ["convert_to_json_schema"]

DIALECT = "https://json-schema.org/draft/2020-12/schema"

NULL_NAMES = ("nil", "Object", "void", "None")
NEVER_NAMES = ("never", "Oneof")
SCALAR_TYPES = {"int": "integer", "float": "number", "str": "string",
                "bool": "boolean"}
ARRAY_CONSTRUCTORS = ("list", "set", "ListLiteral", "SetLiteral")
OBJECT_CONSTRUCTORS = ("dict", "DictLiteral")


class _SchemaErr(Exception):
    """A piece of the design with no JSON Schema to stand for it."""


def convert_to_json_schema(viba_path, entry, strict=None) -> Result:
    """Result[str]: the JSON Schema the entry answers to, or why there is none.

    `viba_path` and `entry` are `check_json_serdeable`'s: the search path an
    import is resolved on, and (`module_name`, name) or (`type_names`, names).
    The judgment runs first — over what is converted — and its error is the
    answer when the design cannot be read back from JSON.

    `strict` says what to do with an entry that is a function (`A <- B`) or a
    call (`f << ...`): None, or False, is the default and passes those
    definitions over — a module that is a program keeps its `__decl__` and
    `__impl__` out of the document — while True leaves them to the judgment,
    which refuses them. A function an extracted type still reaches is no schema
    either way.
    """
    if not (strict is None or isinstance(strict, bool)):
        return Err(f"strict is a bool or nil, not {type(strict).__name__}")
    problem = argument_problem(viba_path, entry)
    if problem is not None:
        return Err(problem)
    tag, value = entry
    tag = tag.lstrip("$")
    builder = _Builder(_Runner(":".join(viba_path) if viba_path else None))
    try:
        definitions = entry_definitions(builder.loader, tag, value, bool(strict))
        checked = check_json_serdeable(viba_path, entry, strict=strict)
        if isinstance(checked, Err):
            return checked
        entries = [(builder.full_name(module, definition), definition, module)
                   for definition, module in definitions]
        document = builder.document(entries)
    except (CheckErr, _SchemaErr) as mistake:
        return Err(str(mistake))
    return Ok(json.dumps(document, ensure_ascii=False, indent=2) + "\n")


class _Builder:
    """Walks the entry and writes the document: one `$defs` entry per definition."""

    def __init__(self, loader):
        self.loader = loader
        self.defs: Dict[str, dict] = {}
        self.root: Optional[str] = None

    # ---- the document ----

    def document(self, definitions) -> dict:
        refs = [self._schema_of(definition, module)
                for _full_name, definition, module in definitions]
        document = {"$schema": DIALECT}
        if refs:
            document["$ref"] = refs[0]["$ref"]
        document["$defs"] = self.defs
        return document

    def _load(self, name: str, near) -> CustomModuleType:
        loaded = self.loader.imported(name, near)
        if isinstance(loaded, Err):
            raise _SchemaErr(f"cannot load the module {name!r}: {loaded.msg}")
        module = loaded.ok_value
        if not isinstance(module, CustomModuleType):
            raise _SchemaErr(
                f"{name!r} names a generic: a generic answers an application, "
                f"not a bare name")
        return module

    def full_name(self, module, definition) -> str:
        name = getattr(module, "name", "") or "?"
        return f"{name}.{definition.name}"

    # ---- definitions ----

    def _schema_of(self, definition, module, bindings: dict = None) -> dict:
        """The definition's schema, as a `$ref`; a definition stands once per binding.

        A definition reached with arguments bound (a generic's answer, a pattern
        file's `type`) is a different schema for another set of arguments, so its
        key carries a digest of them: `report.Box@1f2e3d4c`. One without
        bindings is keyed by its whole name.
        """
        key = self._key_of(definition, module, bindings)
        if key in self.defs:
            return _ref(key)
        self.defs[key] = {}
        schema = self._schema(definition.body, module,
                              with_params(definition, dict(bindings or {})))
        self.defs[key].update(schema)
        return _ref(key)

    def _key_of(self, definition, module, bindings: dict = None) -> str:
        full_name = self.full_name(module, definition)
        if not bindings:
            return full_name
        return f"{full_name}@{_bindings_digest(bindings)}"

    # ---- the pieces ----

    def _schema(self, node, module, bindings: dict) -> dict:
        if isinstance(node, viba_ast.Constant):
            return {"const": node.value}
        if isinstance(node, viba_ast.Nil):
            return {"type": "null"}
        if isinstance(node, viba_ast.Never):
            return False                    # a value of never is no value
        if isinstance(node, (viba_ast.Any, viba_ast.Ellipsis, viba_ast.CodeBlock)):
            raise _SchemaErr(f"{_spell(node)} has no JSON Schema")
        if isinstance(node, viba_ast.TypeRef):
            return self._named(node.name, module, bindings)
        if isinstance(node, viba_ast.Tagged):
            return self._tagged(node.tag, node.type, module, bindings)
        if isinstance(node, viba_ast.MemberTaken):
            return self._member(node, module, bindings)
        if isinstance(node, viba_ast.TypeApp):
            return self._application(node, module, bindings)
        if isinstance(node, viba_ast.Tuple):
            return {"type": "array",
                    "prefixItems": [self._schema(element, module, bindings)
                                    for element in node.elements],
                    "items": False,
                    "minItems": len(node.elements)}
        if isinstance(node, (viba_ast.Partial, viba_ast.Exponent)):
            raise _SchemaErr(f"{_spell(node)} is no JSON value")
        elements = chain_elements(node)
        if elements is None:
            raise _SchemaErr(f"{_spell(node)} has no JSON Schema")
        if isinstance(node, (viba_ast.Sum, viba_ast.SumChain)):
            return {"oneOf": self._sum_options(elements, module, bindings)}
        if isinstance(node, (viba_ast.Product, viba_ast.ProductChain)):
            return self._product(elements, module, bindings)
        raise _SchemaErr(f"{_spell(node)} has no JSON Schema")

    def _sum_options(self, elements, module, bindings: dict) -> List[dict]:
        """One schema per option, a sum that stands in a sum taken together with it."""
        options: List[dict] = []
        for element in elements:
            if isinstance(element, (viba_ast.Sum, viba_ast.SumChain)):
                options.extend(self._sum_options(
                    chain_elements(element) or [], module, bindings))
                continue
            options.append(self._schema(element, module, bindings))
        return options

    def _named(self, name: str, module, bindings: dict) -> dict:
        if name in bindings:
            bound = bindings[name]
            if bound is None:
                raise _SchemaErr(
                    f"the generic parameter {name!r} has no argument")
            return self._schema(bound[0], bound[1], bound[2])
        if name in NULL_NAMES:
            return {"type": "null"}
        if name in NEVER_NAMES:
            return False
        if name in SCALAR_TYPES:
            return {"type": SCALAR_TYPES[name]}
        found = module_get_type(module, name)
        if isinstance(found, Err):
            raise _SchemaErr(
                f"cannot resolve {name!r} in {module_label(module)}: {found.msg}")
        target = found.ok_value
        if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
            raise _SchemaErr(f"{name!r} has no JSON Schema")
        return self._schema_of(target.ast_node, target.container_module)

    def _tagged(self, tag: str, body, module, bindings: dict) -> dict:
        key = _tag_name(tag)
        return {"type": "object",
                "properties": {key: self._schema(body, module, bindings)},
                "required": [key],
                "additionalProperties": False}

    def _member(self, node, module, bindings: dict) -> dict:
        source_path = viba_ast.source_path(node)
        if source_path is not None:
            return self._named(source_path, module, bindings)
        owner = node.owner
        if not isinstance(owner, viba_ast.TypeApp):
            raise _SchemaErr(f"{_spell(node)} has no JSON Schema")
        generic = generic_named(module, owner.constructor)
        if isinstance(generic, Err):
            raise _SchemaErr(
                f"cannot take {owner.constructor!r} in {module_label(module)}: "
                f"{generic.msg}")
        if generic.ok_value is None:
            raise _SchemaErr(f"{_spell(node)} has no JSON Schema")
        got = decide(generic.ok_value, owner.args, module,
                     argument_modules=self._argument_modules(owner, bindings))
        if isinstance(got, Err) or got.ok_value is None:
            raise _SchemaErr(f"cannot decide {_spell(owner)}")
        choice = got.ok_value
        if choice.body is not None:
            raise _SchemaErr(f"{_spell(node)} has no JSON Schema")
        inner = dict(bindings)
        for name, bound in choice.bindings.items():
            inner[name] = (bound.ast_node, bound.container_module, bindings)
        for statement in choice.module.module.body:
            if is_definition(statement) and statement.name == node.name:
                return self._schema_of(statement, choice.module, inner)
        raise _SchemaErr(f"no definition {node.name!r} in {module_label(choice.module)}")

    def _application(self, node, module, bindings: dict) -> dict:
        constructor = node.constructor
        if constructor == TAGGED_NAME:
            symbol = self._symbol(node.args[0], bindings)
            if symbol is None:
                raise _SchemaErr(f"{_spell(node)} has no tag to write")
            return self._tagged(symbol, node.args[1], module, bindings)
        if constructor in ARRAY_CONSTRUCTORS:
            schema = {"type": "array",
                      "items": self._schema(node.args[0], module, bindings)}
            if constructor in ("set", "SetLiteral"):
                schema["uniqueItems"] = True
            return schema
        if constructor in OBJECT_CONSTRUCTORS:
            return {"type": "object",
                    "additionalProperties": self._schema(node.args[1], module,
                                                         bindings)}
        if constructor in NULL_NAMES:
            return {"type": "null"}
        if constructor in SCALAR_TYPES:
            return {"type": SCALAR_TYPES[constructor]}
        generic = generic_named(module, constructor)
        if isinstance(generic, Err):
            raise _SchemaErr(
                f"cannot take {constructor!r} in {module_label(module)}: {generic.msg}")
        if generic.ok_value is None:
            found = module_get_type(module, constructor)
            if isinstance(found, Err):
                raise _SchemaErr(
                    f"cannot resolve {constructor!r} in {module_label(module)}: "
                    f"{found.msg}")
            target = found.ok_value
            if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
                raise _SchemaErr(f"{_spell(node)} has no JSON Schema")
            definition = target.ast_node
            params = list(getattr(definition, "generic_params", None) or ())
            if len(params) != len(node.args):
                raise _SchemaErr(f"{_spell(node)}: {definition.name!r} takes "
                                 f"{len(params)} parameters")
            inner = dict(bindings)
            for param, argument in zip(params, node.args):
                inner[param] = (argument, module, bindings)
            return self._schema_of(definition, target.container_module, inner)
        got = decide(generic.ok_value, node.args, module,
                     argument_modules=self._argument_modules(node, bindings))
        if isinstance(got, Err):
            raise _SchemaErr(f"cannot decide {_spell(node)}: {got.msg}")
        choice = got.ok_value
        if choice is None or choice.body is None:
            raise _SchemaErr(f"{_spell(node)}: the chosen file answers a member")
        inner = dict(bindings)
        for name, bound in choice.bindings.items():
            inner[name] = (bound.ast_node, bound.container_module, bindings)
        return self._schema(choice.body, choice.module, inner)

    def _product(self, elements, module, bindings: dict) -> dict:
        properties: Dict[str, dict] = {}
        required: List[str] = []
        self._members(elements, module, bindings, properties, required)
        return {"type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False}

    def _members(self, elements, module, bindings: dict,
                 properties: Dict[str, dict], required: List[str]) -> None:
        for element in elements:
            if isinstance(element, viba_ast.Tagged):
                key = _tag_name(element.tag)
                properties[key] = self._schema(element.type, module, bindings)
                required.append(key)
                continue
            form, form_module, form_bindings = self._stands_for(
                element, module, bindings)
            if self._is_unit(form) or _is_unit_name(form):
                continue                    # a unit is no member
            inner = chain_elements(form)
            if inner is not None and isinstance(form, (viba_ast.Product,
                                                       viba_ast.ProductChain)):
                head, rest = inner[0], inner[1:]
                if _is_unit_name(head) or self._is_unit(head):
                    inner = rest
                self._members(inner, form_module, form_bindings,
                              properties, required)
                continue
            raise _SchemaErr(
                f"the product's member {_spell(element)} has no tag: "
                f"a JSON object needs a key")

    def _stands_for(self, node, module, bindings: dict):
        """The form an inline slot stands for: (the form, the module, the bindings)."""
        seen = set()
        while True:
            if id(node) in seen:
                return node, module, bindings
            seen.add(id(node))
            if isinstance(node, viba_ast.TypeRef):
                bound = bindings.get(node.name)
                if isinstance(bound, tuple):
                    node, module, bindings = bound
                    continue
                if node.name in bindings:
                    return node, module, bindings
                found = module_get_type(module, node.name)
                if isinstance(found, Err):
                    raise _SchemaErr(
                        f"cannot resolve {node.name!r} in "
                        f"{module_label(module)}: {found.msg}")
                target = found.ok_value
                if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
                    return node, module, bindings
                definition = target.ast_node
                node = definition.body
                module = target.container_module
                bindings = with_params(definition, {})
                continue
            if isinstance(node, viba_ast.TypeApp):
                if node.constructor in (TAGGED_NAME, *ARRAY_CONSTRUCTORS,
                                        *OBJECT_CONSTRUCTORS):
                    return node, module, bindings
                generic = generic_named(module, node.constructor)
                if isinstance(generic, Err):
                    raise _SchemaErr(
                        f"cannot take {node.constructor!r} in "
                        f"{module_label(module)}: {generic.msg}")
                if generic.ok_value is None:
                    return node, module, bindings
                got = decide(generic.ok_value, node.args, module,
                             argument_modules=self._argument_modules(node, bindings))
                if isinstance(got, Err) or got.ok_value is None:
                    return node, module, bindings
                choice = got.ok_value
                if choice.body is None:
                    return node, module, bindings
                inner = dict(bindings)
                for name, bound in choice.bindings.items():
                    inner[name] = (bound.ast_node, bound.container_module, bindings)
                node, module, bindings = choice.body, choice.module, inner
                continue
            return node, module, bindings

    def _is_unit(self, form) -> bool:
        return isinstance(form, (viba_ast.Nil, viba_ast.Never))

    def _symbol(self, node, bindings: dict):
        if isinstance(node, viba_ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, viba_ast.TypeRef):
            bound = bindings.get(node.name)
            if isinstance(bound, tuple):
                return self._symbol(bound[0], bound[2])
        return None

    def _argument_modules(self, node, bindings: dict):
        kept = []
        for argument in node.args:
            where = None
            if isinstance(argument, viba_ast.TypeRef):
                bound = bindings.get(argument.name)
                if isinstance(bound, tuple):
                    where = bound[1]
            kept.append(where)
        return kept if any(where is not None for where in kept) else None


def _is_unit_name(node) -> bool:
    return isinstance(node, viba_ast.TypeRef) and node.name in NULL_NAMES + NEVER_NAMES


def _tag_name(tag: str) -> str:
    """The key a tag writes as: its name, the `$` left off."""
    return tag[1:] if tag.startswith("$") else tag


def _ref(full_name: str) -> dict:
    return {"$ref": "#/$defs/" + full_name.replace("~", "~0").replace("/", "~1")}


def _spell(node) -> str:
    try:
        return viba_ast.unparse_type(node)
    except Exception:
        return type(node).__name__


def _bindings_digest(bindings: dict) -> str:
    """A short, stable key for the arguments a definition stands in."""
    parts = []
    for name in sorted(bindings):
        bound = bindings[name]
        if bound is None:
            parts.append(f"{name}=?")
            continue
        module = getattr(bound[1], "name", "") or ""
        parts.append(f"{name}={viba_ast.dump(bound[0])}@{module}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8]

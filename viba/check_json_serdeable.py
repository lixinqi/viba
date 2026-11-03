"""viba.check_json_serdeable — can this design round-trip through JSON?

    from viba.check_json_serdeable import check_json_serdeable

    check_json_serdeable(["kb"], ("type_names", ["report.Report"]))
    # Ok(None), or Err(the first sum that cannot be read back)

A viba value is written as JSON-looking data: a product is an object, a list is
an array, a scalar is a scalar. Reading a value back means picking an option of
a sum from what the JSON value is — and that only works where the options can be
told apart by what JSON has: null, a boolean, a number, a string, an array, an
object, or the tag an option carries (the key of its object).

So the judgment walks the type the entry names and asks of every sum it reaches
that no two options read as the same thing:

    Good = $a int | $b int     # fine: two tags
    B = int
    C = int
    Bad = B | C                # not: both are a number

A literal and a type of the same kind read as the same thing too (`7` and
`int`): the value 7 could be either. Two literals of one kind are told apart by
their values, so `"a" | "b"` is fine.

The rest of the type has to be a JSON value at all: a function, `Any`, `...`,
a call and a bare code block are none, and a generic parameter with no argument
is none yet. A name that does not resolve is a mistake, and so is a generic
application the source cannot decide.

The product side — one tag per product, inline chains that end — is
`viba.check_tag_and_inline.check_tag_and_inline`, run over the same design
first; a design that fails that check is not judged further.

What is read is the design the `viba_path` reaches, through the interpreter's
own loader, so names resolve where a run would resolve them. Nothing is run.

The unit names are the language's own (`reflect.language_config`): a name that
stands for `nil` or `never` is a unit, and a unit carries no JSON value of its
own beyond `null` (and `never` carries none at all).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from viba import viba_ast
from viba.check_tag_and_inline import check_tag_and_inline
from viba.design_walk import (
    is_function,
    bindings_key,
    chain_elements,
    is_definition,
    module_label,
    split_type_name,
    under_builtin,
    with_params,
)
from viba.interpret import _Runner
from viba.pattern import GenericModuleType, decide, generic_named
from viba.type import (
    AstNodeType,
    CustomModuleType,
    Ok,
    Result,
    Err,
    module_get_type,
)
from viba.viba_ast.tagged import TAGGED_NAME
from viba.viba_type_descriptor import empty_pool, parse_viba_file, pool_add_file

__all__ = ["check_json_serdeable", "entry_definitions", "argument_problem", "CheckErr"]

# The language's units (reflect.language_config): a name standing for `nil` or
# `never`, so a unit rather than a definition of its own.
NIL_EQV = ("Object", "Assert", "Appendix", "Hint")
NEVER_EQV = ("Oneof",)

# The names a value of the two units reads as: null, or nothing at all.
NULL_NAMES = ("nil", "Object", "void", "None")
NO_VALUE_NAMES = ("never", "Oneof")
NUMBER_NAMES = ("int", "float")
SCALAR_NAMES = {"int": "number", "float": "number", "str": "string", "bool": "boolean"}
ARRAY_CONSTRUCTORS = ("list", "set", "ListLiteral", "SetLiteral")
OBJECT_CONSTRUCTORS = ("dict", "DictLiteral")


class CheckErr(Exception):
    """The first thing that keeps this design from being read back from JSON."""


def check_json_serdeable(viba_path, entry, strict=None) -> Result:
    """Result[None]: Ok(None) when every sum the entry reaches reads unambiguously.

    `viba_path` is the search path an import is resolved on, in order.
    `entry` is one of the two branches the design gives:

        ("module_name", "pkg.a")          # every definition of that module
        ("type_names", ["pkg.a.Foo"])     # those definitions

    `strict` says what to do with an entry definition that is a function
    (`A <- B`) or a call (`f << ...`): None, or False, is the default and
    passes those definitions over — a module that is a program is judged by its
    types alone — while True judges them, and a function is no JSON value. A
    function a judged type still reaches is refused either way.

    A `Err` says what is wrong first: a name that does not resolve, a
    value JSON has none of, or two sum options a JSON value cannot tell apart.
    """
    if not (strict is None or isinstance(strict, bool)):
        return Err(f"strict is a bool or nil, not {type(strict).__name__}")
    problem = argument_problem(viba_path, entry)
    if problem is not None:
        return Err(problem)
    tag, value = entry
    tag = tag.lstrip("$")
    loader = _Runner(":".join(viba_path) if viba_path else None)
    checker = _Checker(loader)
    first = None
    try:
        definitions = entry_definitions(loader, tag, value, bool(strict))
        checker.check(definitions)
    except CheckErr as mistake:
        first = mistake
    design = _design_pool(checker.loader)
    if isinstance(design, Err):
        return design
    tagged = check_tag_and_inline(design.ok_value)
    if isinstance(tagged, Err):
        return tagged
    if first is not None:
        return Err(str(first))
    return Ok(None)


def argument_problem(viba_path, entry) -> Optional[str]:
    if viba_path is None:
        return "viba_path is a list of directories"
    if isinstance(viba_path, str):
        return "viba_path is a list of directories, not one directory"
    try:
        paths = list(viba_path)
    except TypeError:
        return f"viba_path is a list of directories, not {type(viba_path).__name__}"
    for one in paths:
        if not isinstance(one, str):
            return f"viba_path holds a directory, not {type(one).__name__}"
    if not (isinstance(entry, tuple) and len(entry) == 2
            and isinstance(entry[0], str)):
        return ('entry is ("module_name", name) or ("type_names", [names])')
    tag, value = entry
    tag = tag.lstrip("$")
    if tag == "module_name":
        if not isinstance(value, str):
            return f"module_name is a string, not {type(value).__name__}"
        return None
    if tag == "type_names":
        if isinstance(value, str):
            return "type_names is a list of names, not one name"
        try:
            names = list(value)
        except TypeError:
            return f"type_names is a list of names, not {type(value).__name__}"
        for one in names:
            if not isinstance(one, str):
                return f"type_names holds a name, not {type(one).__name__}"
        return None
    return f"{tag!r} is no branch of entry: module_name or type_names"


def entry_definitions(loader, tag: str, value, strict: bool):
    """The definitions an entry names: (the definition, the module it is in).

    A module names its every definition; `type_names` names each by its whole
    name. Without `strict` a definition that is a function or a call is passed
    over, and an entry left with nothing is an error.
    """
    if tag == "module_name":
        module = _load(loader, value, None)
        definitions = [(statement, module)
                       for statement in module.module.body
                       if is_definition(statement)]
        if not definitions:
            raise CheckErr(f"the module {value!r} defines no type")
        if not strict:
            definitions = [entry for entry in definitions
                           if not is_function(entry[0])]
            if not definitions:
                raise CheckErr("the entry holds no type")
        return definitions
    else:
        definitions = []
        for name in value:
            module_name, definition_name = split_type_name(name)
            if not module_name:
                raise CheckErr(
                    f"{name!r} is no type name: it is a module and a "
                    f"definition, like 'pkg.mod.Name'")
            module = _load(loader, module_name, None)
            found = module_get_type(module, definition_name)
            if isinstance(found, Err):
                raise CheckErr(f"no type {name!r}: {found.msg}")
            target = found.ok_value
            if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
                raise CheckErr(f"{name!r} is no definition")
            definitions.append((target.ast_node, target.container_module))
    return definitions


def _load(loader, name: str, near) -> CustomModuleType:
    loaded = loader.imported(name, near)
    if isinstance(loaded, Err):
        raise CheckErr(f"cannot load the module {name!r}: {loaded.msg}")
    module = loaded.ok_value
    if not isinstance(module, CustomModuleType):
        raise CheckErr(
            f"{name!r} names a generic: a generic answers an application, "
            f"not a bare name")
    return module


class _Checker:
    """Walks one entry through the design and checks every sum it reaches."""

    def __init__(self, loader):
        self.loader = loader
        self.nil_eqv = set(NIL_EQV)
        self.never_eqv = set(NEVER_EQV)
        self.visited = set()

    # ---- the walk ----

    def check(self, definitions) -> None:
        for definition, module in definitions:
            self._definition(definition, module)

    def _load(self, name: str, near) -> CustomModuleType:
        loaded = self.loader.imported(name, near)
        if isinstance(loaded, Err):
            raise CheckErr(f"cannot load the module {name!r}: {loaded.msg}")
        module = loaded.ok_value
        if not isinstance(module, CustomModuleType):
            raise CheckErr(
                f"{name!r} names a generic: a generic answers an application, "
                f"not a bare name")
        return module

    # ---- the walk ----

    def _definition(self, definition, module: CustomModuleType,
                    bindings: dict = None) -> None:
        bindings = dict(bindings or {})
        key = (module.file_path, definition.name, bindings_key(bindings))
        if key in self.visited:
            return                      # coinduction: the same walk again
        self.visited.add(key)
        self._walk(definition.body, module, with_params(definition, bindings))

    def _walk(self, node, module, bindings: dict) -> None:
        if node is None:
            return
        if isinstance(node, (viba_ast.Constant, viba_ast.Nil, viba_ast.Never)):
            return
        if isinstance(node, (viba_ast.Any, viba_ast.Ellipsis)):
            raise CheckErr(
                f"{_spell(node)} is no JSON value")
        if isinstance(node, viba_ast.CodeBlock):
            raise CheckErr("a code block is no JSON value")
        if isinstance(node, viba_ast.Member):
            raise CheckErr(f"{_spell(node)} is no JSON value")
        if isinstance(node, viba_ast.TypeRef):
            self._name(node.name, module, bindings)
            return
        if isinstance(node, viba_ast.MemberTaken):
            self._member(node, module, bindings)
            return
        if isinstance(node, viba_ast.Tagged):
            self._walk(node.type, module, bindings)
            return
        if isinstance(node, viba_ast.TypeApp):
            self._application(node, module, bindings)
            return
        if isinstance(node, viba_ast.Tuple):
            for element in node.elements:
                self._walk(element, module, bindings)
            return
        if isinstance(node, viba_ast.Partial):
            raise CheckErr("a call is no JSON value")
        if isinstance(node, viba_ast.Exponent):
            raise CheckErr("a function is no JSON value")
        elements = chain_elements(node)
        if elements is None:
            raise CheckErr(f"{_spell(node)} is no JSON value")
        if isinstance(node, (viba_ast.Sum, viba_ast.SumChain)):
            self._sum(node, elements, module, bindings)
            return
        for element in elements:
            self._walk(element, module, bindings)

    def _name(self, name: str, module, bindings: dict) -> None:
        if name in bindings:
            bound = bindings[name]
            if bound is None:
                raise CheckErr(
                    f"the generic parameter {name!r} has no argument: "
                    f"no JSON value yet")
            self._walk(bound[0], bound[1], bound[2])
            return
        if name in self.never_eqv or name in ("never", "Oneof"):
            return                              # a value of never is no value
        if name in self.nil_eqv or name in NULL_NAMES:
            return                              # a unit: null
        if name in SCALAR_NAMES:
            return
        found = module_get_type(module, name)
        if isinstance(found, Err):
            raise CheckErr(
                f"cannot resolve {name!r} in {module_label(module)}: {found.msg}")
        target = found.ok_value
        if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
            return                              # a builtin form: a leaf
        self._definition(target.ast_node, target.container_module)

    def _application(self, node, module, bindings: dict) -> None:
        constructor = node.constructor
        if constructor == TAGGED_NAME:
            for argument in node.args[1:]:
                self._walk(argument, module, bindings)
            return
        if constructor in ARRAY_CONSTRUCTORS or constructor in ("dict", "DictLiteral"):
            for argument in node.args:
                self._walk(argument, module, bindings)
            if constructor in ("dict", "DictLiteral"):
                key = node.args[0]
                form, form_module, form_bindings = self._stands_for(
                    key, module, bindings)
                read = self._read_as(form, form_module, form_bindings)
                if read is None or read[0] != "string":
                    raise CheckErr(
                        f"a JSON object's keys are strings: {_spell(key)} is not")
            return
        if constructor in self.nil_eqv or constructor in self.never_eqv:
            return                              # a unit with a document block
        if constructor in SCALAR_NAMES or constructor in NULL_NAMES:
            return
        self._note_application(node, module, bindings)

    def _note_application(self, node, module, bindings: dict) -> None:
        constructor = node.constructor
        generic = generic_named(module, constructor)
        if isinstance(generic, Err):
            raise CheckErr(
                f"cannot take {constructor!r} in {module_label(module)}: {generic.msg}")
        if generic.ok_value is None:
            # A plain application: a generic definition of this module, a name
            # that stands for a type, or something JSON has no value for.
            found = module_get_type(module, constructor)
            if isinstance(found, Err):
                raise CheckErr(
                    f"cannot resolve {constructor!r} in "
                    f"{module_label(module)}: {found.msg}")
            target = found.ok_value
            if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
                raise CheckErr(
                    f"{_spell(node)} is no JSON value")
            definition = target.ast_node
            params = list(getattr(definition, "generic_params", None) or ())
            if len(params) != len(node.args):
                raise CheckErr(
                    f"{_spell(node)}: {definition.name!r} takes "
                    f"{len(params)} parameters")
            inner = dict(bindings)
            for param, argument in zip(params, node.args):
                inner[param] = (argument, module, bindings)
            self._definition(definition, target.container_module, inner)
            return
        got = decide(generic.ok_value, node.args, module,
                     argument_modules=self._argument_modules(node, bindings))
        if isinstance(got, Err):
            raise CheckErr(
                f"cannot decide {_spell(node)}: {got.msg}")
        choice = got.ok_value
        if choice is None or choice.body is None:
            raise CheckErr(
                f"{_spell(node)}: the chosen file answers a member, not a type")
        inner = dict(bindings)
        for name, bound in choice.bindings.items():
            inner[name] = (bound.ast_node, bound.container_module, bindings)
        self._walk(choice.body, choice.module, inner)

    def _member(self, node, module, bindings: dict) -> None:
        source_path = viba_ast.source_path(node)
        if source_path is not None:
            self._name(source_path, module, bindings)
            return
        owner = node.owner
        if not isinstance(owner, viba_ast.TypeApp):
            raise CheckErr(f"{_spell(node)} is no JSON value")
        generic = generic_named(module, owner.constructor)
        if isinstance(generic, Err):
            raise CheckErr(
                f"cannot take {owner.constructor!r} in {module_label(module)}: "
                f"{generic.msg}")
        if generic.ok_value is None:
            raise CheckErr(f"{_spell(node)} has no JSON form of its own")
        got = decide(generic.ok_value, owner.args, module,
                     argument_modules=self._argument_modules(owner, bindings))
        if isinstance(got, Err):
            raise CheckErr(f"cannot decide {_spell(owner)}: {got.msg}")
        choice = got.ok_value
        if choice is None:
            raise CheckErr(f"{_spell(node)} has no JSON form of its own")
        inner = dict(bindings)
        for name, bound in choice.bindings.items():
            inner[name] = (bound.ast_node, bound.container_module, bindings)
        if choice.body is not None:
            raise CheckErr(f"{_spell(node)} has no JSON form of its own")
        for statement in choice.module.module.body:
            if is_definition(statement) and statement.name == node.name:
                self._definition(statement, choice.module, inner)
                return
        raise CheckErr(f"no definition {node.name!r} in {module_label(choice.module)}")

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

    # ---- sums ----

    def _sum(self, node, elements, module, bindings: dict) -> None:
        options = []
        for element in elements:
            form, form_module, form_bindings = self._stands_for(
                element, module, bindings)
            options.extend(self._flatten(form, form_module, form_bindings,
                                        {id(node)}, _spell(element)))
        reads = []
        for form, form_module, form_bindings, spelling in options:
            read = self._read_as(form, form_module, form_bindings)
            for other_spelling, other_read in reads:
                if _same_read(read, other_read):
                    # The option the source has first is named first.
                    raise CheckErr(_clash(other_spelling, spelling, read))
            reads.append((spelling, read))
        for element in elements:
            self._walk(element, module, bindings)

    def _flatten(self, form, module, bindings, seen, spelling) -> List[tuple]:
        """One sum's options, the sums among them taken together with it."""
        if not isinstance(form, (viba_ast.Sum, viba_ast.SumChain)):
            return [(form, module, bindings, spelling)]
        if id(form) in seen:
            return []                       # a sum that comes back to itself
        seen = seen | {id(form)}
        options = []
        for element in chain_elements(form) or []:
            inner, inner_module, inner_bindings = self._stands_for(
                element, module, bindings)
            options.extend(self._flatten(inner, inner_module, inner_bindings,
                                         seen, _spell(element)))
        return options

    def _stands_for(self, node, module, bindings: dict):
        """The form an option stands for: (the form, the module, the bindings)."""
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
                if node.name in bindings or node.name in self.nil_eqv:
                    return node, module, bindings
                if node.name in self.never_eqv or node.name in SCALAR_NAMES:
                    return node, module, bindings
                found = module_get_type(module, node.name)
                if isinstance(found, Err):
                    raise CheckErr(
                        f"cannot resolve {node.name!r} in {module_label(module)}: "
                        f"{found.msg}")
                target = found.ok_value
                if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
                    return node, module, bindings
                definition = target.ast_node
                node = definition.body
                module = target.container_module
                bindings = with_params(definition, {})
                continue
            if isinstance(node, viba_ast.MemberTaken):
                source_path = viba_ast.source_path(node)
                if source_path is not None:
                    node = viba_ast.TypeRef(source_path)
                    continue
                return node, module, bindings
            if isinstance(node, viba_ast.TypeApp):
                if node.constructor == TAGGED_NAME:
                    return node, module, bindings
                if (node.constructor in ARRAY_CONSTRUCTORS
                        or node.constructor in OBJECT_CONSTRUCTORS):
                    return node, module, bindings
                generic = generic_named(module, node.constructor)
                if isinstance(generic, Err):
                    raise CheckErr(
                        f"cannot take {node.constructor!r} in "
                        f"{module_label(module)}: {generic.msg}")
                if generic.ok_value is not None:
                    got = decide(generic.ok_value, node.args, module,
                                 argument_modules=self._argument_modules(node, bindings))
                    if isinstance(got, Err):
                        raise CheckErr(f"cannot decide {_spell(node)}: {got.msg}")
                    choice = got.ok_value
                    if choice is None or choice.body is None:
                        return node, module, bindings
                    inner = dict(bindings)
                    for name, bound in choice.bindings.items():
                        inner[name] = (bound.ast_node, bound.container_module, bindings)
                    node, module, bindings = choice.body, choice.module, inner
                    continue
                found = module_get_type(module, node.constructor)
                if isinstance(found, Err):
                    raise CheckErr(
                        f"cannot resolve {node.constructor!r} in "
                        f"{module_label(module)}: {found.msg}")
                target = found.ok_value
                if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
                    return node, module, bindings
                definition = target.ast_node
                params = list(getattr(definition, "generic_params", None) or ())
                if len(params) != len(node.args):
                    return node, module, bindings
                inner = dict(bindings)
                for param, argument in zip(params, node.args):
                    inner[param] = (argument, module, bindings)
                node, module, bindings = (definition.body, target.container_module,
                                          with_params(definition, inner))
                continue
            return node, module, bindings

    def _read_as(self, form, module, bindings: dict):
        """What a JSON value of this option is: ("tag", name) or (kind, value)."""
        if isinstance(form, viba_ast.Tagged):
            return ("tag", form.tag)
        if isinstance(form, viba_ast.Constant):
            return (_literal_kind(form.value), form.value)
        if isinstance(form, viba_ast.Nil):
            return ("null", None)
        if isinstance(form, viba_ast.Never):
            return None                     # a value of never is no value
        if isinstance(form, (viba_ast.Product, viba_ast.ProductChain,
                             viba_ast.Tuple)):
            return ("object", None)
        if isinstance(form, viba_ast.TypeRef):
            name = form.name
            if name in bindings:
                bound = bindings[name]
                if bound is None:
                    raise CheckErr(
                        f"the generic parameter {name!r} has no argument: "
                        f"no JSON value yet")
                return self._read_as(bound[0], bound[1], bound[2])
            if name in NULL_NAMES or name in self.nil_eqv:
                return ("null", None)
            if name in NO_VALUE_NAMES or name in self.never_eqv:
                return None
            if name in SCALAR_NAMES:
                return (SCALAR_NAMES[name], None)
            return self._read_resolved(name, module)
        if isinstance(form, viba_ast.TypeApp):
            constructor = form.constructor
            if constructor in ARRAY_CONSTRUCTORS:
                return ("array", None)
            if constructor in OBJECT_CONSTRUCTORS:
                return ("object", None)
            if constructor == TAGGED_NAME:
                symbol = self._symbol(form.args[0], module, bindings)
                return ("tag", symbol) if symbol is not None else None
            return None
        return None

    def _read_resolved(self, name: str, module):
        found = module_get_type(module, name)
        if isinstance(found, Err):
            raise CheckErr(
                f"cannot resolve {name!r} in {module_label(module)}: {found.msg}")
        target = found.ok_value
        if not isinstance(target, AstNodeType) or not is_definition(target.ast_node):
            return None
        definition = target.ast_node
        form, form_module, form_bindings = self._stands_for(
            viba_ast.TypeRef(name), module, {})
        return self._read_as(form, form_module, form_bindings)

    def _symbol(self, node, module, bindings: dict):
        if isinstance(node, viba_ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, viba_ast.TypeRef):
            bound = bindings.get(node.name)
            if isinstance(bound, tuple):
                return self._symbol(bound[0], bound[1], bound[2])
        return None


def _literal_kind(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "literal"


def _read_name(read) -> str:
    tag, value = read
    if tag == "tag":
        return f"the tag {value!r}"
    if tag == "literal":
        return f"the value {value!r}"
    return {"number": "a number", "string": "a string", "boolean": "a boolean",
            "null": "null", "array": "an array", "object": "an object"}.get(tag, tag)


def _clash(one: str, other: str, read) -> str:
    if read is not None and read[0] == "tag":
        return f"the sum's options {one} and {other} carry the same tag {read[1]!r}"
    return (f"the sum's options {one} and {other} both read as "
            f"{_read_name(read)} in JSON: a tag tells them apart")


def _same_read(one, other) -> bool:
    """Can one JSON value be both? Two tags answer yes when they are one tag."""
    if one is None or other is None:
        return False
    if one[0] == "tag" or other[0] == "tag":
        return one == other
    if one[0] != other[0]:
        return False
    one_value, other_value = one[1], other[1]
    if one_value is None or other_value is None:
        return True                         # a kind against a value of that kind
    return one_value == other_value


def _spell(node) -> str:
    try:
        return viba_ast.unparse_type(node)
    except Exception:                        # a form with no spelling: say the kind
        return type(node).__name__


# ----------------------------------------------------------------------
# The design: the modules the walk loaded, for the product check
# ----------------------------------------------------------------------


def _design_pool(loader) -> Result:
    """A pool of the modules this walk loaded, for check_tag_and_inline.

    The builtin vocabulary is not part of it: what the package ships is the
    interpreter's own design, not the caller's. Pattern files are in the
    loader's names (`G.100`), so the pool sees the same generics.
    """
    pool = empty_pool()
    files = set()
    for name, module in list(loader.by_name.items()):
        if isinstance(module, GenericModuleType):
            marker = str(Path(module.directory) / "__generic__.viba")
            if marker in files or under_builtin(marker):
                continue
            added = _add_file(pool, marker, f"{name}.__generic__")
            if isinstance(added, Err):
                continue
            pool = added.ok_value
            files.add(marker)
            continue
        if not isinstance(module, CustomModuleType):
            continue
        file_path = module.file_path
        if not file_path or file_path in files or under_builtin(file_path):
            continue
        added = _add_file(pool, file_path, name)
        if isinstance(added, Err):
            # A file the descriptor pool cannot build holds a program (its
            # `__impl__` is a call), not a design it could check: passed over.
            continue
        pool = added.ok_value
        files.add(file_path)
    return Ok(pool)


def _add_file(pool, file_path: str, module_name: str) -> Result:
    source = Path(file_path).read_text(encoding="utf-8")
    parsed = parse_viba_file(pool, source, file_path, module_name)
    if isinstance(parsed, Err):
        return parsed
    return pool_add_file(pool, parsed.ok_value)

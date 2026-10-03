"""Specialization: a generic is a directory, and one of its files answers.

The four worked examples of `viba-specialize.md` live in `tests/data/specialize/demo/`:

    demo/is_base_type/      100 restricts to bool | int | float | str, 200 extracts
    demo/is_compatable/     four files, an equality among them
    demo/element_type_of/   list[A] -> A, set[A] -> A, dict[A, B] -> (A, B)
    demo/ret_type_of/       A <- (() | nil) -> A, then A <- B -> A, then
                            A <- B <- C -> A: the chain's length picks the file,
                            and the first file's restriction is what the pattern
                            matcher checks on its own (its answer is the second
                            file's answer too, so the order between the two is
                            only observable where the files answer differently —
                            `is_base_type[bool]` is where that is pinned)
    demo/num_generic_args/  dict[A, B] -> 2, list[A] -> 1, A -> 0: the pattern
                            itself holds the structure that is counted
    demo/num_variadic_args/ one file per arity, so [] -> 0, [A] -> 1, [A, B] -> 2

`demo/wrapper/` is the one file whose answer is its own definition, so the
call's parameter has to reach into that definition too. `broken/` and
`loose_specialize.viba` are the mistakes: a file named anything but its order,
a directory with no marker, a file with no `__def__`, a decision that fails, a
pattern count that is not the argument count, a marker that writes `specialize`,
and `specialize` written outside a generic's directory.

Read where a design is read (the judgment and the descriptor pool), a generic
application is the type its decision picked; read where a program runs (the
interpreter), it is that type as viba data.

    python3 tests/test_specialize.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import Checks, Host, value_of

from viba import viba_ast
from viba.interpret import interpret
from viba.is_complete import is_complete
from viba.is_sub_type import is_sub_type
from viba.reflect import VibaData, access, by_tag
from viba.specialize import (GENERIC_FILE, load_generic, patterns_of,
                             structural_pattern_match)
from viba.type import AstNodeType, Ok, VibaProgramErr, custom_module
from viba.viba_type_descriptor import (empty_pool, parse_viba_file,
                                       pool_add_file, pool_find_definition)

CASES = Path(__file__).resolve().parent / "data" / "specialize"

checks = Checks("specialize")
check = checks.check
labelled = checks.labelled


def _case(name: str) -> str:
    return str(CASES / f"{name}.viba")


# (文件, 跑出来该是什么值, 这是哪一条)
RUNTIME_CASES = [
    ("is_base_type_bool", True, "is_base_type[bool]"),
    ("is_base_type_list", False, "is_base_type[list[int]]"),
    ("is_base_type_str", True, "is_base_type[str]"),
    ("is_compatable_bool_int", True, "is_compatable[bool, int]"),
    ("is_compatable_int_bool", True, "is_compatable[int, bool]"),
    ("is_compatable_lists", True, "is_compatable[list[int], list[int]]"),
    ("is_compatable_float_str", False, "is_compatable[float, str]"),
    ("num_generic_args_dict", 2, "num_generic_args[dict[int, str]]"),
    ("num_generic_args_list", 1, "num_generic_args[list[int]]"),
    ("num_generic_args_set", 1, "num_generic_args[set[float]]"),
    ("num_generic_args_bool", 0, "num_generic_args[bool]"),
    ("num_generic_args_function", 0, "num_generic_args[int <- int]"),
    ("num_variadic_args_none", 0, "num_variadic_args[]"),
    ("num_variadic_args_one", 1, "num_variadic_args[bool]"),
    ("num_variadic_args_two", 2, "num_variadic_args[bool, str]"),
    ("num_variadic_args_three", 3, "num_variadic_args[int, str, list[int]]"),
]

# (文件, 答出来的类型写成什么, 这是哪一条)
EXTRACTED_CASES = [
    ("element_of_a_list", "int", "element_type_of[list[int]]"),
    ("element_of_a_set", "float", "element_type_of[set[float]]"),
    ("element_of_a_dict", "(int, str)", "element_type_of[dict[int, str]]"),
    ("element_of_a_local_name", "Local", "the caller's own name is kept"),
    ("element_of_a_wrapped", "$item int", "the chosen file's definition, bound"),
    ("ret_of_int_int", "int", "ret_type_of[int <- int]"),
    ("ret_of_float_unit", "float", "ret_type_of[float <- ()]"),
    ("ret_of_three", "(int, str)", "ret_type_of[(int, str) <- bool <- str]"),
]


def run(scratch: Path):
    _the_runtime_answers()
    _what_was_extracted()
    _the_judgment_reads_it()
    _the_pattern_matcher()
    _the_arity_is_the_file()
    _the_decision_fails_loudly()
    _a_generic_is_no_module()
    _a_function_type_is_no_value()
    _a_design_is_complete_through_it()
    _the_pool_holds_a_generic()
    _the_reflection_reads_it()


def _environ():
    return Host().environ(viba_path=str(CASES))


def _the_runtime_answers():
    """一份模块应用一个泛型：选中的那一支就是它的答案。"""
    environ = _environ()
    for name, want, label in RUNTIME_CASES:
        result = interpret(_case(name), environ)
        check(isinstance(result, Ok) and value_of(result) is want,
              f"{label} -> {want}: {result!r}")


def _what_was_extracted():
    """萃取到的类型，就是答案里写下来的那份。"""
    environ = _environ()
    for name, want, label in EXTRACTED_CASES:
        result = interpret(_case(name), environ)
        got = (viba_ast.unparse_type(result.ok_value.data)
               if isinstance(result, Ok) else repr(result))
        check(isinstance(result, Ok) and got == want,
              f"{label} -> {want}: {got!r}")


# ----------------------------------------------------------------------
# The judgment and the pool: the same application, read as a type
# ----------------------------------------------------------------------


def _environment():
    """Module name -> module, over `tests/data/specialize/`.

    A name is a generic when its directory has the marker, a file otherwise:
    what `_Runner` does for a run, done here without a run.
    """
    cache = {}

    def read(path):
        try:
            return Path(path).read_text()
        except OSError:
            return None

    def listing(directory):
        try:
            return Ok(os.listdir(directory))
        except OSError as exc:
            return VibaProgramErr(str(exc))

    def module_of(path, name, text):
        module = custom_module(text, environment)
        module.imports = _imports(module.module.body)
        cache[name] = module
        return Ok(module)

    def environment(name):
        if name in cache:
            return Ok(cache[name])
        directory = CASES.joinpath(*name.split("."))
        if (directory / GENERIC_FILE).is_file():
            got = load_generic(str(directory), name, read, listing, module_of)
            if isinstance(got, VibaProgramErr):
                return got
            cache[name] = got.ok_value
            return Ok(got.ok_value)
        path = directory.with_suffix(".viba")
        if path.is_file():
            return module_of(str(path), name, path.read_text())
        return VibaProgramErr(f"no module {name!r}")

    return environment


def _imports(body) -> dict:
    return {stmt.alias or stmt.module: stmt.module
            for stmt in body if isinstance(stmt, viba_ast.Import)}


def _module_of(source: str):
    module = custom_module(source, _environment())
    module.imports = _imports(module.module.body)
    return module


def _written(text: str, module):
    return AstNodeType(viba_ast.parse(f"__x__ = {text}").body[0].body, module)


def _judge(source: str, sub: str, sup: str):
    module = _module_of(source)
    got = is_sub_type(_written(sub, module), _written(sup, module))
    return got.ok_value if isinstance(got, Ok) else got


# (源, 左, 右, 想要什么, 这是哪一条)
JUDGMENT_CASES = [
    ("import demo.is_base_type as g\nX = g[bool]\n", "X", "true", True,
     "a known base type answers true"),
    ("import demo.is_base_type as g\nX = g[list[int]]\n", "X", "false", True,
     "anything else answers false"),
    ("import demo.is_base_type as g\nX = g[str]\n", "X", "int", False,
     "a literal bool type is not an int"),
    ("import demo.element_type_of as g\nX = g[list[int]]\n", "X", "int", True,
     "list[A] extracts A"),
    ("import demo.element_type_of as g\nX = g[set[float]]\n", "X", "float", True,
     "set[A] extracts A"),
    ("import demo.element_type_of as g\nX = g[dict[int, str]]\n", "X", "(int, str)", True,
     "dict[A, B] extracts the pair"),
    ("import demo.element_type_of as g\nX = g[list[int]]\n", "X", "str", False,
     "the extraction is not some other type"),
    ("import demo.element_type_of as g\nLocal = int\nX = g[list[Local]]\n",
     "X", "int", True,
     "the extracted name is read in the module that wrote the argument"),
    ("import demo.wrapper as g\nX = g[list[int]]\n", "X", "$item int", True,
     "the chosen file's own definition is read in that file"),
    ("import demo.ret_type_of as g\nX = g[float <- ()]\n", "X", "float", True,
     "a one-argument chain whose argument is a unit"),
    ("import demo.ret_type_of as g\nX = g[int <- int]\n", "X", "int", True,
     "a one-argument chain the first file refuses falls to the second"),
    ("import demo.ret_type_of as g\nX = g[(int, str) <- bool <- str]\n",
     "X", "(int, str)", True,
     "a three-part chain extracts its result"),
    ("import demo.call_it as g\nX = g[list[int]]\n", "X", "int <- int", True,
     "a chosen __def__ may be a function type, its $env not a parameter"),
    ("import demo.call_it as g\nX = g[list[int]]\n", "X", "int <- int <- int", False,
     "and it is that function type and no other"),
    ("import demo.num_generic_args as g\nX = g[dict[int, str]]\n", "X", "2", True,
     "the pattern's own structure is what is counted"),
    ("import demo.num_generic_args as g\nX = g[list[int]]\n", "X", "1", True,
     "one argument, one"),
    ("import demo.num_generic_args as g\nX = g[bool]\n", "X", "0", True,
     "no application at all is none"),
    ("import demo.num_generic_args as g\nX = g[list[int]]\n", "X", "2", False,
     "and the count is the one the file says, not another"),
    ("import demo.is_compatable as g\nX = g[bool, int]\n", "X", "true", True,
     "one restriction for both positions"),
    ("import demo.is_compatable as g\nX = g[float, str]\n", "X", "false", True,
     "an equality that does not hold falls through to the last file"),
    ("import demo.element_type_of as g\nX = g[list[int]]\n", "int", "X", True,
     "the extraction reads the same from the other side"),
    ("import demo.element_type_of as g\nX = g[bool]\n", "X", "never", "error",
     "a decision that fails is a VibaProgramErr"),
]


def _the_judgment_reads_it():
    for source, sub, sup, want, label in JUDGMENT_CASES:
        got = _judge(source, sub, sup)
        if want == "error":
            check(isinstance(got, VibaProgramErr) and "no specialization" in got.err_msg,
                  f"{label}: {got!r}")
            continue
        check(got is want, f"{label}: {sub} <: {sup} -> {got!r}, wanted {want}")


# ----------------------------------------------------------------------
# The pattern matcher itself
# ----------------------------------------------------------------------


def _pattern_case(source: str):
    module = _module_of(source)
    return module, patterns_of(module.module.body)


def _node(text: str):
    return viba_ast.parse(f"__x__ = {text}").body[0].body


def _match(pattern, module, argument: str):
    return structural_pattern_match(pattern, module, _node(argument), module, {})


def _the_pattern_matcher():
    """structural_pattern_match: 萃取到的类型就在这次答的绑定里。"""
    module, patterns = _pattern_case("specialize A\n")
    got = _match(patterns[0], module, "int")
    check(isinstance(got, Ok) and isinstance(got.ok_value, dict)
          and viba_ast.unparse_type(got.ok_value["A"].ast_node) == "int",
          f"a bare parameter takes the whole argument: {got!r}")

    module, patterns = _pattern_case("specialize bool | int | float | str\n")
    check(_match(patterns[0], module, "bool").ok_value == {},
          "a restriction with no parameter has no bindings")
    check(_match(patterns[0], module, "true").ok_value == {},
          "a literal fits its base type (subtyping, not equality)")
    check(_match(patterns[0], module, "list[int]").ok_value is None,
          "a container is not a base type")

    module, patterns = _pattern_case("specialize list[A]\n")
    got = _match(patterns[0], module, "list[int]")
    check(isinstance(got.ok_value, dict)
          and viba_ast.unparse_type(got.ok_value["A"].ast_node) == "int",
          f"a parameter inside a container takes the element: {got!r}")
    check(_match(patterns[0], module, "set[int]").ok_value is None,
          "another container family does not fit")
    check(_match(patterns[0], module, "list[int, str]").ok_value is None,
          "an application of the wrong arity does not fit")

    module, patterns = _pattern_case("specialize A <- (() | nil)\n")
    got = _match(patterns[0], module, "float <- ()")
    check(isinstance(got.ok_value, dict)
          and viba_ast.unparse_type(got.ok_value["A"].ast_node) == "float",
          f"a function pattern reads result and arguments apart: {got!r}")
    check(_match(patterns[0], module, "int <- int").ok_value is None,
          "the argument position has to fit")
    check(_match(patterns[0], module, "int <- int <- int").ok_value is None,
          "a longer chain does not fit a one-argument pattern")

    module, patterns = _pattern_case("specialize A\nspecialize A\n")
    first = _match(patterns[0], module, "int")
    check(isinstance(first.ok_value, dict)
          and viba_ast.unparse_type(first.ok_value["A"].ast_node) == "int",
          f"the first pattern binds the parameter: {first!r}")
    same = structural_pattern_match(patterns[1], module, _node("int"), module,
                                    first.ok_value)
    check(isinstance(same.ok_value, dict),
          f"the same type again fits the equality: {same!r}")
    other = structural_pattern_match(patterns[1], module, _node("str"), module,
                                     dict(first.ok_value))
    check(other.ok_value is None,
          f"another type does not fit the equality: {other!r}")

    module, patterns = _pattern_case("specialize ...\n")
    got = _match(patterns[0], module, "int")
    check(isinstance(got, VibaProgramErr) and "ellipsis" in got.err_msg,
          f"ellipsis is no pattern: {got!r}")


# ----------------------------------------------------------------------
# The arity each file declares
# ----------------------------------------------------------------------


def _the_arity_is_the_file():
    """一个文件写几行 specialize，就收几个实参；没有这个个数的，当场报个数。"""
    check(_judge("import demo.num_variadic_args as g\nX = g[]\n", "X", "0") is True,
          "no line at all is the generic of no parameters")
    check(_judge("import demo.num_variadic_args as g\nX = g[bool, str]\n", "X", "2") is True,
          "two lines are the generic of two")
    got = _judge("import demo.num_variadic_args as g\nX = g[bool, str, int, float]\n",
                 "X", "0")
    check(isinstance(got, VibaProgramErr)
          and "takes 0, 1, 2 or 3 parameters, not 4" in got.err_msg,
          f"an arity no file declares names the counts: {got!r}")


# ----------------------------------------------------------------------
# Mistakes
# ----------------------------------------------------------------------


BROKEN_CASES = [
    ("broken_no_marker", "not found", "a directory without the marker is no generic"),
    ("broken_named_badly", "named by its order", "a file named anything but a number"),
    ("broken_no_def", "__def__", "a specialization with no answer"),
    ("broken_no_answer", "no specialization", "a decision that fails"),
    ("broken_two_params", "takes 2 parameters", "a pattern count that is not the argument count"),
    ("broken_marker", "numbered files", "a marker that writes specialize"),
    ("loose_specialize", "numbered files", "specialize outside a generic's directory"),
]


def _the_decision_fails_loudly():
    environ = _environ()
    for name, want, label in BROKEN_CASES:
        labelled(interpret(_case(name), environ), want, label)


def _a_generic_is_no_module():
    """泛型不是模块：它有应用，没有定义。"""
    got = _judge("import demo.is_base_type as g\nX = g\n", "X", "int")
    check(isinstance(got, VibaProgramErr) and "generic" in got.err_msg,
          f"a bare generic name is not a type: {got!r}")

    labelled(interpret(_case("a_generic_is_no_program"), _environ()),
             "is a generic", "a generic is not a program either")


def _a_function_type_is_no_value():
    """选中的 `__def__` 是函数类型时，值的位置上它照旧不是值。"""
    labelled(interpret(_case("call_it_answer"), _environ()), "cannot compute Exponent",
             "a chosen __def__ that is a function type is read as a type, not a value")


# ----------------------------------------------------------------------
# Completeness, and the pool
# ----------------------------------------------------------------------


def _library():
    """(file path, content) for every .viba file under `CASES`."""
    found = []
    for path in sorted(CASES.rglob("*.viba")):
        rel = path.relative_to(CASES).with_suffix("")
        module = ".".join(rel.parts)
        found.append((str(rel), path.read_text(), module))
    return found


def _a_design_is_complete_through_it():
    library = [(path, source) for path, source, _module in _library()]
    check(is_complete("import demo.is_base_type as g\nX = g[bool]\n", library),
          "a decision that picks a literal is complete")
    check(is_complete("import demo.element_type_of as g\nX = g[list[int]]\n", library),
          "a decision that extracts a leaf name is complete")
    check(not is_complete("import demo.element_type_of as g\nX = g[bool]\n", library),
          "a decision that fails is incomplete")
    check(not is_complete('import demo.element_type_of as g\nX = g[list[{todo}]]\n',
                          library),
          "an extracted piece with no leaf is incomplete")


def _the_pool_holds_a_generic():
    pool = empty_pool()
    for path, source, module in _library():
        if not module.startswith("demo.is_base_type"):
            continue
        parsed = parse_viba_file(pool, source, path, module)
        check(isinstance(parsed, Ok), f"the pool compiles {module}: {parsed!r}")
        added = pool_add_file(pool, parsed.ok_value)
        if isinstance(added, VibaProgramErr):
            check(False, f"the pool takes {module}: {added!r}")
            return
        pool = added.ok_value
    entry = "import demo.is_base_type as g\nX = g[bool]\n"
    parsed = parse_viba_file(pool, entry, "entry.viba", "entry")
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    module = pool.module_environment("entry").ok_value
    got = is_sub_type(_written("X", module), _written("true", module))
    check(isinstance(got, Ok) and got.ok_value is True,
          f"a pool serves the generic its directory holds: {got!r}")


def _the_reflection_reads_it():
    """一份设计里写着泛型应用时，读实例也按决断往下走。"""
    pool = empty_pool()
    wanted = ("demo.element_type_of", "demo.wrapper", "design")
    sources = [(path, source, module) for path, source, module in _library()
               if module.startswith("demo.element_type_of")
               or module.startswith("demo.wrapper")]
    sources.append(("design.viba",
                    "import demo.element_type_of as g\n"
                    "import demo.wrapper as w\n"
                    "Picked = $value g[list[int]]\n"
                    "Boxed = $value w[list[str]]\n", "design"))
    for path, source, module in sources:
        parsed = parse_viba_file(pool, source, path, module)
        if isinstance(parsed, VibaProgramErr):
            check(False, f"the pool compiles {module}: {parsed!r}")
            return
        pool = pool_add_file(pool, parsed.ok_value).ok_value

    definition = pool_find_definition(pool, "design.Picked").ok_value
    node = access.root(definition, VibaData(viba_ast.Tagged("$value", viba_ast.Constant(3))))
    check(isinstance(node, Ok), f"a design whose member is extracted from a list: {node!r}")
    if isinstance(node, Ok):
        leaf = access.leaf(access.get(node.ok_value, by_tag("$value")).ok_value)
        check(isinstance(leaf, Ok) and leaf.ok_value == 3,
              f"an int sits where the extraction said it did: {leaf!r}")

    definition = pool_find_definition(pool, "design.Boxed").ok_value
    boxed = viba_ast.Tagged("$value", viba_ast.Tagged("$item", viba_ast.Constant("hi")))
    node = access.root(definition, VibaData(boxed))
    check(isinstance(node, Ok), f"a design whose member is the chosen file's own type: {node!r}")
    if isinstance(node, Ok):
        leaf = access.leaf(access.get(
            access.get(node.ok_value, by_tag("$value")).ok_value, by_tag("$item")).ok_value)
        check(isinstance(leaf, Ok) and leaf.ok_value == "hi",
              f"the chosen file's $item is where the design says it is: {leaf!r}")


if __name__ == "__main__":
    run(CASES)
    sys.exit(checks.report())

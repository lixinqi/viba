"""闭包：`pattern` 认出一个还没给环境的写下来的调用，`is_closure` 与 `unclosure` 用它。

判定在类型层，读法和 `tests/test_pattern.py` 的判定那一段一样：一份写成类型的东西与另一份比，
看 `<:` 成不成立。

    is_closure[add << $a 1].value                     -> true
    is_closure[add].value                             -> false   （光一个名字，还没有实参）
    is_closure[add << args.env << $a 1 << $b 2].value -> false   （环境给了：它跑了）
    is_closure[0].value                               -> false   （一个叶子）

    unclosure[add << $a 1].f                          -> `add` 那条 api 的类型
    unclosure[add << $a 1].captured                   -> `$a 1`（收下的那份积，写下来是什么就是什么）

`is_closure` 与 `unclosure` 住在 `viba/builtin/` 下（`viba/builtin/is_closure/`、
`viba/builtin/unclosure/`），全名就是 `builtin.is_closure`；裸名 `is_closure` 对每个模块可见，
所以写它们不用 import。

一条调用模式的读法（`viba/pattern.py` 的 `_match_call`）：链头归 `F`，写下来的实参**一个 `<<`
对一个**，按书写顺序归后面的段 —— 模式写几个 `<<`，就认给了几个实参的调用，所以那两份泛型各写了
1..16 段的 16 个文件（照 `viba/apply_impl/` 那样，一个长度一个文件）。还有：

  * 环境（`$env` 那个 tag、或者这一段的类型就是 `Env`）不算实参：它给了就是跑了，不是闭包；
  * 外面的 tag 是这一份的一部分，`pattern F << A` 认不出 `$x (add << $a 1)`，要写 `pattern $x (F << A)`；
  * 16 段以上的调用没有文件接：`is_closure` 落到最后那份兜底给出 false，`unclosure` 当场说没有模式。

用例在 `tests/data/pattern_closure/`：`closure.viba` 是判定用的那份模块（里面写着 `add`），
`square_sum.viba` 是一个模块（模块调用没给环境时也是一个闭包），另外六个目录是小的泛型，
各自钉住上面的一条：`take_apart/`（`F << A << B`）、`take_three/`（`F << A << B << C`）、
`rest_apart/`（`F << A << (B * C)`）、`under_tag/`（`$x (F << A)`）、`by_the_symbol/`
（`tagged[arg, F << A]`）、`same_two/`（写两次的 `A`）。`*.viba` 那几份是跑到程序这一层的用例：
泛型选中那一支给出的类型，就是程序给出的值。

    python3 tests/test_pattern_closure.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import message_of, error_of, Checks, Host, value_of

from viba import viba_ast
from viba.interpret import interpret
from viba.is_sub_type import is_sub_type
from viba.pattern import (GENERIC_FILE, load_generic, structural_pattern_match)
from viba.type import (BUILTIN_CONCEPT_DIR, BUILTIN_DIR, AstNodeType, Ok,
                       VibaProgramErr, custom_module)
from viba.viba_type_descriptor import (empty_pool, parse_viba_file,
                                       pool_add_file)

checks = Checks("pattern_closure")
check = checks.check
labelled = checks.labelled

CASES = Path(__file__).resolve().parent / "data" / "pattern_closure"

# 一份写着 `add` 的源码：几处检查直接拿它拼一个模块。
ADD = """
add =
    int
  <- $env Env
  <- $a int
  <- $b int
  <- { add the two }
"""

# (要看的那条定义, 它该是什么, 这是哪一条)
JUDGMENT_CASES = [
    # 是闭包：给了实参，没给环境
    ("is_a_closure", "true", "a call with an argument and no environment is a closure"),
    ("is_a_full_closure", "true", "every argument captured, no environment: still a closure"),
    ("is_a_long_closure", "true", "one file's pattern reads a call of any length"),
    ("is_a_wrapped_closure", "true", "the parentheses do not change the written call"),
    ("is_a_nested_closure", "true", "an argument that is itself a closure is no environment"),
    ("a_module_closure", "true", "a module call with no environment is a closure too"),
    ("a_qualified_question", "true", "the qualified name asks the same generic"),
    # 不是闭包：环境给了（它跑了），或者根本不是一次调用
    ("is_a_bare_name", "false", "a bare name has no argument yet"),
    ("is_an_executed_call", "false", "a call that was given its environment has run"),
    ("is_an_executed_call_in_the_middle", "false", "the environment may sit anywhere in the call"),
    ("is_a_tagged_environment", "false", "`$env` says it is the environment whatever it holds"),
    ("is_an_environment_by_name", "false",
     "a name denoting the environment is the environment too"),
    ("is_an_executed_sub_env", "false", "a call of `sub_env` given its environment is one"),
    ("is_an_executed_module_call", "false", "a module call given its environment has run"),
    ("is_a_leaf", "false", "a leaf is no call at all"),
    ("is_a_float", "false", "…and a float is a leaf"),
    ("is_a_str", "false", "…and a string is a leaf"),
    ("is_a_bool", "false", "…and a bool is a leaf"),
    ("is_an_empty_product", "false", "…and the empty product is a leaf"),
    ("is_nil", "false", "…and nil is a leaf"),
    ("is_a_type_name", "false", "…and a written type name is no call"),
    ("is_a_chain", "false", "…and a chain is no call"),
    ("is_a_tagged_leaf", "false", "…and a tagged leaf is no call"),
    ("is_a_tagged_call", "false", "a call wearing a tag is not what `F << A` alone reads"),
    # 拆出来：`f` 是链头那条 api，`captured` 是收下的那份积
    ("the_api", "int <- $env Env <- $a int <- $b int", "the head is the api the call is of"),
    ("two_api", "int <- $env Env <- $a int <- $b int",
     "…the same one, however many arguments were given"),
    ("the_captured", "$a 1", "what it captured is the product it holds"),
    ("two_captured", "($a 1 * $b 2)", "however many arguments, they come back as one product"),
    ("three_captured", "($a 1 * $b 2 * $c 3)",
     "…and a call with more arguments than the api has slots is read the same way"),
    ("a_nested_capture", "$a (mul << $b 2)", "a captured closure is one piece, tag and all"),
    ("a_module_capture", "3", "a module closure captured its argument"),
    ("a_qualified_capture", "$a 1", "`builtin.` hands back the same product"),
    # 闭包模式在别的泛型里的读法
    ("first_of_two", "$a 1", "`F << A << B`: the first link takes the first argument"),
    ("second_of_two", "$b 2", "…and the second the second"),
    ("take_apart_api", "int <- $env Env <- $a int <- $b int", "…and its `f` is the api"),
    ("third_of_three", "$c 3", "`F << A << B << C`: a longer pattern reads a longer call"),
    ("take_three_api", "int <- $env Env <- $a int <- $b int", "…and its `f` is the api too"),
    ("the_second", "$b 2", "`F << A << (B * C)` reads an argument that is one product"),
    ("the_third", "$c 3", "…position by position"),
    ("rest_apart_api", "int <- $env Env <- $a int <- $b int", "…and its `f` is the api too"),
    ("the_tagged_call", "$a 1", "`$x (F << A)` reads the call inside the tag"),
    ("the_tagged_api", "int <- $env Env <- $a int <- $b int", "…and takes its head as the api"),
    ("the_symbol_capture", "$a 1", "`tagged[arg, F << A]` is the same reading"),
    ("the_symbol_head", "int <- $env Env <- $a int <- $b int", "…and its head is the api too"),
    ("the_symbol", '"y"', "…and the tag's symbol comes back as a string"),
    ("the_same", "$a 1", "a name written twice takes one type"),
]

# (要看的那条定义, 它不该是什么, 这是哪一条)
NEGATIVE_CASES = [
    ("the_captured", "$a 2", "the captured product keeps its tag and its value"),
    ("the_captured", "$b 1", "…and its tag is the one it was written with"),
    ("is_a_closure", "false", "a closure is not a non-closure"),
    ("is_a_bare_name", "true", "a bare name is not a closure"),
    ("is_a_leaf", "true", "a leaf is not a closure"),
    ("is_a_tagged_call", "true", "a tagged call is not what a bare call pattern reads"),
    ("is_a_tagged_environment", "true", "a call given its environment is not a closure"),
    ("a_module_capture", "4", "the module closure captured 3, not 4"),
    ("the_same", "$b 1", "two arguments of different tags are not one type"),
    ("the_symbol", '"x"', "the symbol is the tag the call wore"),
    ("first_of_two", "$b 2", "the first link is not the second argument"),
    ("second_of_two", "$b 1", "an argument keeps its tag"),
    ("the_third", "$b 2", "a product read apart keeps the written order"),
]

# (要问的那条写法, 决断该说什么, 这是哪一条)
ERROR_CASES = [
    ("unclosure[add]", "no pattern of 'unclosure'", "a bare name is no closure to read apart"),
    ("unclosure[0]", "no pattern of 'unclosure'", "a leaf is no closure to read apart"),
    ("unclosure[add << args.env << $a 1]", "no pattern of 'unclosure'",
     "a call that was given its environment is no closure"),
    ("unclosure[$x (add << $a 1)]", "no pattern of 'unclosure'",
     "a call wearing a tag is no closure to `unclosure`"),
    ("take_apart[add]", "no pattern of 'take_apart'",
     "a call with no argument at all gives the first link nothing"),
    ("take_apart[add << $a 1 << $b 2 << $c 3]", "no pattern of 'take_apart'",
     "a two-link pattern does not read a three-argument call"),
    ("take_apart[add << $a 1]", "no pattern of 'take_apart'",
     "…nor does it read a one-argument call"),
    ("take_apart[add << args.env << $a 1]", "no pattern of 'take_apart'",
     "…and a call that was given its environment is no closure"),
    ("rest_apart[add << $a 1 << $b 2]", "no pattern of 'rest_apart'",
     "an argument that is no product is not read apart"),
    ("under_tag[add << $a 1]", "no pattern of 'under_tag'",
     "a call with no tag is not what a tagged pattern reads"),
    ("under_tag[$y (add << $a 1)]", "no pattern of 'under_tag'",
     "the tag has to be the one written"),
    ("by_the_symbol[add << $a 1]", "no pattern of 'by_the_symbol'",
     "…the same for the reading that writes the tag as a symbol"),
    ("same_two[add << $a 1 << $b 1]", "no pattern of 'same_two'",
     "a name written twice wants one type, tags included"),
    ("is_closure[0, 1]", "takes 1", "the generic takes one argument"),
]

# (用例文件, 跑出来该是什么值, 这是哪一条)
RUNTIME_CASES = [
    ("is_closure_answer", True, "is_closure[square_sum << 3].value"),
    ("is_closure_of_a_local_call", True, "is_closure[add << $a 1].value"),
    ("is_closure_of_two_arguments", True, "…a call with two arguments is another file's"),
    ("is_closure_of_an_executed_call", False, "is_closure[add << $env Env << $a 1].value"),
]

# (用例文件, 给出的类型写成什么, 这是哪一条)
RUNTIME_EXTRACTED_CASES = [
    ("unclosure_answer", "3", "unclosure[square_sum << 3].captured"),
    ("unclosure_api", "square_sum", "unclosure[square_sum << 3].f names the module it was"),
    ("unclosure_api_of_a_call", "add", "unclosure[add << $a 1].f names the api it was"),
    ("unclosure_captured", "$a 1", "unclosure[add << $a 1].captured"),
]


def run():
    _judgments()
    _failed_decisions()
    _the_matcher_reads_a_call()
    _the_length_bound()
    _a_local_definition_wins()
    _the_pool_reads_the_names()
    _the_runtime_answers()
    _what_was_extracted()
    _unclosure_needs_a_closure()


def _module():
    """`closure.viba` 读成一个模块：它的 import 也在这儿对上。"""
    path = CASES / "closure.viba"
    module = custom_module(path.read_text(), _environment())
    module.imports = {stmt.alias or stmt.module: stmt.module
                      for stmt in module.module.body
                      if isinstance(stmt, viba_ast.Import)}
    return module


def _environment():
    """模块名 -> 模块：先看用例目录，再看内建目录与 `builtin/`（两个泛型在那里）。"""
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
        module.imports = {stmt.alias or stmt.module: stmt.module
                          for stmt in module.module.body
                          if isinstance(stmt, viba_ast.Import)}
        cache[name] = module
        return Ok(module)

    def environment(name):
        if name in cache:
            return Ok(cache[name])
        for root in (CASES, BUILTIN_DIR, BUILTIN_CONCEPT_DIR):
            directory = root.joinpath(*name.split("."))
            if (directory / GENERIC_FILE).is_file():
                got = load_generic(str(directory), name, read, listing, module_of)
                if isinstance(error_of(got), VibaProgramErr):
                    return got
                cache[name] = got.ok_value
                return Ok(got.ok_value)
            path = directory.with_suffix(".viba")
            if path.is_file():
                return module_of(str(path), name, path.read_text())
        return VibaProgramErr(f"no module {name!r}")

    return environment


def _written(text: str, module):
    return AstNodeType(viba_ast.parse(f"__x__ = {text}").body[0].body, module)


def _judge(module, sub: str, sup: str):
    """`sub <: sup` in this module, as a `Result` (never a raised judgment)."""
    return is_sub_type(_written(sub, module), _written(sup, module))


def _judgments():
    module = _module()
    for name, want, label in JUDGMENT_CASES:
        got = _judge(module, name, want)
        check(isinstance(got, Ok) and got.ok_value is True,
              f"{label}: {name} <: {want}: {got!r}")
    for name, want, label in NEGATIVE_CASES:
        got = _judge(module, name, want)
        check(isinstance(got, Ok) and got.ok_value is False,
              f"{label}: {name} <: {want} does not hold: {got!r}")


def _failed_decisions():
    """没有模式接得住的那几条：决断当场说没有，而不是讲一声「不是闭包」。"""
    module = _module()
    for text, want, label in ERROR_CASES:
        got = _judge(module, text, "Any")
        check(isinstance(error_of(got), VibaProgramErr) and want in message_of(got),
              f"{label}: expected VibaProgramErr({want!r}), got {got!r}")


def _the_matcher_reads_a_call():
    """调用模式本身：绑定里就是链头加写下来的那些实参，一个 `<<` 对一个实参。

    `take_apart/100.viba` 的模式是 `F << A << B`，这里直接拿它读几份写下来的调用，
    省得只从判定那一层看结果。
    """
    module = _module()
    handed = module.module_environment("take_apart")
    check(isinstance(handed, Ok), f"the generic is loaded next to the module: {handed!r}")
    if not isinstance(handed, Ok):
        return
    entry = handed.ok_value.entries[0]
    read_once = entry.read()
    check(isinstance(read_once, Ok), f"the pattern file is read: {read_once!r}")
    if not isinstance(read_once, Ok):
        return

    def read(text: str):
        return structural_pattern_match(entry.patterns[0], entry.module,
                                        _node(text), module, {})

    got = read("add << $a 1 << $b 2")
    want = {"F": "add", "A": "$a 1", "B": "$b 2"}
    taken = got.ok_value if isinstance(got, Ok) else None
    if not isinstance(taken, dict):
        check(False, f"a two-argument call fits `F << A << B`: {got!r}")
    else:
        shown = {name: viba_ast.unparse_type(one.ast_node)
                 for name, one in taken.items()}
        check(set(taken) == set(want)
              and all(_is_the_same_type(taken[name], _written(one, module))
                      for name, one in want.items()),
              f"one link per written argument: {shown!r}")
    for text, label in [("add << $a 1", "a call with one argument"),
                        ("add << $a 1 << $b 2 << $c 3", "a call with three"),
                        ("add << args.env << $a 1", "a call that was given its environment"),
                        ("add", "a bare name"),
                        ("$x (add << $a 1)", "a call wearing a tag")]:
        got = read(text)
        check(isinstance(got, Ok) and got.ok_value is None,
              f"{label} is no fit for `F << A << B`: {got!r}")


def _node(text: str):
    return viba_ast.parse(f"__x__ = {text}").body[0].body


def _module_of(source: str):
    """一份写下来的源码读成模块：它 import 的名字都从用例目录里取。"""
    module = custom_module(source, _environment())
    module.imports = {}
    return module


def _the_pool_reads_the_names():
    """设计从池子里读时也一样：`builtin.is_closure` 与裸名 `is_closure` 都走到那份泛型。"""
    pool = empty_pool()
    for name in ("is_closure", "unclosure"):
        directory = BUILTIN_CONCEPT_DIR / name
        for path in sorted(directory.iterdir()):
            module_name = f"builtin.{name}.{path.stem}"
            parsed = parse_viba_file(pool, path.read_text(), str(path), module_name)
            if not isinstance(parsed, Ok):
                check(False, f"the pool compiles {module_name}: {parsed!r}")
                return
            added = pool_add_file(pool, parsed.ok_value)
            if isinstance(error_of(added), VibaProgramErr):
                check(False, f"the pool takes {module_name}: {added!r}")
                return
            pool = added.ok_value
    entry = ADD + ("__bare__ = is_closure[add << $a 1].value\n"
                   "__qualified__ = builtin.unclosure[add << $a 1].captured\n")
    parsed = parse_viba_file(pool, entry, "entry.viba", "entry")
    if not isinstance(parsed, Ok):
        check(False, f"the pool compiles the entry: {parsed!r}")
        return
    pool = pool_add_file(pool, parsed.ok_value).ok_value
    module = pool.module_environment("entry").ok_value
    for name, want, label in [("__bare__", "true", "the bare name reaches the generic"),
                              ("__qualified__", "$a 1", "…and so does `builtin.`")]:
        got = _judge(module, name, want)
        check(isinstance(got, Ok) and got.ok_value is True,
              f"{label}: {name} <: {want}: {got!r}")


def _a_local_definition_wins():
    """模块自己写了 `is_closure` 这个名字，给出的就是它，内建目录里的泛型不插嘴。"""
    shadowed = _module_of("is_closure = true\nunclosure = false\n" + ADD)
    plain = _module_of(ADD)
    asked = "is_closure[add << $a 1].value"
    read_apart = "unclosure[add << $a 1].f"
    api = "int <- $env Env <- $a int <- $b int"
    for text, want, label in [(asked, "true", "`is_closure`"),
                              (read_apart, api, "`unclosure`")]:
        got = _judge(plain, text, want)
        check(isinstance(got, Ok) and got.ok_value is True,
              f"{label} answers where nothing shadows it: {text} <: {want}: {got!r}")
        got = _judge(shadowed, text, want)
        check(not (isinstance(got, Ok) and got.ok_value is True),
              f"a local definition wins over the builtin {label}: {got!r}")


def _a_written_call(count: int) -> str:
    """`add << $a0 0 << $a1 1 …`: 一份给了这么多个实参的调用。"""
    return "add " + " ".join(f"<< $a{i} {i}" for i in range(count))


def _the_length_bound():
    """内建那两份各写到 16 段为止：再长 `is_closure` 落到兜底那份给出 false，`unclosure` 没有文件接。"""
    for count in (1, 2, 16):
        module = _module_of(ADD + f"\n__x__ = is_closure[{_a_written_call(count)}].value\n")
        got = _judge(module, "__x__", "true")
        check(isinstance(got, Ok) and got.ok_value is True,
              f"a call with {count} arguments is a closure: {got!r}")
    module = _module_of(ADD + f"\n__x__ = unclosure[{_a_written_call(16)}].captured\n")
    got = _judge(module, "__x__", "Any")
    check(isinstance(got, Ok) and got.ok_value is True,
          f"sixteen arguments are read apart: {got!r}")
    module = _module_of(ADD + f"\n__x__ = is_closure[{_a_written_call(17)}].value\n")
    got = _judge(module, "__x__", "false")
    check(isinstance(got, Ok) and got.ok_value is True,
          f"a call longer than the 16 files write falls to the last one: {got!r}")
    module = _module_of(ADD + f"\n__x__ = unclosure[{_a_written_call(17)}].captured\n")
    got = _judge(module, "__x__", "Any")
    check(isinstance(error_of(got), VibaProgramErr) and "no pattern of 'unclosure'" in message_of(got),
          f"…and `unclosure` has no file for it: {got!r}")


def _is_the_same_type(left: AstNodeType, right: AstNodeType) -> bool:
    """One written type is the other, as the judgment layer reads it."""
    one = is_sub_type(left, right)
    other = is_sub_type(right, left)
    return (isinstance(one, Ok) and one.ok_value is True
            and isinstance(other, Ok) and other.ok_value is True)


def _the_runtime_answers():
    """一份模块应用一个泛型：选中的那一支就是它给出的值。"""
    environ = Host().environ(viba_path=str(CASES))
    for name, want, label in RUNTIME_CASES:
        result = interpret(str(CASES / f"{name}.viba"), environ)
        check(isinstance(result, Ok) and value_of(result) is want,
              f"{label} -> {want!r}: {result!r}")


def _what_was_extracted():
    """萃取到的类型，就是结果里写下来的那份。"""
    environ = Host().environ(viba_path=str(CASES))
    for name, want, label in RUNTIME_EXTRACTED_CASES:
        result = interpret(str(CASES / f"{name}.viba"), environ)
        got = (viba_ast.unparse_type(result.ok_value.data)
               if isinstance(result, Ok) else repr(result))
        check(isinstance(result, Ok) and got == want,
              f"{label} -> {want!r}: {got!r}")


def _unclosure_needs_a_closure():
    """`unclosure` 只认闭包：光一个名字没有模式接它，跑到程序这一层也这么说。"""
    module = _module()
    got = _judge(module, "unclosure[add]", "int <- $env Env <- $a int <- $b int")
    check(isinstance(error_of(got), VibaProgramErr) and "no pattern of 'unclosure'" in message_of(got),
          f"unclosure of a bare name is no decision: {got!r}")
    labelled(interpret(str(CASES / "unclosure_of_a_bare_name.viba"),
                       Host().environ(viba_path=str(CASES))),
             "no pattern of 'unclosure'",
             "the same failure where a program runs")


if __name__ == "__main__":
    run()
    sys.exit(checks.report())

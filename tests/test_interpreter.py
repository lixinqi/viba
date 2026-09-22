"""viba.interpreter 的压测：把 interpret 的边角都过一遍。

语料是设计里那两份可执行模块（add_demo.viba / main.viba），其余模块都在临时目录里
现写现跑，不留在仓库里。

    python tests/test_interpreter.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.interpreter import (Environment, EnvironmentCompute, EnvironmentStorage,
                              interpret)
from viba.reflect import access as reflect_access
from viba.type import Err, Ok

CASES = Path(__file__).resolve().parent / "data" / "interpreter"

PASS = FAIL = 0


def check(ok: bool, label: str):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def value_of(result):
    """The leaf a run answered: None for a nil piece."""
    if not isinstance(result, Ok):
        return result
    return reflect_access.leaf(result.ok_value).ok_value


def labelled(result, want, label: str):
    """`want` is a substring of the Err, or None for Ok."""
    if want is None:
        check(isinstance(result, Ok), f"{label}: {result!r}")
    else:
        check(isinstance(result, Err) and want in result.err_msg,
              f"{label}: expected Err({want!r}), got {result!r}")


class Host:
    """The host side: a few functions, a record of the calls, and knobs."""

    def __init__(self, **knobs):
        self.calls = []
        self.printed = []
        self.knobs = knobs

    def get_func(self, module_path, func_name):
        self.calls.append((module_path, func_name))
        if self.knobs.get("get_func_raises"):
            raise RuntimeError("host broke")
        if func_name == "add":
            return lambda env, a, b: a.value + b.value
        if func_name == "join":
            return lambda env, a, b: a.value + b.value
        if func_name == "print":
            def show(env, x):
                self.printed.append(x)
                return None
            return show
        if func_name == "explode":
            return lambda env: 1 / 0
        if func_name == "inc":
            return lambda env, x: x.value + 1
        if func_name == "twice":
            def twice(env, f, x):
                return f(env, x).value + f(env, x).value
            return twice
        if func_name == "echo":
            return lambda env, x: x
        if func_name == "leaf":
            return lambda env: 7
        if func_name == "text":
            return lambda env: "hi"
        if func_name == "flag":
            return lambda env: True
        if func_name == "ratio":
            return lambda env: 0.5
        if func_name == "nothing":
            return lambda env: None
        if func_name == "zero":
            return lambda env: 0
        if func_name == "empty":
            return lambda env: ""
        if func_name == "falsey":
            return lambda env: False
        if func_name == "answer_a_list":
            return lambda env: [1, 2]
        if func_name == "answer_a_tuple":
            return lambda env: (1, 2)
        if func_name == "answer_a_dict":
            return lambda env: {"a": 1}
        if func_name == "wrong_arity":
            return lambda env: 1
        if func_name == "make_node":
            def make_node(env):
                from viba import viba_ast
                from viba.reflect import VibaNode, access
                from viba.viba_type_descriptor import descriptor_of
                from viba.type import AstNodeType, custom_module
                node = viba_ast.Constant(11)
                return VibaNode(access,
                                descriptor_of(AstNodeType(node, custom_module(""))), node)
            return make_node
        if func_name == "answer_a_function":
            return lambda env: (lambda: 1)
        if func_name == "overfeed":
            def overfeed(env, f, x):
                return f(env, x, 1)
            return overfeed
        if func_name == "inner_value":
            def inner_value(env):
                result = interpret(self.knobs["inner_file"], env)
                if isinstance(result, Err):
                    raise RuntimeError(result.err_msg)
                return result.ok_value
            return inner_value
        if func_name == "feed_a_list":
            def feed_a_list(env, f):
                return f(env, [1, 2])
            return feed_a_list
        if func_name == "interrupt":
            def interrupt(env):
                raise KeyboardInterrupt
            return interrupt
        return None

    def environ(self, path="root"):
        return Environment(EnvironmentStorage(path), EnvironmentCompute(self.get_func))


def run() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="viba-interpreter-stress-"))
    try:
        _spec_modules()
        _values_and_answers(tmp)
        _applications(tmp)
        _environments(tmp)
        _modules(tmp)
        _loose_ends(tmp)
        _names_and_sources(tmp)
        _higher_order_and_answers(tmp)
        _shapes_and_scale(tmp)
        _caching_and_repeats(tmp)
        _more_corners(tmp)
        _virtual_files(tmp)
        _paths(tmp)
    finally:
        for leftover in sorted(tmp.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    print(f"interpreter: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


def _write(tmp: Path, name: str, source: str) -> str:
    path = tmp / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return str(path)


LEAF = """
leaf :=
	int
	<- $env Environment
	<- { answer seven }
"""

TEXT = """
text :=
	str
	<- $env Environment
	<- { answer some text }
"""

ADD = """
add :=
	int
	<- $env Environment
	<- $a int
	<- $b int
	<- { add two integer }
"""


def _spec_modules():
    """设计里那两份：模块当函数、import、sub_env、print。"""
    host = Host()
    environ = host.environ()

    result = interpret(str(CASES / "add_demo.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 1000000,
          f"add_demo's __ret__ is what add answered: {result!r}")
    check(("root", "add") in host.calls,
          f"the implementation is looked up at the environment's path: {host.calls}")

    result = interpret(str(CASES / "main.viba"), environ)
    check(isinstance(result, Ok), f"main runs: {result!r}")
    check(("root/add_demo", "add") in host.calls,
          f"a module called through a sub-environment looks its functions up there: {host.calls}")
    check(len(host.printed) == 1 and getattr(host.printed[0], "value", None) == 1000000,
          f"print got the value main computed: {host.printed!r}")


def _values_and_answers(tmp: Path):
    """宿主返回什么，__ret__ 就是什么；宿主出错就是 Err，不是崩。"""
    host = Host()
    environ = host.environ()
    for func, want, label in (("leaf", 7, "an int"), ("text", "hi", "a str"),
                              ("flag", True, "a bool"), ("ratio", 0.5, "a float"),
                              ("nothing", None, "None lands as nil")):
        path = _write(tmp, f"{func}.viba", f"""
{func} :=
	int
	<- $env Environment
	<- {{ inline }}
__ret__ := {func} << $env environ
""")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"a host function returning {label}: {result!r}")

    boom = _write(tmp, "boom.viba", """
explode :=
	int
	<- $env Environment
	<- { go }
__ret__ := explode << $env environ
""")
    labelled(interpret(boom, environ), "ZeroDivision", "a host function that raises -> Err")
    labelled(interpret(boom, Host(get_func_raises=True).environ()), "raised",
             "a get_func that raises -> Err")

    missing = _write(tmp, "no_impl.viba", """
ghost :=
	int
	<- $env Environment
	<- { nothing implements this }
__ret__ := ghost << $env environ
""")
    labelled(interpret(missing, environ), "no implementation", "get_func says None -> Err")

    echo = _write(tmp, "echo.viba", """
echo :=
	int
	<- $env Environment
	<- $x int
	<- { pass it through }
__ret__ := echo << $env environ << $x 42
""")
    result = interpret(echo, environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"a host function echoing its argument: {result!r}")


def _applications(tmp: Path):
    """`<<` 的边角：少给、多给、给错、说明块、分两步、不是值。"""
    host = Host()
    environ = host.environ()
    cases = [
        ("__ret__ := add << $env environ << $a 1 << $b 2\n", None, "all three given"),
        ("__ret__ := add << $env environ << $a 1\n",
         "waiting for arguments", "one argument short"),
        ("__ret__ := add << $env environ << $a 1 << $b 2 << $b 3\n",
         "is not a function", "one argument too many (the call already answered)"),
        ("__ret__ := add << $env environ << $a 1 << $z 2\n",
         "takes no $z", "an argument the function does not have"),
        ("__ret__ := add << $env environ << $a 1 << $b 2 << { trailing note }\n", None,
         "documentation after the arguments"),
        ("__ret__ := add << $env environ << { a note } << $a 1 << $b 2\n", None,
         "documentation between the arguments"),
        ("__ret__ := add\n", "waiting for arguments", "the function itself is not a value"),
        ("__ret__ := { just a note }\n", "documentation", "a code block is not a value"),
        ("__ret__ := Nope\n", "no definition named", "a name nothing defines"),
    ]
    for index, (body, want, label) in enumerate(cases):
        path = _write(tmp, f"apply{index}.viba", ADD + body)
        labelled(interpret(path, environ), want, label)

    two_steps = _write(tmp, "two_steps.viba", ADD + """
half := add << $env environ << $a 40
__ret__ := half << $b 2
""")
    result = interpret(two_steps, environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"a partially applied function kept in a definition: {result!r}")

    positional = _write(tmp, "positional.viba", ADD + """
__ret__ := add << environ << 40 << 2
""")
    result = interpret(positional, environ)
    check(isinstance(result, Ok) and value_of(result) == 42,
          f"arguments written without tags bind in order: {result!r}")


def _environments(tmp: Path):
    """environment 的边角：必须给、必须是 Environment、子环境带父级 compute。"""
    host = Host()
    environ = host.environ()

    no_env = _write(tmp, "no_env.viba", """
double := int <- $a int <- { double it }
__ret__ := double << $a 21
""")
    labelled(interpret(no_env, environ), "$env Environment",
             "a function without $env -> Err")

    not_given = _write(tmp, "not_given.viba", ADD + "__ret__ := add << $a 1 << $b 2\n")
    labelled(interpret(not_given, environ), "still waiting for arguments",
             "a call that was not given the environment -> Err")

    wrong = _write(tmp, "wrong_env.viba", ADD + "__ret__ := add << $env 7 << $a 1 << $b 2\n")
    labelled(interpret(wrong, environ), "not given an Environment",
             "an environment argument that is not an Environment -> Err")

    child = environ.sub_env("a")
    grand = child.sub_env("b")
    check(child.storage.cur_storage_path == "root/a",
          "a sub-environment's path is <parent>/<name>")
    check(grand.storage.cur_storage_path == "root/a/b", "and it nests")
    check(child.compute is environ.compute and grand.compute is environ.compute,
          "every sub-environment holds the parent's compute")
    check(environ.sub_env("a").storage is child.storage,
          "the same name hands back the same sub-storage")
    check(isinstance(child, Environment), "sub_env answers an Environment")

    host.calls.clear()
    paths = _write(tmp, "paths.viba", ADD + "__ret__ := add << $env environ << $a 1 << $b 2\n")
    labelled(interpret(paths, child), None, "a module runs under a sub-environment")
    check(("root/a", "add") in host.calls,
          f"its path is what get_func sees: {host.calls}")

    labelled(interpret(paths, object()), "needs an Environment",
             "interpret with something that is not an Environment -> Err")

    # 子环境的名来自 viba 那边写下的东西：材料取它的叶子，别的就 str 一下
    seeded = EnvironmentStorage("root", {"a": EnvironmentStorage("root/a")})
    check(Environment(seeded, environ.compute).sub_env("a").storage is
          seeded.sub_storage["a"],
          "a storage handed in ready-made is the one sub_env hands back")
    check(environ.sub_env(7).storage.cur_storage_path == "root/7",
          "a name that is not a string still lands in the path")
    child_a = environ.sub_env("a")
    child_b = environ.sub_env("b")
    check(child_a.sub_env("x").storage.cur_storage_path == "root/a/x" and
          child_b.sub_env("x").storage.cur_storage_path == "root/b/x",
          "the same name under two parents is two storages")

    sub = _write(tmp, "sub.viba", """
go :=
	int
	<- $env Environment
	<- { nobody calls this }
__ret__ := environ.sub_env << "child"
""")
    result = interpret(sub, environ)
    check(isinstance(result, Ok) and isinstance(result.ok_value, Environment) and
          result.ok_value.storage.cur_storage_path == "root/child",
          f"environ.sub_env written in viba names a child: {result!r}")


def _modules(tmp: Path):
    """模块：嵌套 import、dotted import、设计模块、自调用环。"""
    host = Host()
    environ = host.environ()

    _write(tmp, "inner.viba", ADD + "__ret__ := add << $env environ << $a 1 << $b 2\n")
    outer = _write(tmp, "outer.viba", """
import inner as inner
__ret__ := inner << (environ.sub_env << "inner")
""")
    result = interpret(outer, environ)
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"a module imported by a module: {result!r}")

    _write(tmp, "pkg/mod.viba", """
join :=
	str
	<- $env Environment
	<- $a str
	<- $b str
	<- { join two strings }
__ret__ := join << $env environ << $a "a" << $b "b"
""")
    dotted = _write(tmp, "dotted.viba", """
import pkg.mod as mod
__ret__ := mod << environ
""")
    result = interpret(dotted, environ)
    check(isinstance(result, Ok) and value_of(result) == "ab",
          f"a dotted import finds pkg/mod.viba: {result!r}")

    _write(tmp, "design_only.viba", "Only := $x int\n")
    design = _write(tmp, "use_design.viba", """
import design_only as d
__ret__ := d << environ
""")
    labelled(interpret(design, environ), "has no __ret__",
             "calling a module that is design only -> Err")

    _write(tmp, "loop.viba", "import loop as loop\n__ret__ := loop << environ\n")
    labelled(interpret(str(tmp / "loop.viba"), environ), "already running",
             "a module that calls itself -> Err")

    missing = _write(tmp, "missing_import.viba", "import nope as n\n__ret__ := n << environ\n")
    labelled(interpret(missing, environ), "not found", "an import that names no file -> Err")

    labelled(interpret(str(tmp / "nothing_here.viba"), environ), "no such file",
             "a main file that is not there -> Err")


def _loose_ends(tmp: Path):
    """再压一批：没写 as 的 import、字面量当 __ret__、坏路径、坏答案、A→B→A。"""
    host = Host()
    environ = host.environ()

    # import 没写 as：绑定的就是模块全名
    _write(tmp, "plain.viba", LEAF + "__ret__ := leaf << $env environ\n")
    plain = _write(tmp, "plain_user.viba", """
import plain
__ret__ := plain << environ
""")
    result = interpret(plain, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"an import without as binds its whole name: {result!r}")

    # 字面量、nil、Any 当 __ret__
    for body, want, label in (("42", 42, "a literal int"), ('"hi"', "hi", "a literal str"),
                              ("true", True, "a literal bool")):
        path = _write(tmp, f"lit_{want}.viba", f"__ret__ := {body}\n")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"__ret__ written as {label}: {result!r}")
    for body, label in (("nil", "nil"), ("never", "never"), ("Any", "Any")):
        path = _write(tmp, f"unit_{label}.viba", f"__ret__ := {body}\n")
        result = interpret(path, environ)
        check(isinstance(result, Ok), f"__ret__ written as {label}: {result!r}")
    for body, label in (("(1 | 2)", "a sum"), ("(1 * 2)", "a product")):
        path = _write(tmp, f"shape_{label.split()[-1]}.viba", f"__ret__ := {body}\n")
        labelled(interpret(path, environ), "cannot compute", f"__ret__ written as {label} -> Err")

    # 坏路径 / 坏答案 / 坏 environment
    labelled(interpret(str(tmp), environ), "cannot read", "the main path is a directory -> Err")
    labelled(interpret(str(tmp / "gone.viba"), environ), "no such file", "no such file -> Err")
    labelled(interpret(str(tmp / "plain.viba"), Environment(None, None)), "compute",
             "an environment with no compute side -> Err")

    host.knobs["get_func_raises"] = False
    bad = Host()
    bad.get_func = lambda p, n: "not callable"
    weird = _write(tmp, "weird.viba", LEAF + "__ret__ := leaf << $env environ\n")
    labelled(interpret(weird, bad.environ()), "raised", "a non-callable implementation -> Err")

    # A -> B -> A 的环
    _write(tmp, "cycle_a.viba", "import cycle_b as b\n__ret__ := b << environ\n")
    _write(tmp, "cycle_b.viba", "import cycle_a as a\n__ret__ := a << environ\n")
    labelled(interpret(str(tmp / "cycle_a.viba"), environ), "already running",
             "a module call cycle A->B->A -> Err")


def _names_and_sources(tmp: Path):
    """编译不过的源、写错的名字、不是函数的东西、模块没给 Environment。"""
    host = Host()
    environ = host.environ()

    broken = _write(tmp, "broken.viba", "add :=\n\tint\n\t<- )\n")
    labelled(interpret(broken, environ), "cannot parse",
             "a main file that does not compile -> Err")
    bad_import = _write(tmp, "bad_import.viba", "import broken as b\n__ret__ := b << environ\n")
    labelled(interpret(bad_import, environ), "cannot parse",
             "an imported module that does not compile -> Err")

    _write(tmp, "design_two.viba", "Only := int <- $env Environment\n")
    for body, want, label in (
            ("d.nope", "has no 'nope'", "a name the imported module does not have"),
            ("environ.nope", "environment has no 'nope'", "a name the environment does not have"),
            ("environ.sub_env", "still waiting for arguments",
             "environ.sub_env with no module name"),
            ("1 << $x 2", "is not a function", "giving an argument to a number"),
            ("nope << $env environ", "no definition named", "a name nobody defined")):
        path = _write(tmp, f"names_{abs(hash(body))}.viba",
                      f"import design_two as d\n__ret__ := {body}\n")
        labelled(interpret(path, environ), want, f"{label} -> Err")

    _write(tmp, "plain_two.viba", LEAF + "__ret__ := leaf << $env environ\n")
    wrong = _write(tmp, "wrong_arg.viba", "import plain_two as p\n__ret__ := p << 7\n")
    labelled(interpret(wrong, environ), "needs an Environment",
             "a module called without an Environment -> Err")


def _higher_order_and_answers(tmp: Path):
    """高阶（宿主调用拿到的 viba 函数）、宿主自己造的节点、以及少见的答案。"""
    host = Host()
    environ = host.environ()

    higher = _write(tmp, "higher.viba", """
inc :=
	int
	<- $env Environment
	<- $x int
	<- { add one }
twice :=
	int
	<- $env Environment
	<- $f (int <- $env Environment <- $x int)
	<- $x int
	<- { call f twice }
__ret__ := twice << $env environ << $f inc << $x 10
""")
    result = interpret(higher, environ)
    check(isinstance(result, Ok) and value_of(result) == 22,
          f"a viba function handed to the host is callable: {result!r}")

    made = _write(tmp, "made.viba", """
make_node :=
	int
	<- $env Environment
	<- { make a node }
__ret__ := make_node << $env environ
""")
    result = interpret(made, environ)
    check(isinstance(result, Ok) and value_of(result) == 11,
          f"a node the host built itself: {result!r}")

    afunc = _write(tmp, "afunc.viba", """
answer_a_function :=
	int
	<- $env Environment
	<- { answer a function }
__ret__ := answer_a_function << $env environ
""")
    labelled(interpret(afunc, environ), "no leaf",
             "a host answer that is a function -> Err")

    arity = _write(tmp, "arity.viba", """
wrong_arity :=
	int
	<- $env Environment
	<- $a int
	<- $b int
	<- { takes two }
__ret__ := wrong_arity << $env environ << $a 1 << $b 2
""")
    labelled(interpret(arity, environ), "raised", "a host function of the wrong arity -> Err")

    nested = _write(tmp, "nested.viba", ADD + """
unused :=
	int
	<- $env Environment
	<- { never asked for }
__ret__ := add << $env environ << $a 1 << $b 2
""")
    result = interpret(nested, environ)
    check(isinstance(result, Ok) and value_of(result) == 3 and
          ("root", "unused") not in host.calls,
          f"a definition nobody asks for is never run: {result!r}")

    # 宿主调用拿到手的 viba 函数：给多了是 Err，不是崩
    overfeed = _write(tmp, "overfeed.viba", """
overfeed :=
	int
	<- $env Environment
	<- $f (int <- $env Environment <- $x int)
	<- $x int
	<- { give f one argument too many }
inc :=
	int
	<- $env Environment
	<- $x int
	<- { add one }
__ret__ := overfeed << $env environ << $f inc << $x 10
""")
    labelled(interpret(overfeed, environ), "raised",
             "a host that gives the viba function too many arguments -> Err")

    # 宿主里面再跑一次 interpret：两个 run 互不干扰
    inner = _write(tmp, "inner_module.viba", LEAF + "__ret__ := leaf << $env environ\n")
    outer = _write(tmp, "outer_run.viba", """
inner_value :=
	int
	<- $env Environment
	<- { run another module from inside the host }
__ret__ := inner_value << $env environ
""")
    host.knobs["inner_file"] = inner
    result = interpret(outer, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a host function that runs interpret itself: {result!r}")
    host.knobs.pop("inner_file")

    # 宿主抛的不是 Exception：interpret 不吞
    interrupt = _write(tmp, "interrupt.viba", """
interrupt :=
	int
	<- $env Environment
	<- { interrupt }
__ret__ := interrupt << $env environ
""")
    try:
        interpret(interrupt, environ)
        check(False, "a host raising KeyboardInterrupt is not swallowed")
    except KeyboardInterrupt:
        check(True, "a host raising KeyboardInterrupt is not swallowed")


def _shapes_and_scale(tmp: Path):
    """值形状、规模：元组/泛型应用不是值；长链、深 import。"""
    host = Host()
    environ = host.environ()
    for body, label in ((u"(1, 2)", "a tuple"), ("list[int]", "a generic application"),
                        ("int <- $x int", "an exponent")):
        path = _write(tmp, f"shape_{abs(hash(body))}.viba", f"__ret__ := {body}\n")
        labelled(interpret(path, environ), "cannot compute", f"__ret__ written as {label} -> Err")

    many = "".join(f" << $a{i} {i}" for i in range(1, 21))
    long_chain = _write(tmp, "long_chain.viba", """
sum :=
	int
	<- $env Environment
	<- $a1 int <- $a2 int <- $a3 int <- $a4 int <- $a5 int
	<- $a6 int <- $a7 int <- $a8 int <- $a9 int <- $a10 int
	<- $a11 int <- $a12 int <- $a13 int <- $a14 int <- $a15 int
	<- $a16 int <- $a17 int <- $a18 int <- $a19 int <- $a20 int
	<- { add them all }
__ret__ := sum << $env environ""" + many + "\n")
    host.knobs["answers"] = {}
    original = host.get_func
    def summing(path, func_name):
        if func_name == "sum":
            return lambda env, *rest: sum(v.value for v in rest)
        return original(path, func_name)
    host.get_func = summing
    host2 = Environment(EnvironmentStorage("root"), EnvironmentCompute(host.get_func))
    result = interpret(long_chain, host2)
    check(isinstance(result, Ok) and value_of(result) == 210,
          f"a twenty-argument chain: {result!r}")

    # 深 import：五层
    depth = tmp / "deep"
    depth.mkdir(exist_ok=True)
    _write(depth, "leaf5.viba", LEAF + "__ret__ := leaf << $env environ\n")
    previous = "leaf5"
    for level in range(4, 0, -1):
        name = f"leaf{level}"
        _write(depth, f"{name}.viba",
               f"import {previous} as down\n__ret__ := down << environ\n")
        previous = name
    result = interpret(str(depth / "leaf1.viba"), environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a five-deep import chain: {result!r}")


def _caching_and_repeats(tmp: Path):
    """一个文件只编一次；同一个 tag 给两次，后给的算。"""
    import viba.interpreter as interpreter_module

    host = Host()
    environ = host.environ()

    _write(tmp, "shared.viba", LEAF + "__ret__ := leaf << $env environ\n")
    _write(tmp, "left.viba", "import shared as s\n__ret__ := s << environ\n")
    _write(tmp, "right.viba", "import shared as r\n__ret__ := r << environ\n")
    top = _write(tmp, "top.viba", ADD + """
import left as l
import right as r
__ret__ := add << $env environ << $a (l << environ) << $b (r << environ)
""")

    parsed = []
    original = interpreter_module.custom_module

    def counting(source):
        parsed.append(1)
        return original(source)

    interpreter_module.custom_module = counting
    try:
        result = interpret(top, environ)
    finally:
        interpreter_module.custom_module = original
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"one module reached by two importers runs for both: {result!r}")
    check(len(parsed) == 4,
          f"each file is parsed once, however many importers it has: {len(parsed)}")

    twice = _write(tmp, "twice_tag.viba", ADD + """
__ret__ := add << $env environ << $a 1 << $a 2 << $b 3
""")
    result = interpret(twice, environ)
    check(isinstance(result, Ok) and value_of(result) == 5,
          f"a tag given twice: the later value stands: {result!r}")


def _more_corners(tmp: Path):
    """再压一层：书写次序、假值答案、重复与次序无关、environment 的边角。"""
    host = Host()
    environ = host.environ()

    # tag 给的顺序不影响结果
    out_of_order = _write(tmp, "out_of_order.viba", ADD + """
__ret__ := add << $b 2 << $env environ << $a 1
""")
    result = interpret(out_of_order, environ)
    check(isinstance(result, Ok) and value_of(result) == 3,
          f"tags may be given in any order: {result!r}")

    # 环境那一格必须写成 $env：不带 tag 的 Environment 位不算数
    positional = _write(tmp, "positional_env.viba", """
f :=
	int
	<- Environment
	<- $x int
	<- { inline }
__ret__ := f << environ << $x 1
""")
    labelled(interpret(positional, environ), "takes no $env Environment",
             "an environment slot written without a tag -> Err")

    # 假值但不是 nil：0、空串、false 都是叶子
    for func, want, label in (("zero", 0, "0"), ("empty", "", "an empty str"),
                              ("falsey", False, "false")):
        path = _write(tmp, f"falsy_{func}.viba", f"""
{func} :=
	int
	<- $env Environment
	<- {{ inline }}
__ret__ := {func} << $env environ
""")
        result = interpret(path, environ)
        check(isinstance(result, Ok) and value_of(result) == want,
              f"a host answer of {label} is a leaf, not nil: {result!r}")

    # 答不出叶子的容器
    for func, label in (("answer_a_list", "a list"), ("answer_a_tuple", "a tuple"),
                        ("answer_a_dict", "a dict")):
        path = _write(tmp, f"{func}.viba", f"""
{func} :=
	int
	<- $env Environment
	<- {{ inline }}
__ret__ := {func} << $env environ
""")
        labelled(interpret(path, environ), "no leaf",
                 f"a host answer that is {label} -> Err")

    # 重复定义：先写的那个算
    twice_defined = _write(tmp, "twice_defined.viba", "x := 1\nx := 2\n__ret__ := x\n")
    result = interpret(twice_defined, environ)
    check(isinstance(result, Ok) and value_of(result) == 1,
          f"a name defined twice: the first definition stands: {result!r}")

    # 定义的次序无关：先用后写也行
    defined_after = _write(tmp, "defined_after.viba", "__ret__ := x\nx := 7\n")
    result = interpret(defined_after, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a definition written after its use: {result!r}")

    # import 写在定义之后也算
    _write(tmp, "late_lib.viba", LEAF + "__ret__ := leaf << $env environ\n")
    late = _write(tmp, "late_import.viba",
                  "__ret__ := late_lib << environ\nimport late_lib\n")
    result = interpret(late, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"an import written at the end of the file: {result!r}")

    # 本地定义压过 import 的别名
    shadow = _write(tmp, "shadow.viba",
                    "import late_lib as late_lib\nlate_lib := 5\n__ret__ := late_lib\n")
    result = interpret(shadow, environ)
    check(isinstance(result, Ok) and value_of(result) == 5,
          f"a local definition shadows an import alias: {result!r}")

    # 把 import 绑到 environ 上，也压不过内建的那个环境
    env_alias = _write(tmp, "env_alias.viba",
                       "import late_lib as environ\n__ret__ := environ\n")
    result = interpret(env_alias, environ)
    check(isinstance(result, Ok) and isinstance(result.ok_value, Environment),
          f"environ stays the built-in environment even imported as one: {result!r}")

    # environ 上不是函数的东西
    storage = _write(tmp, "env_member.viba", "__ret__ := environ.storage\n")
    labelled(interpret(storage, environ), "environment has no 'storage'",
             "an environment member that is not callable -> Err")

    # 已经答完的 sub_env 再给参数：那不是函数
    answered = _write(tmp, "env_answered.viba",
                      '__ret__ := environ.sub_env << "a" << "b"\n')
    labelled(interpret(answered, environ), "is not a function",
             "another argument given to an answered sub_env -> Err")

    # __ret__ 就是环境本身
    the_env = _write(tmp, "the_env.viba", "__ret__ := environ\n")
    result = interpret(the_env, environ)
    check(isinstance(result, Ok) and result.ok_value is environ,
          f"a module whose __ret__ is the environment: {result!r}")

    # __ret__ 是 import 进来的模块：还缺参数，不是值
    module_value = _write(tmp, "module_value.viba",
                          "import late_lib as lib\n__ret__ := lib\n")
    labelled(interpret(module_value, environ), "still waiting for arguments",
             "__ret__ written as a module -> Err")

    # 内建类型名不是值
    builtin_value = _write(tmp, "builtin_value.viba", "__ret__ := str\n")
    labelled(interpret(builtin_value, environ), "no definition named",
             "a builtin type name used as a value -> Err")

    # 词法上就没有这个词：'-' 不能被悄悄跳过，否则 -5 会跑成 5
    negative = _write(tmp, "negative.viba", "__ret__ := -5\n")
    result = interpret(negative, environ)
    check(isinstance(result, Err) and "cannot parse" in result.err_msg
          and "illegal character" in result.err_msg,
          f"a character with no token of its own -> Err: {result!r}")

    # CRLF 只是行尾：写得跟 LF 一样读
    crlf = _write(tmp, "crlf.viba",
                  (LEAF + "__ret__ := leaf << $env environ\n").replace("\n", "\r\n"))
    result = interpret(crlf, environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a module written with CRLF line endings: {result!r}")

    # 带 tag 的说明块不是值
    tagged_note = _write(tmp, "tagged_note.viba",
                         ADD + "__ret__ := add << $env environ << $a { note } << $b 2\n")
    labelled(interpret(tagged_note, environ), "documentation",
             "a tagged code block where an argument goes -> Err")

    # 把没给全参数的函数当实参：宿主拿到的就是那个函数，回手就被拒
    handed = _write(tmp, "handed.viba", LEAF + """
echo :=
	int
	<- $env Environment
	<- $x int
	<- { hand the argument back }
__ret__ := echo << $env environ << $x leaf
""")
    labelled(interpret(handed, environ), "no leaf",
             "a viba function handed where a value is expected, echoed back -> Err")

    # 参数出错：那个函数根本不会被调用
    argument_boom = _write(tmp, "argument_boom.viba", ADD + """
explode :=
	int
	<- $env Environment
	<- { go }
__ret__ := add << $env environ << $a (explode << $env environ) << $b 2
""")
    host.calls.clear()
    labelled(interpret(argument_boom, environ), "raised", "an argument that blows up -> Err")
    check(("root", "add") not in host.calls,
          f"the call itself never happens: {host.calls}")

    # 一个定义算一次：两处用它，宿主只被叫一次
    memo = _write(tmp, "memo.viba", ADD + LEAF + """
half := leaf << $env environ
__ret__ := add << $env environ << $a half << $b half
""")
    host.calls.clear()
    result = interpret(memo, environ)
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"a definition used twice: {result!r}")
    check(host.calls.count(("root", "leaf")) == 1,
          f"a definition is computed once: {host.calls}")

    # 同一个模块被用了两次就跑两次（每次都是一个新调用）
    host.calls.clear()
    twice_module = _write(tmp, "twice_module.viba", ADD + """
import late_lib as lib
__ret__ := add << $env environ << $a (lib << environ) << $b (lib << environ)
""")
    result = interpret(twice_module, environ)
    check(isinstance(result, Ok) and value_of(result) == 14,
          f"one module called twice: {result!r}")
    check(host.calls.count(("root", "leaf")) == 2,
          f"each call runs the module again: {host.calls}")

    # module.MyType.Inner：点到底也还是找那个定义
    dotted_member = _write(tmp, "dotted_member.viba",
                           "import late_lib as lib\n__ret__ := lib.Only.More\n")
    labelled(interpret(dotted_member, environ), "has no 'Only.More'",
             "a dotted rest that names no definition -> Err")

    # 点分 import 的最长前缀赢：a.b 与 a.b.c 各是各的模块
    dotted_dir = tmp / "dotted"
    dotted_dir.mkdir(exist_ok=True)
    _write(dotted_dir, "a/b.viba", "X := 1\n__ret__ := 1\n")
    _write(dotted_dir, "a/b/c.viba", "X := 2\n__ret__ := 2\n")
    both = _write(tmp, "both_dotted.viba",
                  "import a.b\nimport a.b.c\n__ret__ := a.b.c << environ\n")
    result = interpret(both, environ, viba_path=str(dotted_dir))
    check(isinstance(result, Ok) and value_of(result) == 2,
          f"the longest import prefix wins: {result!r}")

    # 点分名也可以是一个带点的平面文件：pkg.inner.viba
    flat_dir = tmp / "flatdotted"
    flat_dir.mkdir(exist_ok=True)
    _write(flat_dir, "pkg.inner.viba", LEAF + "__ret__ := leaf << $env environ\n")
    flat_use = _write(tmp, "flat_dotted.viba",
                      "import pkg.inner\n__ret__ := pkg.inner << environ\n")
    result = interpret(flat_use, environ, viba_path=str(flat_dir))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a dotted import that is one file named pkg.inner.viba: {result!r}")

    # 主文件也可以是相对路径
    _write(tmp, "relative_main.viba", LEAF + "__ret__ := leaf << $env environ\n")
    here = os.getcwd()
    try:
        os.chdir(tmp)
        result = interpret("relative_main.viba", environ)
    finally:
        os.chdir(here)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a main file named by a relative path: {result!r}")

    # 主文件也可以直接给一个 Path
    result = interpret(Path(tmp) / "relative_main.viba", environ)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a main file given as a Path: {result!r}")

    # 环境上的成员：宿主给自己的 Environment 加方法，viba 那边就能调
    class Shouting(Environment):
        __slots__ = ()

        def shout(self, x):
            return f"{x.value}!"

    shouting = Shouting(EnvironmentStorage("root"), EnvironmentCompute(host.get_func))
    said = _write(tmp, "shout.viba", "__ret__ := environ.shout << \"hi\"\n")
    result = interpret(said, shouting)
    check(isinstance(result, Ok) and value_of(result) == "hi!",
          f"a method the host hung on its environment: {result!r}")

    # 宿主把没有叶子的东西喂给（它手里那个）viba 函数
    fed = _write(tmp, "fed.viba", """
feed_a_list :=
	int
	<- $env Environment
	<- $f (int <- $env Environment <- $x int)
	<- { hand f a list }
inc :=
	int
	<- $env Environment
	<- $x int
	<- { add one }
__ret__ := feed_a_list << $env environ << $f inc
""")
    labelled(interpret(fed, environ), "raised",
             "a host handing a viba function something with no leaf -> Err")

    # 一个参数位都没有的函数：不带 tag 的实参就是给多了
    no_slots = _write(tmp, "no_slots.viba", """
f :=
	int
	<- { a function with no argument slot }
__ret__ := f << 1
""")
    labelled(interpret(no_slots, environ), "takes no more arguments",
             "an untagged argument to a function with no slots -> Err")

    # storage 都没有的环境：宿主那边崩了也是 Err，不是把异常扔出来
    headless = Environment(None, EnvironmentCompute(host.get_func))
    headless_use = _write(tmp, "headless.viba", '__ret__ := environ.sub_env << "a"\n')
    labelled(interpret(headless_use, headless), "raised",
             "an environment with no storage -> Err, not a crash")


def _virtual_files(tmp: Path):
    """get_file：源从宿主手里来，一个字节都不碰文件系统。"""
    host = Host()
    environ = host.environ()

    files = {
        "/vfs/main.viba": "import pkg.inner\n__ret__ := pkg.inner << environ\n",
        "/vfs/pkg/inner.viba": LEAF + "__ret__ := leaf << $env environ\n",
    }
    asked = []

    def get_file(path):
        asked.append(path)
        return files.get(path)

    result = interpret("/vfs/main.viba", environ, get_file=get_file)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a run served out of a dict: {result!r}")
    check("/vfs/pkg/inner.viba" in asked,
          f"the hook is asked for the module next to the importer: {asked}")
    check(all(isinstance(path, str) for path in asked),
          f"every path the hook sees is written as a string: {asked}")

    # 主文件也在宿主手里：文件系统上没有这个路径
    check(not Path("/vfs/main.viba").exists(),
          "the paths the hook serves are not on this filesystem")

    # 找不到的名字：每个地方都问过，然后说 not found。主文件也在手里那份字典里。
    asked.clear()
    missing = "/vfs/wants_nope.viba"
    files[missing] = "import nope\n__ret__ := nope << environ\n"
    labelled(interpret(missing, environ, get_file=get_file), "not found",
             "a name the hook does not serve -> not found")
    check(asked == ["/vfs/wants_nope.viba", "/vfs/nope.viba"],
          f"the main file, then one place for the name that is not there: {asked}")

    # 模块已经加载过就不再问
    asked.clear()
    twice = "/vfs/twice.viba"
    lib_path = "/vfs/lib.viba"
    files[twice] = ADD + ("import lib as one\nimport lib as two\n"
                          "__ret__ := add << $env environ"
                          " << $a (one << environ) << $b (two << environ)\n")
    files[lib_path] = LEAF + "__ret__ := leaf << $env environ\n"
    result = interpret(twice, environ, get_file=get_file)
    check(isinstance(result, Ok) and value_of(result) == 14 and
          asked.count(lib_path) == 1,
          f"a module already loaded is not asked for again: {result!r} {asked}")

    # "这里没有"的三种说法都行：None、FileNotFoundError，以及别的报错
    def with_main(behaviour):
        """主文件照样答，别的地方交给 behaviour。"""
        def hook(path):
            return files[missing] if path == missing else behaviour(path)
        return hook

    def raising(path):
        raise FileNotFoundError(path)

    labelled(interpret(missing, environ, get_file=with_main(raising)), "not found",
             "a hook that raises FileNotFoundError -> not found")
    labelled(interpret(missing, environ, get_file=with_main(
        lambda path: (_ for _ in ()).throw(ValueError("数据库连不上")))), "raised",
        "a hook that raises something else -> Err")
    labelled(interpret(missing, environ, get_file=with_main(lambda path: b"bytes")),
             "not the file's text", "a hook that answers bytes -> Err")
    labelled(interpret(missing, environ, get_file=with_main(lambda path: "X := (")),
             "cannot parse",
             "a hook that answers something that does not compile -> Err")

    # 主文件也要走 hook
    labelled(interpret("/vfs/nowhere.viba", environ, get_file=get_file), "no such file",
             "a main file the hook does not serve -> no such file")

    # hook 在场时不用文件系统：磁盘上那份真的不再被读
    real = _write(tmp, "real_on_disk.viba", LEAF + "__ret__ := leaf << $env environ\n")
    served = dict(files)
    served[real] = TEXT + "__ret__ := text << $env environ\n"
    result = interpret(real, environ, get_file=served.get)
    check(isinstance(result, Ok) and value_of(result) == "hi",
          f"get_file wins over the filesystem: {result!r}")

    # 不是函数的 get_file
    labelled(interpret(str(missing), environ, get_file=7), "get_file is a function",
             "a get_file that is not callable -> Err")


def _paths(tmp: Path):
    """VIBA_PATH：按顺序找，import 旁边的先赢。"""
    host = Host()
    environ = host.environ()
    first = tmp / "one"
    second = tmp / "two"
    first.mkdir(exist_ok=True)
    second.mkdir(exist_ok=True)
    _write(first, "lib.viba", LEAF + "__ret__ := leaf << $env environ\n")
    _write(second, "lib.viba", TEXT + "__ret__ := text << $env environ\n")

    user = _write(tmp, "uses_path.viba", "import lib as lib\n__ret__ := lib << environ\n")
    result = interpret(user, environ, viba_path=f"{first}:{second}")
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"the first directory on VIBA_PATH wins: {result!r}")

    near = _write(second, "near.viba", "import lib as lib\n__ret__ := lib << environ\n")
    result = interpret(near, environ, viba_path=f"{first}:{second}")
    check(isinstance(result, Ok) and value_of(result) == "hi",
          f"a module next to the importer beats VIBA_PATH: {result!r}")

    # 空条目、不存在的目录：跳过，不炸
    flat = tmp / "flat"
    flat.mkdir(exist_ok=True)
    _write(flat, "pkg/inner.viba", LEAF + "__ret__ := leaf << $env environ\n")
    ragged = f":{first}:{tmp / 'missing'}::{flat}:"
    dotted = _write(tmp, "dotted.viba",
                    "import pkg.inner\n__ret__ := pkg.inner << environ\n")
    result = interpret(dotted, environ, viba_path=ragged)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a dotted module found on VIBA_PATH, empty and missing entries skipped: {result!r}")

    aliased = _write(tmp, "aliased.viba",
                     "import pkg.inner as inner\n__ret__ := inner << environ\n")
    result = interpret(aliased, environ, viba_path=ragged)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"the same module under an alias: {result!r}")

    # 相对路径按当前目录算
    relative = os.path.relpath(str(flat), os.getcwd())
    result = interpret(dotted, environ, viba_path=relative)
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a relative VIBA_PATH entry: {result!r}")

    # 这一段用的是仓库里现成的两份 .viba（data/interpreter/paths）：main.viba
    # 旁边没有 pkg/，模块只在给进来的那个目录里——一个 Path 就是一个目录；
    # 别的类型说清楚要的是字符串。
    paths_case = CASES / "paths"
    main_file = str(paths_case / "main.viba")
    result = interpret(main_file, environ, viba_path=paths_case / "elsewhere")
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a dotted module on a VIBA_PATH that is one Path: {result!r}")
    labelled(interpret(main_file, environ, viba_path=7), "viba_path is a string",
             "a VIBA_PATH that is not a string or a path -> Err")
    labelled(interpret(main_file, environ), "not found",
             "the same module with no VIBA_PATH at all -> Err")


if __name__ == "__main__":
    sys.exit(run())

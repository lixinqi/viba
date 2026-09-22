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
        _higher_order_and_answers(tmp)
        _shapes_and_scale(tmp)
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


if __name__ == "__main__":
    sys.exit(run())

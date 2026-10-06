"""回环：先缺一步 → 拿到 `$call` → 补上实现 → 把这次调用再跑一遍。

每一份用例都走同一条路（`viba-interpreter.md`「把一次调用写成可执行的」）：

    1. `interpret` 跑一份程序，宿主那里**没有**这一步的实现 → `$not_implemented_err`；
    2. 从错误里拿出 `$call`（`error.call`）—— 它是**闭包**：类型 `Any <- $env Env`，环境还没给；
    3. 宿主补上这一步（同一个环境对象：实现表是可变的，`get_func` 每次都读它）；
    4. 把那份 `$call` **当数据**交给一次运行（`tests/data/roundtrip/runner.viba`：
       `held = hold << $env args.env`，再 `held << args.env`），这次运行给它一个环境 ——
       于是它就跑起来了，答案就是这一步本来会给出的值。

第 4 步给它的环境在那条**数据路径**上：错误里的 `$module_path` 说这次调用发生在哪儿，
同一个定义、另一条数据路径是另一步（`two_paths` 那份用例把这条钉住）。`$call` 本身不是一份模块，
所以第 4 步不把它写成代码：它是可序列化数据。另有一条路也测：用 `serialize` 把 `$call` 写成一份
viba 模块源代码（`call = …`）再 `exec`，答案一样。

`__dyn_method__` 那两份（`member_value`、`member_by_name`）只走第三条路：那种调用的那份值里写着
**写它的模块里的名字**，所以它拿回自己的模块里读（`test_interpreter_dyn.py` 钉住了它的写法）。

    python3 tests/test_interpreter_roundtrip.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import error_of, Checks, value_of

from viba import serialize, viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            exec, interpret, sub_env)
from viba.reflect import VibaNode, access as reflect_access
from viba.type import (AstNodeType, NOT_IMPLEMENTED_TAG, Ok, UnderlyingOpErr,
                       custom_module)
from viba.viba_type_descriptor import descriptor_of

CASES = Path(__file__).resolve().parent / "data" / "roundtrip"
RUNNER = CASES / "runner.viba"

checks = Checks("interpreter_roundtrip")
check = checks.check

HELD = "hold"

# 一条把 `$call` 写成代码的路用到的开头：主模块、它的环境，然后是那份 `$call` 的定义。
SOURCE_HEAD = ("__decl__ =\n    Any\n  <- $env Env\n\n"
               "args = __get_args__ << __decl__\n\n")


class Host:
    """一个可以先没有、后来补上的宿主：实现表可变，`get_func` 每次都读它。

    同一个对象从头用到尾 —— 先跑那份程序（缺一步），补上实现，再跑 `$call`：
    这就是「准备好了 Environment 之后」。
    """

    def __init__(self, path: str = "root"):
        self.path = path
        self.table = {}                  # func_name -> (module_path -> step | None)
        self.holding = None              # what the `hold` step hands back
        self.calls = []
        self.environ = Environment(EnvironmentStorage(path),
                                   EnvironmentCompute(self.get_func))

    def get_func(self, module_path, func_name):
        self.calls.append((module_path, func_name))
        if func_name == HELD:
            # 那条闭包就在手里：这个步骤把它交回给运行。
            held = self.holding
            return (lambda env: held) if held is not None else None
        rule = self.table.get(func_name)
        return rule(module_path) if rule is not None else None

    def prepare(self, table):
        """补上这批实现。"""
        self.table.update(table)

    def at(self, path: str):
        """同一条数据链上走到那条路径：`$module_path` 说的就是这条路径。"""
        environ = self.environ
        if path != self.path:
            assert path.startswith(self.path + "/"), path
            for segment in path[len(self.path) + 1:].split("/"):
                environ = sub_env(environ, segment)
        return environ


def _everywhere(**steps):
    """这批实现在哪儿都有：`{名字: (那条路径) -> 那一步}`。"""
    return {name: (lambda path, step=step: step) for name, step in steps.items()}


def _only_at(path, **steps):
    """这批实现只在那条数据路径上：另一条路径上没有这一步。"""
    return {name: (lambda seen, step=step: step if seen == path else None)
            for name, step in steps.items()}


class Case:
    """一份回环用例：跑哪份程序、缺哪一步、怎么补、补完该答什么。"""

    def __init__(self, name, missing, after, want, *, before=None, path="root",
                 whole_program=True, strict_path=False, as_source=False,
                 as_exec=False, want_text=None, member=False):
        self.name = name
        self.missing = missing
        self.after = after
        self.want = want
        self.before = before or {}
        self.path = path
        self.whole_program = whole_program
        self.strict_path = strict_path
        self.as_source = as_source
        self.as_exec = as_exec
        self.want_text = want_text
        self.member = member

    @property
    def file(self) -> Path:
        return CASES / f"{self.name}.viba"


def _add(env, a, b):
    return a.value + b.value


def _a_product():
    """一份积，宿主自己造的：`$a 1 * $b 2`。"""
    node = viba_ast.Product(viba_ast.Tagged("$a", viba_ast.Constant(1)),
                            viba_ast.Tagged("$b", viba_ast.Constant(2)))
    return VibaNode(reflect_access,
                    descriptor_of(AstNodeType(node, custom_module(""))), node)


# 二十六份用例，每份换一个说法：参数怎么写、答案是什么类型、这一步在哪条数据路径上。
CASES_LIST = [
    Case("tagged_two", "add", _everywhere(add=_add), 3, as_source=True),
    Case("positional_two", "add", _everywhere(add=_add), 3, as_exec=True),
    Case("out_of_order", "add", _everywhere(add=_add), 3),
    Case("no_arguments", "tick", _everywhere(tick=lambda env: 7), 7, as_source=True),
    Case("one_product", "sum_of",
         _everywhere(sum_of=lambda env, both: both.get_a().value + both.get_b().value), 3),
    Case("mixed_product", "pack",
         _everywhere(pack=lambda env, both: f"{both.get_a().value}{both.get_b().value}"), "1x"),
    Case("string_argument", "shout",
         _everywhere(shout=lambda env, x: x.value.upper()), "HI"),
    Case("bool_argument", "flip",
         _everywhere(flip=lambda env, x: not x.value), False),
    Case("float_argument", "half",
         _everywhere(half=lambda env, x: x.value / 2), 1.75),
    Case("nil_argument", "count_nothing",
         _everywhere(count_nothing=lambda env, x: 0), 0),
    Case("sum_typed_argument", "widen",
         _everywhere(widen=lambda env, x: x.value + 1), 2),
    Case("bare_builtin", "builtin.concat",
         _everywhere(**{"builtin.concat": lambda env, x, y: x.value + y.value}),
         "ab", as_source=True),
    Case("dotted_builtin", "builtin.mul",
         _everywhere(**{"builtin.mul": lambda env, x, y: x.value * y.value}), 42),
    Case("imported_step", "inc", _everywhere(inc=lambda env, x: x.value + 1), 42,
         path="root/inner", as_source=True),
    Case("kid_step", "inc", _everywhere(inc=lambda env, x: x.value + 1), 42,
         path="root/kid"),
    Case("two_level", "deep", _everywhere(deep=lambda env, x: x.value), 5,
         path="root/mid/leaf"),
    Case("tmp_path", "inc", _everywhere(inc=lambda env, x: x.value + 1), 42,
         path="root/"),          # the exact path is the one the error names
    Case("computed_argument", "not_of",
         _everywhere(not_of=lambda env, x: not x.value), False,
         before=_everywhere(lt=lambda env, x, y: x.value < y.value)),
    Case("three_arguments", "add3",
         _everywhere(add3=lambda env, a, b, c: a.value + b.value + c.value), 6),
    Case("module_arguments", "total",
         _everywhere(total=lambda env, x: x.value + 1), 42, path="root/with_args"),
    Case("nil_answer", "noop", _everywhere(noop=lambda env: None), None),
    Case("product_answer", "pair", _everywhere(pair=lambda env, x: _a_product()),
         None, want_text="$a 1 * $b 2"),
    Case("two_paths", "inc", _only_at("root/b", inc=lambda env, x: x.value + 2), 4,
         before=_only_at("root/a", inc=lambda env, x: x.value + 1),
         path="root/b", whole_program=False, strict_path=True),
    Case("member_value", "inc", _everywhere(inc=lambda box, env, x: x.value + 1), 2,
         member=True),
    Case("member_by_name", "bump",
         _everywhere(bump=lambda box, env, x: x.value + 1), 2, member=True),
    Case("nested_calls", "mul", _everywhere(mul=lambda env, x, y: x.value * y.value),
         6, whole_program=False),
]


def run(tmp: Path):
    for case in CASES_LIST:
        if case.member:
            _a_member_call(case, tmp)
        else:
            _one_case(case)
    _one_step_at_a_time()
    _both_entries_agree_on_the_call()


def _one_case(case: Case):
    """一份用例的整条回环：缺一步、补上、把 `$call` 跑一遍。"""
    host = Host()
    host.prepare(case.before)
    error = _the_missing_step(case, host)
    if error is None:
        return                          # it already said what went wrong
    check(_at_the_path(error.module_path, case),
          f"{case.name}: the step that stopped is the one at that data path: "
          f"{error.module_path!r}")
    host.prepare(case.after)

    # 把那份闭包当数据交给一次运行，这次运行给它一个环境。
    host.holding = error.call
    replayed = interpret(str(RUNNER), host.at(error.module_path))
    check(_the_wanted_answer(replayed, case),
          f"{case.name}: the call the error carried runs, given that environment: "
          f"{replayed!r}")

    # 补上之后，原程序自己也跑得通。
    if case.whole_program:
        again = _first_run(case, host)
        check(_the_wanted_answer(again, case),
              f"{case.name}: and the program itself runs now: {again!r}")

    # 那条数据路径是这次调用的一部分：换一条就没有这一步。
    if case.strict_path:
        elsewhere = Host()
        elsewhere.prepare(case.after)
        elsewhere.holding = error.call
        wrong = interpret(str(RUNNER), elsewhere.environ)
        check("no implementation" in getattr(error_of(wrong), "msg", ""),
              f"{case.name}: another data path is another step: {wrong!r}")

    # 另一条路：把 `$call` 写成一份 viba 模块源代码，再 `exec` 跑它。
    if case.as_source:
        source = serialize.serialize("call", error.call)
        check(isinstance(source, Ok),
              f"{case.name}: the call is serializable viba data: {source!r}")
        code = SOURCE_HEAD + source.ok_value + "\n__impl__ = call << args.env\n"
        written_out = exec(code, host.at(error.module_path))
        check(_the_wanted_answer(written_out, case),
              f"{case.name}: the same call written out as source runs too: "
              f"{written_out!r}")


def _a_member_call(case: Case, tmp: Path):
    """`__dyn_method__` 那两份：那份值里写着它的模块里的名字，所以拿回它的模块里读。"""
    host = Host()
    error = _the_missing_step(case, host)
    if error is None:
        return
    host.prepare(case.after)
    call = " ".join(viba_ast.unparse_type(error.call.data).split())
    check(call.startswith('__dyn_method__ << "'),
          f"{case.name}: the call keeps the member layer: {call!r}")

    # 把 `$call` 放进它自己的那份模块（定义都在），再跑一遍。
    definitions = case.file.read_text().split("__impl__ =", 1)[0]
    replay = tmp / f"{case.name}_again.viba"
    replay.write_text(definitions + "__impl__ = " + call + " << args.env\n")
    ran = interpret(str(replay), host.environ)
    check(_the_wanted_answer(ran, case),
          f"{case.name}: read in the module it was written in, it runs: {ran!r}")


def _one_step_at_a_time():
    """缺的那一步补上，下一次缺的是外面那一步：一轮一轮地走完。"""
    host = Host()
    first = error_of(interpret(str(CASES / "nested_calls.viba"), host.environ))
    check(isinstance(first, UnderlyingOpErr) and first.func_name == "mul",
          f"the innermost step stops first: {first!r}")
    host.prepare(_everywhere(mul=lambda env, x, y: x.value * y.value))
    host.holding = first.call
    inner = interpret(str(RUNNER), host.at(first.module_path))
    check(value_of(inner) == 6, f"the inner call runs once it is implemented: {inner!r}")

    second = error_of(interpret(str(CASES / "nested_calls.viba"), host.environ))
    check(isinstance(second, UnderlyingOpErr) and second.func_name == "add",
          f"the run then stops one step further out: {second!r}")
    check(" ".join(viba_ast.unparse_type(second.call.data).split()) ==
          '__dyn_call__ << "add" << $a 6 << $b 4',
          f"and that call carries the argument the first one answered: "
          f"{viba_ast.unparse_type(second.call.data)!r}")
    host.prepare(_everywhere(add=_add))
    host.holding = second.call
    outer = interpret(str(RUNNER), host.at(second.module_path))
    check(value_of(outer) == 10, f"the outer call runs too: {outer!r}")

    whole = interpret(str(CASES / "nested_calls.viba"), host.environ)
    check(value_of(whole) == 10, f"and the program runs to the end: {whole!r}")


def _both_entries_agree_on_the_call():
    """`interpret` 与 `exec` 跑同一份程序，缺的那一步带回来的 `$call` 是同一份数据。"""
    host = Host()
    by_file = error_of(interpret(str(CASES / "tagged_two.viba"), host.environ))
    by_code = error_of(exec((CASES / "tagged_two.viba").read_text(), host.environ))
    check(" ".join(viba_ast.unparse_type(by_file.call.data).split()) ==
          " ".join(viba_ast.unparse_type(by_code.call.data).split()),
          f"one file, one entry: {by_file.call!r} != {by_code.call!r}")


def _first_run(case: Case, host: Host):
    """跑那份程序：`as_exec` 的走 `exec`（同一段代码，另一个人口）。"""
    if case.as_exec:
        return exec(case.file.read_text(), host.environ)
    return interpret(str(case.file), host.environ)


def _the_missing_step(case: Case, host: Host):
    """跑那份程序，确认它缺的就是那一步，并把错误交回来。"""
    stopped = _first_run(case, host)
    error = error_of(stopped)
    if not (isinstance(error, UnderlyingOpErr) and error.tag == NOT_IMPLEMENTED_TAG
            and error.func_name == case.missing):
        check(False, f"{case.name}: the run stops at the step nothing implements "
                     f"({case.missing!r}): {stopped!r}")
        return None
    return error


def _at_the_path(path: str, case: Case) -> bool:
    if case.path is None:
        return True
    if case.path == "root/":            # a temporary environment: the path is a fresh name
        return path.startswith("root/tmp_")
    return path == case.path


def _the_wanted_answer(result, case: Case) -> bool:
    if case.want_text is not None:
        data = getattr(getattr(result, "ok_value", None), "data", None)
        return (" ".join(viba_ast.unparse_type(data).split()) == case.want_text
                if data is not None else False)
    return value_of(result) == case.want


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-roundtrip-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

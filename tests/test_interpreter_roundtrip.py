"""回环：先缺一步 → 拿到 `$call` → 补上实现 → 把这次调用再跑一遍。

每一份用例都走同一条路（`viba-interpreter.md`「把一次调用写成可执行的」）：

    1. `interpret` 跑一份程序，宿主那里**没有**这一步的实现 → `$not_implemented_err`；
    2. 从那次停止里拿出 `$call`（`stop_node(result, "$call")`）—— 它是**闭包**：类型
       `Any <- $env Env`，环境还没给；
    3. 宿主补上这一步（同一个环境对象：实现表是可变的，`get_func` 每次都读它）；
    4. 把那份 `$call` **当数据**交给一次运行（`tests/data/roundtrip/runner.viba`：
       `held = hold << $env args.env`，再 `held << args.env`），这次运行给它一个环境 ——
       于是它就跑起来了，答案就是这一步本来会给出的值。

第 4 步给它的环境在**它原来跑的那条数据路径**上（用例自己知道；给在哪儿由给它的环境说了算）。
那次停止里的 `$module_path` 说的是**这一步声明在哪个模块**：就地写的步骤就是这份文件，import 进来的就是
那个模块（`/inner`、`/leaf`），而 `$full_qualified_func_name` 是这一步的整名（`inner.inc`：模块加它在那儿叫的名字）
—— 名字带着模块，所以交给别的运行也知道是哪一步。`$call` 本身不是一份模块，
所以第 4 步不把它写成代码：它是可序列化数据。另外两条路也测：用 `serialize` 把 `$call` 写成一份
viba 模块源代码（`call = …`）再 `exec`；以及把那份 `$call` **直接当节点**交给 `exec`（它本来就是
节点，`exec(viba_code, environ)` 收文本，也收节点），答案一样。

    python3 tests/test_interpreter_roundtrip.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import (answer_of, full_name_of, module_of, module_path_of,
                                  is_ok, stop_node, stop_tag, stop_text,
                                  Checks, value_of)

from viba import serialize, viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            exec, interpret, sub_env)
from viba.reflect import VObject, access as reflect_access
from viba.type import (AstNodeType, NOT_IMPLEMENTED_TAG, Ok, custom_module)
from viba.viba_type_descriptor import descriptor_of

CASES = Path(__file__).resolve().parent / "data" / "roundtrip"
RUNNER = CASES / "runner.viba"

checks = Checks("interpreter_roundtrip")
check = checks.check

HELD = "hold"
CODE_LABEL = "<viba_code>"

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
        # 一步是按“模块 + 它自己的名字”问的；这张表按名字记，模块由那一步自己看。
        self.calls.append((module_path, func_name))
        if func_name == HELD:
            # 那条闭包就在手里：这个步骤把它交回给运行。
            held = self.holding
            return (lambda env: held) if held is not None else None
        rule = self.table.get(func_name)
        if rule is None:
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
                 module=None, replay_at=None, whole_program=True, as_source=False,
                 as_exec=False, want_text=None, as_node=False):
        self.name = name
        self.missing = missing
        self.after = after
        self.want = want
        self.before = before or {}
        self.path = path                   # where the call ran (the data path)
        self.module = module               # the path of the module that declares the step
        self.replay_at = replay_at         # where to give the `$call` an environment
        self.whole_program = whole_program
        self.as_source = as_source
        self.as_exec = as_exec
        self.want_text = want_text
        self.as_node = as_node

    @property
    def file(self) -> Path:
        return CASES / f"{self.name}.viba"

    @property
    def missing_name(self) -> str:
        """The whole name of the step that is missing: the module, then the name there."""
        if self.module is not None:
            module = self.module.lstrip("/").replace("/", ".")
        elif self.as_exec:
            module = CODE_LABEL
        else:
            module = module_of(self.file)
        return f"{module}.{self.missing}"

    @property
    def module_path(self) -> str:
        """What `$module_path` should say: the module that declares the step.

        A step of the case file itself is that file; a step of a module the case
        imported is that module (`/inner`, `/leaf`); a builtin member is the
        built-in vocabulary (`/builtin`); and a run started from code has no file,
        so its module is the label it was compiled under.
        """
        if self.module is not None:
            return self.module
        if self.as_exec:
            return "/<viba_code>"
        return module_path_of(self.file)


def _add(env, a, b):
    return a.value + b.value


def _a_product():
    """一份积，宿主自己造的：`$a 1 * $b 2`。"""
    node = viba_ast.Product(viba_ast.Tagged("$a", viba_ast.Constant(1)),
                            viba_ast.Tagged("$b", viba_ast.Constant(2)))
    return VObject(reflect_access,
                    descriptor_of(AstNodeType(node, custom_module(""))), node)


# 二十七份用例，每份换一个说法：参数怎么写、答案是什么类型、这一步在哪条数据路径上。
CASES_LIST = [
    Case("tagged_two", "add", _everywhere(add=_add), 3, as_source=True, as_node=True),
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
    Case("bare_builtin", "concat",
         _everywhere(concat=lambda env, x, y: x.value + y.value),
         "ab", module="/builtin", as_source=True),
    Case("dotted_builtin", "mul",
         _everywhere(mul=lambda env, x, y: x.value * y.value), 42,
         module="/builtin"),
    Case("imported_step", "inc", _everywhere(inc=lambda env, x: x.value + 1), 42,
         path="root/inner", module="/inner", as_source=True, as_node=True),
    Case("kid_step", "inc", _everywhere(inc=lambda env, x: x.value + 1), 42,
         path="root/kid", module="/inner"),
    Case("two_level", "deep", _everywhere(deep=lambda env, x: x.value), 5,
         path="root/mid/leaf", module="/leaf"),
    Case("tmp_path", "inc", _everywhere(inc=lambda env, x: x.value + 1), 42,
         path="root/tmp_", module="/inner", replay_at="root"),
    Case("computed_argument", "not_of",
         _everywhere(not_of=lambda env, x: not x.value), False,
         before=_everywhere(lt=lambda env, x, y: x.value < y.value)),
    Case("three_arguments", "add3",
         _everywhere(add3=lambda env, a, b, c: a.value + b.value + c.value), 6,
         as_node=True),
    Case("module_arguments", "total",
         _everywhere(total=lambda env, x: x.value + 1), 42, path="root/with_args",
         module="/with_args"),
    Case("nil_answer", "noop", _everywhere(noop=lambda env: None), None),
    Case("product_answer", "pair", _everywhere(pair=lambda env, x: _a_product()),
         None, want_text="$a 1 * $b 2", as_node=True),
    Case("function_argument", "apply", _everywhere(apply=lambda env, f: 7), 7,
         as_source=True, as_node=True),
    Case("member_value", "inc", _everywhere(inc=lambda box, env, x: x.value + 1), 2,
         as_source=True, as_node=True),
    Case("member_by_name", "bump",
         _everywhere(bump=lambda box, env, x: x.value + 1), 2, as_node=True),
    Case("nested_calls", "mul", _everywhere(mul=lambda env, x, y: x.value * y.value),
         6, whole_program=False),
]


def run():
    for case in CASES_LIST:
        _one_case(case)
    _one_step_at_a_time()
    _both_entries_agree_on_the_call()


def _one_case(case: Case):
    """一份用例的整条回环：缺一步、补上、把 `$call` 跑一遍。"""
    host = Host()
    host.prepare(case.before)
    stopped = _the_missing_step(case, host)
    if stopped is None:
        return                          # it already said what went wrong
    check(stop_text(stopped, "$module_path") == case.module_path,
          f"{case.name}: the step that stopped is the one declared in that module: "
          f"{stop_text(stopped, '$module_path')!r}")
    call = stop_node(stopped, "$call")
    host.prepare(case.after)

    # 把那份闭包当数据交给一次运行，这次运行给它一个环境：给在它原来跑的那条数据路径上
    # —— `$module_path` 说的是这一步声明在哪个模块，跑在哪儿由给它的环境说了算。
    where = case.replay_at or case.path
    host.holding = call
    replayed = interpret(str(RUNNER), host.at(where))
    check(_the_wanted_answer(replayed, case),
          f"{case.name}: the call the stop carried runs, given that environment: "
          f"{replayed!r}")

    # 补上之后，原程序自己也跑得通。
    if case.whole_program:
        again = _first_run(case, host)
        check(_the_wanted_answer(again, case),
              f"{case.name}: and the program itself runs now: {again!r}")

    # 另一条路：把 `$call` 写成一份 viba 模块源代码，再 `exec` 跑它。
    if case.as_source:
        source = serialize.serialize("call", call)
        check(isinstance(source, Ok),
              f"{case.name}: the call is serializable viba data: {source!r}")
        code = SOURCE_HEAD + source.ok_value + "\n__impl__ = call << args.env\n"
        written_out = exec(code, host.at(where))
        check(_the_wanted_answer(written_out, case),
              f"{case.name}: the same call written out as source runs too: "
              f"{written_out!r}")

    # 第三条路：那份 `$call` 本来就是节点，直接交给 `exec` —— 不用写出来读回去。
    if case.as_node:
        handed_over = exec(call, host.at(where))
        check(_the_wanted_answer(handed_over, case),
              f"{case.name}: the call handed over as the node it already is runs too: "
              f"{handed_over!r}")


def _one_step_at_a_time():
    """缺的那一步补上，下一次缺的是外面那一步：一轮一轮地走完。"""
    host = Host()
    first = interpret(str(CASES / "nested_calls.viba"), host.environ)
    check(stop_tag(first) == NOT_IMPLEMENTED_TAG and
          stop_text(first, "$full_qualified_func_name") ==
          full_name_of(CASES / "nested_calls", "mul"),
          f"the innermost step stops first: {first!r}")
    host.prepare(_everywhere(mul=lambda env, x, y: x.value * y.value))
    host.holding = stop_node(first, "$call")
    inner = interpret(str(RUNNER), host.environ)
    check(is_ok(inner) and value_of(inner) == 6,
          f"the inner call runs once it is implemented: {inner!r}")

    second = interpret(str(CASES / "nested_calls.viba"), host.environ)
    check(stop_tag(second) == NOT_IMPLEMENTED_TAG and
          stop_text(second, "$full_qualified_func_name") ==
          full_name_of(CASES / "nested_calls", "add"),
          f"the run then stops one step further out: {second!r}")
    check(" ".join(viba_ast.unparse_type(stop_node(second, "$call").data).split()) ==
          '__dyn_call__ << "nested_calls.add" << $a 6 << $b 4',
          f"and that call carries the argument the first one answered: "
          f"{viba_ast.unparse_type(stop_node(second, '$call').data)!r}")
    host.prepare(_everywhere(add=_add))
    host.holding = stop_node(second, "$call")
    outer = interpret(str(RUNNER), host.environ)
    check(is_ok(outer) and value_of(outer) == 10,
          f"the outer call runs too: {outer!r}")

    whole = interpret(str(CASES / "nested_calls.viba"), host.environ)
    check(is_ok(whole) and value_of(whole) == 10,
          f"and the program runs to the end: {whole!r}")


def _both_entries_agree_on_the_call():
    """`interpret` 与 `exec` 跑同一份程序：缺的是同一步，名字各自说自己那个模块。

    一份文件跑起来时这一步声明在那份文件的模块里（`tagged_two.add`）；`exec` 那段代码没有文件，
    它的模块就是编译用的标签（`<viba_code>.add`）。名字不同正是「这一步在哪个模块」的意思。
    """
    host = Host()
    by_file = interpret(str(CASES / "tagged_two.viba"), host.environ)
    by_code = exec((CASES / "tagged_two.viba").read_text(), host.environ)

    def written(result):
        return " ".join(viba_ast.unparse_type(stop_node(result, "$call").data).split())

    check(written(by_file) == '__dyn_call__ << "tagged_two.add" << $a 1 << $b 2' and
          written(by_code) == '__dyn_call__ << "<viba_code>.add" << $a 1 << $b 2',
          f"one file, one step, each named by its own module: "
          f"{written(by_file)!r} != {written(by_code)!r}")


def _first_run(case: Case, host: Host):
    """跑那份程序：`as_exec` 的走 `exec`（同一段代码，另一个人口）。"""
    if case.as_exec:
        return exec(case.file.read_text(), host.environ)
    return interpret(str(case.file), host.environ)


def _the_missing_step(case: Case, host: Host):
    """跑那份程序，确认它缺的就是那一步，并把整份结果交回来。"""
    stopped = _first_run(case, host)
    if not (stop_tag(stopped) == NOT_IMPLEMENTED_TAG
            and stop_text(stopped, "$full_qualified_func_name") == case.missing_name):
        check(False, f"{case.name}: the run stops at the step nothing implements "
                     f"({case.missing!r}): {stopped!r}")
        return None
    return stopped


def _the_wanted_answer(result, case: Case) -> bool:
    if case.want_text is not None:
        return (is_ok(result)
                and " ".join(viba_ast.unparse_type(answer_of(result).data).split())
                == case.want_text)
    return value_of(result) == case.want


if __name__ == "__main__":
    run()
    sys.exit(checks.report())

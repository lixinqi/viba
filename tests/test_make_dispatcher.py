"""make_dispatcher：给 handler 的类型，拿回一个能跑的函数。

`make_dispatcher[f0, f1].type` 是 `Either[Ret, DispatchError] <- (A0|B0) <- ...`
这个函数类型：结果一致、参数名逐位一致、参数数量一致，由 pattern 里的同一个
`Ret`、同一个 `name0`..`nameN` 和同样的 `<-` 位置数钉住。

叫它的时候（`make_dispatcher[f0, f1] << $env … << $a 1 << $b true`）按这次给的参数选
第一个参数积收得下的 handler，把调用交给它；一个都收不下，答案是一个
`DispatchError`（`$dispatch_error_message "no handler matches"`）。

两份泛型各自 32×8 个文件：

* `make_dispatcher/<count*100+params>.viba` 收 handler 一次，给出 `.type`，并把调用交给实现；
* `make_dispatcher_impl/<count*100+params>.viba` 把每个 handler 收两次（一次拆类型、
  一次用来调用），决断与调用都在这里。

    python3 tests/test_make_dispatcher.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, answer_of, is_ok, message_of

from viba import viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                              interpret)

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "viba" / "builtin" / "make_dispatcher"
IMPL = ROOT / "viba" / "builtin" / "make_dispatcher_impl"

checks = Checks("make_dispatcher")
check = checks.check

NO_HANDLER = '$dispatch_error_message "no handler matches"'

HANDLERS = {
    "f0.viba": """
__decl__ =
    int
  <- $env Env
  <- $a int
  <- $b bool

args = __get_args__ << __decl__

__impl__ = args.a
""",
    "f1.viba": """
__decl__ =
    int
  <- $env Env
  <- $a bool
  <- $b str

__impl__ = 2
""",
    "g0.viba": """
__decl__ =
    int
  <- $env Env
  <- $a int

args = __get_args__ << __decl__

__impl__ = args.a
""",
    "g1.viba": """
__decl__ =
    int
  <- $env Env
  <- $a bool

__impl__ = 2
""",
    "c0.viba": """
__decl__ = int <- $env Env <- $a int <- $b int <- $c int
__impl__ = 100
""",
    "c1.viba": """
__decl__ = int <- $env Env <- $a str <- $b str <- $c str
__impl__ = 200
""",
    "c2.viba": """
__decl__ = int <- $env Env <- $a bool <- $b bool <- $c bool
__impl__ = 300
""",
    "w0.viba": """
__decl__ = int <- $env Env <- $a Any
__impl__ = 10
""",
    "w1.viba": """
__decl__ = int <- $env Env <- $a str
__impl__ = 20
""",
    "h0.viba": """
__decl__ = int <- $env Env <- $a int
__impl__ = 1
""",
    "h1.viba": """
__decl__ = int <- $env Env <- $b bool
__impl__ = 2
""",
}

CALLERS = {
    "type_two.viba": """
import f0
import f1

__decl__ = Any <- $env Env
__impl__ = make_dispatcher[f0, f1].type
""",
    "type_one_param.viba": """
import g0
import g1

__decl__ = Any <- $env Env
__impl__ = make_dispatcher[g0, g1].type
""",
    "type_one_handler.viba": """
import f0

__decl__ = Any <- $env Env
__impl__ = make_dispatcher[f0].type
""",
    "type_three.viba": """
import c0
import c1
import c2

__decl__ = Any <- $env Env
__impl__ = make_dispatcher[c0, c1, c2].type
""",
    "type_same.viba": """
__decl__ = Any <- $env Env
__impl__ = Either[int, int]
""",
    "type_diff.viba": """
__decl__ = Any <- $env Env
__impl__ = Either[int, str]
""",
    "fail_names.viba": """
import h0
import h1

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[h0, h1] << args.env << $a 1
""",
    "fail_names_type.viba": """
import h0
import h1

__decl__ = Any <- $env Env
__impl__ = make_dispatcher[h0, h1].type
""",
    "dispatch_first.viba": """
import f0
import f1

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[f0, f1] << args.env << $a 1 << $b true
""",
    "dispatch_second.viba": """
import f0
import f1

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[f0, f1] << args.env << $a true << $b "x"
""",
    "dispatch_none.viba": """
import f0
import f1

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[f0, f1] << args.env << $a 1 << $b "x"
""",
    "dispatch_one_param.viba": """
import g0
import g1

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[g0, g1] << args.env << $a 5
""",
    "dispatch_one_param_second.viba": """
import g0
import g1

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[g0, g1] << args.env << $a false
""",
    "dispatch_three.viba": """
import c0
import c1
import c2

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[c0, c1, c2] << args.env << $a "x" << $b "y" << $c "z"
""",
    "dispatch_any_first.viba": """
import w0
import w1

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[w0, w1] << args.env << $a "x"
""",
    "dispatch_str_first.viba": """
import w0
import w1

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[w1, w0] << args.env << $a "x"
""",
    "dispatch_three_none.viba": """
import c0
import c1
import c2

__decl__ = Any <- $env Env
args = __get_args__ << __decl__

__impl__ = make_dispatcher[c0, c1, c2] << args.env << $a 1 << $b 2 << $c "z"
""",
}

TYPES = [
    ("type_two.viba", "两参数、两个 handler"),
    ("type_one_param.viba", "一参数、两个 handler"),
    ("type_one_handler.viba", "两参数、一个 handler"),
    ("type_three.viba", "三参数、三个 handler"),
]

EITHER_TYPES = [("type_same.viba", "int"), ("type_diff.viba", "int | str")]

ANSWERS = [
    ("dispatch_first.viba", 1),
    ("dispatch_second.viba", 2),
    ("dispatch_one_param.viba", 5),
    ("dispatch_one_param_second.viba", 2),
    ("dispatch_three.viba", 200),
    ("dispatch_any_first.viba", 10),
    ("dispatch_str_first.viba", 20),
]


def _write(directory: Path, files: dict):
    for name, text in files.items():
        (directory / name).write_text(text)


def _answer_type(result):
    return " ".join(viba_ast.unparse_type(answer_of(result).data).split())


def _value(result):
    return answer_of(result).data.value


def _environ(directory: Path):
    """The host side: the two switches `if` is made of, and nothing else."""
    def get_func(module_path, func_name):
        if module_path == "builtin" and func_name == "echo_or_never":
            return branch.echo_or_never
        if module_path == "builtin" and func_name == "never_or_echo":
            return branch.never_or_echo
        return None
    return Environment(EnvironmentStorage("root", None, None),
                       EnvironmentCompute(get_func), viba_path=str(directory))


def _file_shapes():
    """每份文件：pattern 行数对得上 handler 数，type 的参数位对得上参数个数。"""
    for count in range(1, 33):
        for params in range(1, 9):
            name = f"{count * 100 + params}.viba"
            public = (PUBLIC / name).read_text()
            impl = (IMPL / name).read_text()
            patterns = [line for line in public.splitlines()
                        if line.startswith("pattern ")]
            check(len(patterns) == count,
                  f"make_dispatcher/{name}: {count} 个 handler，pattern 有 {len(patterns)} 行")
            doubled = ", ".join(f"f{h}, f{h}" for h in range(count))
            type_line = next(line for line in public.splitlines()
                             if line.startswith("type = "))
            check(type_line == f"type = make_dispatcher_impl[{doubled}].type",
                  f"make_dispatcher/{name}: .type 从实现取，并且每个 handler 给两次")
            handlers = [line for line in impl.splitlines()
                        if line.startswith("pattern (")]
            check(len(handlers) == count,
                  f"make_dispatcher_impl/{name}: {count} 个 handler，拆类型的 pattern 有 "
                  f"{len(handlers)} 行")
            called = [line for line in impl.splitlines()
                      if line.startswith("pattern f")]
            check(len(called) == count,
                  f"make_dispatcher_impl/{name}: {count} 个 handler，用来调用的 pattern 有 "
                  f"{len(called)} 行")
            impl_type = next(line for line in impl.splitlines()
                             if line.startswith("type = "))
            check(len(impl_type.split(" <- ")) == params + 1,
                  f"make_dispatcher_impl/{name}: {params} 个参数位，type 有 "
                  f"{len(impl_type.split(' <- ')) - 1} 个")
            case_line = next(line for line in impl.splitlines()
                             if line.startswith("arg0 = "))
            terms = case_line.split(" = ", 1)[1].split(" | ")
            check(len(terms) == count,
                  f"make_dispatcher_impl/{name}: {count} 个 handler，第一个参数位有 "
                  f"{len(terms)} 项")


def run(directory: Path):
    _write(directory, HANDLERS)
    _write(directory, CALLERS)
    many = ", ".join(["f0"] * 32)
    (directory / "type_32.viba").write_text(
        "import f0\n\n__decl__ = Any <- $env Env\n"
        f"__impl__ = make_dispatcher[{many}].type\n")
    environ = _environ(directory)
    for name, label in TYPES + [("type_32.viba", "32 个 handler")]:
        result = interpret(str(directory / name), environ)
        taken = answer_of(result).data if is_ok(result) else None
        check(is_ok(result)
              and type(taken).__name__ == "MemberTaken"
              and taken.name == "type"
              and getattr(taken.owner, "constructor", None) == "make_dispatcher_impl",
              f"{label}：{result!r}")
    for name, want in EITHER_TYPES:
        result = interpret(str(directory / name), environ)
        check(is_ok(result) and _answer_type(result) == want,
              f"{name} -> {want}：{result!r}")
    failed = interpret(str(directory / "fail_names.viba"), environ)
    check(not is_ok(failed)
          and "no pattern of 'make_dispatcher_impl' matches" in message_of(failed),
          f"参数名不一致让决断失败：{failed!r}")
    failed_type = interpret(str(directory / "fail_names_type.viba"), environ)
    check(not is_ok(failed_type)
          and "no pattern of 'make_dispatcher_impl' matches" in message_of(failed_type),
          f"取 .type 就能测出不能并列：{failed_type!r}")
    for name, want in ANSWERS:
        result = interpret(str(directory / name), environ)
        check(is_ok(result) and _value(result) == want,
              f"{name} 给出 {want}：{result!r}")
    for name in ("dispatch_none.viba", "dispatch_three_none.viba"):
        result = interpret(str(directory / name), environ)
        check(is_ok(result) and _answer_type(result) == NO_HANDLER,
              f"{name} 一个 handler 都收不下：{result!r}")
    _file_shapes()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="viba-make-dispatcher-") as scratch:
        run(Path(scratch))
    sys.exit(checks.report())

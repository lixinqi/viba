"""惰性参数：`ParametersLazyEvaluated` 标记的函数，实参不先算。

被标记的函数（`branch.viba` 的两个开关就是一个例子）不按平常的方式调用：`interpret` 把每个
写下来的实参包成一个无参 lambda 交给宿主，宿主叫哪个才算哪个。于是"没走的那一支"不会被求值——
这才是 if/else；不然两条都算完再丢掉一条，只是结果一样。

    python3 tests/test_interpreter_lazy.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, Host, value_of, write

from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret
from viba.type import NotMyDutyException, Ok, VibaProgramErr

checks = Checks("interpreter_lazy")
check = checks.check
labelled = checks.labelled


def host_for(calls):
    """计数宿主：`tick` / `tock` 每被算一次就记一笔。"""
    def get_func(module_path, func_name):
        if func_name == "tick":
            def tick(env):
                calls.append("tick")
                return 1
            return tick
        if func_name == "tock":
            def tock(env):
                calls.append("tock")
                return 2
            return tock
        if func_name == "ignore_x":
            # 完全不碰 get_x：那个实参不该被算
            return lambda get_env, get_x: 7
        if func_name == "take_x":
            # 叫了 get_x：那个实参这时才算
            return lambda get_env, get_x: get_x().value
        if func_name == "eager_pair":
            # 没被标记的函数：两个实参都要算（老行为）
            return lambda env, x, y: 1
        if func_name == "ge":
            return lambda env, x, y: x.value >= y.value
        if func_name == "explode":
            def explode(env):
                raise ZeroDivisionError("poison")
            return explode
        if func_name == "broken_lookup":
            raise RuntimeError("the router broke")
        if func_name == "answer_a_list":
            return lambda env: [1, 2]
        if func_name == "inner_lambda_record":
            # 分支值本身也是一次调用：它被算过就说明那一支走了
            def inner_lambda_record(env, label):
                calls.append(label.value)
                return 0
            return inner_lambda_record
        if func_name == "ask_twice":
            # 同一个 getter 问两次：只该算一次
            def ask_twice(get_env, get_x):
                first = get_x().value
                second = get_x().value
                return first + second
            return ask_twice
        if func_name == "ask_nothing":
            # 两个实参都不问
            return lambda get_env, get_a, get_b: 0
        if func_name == "positional":
            # 三个槽位都收，按写下来的顺序给 getter
            def positional(get_env, get_condition, get_v):
                return get_v().value if get_condition().value else 0
            return positional
        if func_name == "boom_when_asked":
            def boom_when_asked(get_env):
                calls.append("boom")
                raise ZeroDivisionError("boom")
            return boom_when_asked
        return branch.get_func(module_path, func_name)
    return get_func


def environ_for(calls, store):
    # `import branch` finds branch.viba the way the branch suite does: the
    # checkout root is on the module search path.
    return Environment(EnvironmentStorage("root", None, str(store)),
                       EnvironmentCompute(host_for(calls)),
                       viba_path=str(Path(__file__).resolve().parent.parent))


BRANCHES = """
import branch

tick =
    int <- $env Environment <- { the branch that is taken first }
tock =
    int <- $env Environment <- { the branch that is taken second }
ge =
    bool <- $env Environment <- $x int <- $y int <- { x >= y }

condition = ge << $env environ << $x 1 << $y THRESHOLD
__ret__ =
  Oneof
  | (branch.id_or_never << $env environ << $condition condition << $v (tick << $env environ))
  | (branch.never_or_nil << $env environ << $condition condition << $v (tock << $env environ))
"""


def run(tmp: Path):
    _only_the_taken_branch_is_computed(tmp)
    _an_ignored_argument_is_never_computed(tmp)
    _a_branch_value_is_its_own_call(tmp)
    _an_argument_is_computed_at_most_once(tmp)
    _poison_in_the_untaken_branch(tmp)
    _the_getters_follow_the_slots(tmp)
    _an_unmarked_function_is_still_eager(tmp)
    _a_called_getter_carries_what_stopped(tmp)
    _the_marker_marks_functions(tmp)


def _only_the_taken_branch_is_computed(tmp: Path):
    """branch.viba 的开关：只有选中那一支的实参被算。"""
    taken_first = write(tmp, "taken_first.viba", BRANCHES.replace("THRESHOLD", "0"))
    calls = []
    result = interpret(taken_first, environ_for(calls, tmp / "store-a"))
    check(isinstance(result, Ok) and value_of(result) == 1, f"the first branch: {result!r}")
    check(calls == ["tick"],
          f"and only the first branch's value was computed: {calls}")

    taken_second = write(tmp, "taken_second.viba", BRANCHES.replace("THRESHOLD", "5"))
    calls = []
    result = interpret(taken_second, environ_for(calls, tmp / "store-b"))
    check(isinstance(result, Ok) and value_of(result) == 2, f"the second branch: {result!r}")
    check(calls == ["tock"],
          f"and only the second branch's value was computed: {calls}")


def _an_ignored_argument_is_never_computed(tmp: Path):
    """宿主不叫那个 getter，实参就不算——哪怕它根本没有实现。"""
    program = write(tmp, "ignored.viba", """
ignore_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer seven without looking at x }
    ]

ghost =
    int <- $env Environment <- { nothing implements this }

__ret__ = ignore_x << $env environ << $x (ghost << $env environ)
""")
    calls = []
    result = interpret(program, environ_for(calls, tmp / "store-c"))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"an argument nobody asks for is not computed, so its missing "
          f"implementation never shows: {result!r}")

    # 先给一半、再给另一半：惰性跟着走
    half = write(tmp, "half.viba", """
ignore_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer seven without looking at x }
    ]

ghost =
    int <- $env Environment <- { nothing implements this }

half = ignore_x << $env environ
__ret__ = half << $x (ghost << $env environ)
""")
    calls = []
    result = interpret(half, environ_for(calls, tmp / "store-d"))
    check(isinstance(result, Ok) and value_of(result) == 7,
          f"a marked function stays lazy through a partial application: {result!r}")


def _a_branch_value_is_its_own_call(tmp: Path):
    """分支值写成一次调用（`inner_lambda_record << env << "true_branch"`）时，
    只有走的那一支会留下记录。"""
    source = """
import branch

ge =
    bool <- $env Environment <- $x int <- $y int <- { x >= y }
inner_lambda_record =
    int <- $env Environment <- $label str <- { record which branch was taken }
ghost =
    int <- $env Environment <- { nothing implements this }

condition = ge << $env environ << $x 1 << $y THRESHOLD
__ret__ =
    Oneof
  | (branch.id_or_never << environ << condition << (inner_lambda_record << environ << "true_branch"))
  | (branch.never_or_nil << environ << condition << (inner_lambda_record << environ << "false_branch"))
"""
    calls = []
    result = interpret(write(tmp, "recorded_true.viba", source.replace("THRESHOLD", "0")),
                       environ_for(calls, tmp / "store-true"))
    check(isinstance(result, Ok) and value_of(result) == 0, f"the true branch: {result!r}")
    check(calls == ["true_branch"],
          f"and only it was recorded: {calls}")

    calls = []
    result = interpret(write(tmp, "recorded_false.viba", source.replace("THRESHOLD", "5")),
                       environ_for(calls, tmp / "store-false"))
    check(isinstance(result, Ok) and value_of(result) == 0, f"the false branch: {result!r}")
    check(calls == ["false_branch"],
          f"and only it was recorded: {calls}")

    # 分支背后那一步没有实现：递延报的是那一步，而且它没被算过
    missing = write(tmp, "recorded_missing.viba", source
                    .replace("THRESHOLD", "0")
                    .replace('(inner_lambda_record << environ << "true_branch")',
                             '(ghost << environ)'))
    calls = []
    result = interpret(missing, environ_for(calls, tmp / "store-missing"))
    check(isinstance(result, NotMyDutyException),
          f"an unimplemented step behind a branch defers: {result!r}")
    check(calls == [],
          f"and nothing behind that branch was computed: {calls}")


def _an_argument_is_computed_at_most_once(tmp: Path):
    """一个实参只算一次：问两次不等于做两遍。

    惰性是"要的时候才算"，不是"每次问都算"——原来 eager 调用里那个实参也只求值一次，
    宿主问两次不该让副作用发生两次。
    """
    twice = write(tmp, "ask_twice.viba", """
ask_twice =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { add x to itself, asking for x twice }
    ]

tick =
    int <- $env Environment <- { a value with a side effect }

__ret__ = ask_twice << $env environ << $x (tick << $env environ)
""")
    calls = []
    result = interpret(twice, environ_for(calls, tmp / "store-once"))
    check(isinstance(result, Ok) and value_of(result) == 2,
          f"the answer says the argument was asked for twice: {result!r}")
    check(calls == ["tick"],
          f"and computed once: {calls}")

    nothing = write(tmp, "ask_nothing.viba", """
ask_nothing =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $a int
      <- $b int
      <- { answer zero, asking for neither argument }
    ]

tick =
    int <- $env Environment <- { one }
tock =
    int <- $env Environment <- { the other }

__ret__ = ask_nothing << $env environ << $a (tick << $env environ) << $b (tock << $env environ)
""")
    calls = []
    result = interpret(nothing, environ_for(calls, tmp / "store-none"))
    check(isinstance(result, Ok) and value_of(result) == 0,
          f"a host that asks for nothing still answers: {result!r}")
    check(calls == [], f"and nothing was computed: {calls}")

    # 失败也只算一次：第二次问拿到的是同一个结果，不是重新求值
    failing = write(tmp, "ask_twice_failing.viba", """
ask_twice =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { add x to itself, asking for x twice }
    ]

boom_when_asked =
    int <- $env Environment <- { a step whose implementation raises }

__ret__ = ask_twice << $env environ << $x (boom_when_asked << $env environ)
""")
    calls = []
    checks.failed(interpret(failing, environ_for(calls, tmp / "store-fail-once")),
                  "boom_when_asked raised",
                  "a getter that stops stops the call the first time it is asked")
    check(calls == ["boom"],
          f"and the step behind it ran once, not once per ask: {calls}")


def _the_getters_follow_the_slots(tmp: Path):
    """位置实参与乱序 tag：getter 仍按槽位交给宿主。"""
    positional = write(tmp, "positional.viba", """
positional =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $condition bool
      <- $v int
      <- { the value when the condition holds }
    ]

tick =
    int <- $env Environment <- { one }
ge =
    bool <- $env Environment <- $x int <- $y int <- { x >= y }

value = tick << $env environ
condition = ge << $env environ << $x value << $y 0
__ret__ = positional << environ << condition << value
""")
    calls = []
    result = interpret(positional, environ_for(calls, tmp / "store-pos"))
    check(isinstance(result, Ok) and value_of(result) == 1,
          f"arguments written without tags reach the right slots: {result!r}")

    out_of_order = write(tmp, "out_of_order.viba", """
positional =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $condition bool
      <- $v int
      <- { the value when the condition holds }
    ]

tick =
    int <- $env Environment <- { one }
ge =
    bool <- $env Environment <- $x int <- $y int <- { x >= y }

condition = ge << $env environ << $x 1 << $y 0
__ret__ = positional << $v (tick << $env environ) << $condition condition << $env environ
""")
    calls = []
    result = interpret(out_of_order, environ_for(calls, tmp / "store-ooo"))
    check(isinstance(result, Ok) and value_of(result) == 1,
          f"tags given in another order reach the same slots: {result!r}")
    check(calls == ["tick"], f"and the value was computed: {calls}")


# 毒剂：放进"走不到的那一支"的表达式。它们各自本来会怎样，写在标签里——验的是结果：
# 走的那一支的值照常出来，走不到的那一支一次都不算。
POISONS = [
    ("a step with no implementation (would defer)", "",
     "ghost << $env environ"),
    ("a step whose implementation raises (would fail)", "",
     "explode << $env environ"),
    ("a step whose get_func raises (would fail)", "",
     "broken_lookup << $env environ"),
    ("a step answering something with no leaf (would fail)", "",
     "answer_a_list << $env environ"),
    ("a module that is not there (would be a program error)", "",
     "missing_module << (environ.tmp_sub_env << ())"),
    ("a name nothing defines (would be a program error)", "",
     "nope"),
    ("an argument given to a number (would be a program error)", "",
     "1 << $x 2"),
    ("a product factor that is poison (would defer)", "",
     "(ghost << $env environ) * 1"),
    ("a sum branch that is poison (would defer)", "",
     "(ghost << $env environ) | 1"),
    ("a branch that would take poison itself (would defer)", "",
     "(branch.id_or_never << $env environ"
     " << $condition (ge << $env environ << $x 1 << $y 0)"
     " << $v (ghost << $env environ))"),
]

POISON_PROGRAM = """{imports}import branch

ge =
    bool <- $env Environment <- $x int <- $y int <- {{ x >= y }}
ghost =
    int <- $env Environment <- {{ nothing implements this }}
explode =
    int <- $env Environment <- {{ this implementation raises }}
broken_lookup =
    int <- $env Environment <- {{ get_func itself raises for this name }}
answer_a_list =
    int <- $env Environment <- {{ answers a list, which is no leaf }}

healthy = 42
condition = ge << $env environ << $x 1 << $y 0
__ret__ =
    Oneof
  | (branch.id_or_never << $env environ << $condition condition << $v {one})
  | (branch.never_or_nil << $env environ << $condition condition << $v {two})
"""


def _poison_in_the_untaken_branch(tmp: Path):
    """走不到的那一支放毒：只有走的那一支的值出来，毒一次都不发作。

    验的是**结果**——`Ok(42)`，也就是另一支的值；毒如果被算过，这十条里任何一条都会变成递延、
    失败或程序错误，所以这个结果本身就是"那一支没被算"的证明。
    """
    for index, (label, imports, poison) in enumerate(POISONS):
        source = POISON_PROGRAM.format(imports=imports, one="(healthy)", two=f"({poison})")
        program = write(tmp, f"poison_{index}.viba", source)
        result = interpret(program, environ_for([], tmp / f"store-poison{index}"))
        check(isinstance(result, Ok) and value_of(result) == 42,
              f"poison in the branch that is not taken ({label}): {result!r}")


def _an_unmarked_function_is_still_eager(tmp: Path):
    """没标记的函数一切照旧：每个实参都先算出来。"""
    program = write(tmp, "eager.viba", """
eager_pair =
    int
  <- $env Environment
  <- $x int
  <- $y int
  <- { ignore both arguments, but they are computed first }

tick =
    int <- $env Environment <- { a value with a side effect }

__ret__ = eager_pair << $env environ << $x (tick << $env environ) << $y 2
""")
    calls = []
    result = interpret(program, environ_for(calls, tmp / "store-e"))
    check(isinstance(result, Ok) and value_of(result) == 1,
          f"an unmarked call still answers: {result!r}")
    check(calls == ["tick"],
          f"and its arguments were computed eagerly, the way they always were: {calls}")


def _a_called_getter_carries_what_stopped(tmp: Path):
    """宿主叫了那个 getter，实参算的时候出的事就照常报出来。"""
    missing = write(tmp, "called_missing.viba", """
take_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer whatever x is }
    ]

ghost =
    int <- $env Environment <- { nothing implements this }

__ret__ = take_x << $env environ << $x (ghost << $env environ)
""")
    calls = []
    result = interpret(missing, environ_for(calls, tmp / "store-f"))
    check(isinstance(result, NotMyDutyException),
          f"a getter that is called and has no implementation defers, as ever: {result!r}")

    boomed = write(tmp, "called_boom.viba", """
take_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer whatever x is }
    ]

__ret__ = take_x << $env environ << $x (1 << $x 2)
""")
    calls = []
    labelled(interpret(boomed, environ_for(calls, tmp / "store-g")), "is not a function",
             "an argument that is broken when the getter runs -> the same err, "
             "not a failure of the host that asked for it")


def _the_marker_marks_functions(tmp: Path):
    """标记只标函数；漏了 environ 也照旧是错。"""
    not_a_function = write(tmp, "not_a_function.viba", """
wrong =
    ParametersLazyEvaluated[int]

__ret__ = wrong << $env environ
""")
    calls = []
    result = interpret(not_a_function, environ_for(calls, tmp / "store-h"))
    check(isinstance(result, VibaProgramErr) and
          "ParametersLazyEvaluated marks a function" in result.err_msg,
          f"the marker around a non-function says so: {result!r}")

    no_env = write(tmp, "no_env.viba", """
ignore_x =
    ParametersLazyEvaluated[
        int
      <- $env Environment
      <- $x int
      <- { answer seven without looking at x }
    ]

__ret__ = ignore_x << 5 << 1
""")
    calls = []
    labelled(interpret(no_env, environ_for(calls, tmp / "store-i")),
             "was not given an Environment",
             "a marked function still needs its environment")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-interpreter-lazy-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    sys.exit(checks.report())

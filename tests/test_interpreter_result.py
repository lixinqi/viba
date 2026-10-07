"""一次执行的结果：就是那个节点，而且它落在 `viba/interpret_result.viba` 这个定义里。

`interpret` 与 `exec` 回答的是一份 viba 数据（声明里的 `InterpretResult`），这份数据按 tag 读得出来，
也能拿声明本身判：`is_interpret_result` 用 `is_sub_type` 判它确实落在那份定义里。这里四种停法各钉一遍
（值、没有实现、实现坏了、程序/环境不行、环境上的 api 收不下），并钉住宿主交回失败数据那条路。

    python3 tests/test_interpreter_result.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interpreter_support import (answer_of, branch_of, leaf_of, message_of,
                                  stop_node, stop_tag, stop_text, Checks, Host,
                                  is_ok, value_of)

from viba import serialize, viba_ast
from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            exec as viba_exec, interpret, is_interpret_result,
                            not_implemented)
from viba.type import (ENVIRONMENT_API_TAG, FAILURE_TAG, NOT_IMPLEMENTED_TAG,
                       OK_TAG, PROGRAM_ERR_TAG, Ok)

checks = Checks("interpreter_result")
check = checks.check

CASES = Path(__file__).resolve().parent / "data" / "result"

# 一个什么都不实现、而且没有 storage 的宿主：没有实现的那一支、以及 api 收不下的那一支都用它。
def _headless():
    return Environment(None, EnvironmentCompute(lambda module, name: None))


def _case(name: str) -> str:
    return str(CASES / f"{name}.viba")


def run():
    _a_value()
    _a_value_with_members()
    _a_step_with_no_implementation()
    _a_host_that_hands_the_failure_back()
    _a_step_that_broke()
    _a_program_error()
    _the_chain_it_stopped_in()
    _an_api_that_cannot_take_it()
    _what_is_no_result()


def _fits(result, label: str):
    """`is_sub_type` 判它落在声明里 —— 每一支都要过这一关。"""
    check(is_interpret_result(result),
          f"{label}: the answer does not fit InterpretResult: {result!r}")
    written = serialize.serialize("result", result)
    check(isinstance(written, Ok) and written.ok_value.startswith("result ="),
          f"{label}: and it writes out as viba source: {written!r}")


def _a_value():
    """`$ok` 那一支：值是它自己，整份结果对得上声明。"""
    result = interpret(_case("value"), Host().environ())
    check(branch_of(result) == OK_TAG and is_ok(result) and value_of(result) == 42,
          f"a run that answered is on $ok, and the leaf reads: {result!r}")
    _fits(result, "a value")


def _a_value_with_members():
    """值的读法跟着值走：一份积按它自己的成员读。"""
    result = interpret(_case("product"), Host().environ())
    answer = answer_of(result) if is_ok(result) else None
    check(answer is not None and leaf_of(answer.by_tag("$x")) == 1
          and leaf_of(answer.by_tag("$name")) == "b",
          f"the answer is read through its own members: {result!r}")
    _fits(result, "a product")


def _a_step_with_no_implementation():
    """`$not_implemented_err`：步名、这次调用、为什么都在，`$call` 拿回来能再跑一遍。"""
    result = interpret(_case("no_impl"), Host().environ())
    check(stop_tag(result) == NOT_IMPLEMENTED_TAG
          and message_of(result) == "no implementation",
          f"a step nobody implements is that stop: {result!r}")
    check(stop_text(result, "$module_path") == "/no_impl"
          and stop_text(result, "$full_qualified_func_name") == "no_impl.ghost",
          f"and it names the step: {stop_text(result, '$module_path')!r}, "
          f"{stop_text(result, '$full_qualified_func_name')!r}")
    call = stop_node(result, "$call")
    check(call is not None and isinstance(call.data, viba_ast.Partial),
          f"and it carries the call, as the call it is: {call!r}")
    _fits(result, "a step with no implementation")

    # 补上这一步，照着 `$call` 再跑一遍：这份结果本身就够
    host = Host()
    original = host.get_func

    def with_ghost(module, func_name):
        if func_name == "ghost":
            return lambda env: 7
        return original(module, func_name)

    host.get_func = with_ghost
    again = viba_exec(call, host.environ())
    check(is_ok(again) and value_of(again) == 7,
          f"the carried call runs again once the step is implemented: {again!r}")


def _a_host_that_hands_the_failure_back():
    """宿主交回失败数据：它说的留着，缺的（步名、`$call`）由 run 补上。"""
    result = interpret(_case("no_impl"), Host(refuse=("ghost",)).environ())
    check(stop_tag(result) == NOT_IMPLEMENTED_TAG
          and stop_text(result, "$module_path") == "/no_impl"
          and stop_text(result, "$full_qualified_func_name") == "no_impl.ghost"
          and message_of(result) == "no implementation"
          and stop_node(result, "$call") is not None,
          f"get_func handed the failure back, and the run filled it in: {result!r}")
    _fits(result, "a failure a get_func handed back")

    # 宿主自己带了话：话留着，步名照补
    said = not_implemented("no ghost here", module_path="root/elsewhere")
    result = interpret(_case("no_impl"), Host(hands_back=said).environ())
    check(message_of(result) == "no ghost here"
          and stop_text(result, "$module_path") == "root/elsewhere"
          and stop_text(result, "$full_qualified_func_name") == "no_impl.ghost",
          f"what the host wrote is kept, the step name is filled in: {result!r}")
    _fits(result, "a failure a get_func wrote itself")

    # 实现自己交回它（不是 get_func 交回的）
    result = interpret(_case("no_impl"), Host(says_no=("ghost",)).environ())
    check(stop_tag(result) == NOT_IMPLEMENTED_TAG
          and stop_text(result, "$full_qualified_func_name") == "no_impl.ghost",
          f"an implementation that hands it back is the same stop: {result!r}")
    _fits(result, "a failure an implementation handed back")


def _a_step_that_broke():
    """`$underlying_viba_op_err`：实现抛了，话以 `raised:` 开头。"""
    result = interpret(_case("boom"), Host().environ())
    check(stop_tag(result) == FAILURE_TAG and message_of(result).startswith("raised:"),
          f"an implementation that raised is that stop: {result!r}")
    check(stop_text(result, "$module_path") == "/boom"
          and stop_text(result, "$full_qualified_func_name") == "boom.explode",
          f"and it names the step: {result!r}")
    _fits(result, "a step that broke")


def _a_program_error():
    """`$viba_program_err`：文件不在，或者回答的是环境；`$stack` 是走过的调用链。"""
    result = interpret(str(CASES / "nowhere.viba"), Host().environ())
    check(stop_tag(result) == PROGRAM_ERR_TAG and message_of(result).startswith("no such file"),
          f"a file that is not there is that stop: {result!r}")
    _fits(result, "a program error")

    result = interpret(_case("answers_the_environment"), Host().environ())
    check(stop_tag(result) == PROGRAM_ERR_TAG
          and "the run answered the environment" in message_of(result),
          f"answering the environment is that stop too: {result!r}")
    _fits(result, "a run that answered the environment")


def _the_chain_it_stopped_in():
    """`$stack`：执行里停下来时，一帧就是写这次调用的地方，最外那帧是主文件。"""
    result = interpret(_case("chain"), Host().environ())
    check(stop_tag(result) == PROGRAM_ERR_TAG
          and message_of(result).startswith("no definition named"),
          f"a name that resolves to nothing is that stop: {result!r}")
    frames = stop_node(result, "$stack")
    check(len(frames) == 2,
          f"and the chain it stopped in has two frames: {frames!r}")
    outer = frames.at_index(0)
    check(str(leaf_of(outer.by_tag("$file_path"))).endswith("chain.viba")
          and leaf_of(outer.by_tag("$lineno")) == 0,
          f"the outermost frame is the main file, which no one called: {outer!r}")
    entered = frames.at_index(1)
    check(str(leaf_of(entered.by_tag("$file_path"))).endswith("chain.viba")
          and leaf_of(entered.by_tag("$lineno")) == 10,
          f"and the next one is the call that entered the inner module, "
          f"written in chain.viba on line 10: {entered!r}")
    _fits(result, "a program error with a chain")


def _an_api_that_cannot_take_it():
    """`$environment_api_invalid_argument_err`：哪一个 api、给了它什么，都在。"""
    result = interpret(_case("api_refusal"), _headless())
    check(stop_tag(result) == ENVIRONMENT_API_TAG
          and stop_text(result, "$api_name") == "Environment.sub_env",
          f"an api that cannot take what it was given is that stop: {result!r}")
    check(leaf_of(stop_node(result, "$args")) == "kid",
          f"and what it was given is recorded: {stop_node(result, '$args')!r}")
    _fits(result, "an api refusal")


def _what_is_no_result():
    """不是结果的就不是：一份普通数据判它，是 False。"""
    nothing = interpret(_case("value"), Host().environ()).by_tag(OK_TAG)
    check(not is_interpret_result(nothing),
          f"an answer is no InterpretResult of its own: {nothing!r}")
    check(not is_interpret_result(viba_ast.Constant(1)),
          "and neither is a bare piece")


if __name__ == "__main__":
    run()
    sys.exit(checks.report())

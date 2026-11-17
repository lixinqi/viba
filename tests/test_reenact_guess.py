"""猜数字：重演让 candidate_b 按上一轮的结果改返回值，直到 compare 返回 0。

两局：0..100（谜底 73）和 0..2^30（谜底 2^30-1）。reference_c 是谜底，candidate_b 是玩家，
compare_b_c 比较大小（-1 / 0 / 1）。录制跑一次玩家的 50；之后每次重演都把 candidate_b 换成
二分出来的下一个猜测，拿上一轮的 compare_b_c 返回值决定往哪边收，直到返回 0。reference_c 只在
录制时跑一次：录制完把宿主的谜底改掉，重演仍收敛到原来那个，说明用的是录下来的谜底。重演的
store 每次新建，录制的 store 只读。

    python3 tests/test_reenact_guess.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, is_ok, value_of

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret, make_referenced_env)

CASES = Path(__file__).resolve().parent / "data"
ENTRY = str(CASES / "reenact_guess_entry.viba")
REPLAY = str(CASES / "reenact_guess_replay.viba")
TAG = "guess_game"

checks = Checks("reenact_guess")
check = checks.check

CALLS = {"reference_c": 0, "candidate_b": 0, "compare_b_c": 0}
STATE = {"range": 100, "guess": 50, "secret": 73}


def host_for():
    """reference_c 返回谜底；candidate_b 返回这一轮的猜测；compare_b_c 比较大小。"""
    def get_func(module_path, func_name):
        if func_name in ("echo_or_never", "never_or_echo"):
            return branch.get_func(module_path, func_name)
        if func_name == "echo":
            return lambda env, x: x
        if func_name == "range_of":
            return lambda env: STATE["range"]
        if func_name == "reference_c":
            def reference_c(env, range_):
                CALLS["reference_c"] += 1
                return STATE["secret"]
            return reference_c
        if func_name == "candidate_b":
            def candidate_b(env, range_):
                CALLS["candidate_b"] += 1
                return STATE["guess"]
            return candidate_b
        if func_name == "compare_b_c":
            def compare_b_c(env, b, c):
                CALLS["compare_b_c"] += 1
                if b.value < c.value:
                    return -1
                if b.value > c.value:
                    return 1
                return 0
            return compare_b_c
        return None
    return get_func


def environ(store, referenced=None):
    env = Environment(EnvironmentStorage("root", None, str(store)),
                      EnvironmentCompute(host_for()), viba_path=str(CASES))
    return make_referenced_env(env, str(referenced)) if referenced else env


def reset(range_, secret):
    for name in CALLS:
        CALLS[name] = 0
    STATE["range"] = range_
    STATE["guess"] = 50
    STATE["secret"] = secret


def answer_list(result):
    """The list the replay returned, as Python values."""
    node = result.by_tag("$ok")
    return [node.at_index(index).value for index in range(len(node))]


def files_under(root):
    """{relative path: text} for every file under the store."""
    return {str(path.relative_to(root)): path.read_text()
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def run_case(tmp: Path, label: str, range_: int, secret: int, exact=None):
    recording = tmp / f"{label}-recording"

    reset(range_, secret)
    first = interpret(ENTRY, environ(recording))
    check(is_ok(first), f"{label}: the recording returns a value: {first!r}")
    result = value_of(first)
    check(result == -1, f"{label}: the first guess 50 is below the secret: {result!r}")
    before = files_under(recording)

    # The replay must use the recorded secret: a fresh reference_c now would
    # answer another number, and the guesses would stop at it.
    STATE["secret"] = secret + 1

    guesses = [STATE["guess"]]
    results = [result]
    low, high = 0, range_
    replays = 0
    while result != 0:
        if result < 0:
            low = STATE["guess"] + 1
        else:
            high = STATE["guess"] - 1
        STATE["guess"] = (low + high) // 2
        guesses.append(STATE["guess"])
        replays += 1
        replay = interpret(REPLAY, environ(tmp / f"{label}-replay-{replays}", recording))
        check(is_ok(replay), f"{label}: replay {replays} returns a value: {replay!r}")
        answers = answer_list(replay)
        check(len(answers) == 1, f"{label}: replay {replays} answers one call: {answers}")
        result = answers[0]
        results.append(result)

    check(STATE["guess"] == secret,
          f"{label}: the player guessed the secret: {STATE['guess']}")
    if exact is not None:
        check(guesses == exact[0], f"{label}: each replay tightens the range: {guesses}")
        check(results == exact[1], f"{label}: the comparisons drive the guesses: {results}")
    else:
        check(results[:-1] == [-1] * (len(results) - 1) and results[-1] == 0,
              f"{label}: every guess but the last is low: {results}")
        check(len(guesses) == len(results) == 30,
              f"{label}: the 0..2^30 game takes 30 rounds: {len(guesses)}")

    rounds = 1 + replays
    check(CALLS == {"reference_c": 1, "candidate_b": rounds, "compare_b_c": rounds},
          f"{label}: the secret is recorded once; the player and the comparison run "
          f"every round: {CALLS}")
    check(files_under(recording) == before,
          f"{label}: the replays do not write into the recording's store")

    recorded = Path(recording) / "root" / "game" / "reenact" / TAG
    check((recorded / "count.viba").read_text().strip() == "1",
          f"{label}: one call is recorded")
    args_text = (recorded / "0" / "args.viba").read_text()
    check(str(range_) in args_text, f"{label}: the range is recorded: {args_text!r}")
    references = list(Path(recording).rglob("cached_reference_c.viba"))
    check(len(references) == 1 and str(secret) in references[0].read_text(),
          f"{label}: the secret is stored once: {references}")
    for index in range(1, replays + 1):
        answer = Path(tmp / f"{label}-replay-{index}") / "root" / "game" / "reenact" / TAG / "0" / "answer.viba"
        check(answer.is_file(), f"{label}: replay {index} keeps its answer: {answer}")


def main():
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        run_case(tmp, "small", 100, 73,
                 exact=([50, 75, 62, 68, 71, 73], [-1, 1, -1, -1, -1, 0]))
        run_case(tmp, "big", 2 ** 30, 2 ** 30 - 1)
    print(f"reenact_guess: {checks.passed} passed, {checks.failures} failed")
    return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(main())

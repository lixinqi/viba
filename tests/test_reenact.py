"""重演：把入口的调用录下来，再按参照重跑（`viba/exec/viba-reenact.md`）。

录制跑两遍：reference_c（`reenact_with_reference`）带状态，两次各返回一个值；candidate_b
（被考核的那一个）两次各跑一遍。重演跑一遍：reference_c 一次都不跑，参照从录制的 store
里取；candidate_b、compare_b_c 各再跑一遍，返回值由这次算出来的 candidate_b 和录下来的
reference_c 决定。重演用自己的 store，录制的 store 只读。

    python3 tests/test_reenact.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import branch

from interpreter_support import Checks, is_ok, message_of

from viba.interpret import (Environment, EnvironmentCompute, EnvironmentStorage,
                            interpret, make_referenced_env)

CASES = Path(__file__).resolve().parent / "data" / "reenact"
ENTRY = str(CASES / "entry.viba")
REPLAY = str(CASES / "replay.viba")
TAG = "reenacted_func_a"

checks = Checks("reenact")
check = checks.check

CALLS = {"reference_c": 0, "candidate_b": 0, "compare_b_c": 0}


def host_for():
    """reference_c 带状态，candidate_b 每跑一次换一个返回值，compare_b_c 拿两个返回值算一个结果。"""
    def get_func(module_path, func_name):
        if func_name in ("echo_or_never", "never_or_echo"):
            return branch.get_func(module_path, func_name)
        if func_name == "echo":
            return lambda env, x: x
        if func_name == "reference_c":
            def reference_c(env, x):
                CALLS["reference_c"] += 1
                return 1001 * CALLS["reference_c"]
            return reference_c
        if func_name == "candidate_b":
            def candidate_b(env, x):
                CALLS["candidate_b"] += 1
                return x.value * 10 + CALLS["candidate_b"]
            return candidate_b
        if func_name == "compare_b_c":
            def compare_b_c(env, b, c):
                CALLS["compare_b_c"] += 1
                return b.value - c.value
            return compare_b_c
        return None
    return get_func


def environ(store, referenced=None):
    env = Environment(EnvironmentStorage("root", None, str(store)),
                      EnvironmentCompute(host_for()), viba_path=str(CASES))
    return make_referenced_env(env, str(referenced)) if referenced else env


def reset():
    for name in CALLS:
        CALLS[name] = 0


def answer_list(result):
    """The list the replay returned, as Python values."""
    node = result.by_tag("$ok")
    return [node.at_index(index).value for index in range(len(node))]


def files_under(root):
    """{relative path: text} for every file under the store."""
    return {str(path.relative_to(root)): path.read_text()
            for path in sorted(Path(root).rglob("*")) if path.is_file()}


def run(tmp: Path):
    recording = tmp / "recording"
    replay_store = tmp / "replay"

    reset()
    first = interpret(ENTRY, environ(recording))
    second = interpret(ENTRY, environ(recording))
    check(is_ok(first) and is_ok(second),
          f"the two recordings return a value: {first!r} {second!r}")
    check(CALLS == {"reference_c": 2, "candidate_b": 2, "compare_b_c": 2},
          f"each recording runs reference_c, candidate_b and compare_b_c once: {CALLS}")

    recorded = Path(recording) / "root" / "reenact" / TAG
    check((recorded / "count.viba").read_text().strip() == "2",
          "the tag counts the two calls")
    check((recorded / "0" / "args.viba").is_file(), "the arguments are stored")
    check('value =\n  "reenacted_func_a_impl"' in (recorded / "0" / "entry.viba").read_text(),
          "the entry's name is stored")
    references = sorted(path for path in Path(recording).rglob("cached_reference_c.viba"))
    check(len(references) == 2, f"one reference per recording: {references}")

    before = files_under(recording)

    replayed = interpret(REPLAY, environ(replay_store, recording))
    check(is_ok(replayed), f"the replay returns a value: {replayed!r}")
    check(CALLS == {"reference_c": 2, "candidate_b": 4, "compare_b_c": 4},
          f"the replay runs candidate_b and compare_b_c, never reference_c: {CALLS}")
    answers = answer_list(replayed)
    check(answers == [-948, -1948],
          f"each answer uses the fresh candidate_b and the recorded reference_c: {answers}")
    check(files_under(recording) == before,
          "the replay does not write into the recording's store")

    replayed_paths = Path(replay_store) / "root" / "reenact" / TAG
    check((replayed_paths / "0" / "answer.viba").is_file() and
          (replayed_paths / "1" / "answer.viba").is_file(),
          "each replayed call keeps its answer in the replay's store")

    plain = interpret(REPLAY, environ(tmp / "plain"))
    check(not is_ok(plain) and "referenced environment" in message_of(plain),
          f"a replay on a plain environment stops: {plain!r}")

    missing = tmp / "missing"
    shutil.copytree(recording, missing)
    for path in Path(missing).rglob("cached_reference_c.viba"):
        path.unlink()
    stopped = interpret(REPLAY, environ(tmp / "missing-replay", missing))
    check(not is_ok(stopped) and "no reference" in message_of(stopped),
          f"a missing reference stops instead of running reference_c: {stopped!r}")


def main():
    with tempfile.TemporaryDirectory() as directory:
        run(Path(directory))
    print(f"reenact: {checks.passed} passed, {checks.failures} failed")
    return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(main())

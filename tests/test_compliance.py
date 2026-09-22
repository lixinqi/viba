"""Rule/Witness 用 interpret 重做的第一批：跑规则、Prepare 与回放。

一个规则是程序（`environ` 进、`bool` 出），判定就是跑它；witness 是它判的材料。
不纯的那一步（量距离）走 `measure`：Prepare 是调用（参数定了、结果声明了）加量出来的
值，放在 storage 里——运行之前就备份好的那份直接回放，没有的才当场量、写进本轮 store。

    python3 tests/test_compliance.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from viba import viba_ast, serialize
from viba.compliance import (PreparedStorage, is_compliant, measured_of, prepare_path,
                             prepare_run, read_prepare, record_prepare)
from viba.compliance.demo.host import RULE, DistanceHost
from viba.interpreter import Environment, EnvironmentCompute
from viba.reflect import access as reflect_access
from viba.type import Err, Ok

BACKUP = Path(__file__).resolve().parent / "data" / "compliance" / "backup"

PASS = FAIL = 0


def check(ok: bool, label: str):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def labelled(result, want, label: str):
    if want is None:
        check(isinstance(result, Ok), f"{label}: {result!r}")
    else:
        check(isinstance(result, Err) and want in result.err_msg,
              f"{label}: expected Err({want!r}), got {result!r}")


def environ_for(store, backup=None, host=None):
    host = host or DistanceHost()
    storage = PreparedStorage("root", None, str(store),
                              None if backup is None else str(backup))
    return Environment(storage, EnvironmentCompute(host.get_func)), host


def write(tmp: Path, name: str, source: str) -> str:
    path = tmp / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return str(path)


def run(tmp: Path):
    _judge(tmp)
    _prepare_and_record(tmp)
    _replay_from_a_shipped_backup(tmp)
    _the_store_wins(tmp)
    _storage(tmp)
    _refusals(tmp)


def _judge(tmp: Path):
    """跑一个规则：判定就是 __ret__，材料是 witness 那个程序给的。"""
    store = tmp / "judge-store"
    env, host = environ_for(store)
    verdict = is_compliant(str(RULE), env)
    check(isinstance(verdict, Ok) and verdict.ok_value is True,
          f"the rule answered true: {verdict!r}")
    check(host.judged == [5], f"its predicate saw the measured value: {host.judged}")
    check(host.measured == [((0, 0), (3, 4))],
          f"and the measurement ran once: {host.measured}")

    # 同一份材料，另一个 witness：量出来 3，判定为假
    write(tmp, "case_near.viba", """
__ret__ :=
    $victim ($x 0 * $y 0)
  * $suspect ($x 0 * $y 3)
  * $at "12:30"
""")
    near_rule = write(tmp, "rule_near.viba", """
import case_near as witness

Point := $x int * $y int
Case := $victim Point * $suspect Point * $at str

measure_distance :=
	int
	<- $env Environment
	<- $case Case
	<- { measure }

distance_at_least_5 :=
	bool
	<- $env Environment
	<- $d int
	<- { at least 5? }

__ret__ :=
	distance_at_least_5
	<< $env environ
	<< $d (
		measure_distance
		<< $env environ
		<< $case (witness << (environ.sub_env << "witness"))
	)
""")
    env, host = environ_for(tmp / "near-store")
    verdict = is_compliant(near_rule, env)
    check(isinstance(verdict, Ok) and verdict.ok_value is False,
          f"a case 3 apart answers false: {verdict!r}")
    check(host.judged == [3], f"and the predicate saw 3: {host.judged}")


def _prepare_and_record(tmp: Path):
    """Prepare：本轮量出来的那份写进 store；prepare_run 再把它记进备份。"""
    store = tmp / "record-store"
    backup = tmp / "record-backup"
    env, host = environ_for(store, backup)
    check(isinstance(is_compliant(str(RULE), env), Ok), "the run judges")
    check(host.measured == [((0, 0), (3, 4))], "and measures")

    # 本轮 store 里那份：调用的参数定了、结果也写上了
    prepare = read_prepare(env, "distance")
    value, recorded = measured_of(prepare)
    check(recorded and value == 5, f"the run's own Prepare carries the value: {prepare!r}")

    check(isinstance(prepare_run(str(RULE), env), Ok), "prepare_run judges too")
    check(host.measured == [((0, 0), (3, 4))],
          f"and replays the value it just measured: {host.measured}")
    recorded_file = backup / "root" / "prepare" / "distance.viba"
    check(recorded_file.is_file(), f"the backup now holds the Prepare: {recorded_file}")
    text = recorded_file.read_text()
    check(text.startswith("value :="), f"as viba source: {text!r}")
    check("* $measured 5" in text, f"carrying what was measured: {text!r}")


def _replay_from_a_shipped_backup(tmp: Path):
    """仓库里备份好的那份 Prepare：读它，不重量。"""
    env, host = environ_for(tmp / "replay-store", BACKUP)
    verdict = is_compliant(str(RULE), env)
    check(isinstance(verdict, Ok) and verdict.ok_value is True,
          f"a backed-up Prepare answers the verdict: {verdict!r}")
    check(host.measured == [],
          f"and the impure step is not taken: {host.measured}")
    check(host.judged == [5],
          f"the pure part still runs, on the replayed value: {host.judged}")
    value, recorded = measured_of(read_prepare(env, "distance"))
    check(recorded and value == 5, f"the Prepare it read carries the value: {value!r}")

    # 备份是只读的证据：本轮的目录里没有 prepare，备份文件也没被动过
    check(not (tmp / "replay-store" / "root" / "prepare" / "distance.viba").exists(),
          "a replayed run writes no Prepare of its own")
    check((BACKUP / "root" / "prepare" / "distance.viba").is_file(),
          "and the backup is still there")


def _the_store_wins(tmp: Path):
    """本轮写的那份压过备份：先看自己，再看备份。"""
    store = tmp / "own-store"
    env, host = environ_for(store, BACKUP)
    record_prepare(env, "distance", viba_ast.Constant(0), 7)
    value, recorded = measured_of(read_prepare(env, "distance"))
    check(recorded and value == 7, f"the run's own Prepare is what is read: {value!r}")

    # 备份里还是原来那份
    backup_disk = (BACKUP / "root" / "prepare" / "distance.viba").read_text()
    check("* $measured 5" in backup_disk, "the backup is untouched by the run")


def _storage(tmp: Path):
    """storage 的 Prepare 规矩：哪条路径算 Prepare，子 storage 带着两个根。"""
    store = tmp / "plain-store"
    backup = tmp / "plain-backup"
    storage = PreparedStorage("root", None, str(store), str(backup))
    check(storage.is_prepare("root/prepare/distance.viba") and
          not storage.is_prepare("root/distance.viba"),
          "a Prepare is what sits under a prepare segment")
    child = storage.sub("a")
    check(child.prepare_root_dir == str(backup) and
          child.store_root_dir == str(store) and
          child.is_prepare("a/prepare/x.viba"),
          "a child storage keeps both roots, and still knows a Prepare")

    # 不是 Prepare 的路径：照旧读写自己的 store，备份那边不沾
    storage.write_text("notes.txt", "hello")
    check(storage.read_text("notes.txt") == "hello", "an ordinary file reads back")
    check(storage.read_text("prepare/absent.viba") is None,
          "a Prepare that neither side has answers None")
    check(not backup.exists() or not any(backup.rglob("*")),
          "and nothing was written into the backup")

    # 没有备份的 storage：Prepare 路径就是普通路径
    plain = PreparedStorage("root", None, str(tmp / "plain-only"))
    plain.write_text("prepare/x.viba", "value := 1\n")
    check(plain.read_text("prepare/x.viba") == "value := 1\n",
          "without a backup a Prepare is just a file in the store")


def _refusals(tmp: Path):
    """判定与 Prepare 的错：不是 bool、没有 __ret__、编不过、记不进备份。"""
    env, host = environ_for(tmp / "refuse-store")

    not_a_verdict = write(tmp, "not_a_verdict.viba", '__ret__ := "yes"\n')
    labelled(is_compliant(not_a_verdict, env), "a verdict is a bool",
             "a rule that answers a string -> Err")

    design = write(tmp, "design_only.viba", "Only := $x int\n")
    labelled(is_compliant(design, env), "has no __ret__",
             "a rule that is not a program -> Err")

    broken = write(tmp, "broken.viba", "__ret__ := -1\n")
    labelled(is_compliant(broken, env), "cannot parse",
             "a rule that does not compile -> Err")

    write(tmp, "boom_case.viba", "__ret__ := $victim ($x 0 * $y 0) * $at \"12:30\"\n")
    boom_rule = write(tmp, "boom_rule.viba", """
import boom_case as witness

measure_distance :=
	int
	<- $env Environment
	<- $case Any
	<- { measure }

__ret__ := measure_distance << $env environ << $case (witness << (environ.sub_env << "witness"))
""")
    labelled(is_compliant(boom_rule, env), "raised",
             "a measurement that blows up -> Err")

    plain = PreparedStorage("root", None, str(tmp / "no-backup-store"))
    no_backup = Environment(plain, EnvironmentCompute(DistanceHost().get_func))
    labelled(prepare_run(str(RULE), no_backup), "cannot record a Prepare",
             "prepare_run without a backup to write -> Err")


if __name__ == "__main__":
    scratch = Path(tempfile.mkdtemp(prefix="viba-compliance-"))
    try:
        run(scratch)
    finally:
        for leftover in sorted(scratch.rglob("*"), reverse=True):
            leftover.unlink() if leftover.is_file() else leftover.rmdir()
    print(f"compliance: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)

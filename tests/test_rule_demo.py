"""The rule demo end to end: viba/rule/demo/*.viba.

Three files: demo_metric_func (the metric function and its prepared call),
demo_rule (a rule over that measurement), demo_witness (the evidence). The
verdicts the files claim in their comments are judged here, so the demo
cannot rot.

    python3 tests/test_rule_demo.py
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import viba_ast
from viba.check_tag_and_inline import check_tag_and_inline
from viba.is_sub_type import is_sub_type
from viba.rule import (check_rule_coding_style, is_compliant,
                       reset_predication_by_python_code)
from viba.type import AstNodeType, CustomModuleType, Err, Ok
from viba.viba_type_descriptor import (empty_pool, parse_viba_file, pool_add_file)

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "viba" / "rule" / "demo"
FILES = (
    (ROOT / "viba" / "rule" / "metric.viba", "viba.rule.metric"),
    (ROOT / "viba" / "rule" / "rule.viba", "viba.rule.rule"),
    (DEMO / "demo_metric_func.viba", "viba.rule.demo.demo_metric_func"),
    (DEMO / "demo_rule.viba", "viba.rule.demo.demo_rule"),
    (DEMO / "demo_witness.viba", "viba.rule.demo.demo_witness"),
)

PASS = FAIL = 0


def check(ok: bool, label: str):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def _import_locals(tree) -> dict:
    return {stmt.alias or stmt.module.split(".")[-1]: stmt.module
            for stmt in tree.body if isinstance(stmt, viba_ast.Import)}


def _modules():
    """The five files as modules that resolve each other through imports."""
    built = {}

    def environment(name):
        return Ok(built[name]) if name in built else Err(f"module {name!r} not found")

    for path, module_name in FILES:                 # dependency order
        tree = viba_ast.parse(path.read_text())
        built[module_name] = CustomModuleType(tree, environment, _import_locals(tree))
    return built


def _definition(module, name):
    for node in module.module.body:
        if getattr(node, "name", None) == name:
            return AstNodeType(node.body, module)
    raise AssertionError(f"{name} not found")


def _pool():
    pool = empty_pool()
    for path, module_name in FILES:
        parsed = parse_viba_file(pool, path.read_text(), path.name, module_name)
        assert isinstance(parsed, Ok), (module_name, parsed)
        added = pool_add_file(pool, parsed.ok_value)
        assert isinstance(added, Ok), (module_name, added)
        pool = added.ok_value
    return pool


def _metric_func_code(get_distance):
    """The code block the metric function's trailing Hint carries."""
    for node in viba_ast.walk(get_distance.ast_node):
        if isinstance(node, viba_ast.TypeApp) and node.constructor.split(".")[-1] == "Hint":
            return node.args[0].type.code
    raise AssertionError("the metric function carries no Hint")


def _assertion_of(witness):
    """The constructor of the witness's assertion field: Predicate while it is
    still positive, PredicationFailed once the code has run."""
    for element in viba_ast.walk(witness.ast_node):
        if isinstance(element, viba_ast.Tagged) and element.tag == "$assert_distance_ge_5":
            return element.type.constructor
    raise AssertionError("the witness carries no assertion field")


def main() -> int:
    modules = _modules()
    metric = modules["viba.rule.metric"]
    functions = modules["viba.rule.demo.demo_metric_func"]
    designs = modules["viba.rule.demo.demo_rule"]
    materials = modules["viba.rule.demo.demo_witness"]

    reviewed = check_tag_and_inline(_pool())
    check(isinstance(reviewed, Ok), f"check_tag_and_inline over the demo: {reviewed!r}")

    get_distance = _definition(functions, "GetDistance")
    verdict = is_sub_type(get_distance, _definition(metric, "MetricFuncInterface"))
    check(isinstance(verdict, Ok) and verdict.ok_value is True,
          f"GetDistance <: MetricFuncInterface: {verdict!r}")

    rule = _definition(designs, "DemoRule")
    style = check_rule_coding_style(rule)
    check(isinstance(style, Ok), f"check_rule_coding_style(DemoRule): {style!r}")

    passing = _definition(materials, "DemoWitnessPass")
    verdict = is_compliant(passing, rule)
    check(isinstance(verdict, Ok) and verdict.ok_value is True,
          f"DemoWitnessPass <: DemoRule: {verdict!r}")

    low = _definition(materials, "DemoWitnessLow")
    check(_assertion_of(low) == "Predicate", "DemoWitnessLow is written positive")
    verdict = is_compliant(low, rule)
    check(isinstance(verdict, Ok) and verdict.ok_value is True,
          f"DemoWitnessLow as written judges True (no code has run): {verdict!r}")

    ran = reset_predication_by_python_code(low)
    check(_assertion_of(ran) == "PredicationFailed",
          "running the code flips the low witness to the poison")
    verdict = is_compliant(ran, rule)
    check(isinstance(verdict, Ok) and verdict.ok_value is False,
          f"3 < 5 after the code ran: {verdict!r}")

    failed = _definition(materials, "DemoWitnessFailed")
    check(_assertion_of(failed) == "PredicationFailed",
          "DemoWitnessFailed is the post-run form")
    verdict = is_compliant(failed, rule)
    check(isinstance(verdict, Ok) and verdict.ok_value is False,
          f"DemoWitnessFailed <: DemoRule: {verdict!r}")

    kept = reset_predication_by_python_code(passing)
    check(_assertion_of(kept) == "Predicate", "5 >= 5 flips nothing")
    verdict = is_compliant(kept, rule)
    check(isinstance(verdict, Ok) and verdict.ok_value is True,
          f"the passing witness stays positive: {verdict!r}")

    namespace = {}
    exec(compile(_metric_func_code(get_distance), "<metric_func>", "exec"), namespace)
    point = types.SimpleNamespace
    measured = namespace["metric_func"](point(x=0, y=0), point(x=3, y=4), "12:30")
    check(measured == 5.0, f"metric_func over (0,0) and (3,4) is 5, got {measured!r}")

    print(f"rule_demo: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

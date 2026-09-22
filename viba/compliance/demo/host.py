"""The demo's host side: what the rule's functions actually do.

`DistanceHost` is the `get_func` of a run of `rule_distance.viba` — the case of
the victim at (0,0) and the suspect at (3,4) at 12:30. It records what it was
asked, which is how a test tells a replayed run from a measured one:

- `measured` gets one entry per *measurement actually taken* — a run that
  replayed its Prepare leaves it empty;
- `judged` gets one entry per verdict asked for, which every run does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from viba.compliance import measure
from viba.reflect import access as reflect_access

RULE = Path(__file__).resolve().parent / "rule_distance.viba"
WITNESS = Path(__file__).resolve().parent / "case_at_1230.viba"


class DistanceHost:
    """The demo rule's implementations, and a record of what they were asked."""

    def __init__(self):
        self.measured = []
        self.judged = []

    def get_func(self, module_path, func_name):
        if func_name == "measure_distance":
            return self.measure_distance
        if func_name == "distance_at_least_5":
            return self.distance_at_least_5
        return None

    def measure_distance(self, env, evidence, case):
        """How far apart the two were: the impure step, through `measure`.

        `env` is the environment the call runs under — the rule hands a
        temporary one, since a call has no address of its own. `evidence` is the
        case's environment, and that is where the Prepare is kept, so the
        evidence says which case it belongs to: `root/case_at_1230/prepare/
        measure_distance.viba`.
        """
        def compute(prepared):
            victim = _point(prepared, "victim")
            suspect = _point(prepared, "suspect")
            self.measured.append((victim, suspect))
            return int(round(((victim[0] - suspect[0]) ** 2
                              + (victim[1] - suspect[1]) ** 2) ** 0.5))
        return measure(env, "measure_distance", case, compute, evidence=evidence)

    def distance_at_least_5(self, env, d):
        """The predicate: pure, so it runs on every judgment."""
        self.judged.append(d.value)
        return d.value >= 5


def _point(node, tag):
    """(x, y) of a point member, read through the reflection protocol."""
    point = node.by_tag(tag)
    return (reflect_access.leaf(point.by_tag("x")).ok_value,
            reflect_access.leaf(point.by_tag("y")).ok_value)

"""Compliance: rules and witnesses, as executable viba modules.

A rule is a program (`environ` in, a `bool` verdict out) and a witness is the
program that answers the material it judges, so a judgment is a run —
`is_compliant(rule_file, environ)`.
The impure part of a rule (measuring something) goes through `measure`, which
replays a **Prepare** that was backed up before the run and records one that was
not (see `viba/compliance/storage.py`).

The worked example is `viba/compliance/demo/`; the story is in
`viba-compliance.md`.
"""

from viba.compliance.judge import (CALL_TAG, MEASURED_TAG, call_of, is_compliant,
                                   measure, measured_of, prepare_material,
                                   prepare_path, prepare_run, read_prepare,
                                   record_prepare)
from viba.compliance.storage import PREPARE_PREFIX, PreparedStorage

__all__ = [
    "CALL_TAG", "MEASURED_TAG", "PREPARE_PREFIX", "PreparedStorage",
    "call_of", "is_compliant", "measure", "measured_of", "prepare_material",
    "prepare_path", "prepare_run", "read_prepare", "record_prepare",
]

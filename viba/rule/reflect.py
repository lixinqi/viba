"""viba.rule.reflect — the rule layer's accessor.

The access side lives in ``viba/reflect.py``. Here is only what the rule layer
adds: its own unit names (viba-rule.md's markers and predicate forms) and its
word for a material, ``Witness``.

    from viba.rule import reflect

    reflect.access.get_by_path(node, [reflect.by_tag("len")])
    reflect.access.list_fields(node, definition)
"""

from viba.reflect import (
    Config,
    VibaAccess,
    VibaData,
    VibaNode,
    VibaPath,
    VibaReflectError,
    VibaStep,
    at_index,
    at_key,
    by_field_index,
    by_tag,
)

# viba-rule.md calls one material a Witness; the generic layer binds Data to
# VibaData.
Witness = VibaData

# The rule layer's units: viba-rule.md's rule markers and the predicate forms.
access = VibaAccess(Config(never_eqv={"Oneof"},
                           nil_eqv={"Object", "RuleObject", "Predicate",
                                    "PredicationFailed"}))

__all__ = [
    "access", "Config",
    "VibaAccess", "VibaData", "VibaNode", "VibaStep", "VibaPath", "Witness",
    "VibaReflectError",
    "by_tag", "by_field_index", "at_index", "at_key",
]

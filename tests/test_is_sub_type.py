"""Round-trip and semantics tests for viba.is_sub_type.

Run directly:  python3 tests/test_is_sub_type.py
Each entry_type("...") below is a *type*; `check` asserts the expected
judgment. The round-trip identities per design:

    A <: A                    (reflexivity)
    Cloned(A) <: A   -> False (nominal: a renamed copy is NOT A)
    A <: Cloned(A)   -> False
    parse(unparse(A)) <: A    (serialization is judgment-preserving)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import ast as viba_ast
from viba.type import (
    BUILTIN_MODULE,
    CustomModuleType,
    Err,
    Ok,
    RuleContainsPoisonError,
    custom_module,
    entry_type,
)
from viba.is_sub_type import is_sub_type

PASS = FAIL = 0


def check(sub, sup, expected: bool, label: str):
    global PASS, FAIL
    got = is_sub_type(sub, sup)
    if got is expected:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}: expected {expected}, got {got}")


def cloned(entry):
    """The same inline structure wrapped as if from another module."""
    return entry_type(viba_ast.unparse(viba_ast.parse(f"X := {entry}")).split(":=")[1].strip())


def roundtrip(entry):
    """parse -> unparse -> re-parse; must preserve the judgment."""
    canon = viba_ast.unparse(viba_ast.parse(f"X := {entry}"))
    re_parsed = canon.split(":=")[1].strip()
    return entry_type(re_parsed)


# ----------------------------------------------------------------------
# Round-trip identities
# ----------------------------------------------------------------------

CASES = [
    "$x int",
    "$x int * $y str",
    "$ok int | $err never",
    "(int, str)",
    "str <- int",
    "int <- X <- str",
    "$head int * $tail List[int] | void",
    "int | ...",
]

for src in CASES:
    a = entry_type(src)
    check(a, a, True, f"reflexive {src}")
    check(roundtrip(src), a, True, f"parse(unparse(A)) <: A   [{src}]")
    check(roundtrip(src), roundtrip(src), True, f"roundtrip reflexive   [{src}]")

# Nominal: a differently-NAMED module (different module object) holding
# the same structure is a different type.
m1 = custom_module("Holder := $x int")
m2 = custom_module("Holder := $x int")
t1 = entry_type("Holder", m1)
t2 = entry_type("Holder", m2)
check(t1, t1, True, "same module object, same name")
check(t1, t2, False, "different module object, same name -> nominal false")

# ----------------------------------------------------------------------
# Literals vs base types
# ----------------------------------------------------------------------

check(entry_type("42"), entry_type("int"), True, "42 <: int")
check(entry_type("42"), entry_type("float"), False, "42 /<: float (nominal basics)")
check(entry_type("3.5"), entry_type("float"), True, "3.5 <: float")
check(entry_type('"hi"'), entry_type("str"), True, '"hi" <: str')
check(entry_type("true"), entry_type("bool"), True, "true <: bool")
check(entry_type("true"), entry_type("int"), False, "true /<: int (bool not int)")
check(entry_type("42"), entry_type("43"), False, "42 /<: 43 (literal equality)")
check(entry_type("42"), entry_type("42"), True, "42 <: 42")

# ----------------------------------------------------------------------
# Poison leaf
# ----------------------------------------------------------------------

check(entry_type("AssertionViolated"), entry_type("int"), False,
      "AssertionViolated <: int -> False")
try:
    is_sub_type(entry_type("42"), entry_type("AssertionViolated"))
    print("FAIL: poison on sup side must raise")
    FAIL += 1
except RuleContainsPoisonError:
    PASS += 1

# ----------------------------------------------------------------------
# Structure: sums, products, tuples, exponents, ellipsis
# ----------------------------------------------------------------------

check(entry_type("$a int | $b str"), entry_type("$a int | $b str"), True, "sum skeleton")
check(entry_type("$a 42"), entry_type("$a int | $b str"), True, "witness picks a branch")
check(entry_type("$c 42"), entry_type("$a int | $b str"), False, "unknown tag")
check(entry_type("$b 'x' * $a 42"), entry_type("$a int * $b str"), True,
      "product tag order irrelevant (commutative)")
check(entry_type("$a 42"), entry_type("$a int * $b str"), False,
      "product missing a sup field")
check(entry_type("$a 42 * $extra 'z'"), entry_type("$a int"), True,
      "width: sub may carry extra fields")
check(entry_type("(42, 'x')"), entry_type("(int, str)"), True, "tuple positional")
check(entry_type("(42, 'x')"), entry_type("(str, int)"), False, "tuple order matters")
check(entry_type("(42, 'x')"), entry_type("(int, str, float)"), False, "tuple arity")
check(entry_type("(42, 'x')"), entry_type("int * str"), False, "tuple is not product")
check(entry_type("int <- str"), entry_type("int <- str"), True, "exponent")
check(entry_type("42"), entry_type("..."), True, "ellipsis absorbs")
check(entry_type("$x never"), entry_type("$x never"), True, "never copied verbatim")
check(entry_type("$x 1"), entry_type("$x never"), False, "value in never branch -> forbidden")

# ----------------------------------------------------------------------
# Definitions, generics, recursion, cross-module
# ----------------------------------------------------------------------

mod = custom_module("""
List[T] := $head T * $tail List[T] | void
IntList := List[int]
MyList[T] := $head T * $tail MyList[T] | void
""")

check(entry_type("List[int]", mod), entry_type("List[int]", mod), True,
      "recursive generic, same instantiation")
check(entry_type("List[int]", mod), entry_type("List[float]", mod), False,
      "List[int] /<: List[float] (nominal args)")
check(entry_type("MyList[int]", mod), entry_type("List[int]", mod), False,
      "structural clone is NOT List (nominal)")
check(entry_type("IntList", mod), entry_type("List[int]", mod), False,
      "alias name is not the generic's name (nominal)")
check(entry_type("List", BUILTIN_MODULE), entry_type("List", BUILTIN_MODULE), True,
      "builtin List constructor nominal self-check")

other = custom_module("Widget := $w int")
main = custom_module(
    "Gadget := $g Widget",
    environment=lambda name: Ok(other) if name == "ext" else Err("nope"),
)
# TypeRef `Widget` is not in main; environment maps *module* names, so
# reference it through the imported name resolution path:
main_with_ref = custom_module(
    "Gadget := $g Widget",
    environment=lambda name: Ok(other) if name == "Widget" else Err("nope"),
)
check(entry_type("Gadget", main_with_ref), entry_type("Gadget", main_with_ref), True,
      "cross-module TypeRef via environment resolves")

# Bulk reflexivity: every parser test case compares equal to itself.
import ast as _py_ast

tree_src = _py_ast.parse(open("viba/parser.py").read())
count = 0
for node in _py_ast.walk(tree_src):
    if isinstance(node, _py_ast.Assign) and getattr(node.targets[0], "id", "") == "test_cases":
        for tup in node.value.elts:
            if isinstance(tup, _py_ast.Tuple) and isinstance(tup.elts[0], _py_ast.Constant):
                src = tup.elts[0].value
                try:
                    body = viba_ast.parse(src).body[0]
                    if isinstance(body, viba_ast.Import):
                        continue
                    text = viba_ast.unparse(viba_ast.Module([body]))
                    t = entry_type(text.split(":=")[1].strip())
                    if not is_sub_type(t, t):
                        print(f"FAIL: reflexivity of suite case {src!r}")
                        FAIL += 1
                    else:
                        PASS += 1
                    count += 1
                except Exception as e:
                    print(f"FAIL: suite case {src!r} raised {type(e).__name__}: {e}")
                    FAIL += 1
print(f"bulk reflexivity on {count} suite definition bodies")

# ----------------------------------------------------------------------
# White-box cases: one per checker branch (see viba/is_sub_type.py)
# ----------------------------------------------------------------------

# Poison beats the ellipsis wildcard; nested poison on sub is False,
# nested poison on sup raises.
check(entry_type("AssertionViolated"), entry_type("..."), False,
      "poison vs ellipsis -> False (poison beats wildcard)")
check(entry_type("$a 1 | $b AssertionViolated"), entry_type("$a int | $b str"), False,
      "poison nested in sub sum -> False, no raise")
try:
    is_sub_type(entry_type("$a 1"), entry_type("$a int | $b AssertionViolated"))
    print("FAIL: poison nested in sup sum must raise")
    FAIL += 1
except RuleContainsPoisonError:
    PASS += 1

# Unit/bottom nodes lift to UnitType/NeverType: () IS void by name.
check(entry_type("()"), entry_type("void"), True, "() <: void (unit identity)")
check(entry_type("()"), entry_type("42"), False, "() /<: 42")
check(entry_type("never"), entry_type("never"), True, "never node <: never ref")
check(entry_type("never"), entry_type("int"), True, "never node is bottom")

# Exponent flattening: raw left-nested binaries must agree with
# canonical ExponentChain, and arity direction is long <: short.
check(entry_type("int <- X <- str"), entry_type("int <- str"), True,
      "extra leading-fed arg: long chain <: short chain")
check(entry_type("int <- X <- float"), entry_type("int <- str"), False,
      "long chain with mismatched shared arg")
check(entry_type("int <- str"), entry_type("int <- X <- str"), False,
      "short chain is NOT <: long chain")
canon_exp = viba_ast.unparse(viba_ast.parse("E := int <- X <- str"))
chain_exp = entry_type(canon_exp.split(":=")[1].strip())
check(chain_exp, entry_type("int <- X <- str"), True,
      "canonical ExponentChain == raw nested Exponent")

# A resolved named definition is not its own expansion (nominal).
shapes = custom_module("Shape := $w int * $h int")
check(entry_type("Shape", shapes), entry_type("$w int * $h int", shapes), False,
      "named definition is NOT structurally its body")
check(entry_type("Shape", shapes), entry_type("Shape", shapes), True,
      "named definition reflexive")

# Same name resolved through different module objects -> nominal False.
env_target = custom_module("Widget := $w int")
env_a = custom_module("", environment=lambda name: Ok(env_target))
check(entry_type("Widget", env_a), entry_type("Widget", other), False,
      "same name via env vs local: different module objects -> False")

# OpaqueType atoms are free variables: identity is the name alone.
opaque_mod = custom_module("")
check(entry_type("T", opaque_mod), entry_type("T", opaque_mod), True,
      "opaque atom same module same name")
check(entry_type("T", opaque_mod), entry_type("T", custom_module("")), True,
      "opaque atom is a free variable: name only, module-agnostic")
check(entry_type("T", opaque_mod), entry_type("U", opaque_mod), False,
      "opaque atom different name -> False")

# Tagged body that is itself a product.
check(entry_type("$cfg ($a int * $b str)"), entry_type("$cfg ($a int * $b str)"),
      True, "tagged body is a product")
check(entry_type("$cfg ($a int * $b str)"), entry_type("$cfg ($a int * $b float)"),
      False, "tagged product body mismatch")

# TypeApp arity and argument order.
check(entry_type("List[int]"), entry_type("List[int, str]"), False,
      "TypeApp arity mismatch")
check(entry_type("List[str, int]"), entry_type("List[int, str]"), False,
      "TypeApp argument order matters")

# Builtin generic vs a user-defined generic of compatible shape:
# nominal, so they never mix.
generics = custom_module("MyList[T] := $head T * $tail MyList[T] | void")
check(entry_type("MyList[int]", generics), entry_type("List[int]", generics),
      False, "builtin List vs user MyList: nominal, never mix")
box_a = custom_module("Box[T] := $value T")
box_b = custom_module("Box[T] := $value T")
check(entry_type("Box[int]", box_a), entry_type("Box[int]", box_b), False,
      "same generic name in different modules -> False")

# Tuple vs product stay distinct at the judgment layer.
check(entry_type("(int, str)"), entry_type("$a int * $b str"), False,
      "tuple is not a tagged product")

print(f"\npassed {PASS}, failed {FAIL}")
sys.exit(1 if FAIL else 0)

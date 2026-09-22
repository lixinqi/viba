# Viba

A DSL for defining types using algebraic operations — sum (`|`), product (`*`), exponent (`<-`), and partial computation (`<<`).

## Language Reference

### Syntax

```
Name[T, U, ...] := body
```

| Operator | Syntax | Meaning |
|----------|--------|---------|
| Assign | `Name := body` | Type definition |
| Sum | `A \| B` | Either A or B |
| Product | `A * B` | Both A and B |
| Exponent | `B <- A` | Function from A to B |
| Partial | `T << $a A` | Function T with that written argument given: `(B <- $a A) << $a A` is `B`; what is given must fit the slot (`A' <: A`) |
| Generic | `Name[T]` | Parameterized type |
| Tag | `$label T` | Named field / variant |
| Nil | `nil` | Product identity (`A * nil = A`); `void` and `None` are aliases |
| Never | `never` | Sum identity (`A \| never = A`), the bottom: it is a subtype of everything |
| Any | `Any` | The top: every type is its subtype, and only Any (or a shape equal to it, e.g. `Any \| int`) is below it |
| Ellipsis | `...` | Open/variadic type |
| Tuple | `(A, B, C)` | Positional product (order matters); not sugar for the tagged `*` |
| Code block | `{ ... }` | Arbitrary text, supports nesting |
| Import | `import a.b [as c]` | Module reference (top level) |

### Strings

- `"double quoted"` — standard string
- `'single quoted'` — single-quoted string
- `'''triple quoted'''` — preserves newlines and whitespace

### Comments

`# to end of line`

### Operator Precedence (low to high)

1. `<<` (partial computation) — left-associative, loosest
2. `|` (sum) — left-associative
3. `*` (product) — left-associative
4. `<-` (exponent) — left-associative

All binary operators are left-associative. `T << X` is not a constructor: it reduces a
written function on the spot, and giving every argument leaves the result itself.

### Examples

```viba
# Standard ADTs
Option[T] := $some T | nil
Result[T, E] := $ok T | $err E

# Function types
Map[A, B] := B <- A
Curried := C <- B <- A

# Struct (product of tagged fields)
MatchContext :=
  Object
  * $match_result MatchResult
  * $target fx.GraphModule

# Open sum type
Variadic := A | B | ...

# Literals
Config := "fast" * 42 * 3.14

# Code block
Handler := {def forward(self, x): return x}
```

## Docs

| Document | Subject |
|----------|---------|
| `viba-reflect.md` | The reflection protocol: addressing a design, reading a material |
| `viba-interpreter.md` | Running a module: `environ` in, `__ret__` out — the executable reading |
| `viba-compliance.md` | Rules and witnesses as programs: judging, Prepare, replay |
| `viba_builder.md` | Writing .viba source from Python expressions |
| `contract-enough.md` | What "the implementation is enough" means |
| `viba-rule.md` | **Deprecated** — the old rule layer: rules, witnesses, judgment |

## Modules

`viba/` contains only the parser and the `ast` package — the parser's
grammar actions build `viba.viba_ast.nodes` objects directly; there is no
intermediate representation.

| Module | Summary |
|--------|---------|
| `parser.py` | Lexer + parser (PLY) — .viba source straight to `viba.viba_ast` nodes |
| `ast/nodes.py` | AST node classes (cf. `ast.AST`) |
| `ast/chain.py` | Flattens nested binary trees into Sum/Product/Exponent chains and back |
| `ast/unparse.py` | Code generator — nodes back to canonical .viba source |
| `ast/_match.py` | Keyword-argument pattern matching over node classes |
| `ast/__init__.py` | Public API: `parse`, `unparse`, `canonical`, `dump`, `walk`, visitors |
| `builder.py` | Writes .viba source from Python expressions — see `viba_builder.md` |
| `partial.py` | Reduces `T << X`: the function with that written argument given |
| `interpreter.py` | Runs a module: `environ` in, `__ret__` out — see `viba-interpreter.md` |
| `builtin.viba` | Builtin vocabulary visible from every module — the environment shapes among them |
| `compliance/` | Rules and witnesses as programs — see `viba-compliance.md` |
| `api.viba` | The package's top-level API, as viba signatures |

`viba/rule/` is the old rule layer — a Viba application built on the core. It is
**deprecated and no longer maintained**: nothing in the core depends on it, it is
not the way to write rules any more, and it does not follow the core as it moves.
The replacement is the executable reading of the same files (`interpreter.py`,
`viba-interpreter.md`): a rule, its evidence and its metric are written as a module
that runs, and judgment is `<<` plus the type check. The vocabulary below stays in
the tree for reference only.

The rule vocabulary (`RuleObject`, `Predicate`, `Metric`,
`PredicationFailed`, `not`) lives in `builtin.viba`; these modules carry
the accumulated API:

| Module | Summary |
|--------|---------|
| `rule/generate_witnesses.py` | Witnesses of a rule: random ones, and the measured one a prepared call yields |
| `rule/markers.py` | The rule marker (`RuleObject`) and the definitions that carry it |
| `rule/demo/` | The worked example: metric function, prepared call, rule |
| `rule/reset_predication_by_python_code.py` | Runs each `Predicate`'s `$python_code`; a false predication becomes the poison |
| `rule/is_compliant.py` | `witness <: rule` |
| `rule/check_rule_coding_style.py` | Checks a rule against `viba-rule.md` |
| `rule/check_determinate.py` | Well-formed, predicate code runs, every witness judges without error |
| `is_sub_type.py`, `type.py`, `parser.py`, `viba_ast/` | Core: subtype judgment, Type model, syntax |

```python
from viba.rule import check_determinate, generate_witnesses, is_compliant
```

## Usage

```python
from viba import viba_ast

tree = viba_ast.parse("Option[T] := $some T | nil")
print(viba_ast.dump(tree))
print(viba_ast.unparse(tree))   # canonical chain-style source
```

## Installation

```bash
pip install ply
python -m viba.parser                    # parser test suite
python -m viba.viba_ast                  # ast round-trip checks
python tests/corpus/generate_corpus.py --check   # 130-file corpus round-trip
```

## Demo

```viba
# Optional value
Option[T] := $some T | nil

# Result with error
Result[T, E] := $ok T | $err E

# Linked list
List[T] := T * List[T] | nil

# Dictionary entry
Pair[K, V] := $key K * $value V

# HTTP handler
Handler := Response <- Request

# Parse pipeline
Parser := AST <- Tokens <- String

# 2D point
Point := $x float * $y float

# Color enum
Color := $red int | $green int | $blue int
```

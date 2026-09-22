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

## Usage

```python
from viba import viba_ast

tree = viba_ast.parse("Option[T] := $some T | nil")
print(viba_ast.dump(tree))
print(viba_ast.unparse(tree))   # canonical chain-style source
```

The same file is also a program: give it an environment and it answers its `__ret__`.

```python
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret

environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))
interpret("add_demo.viba", environ)      # -> Result[VibaNode]
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

## Docs

| Document | Subject |
|----------|---------|
| [`viba-reflect.md`](viba-reflect.md) | The reflection protocol: addressing a design, reading a material |
| [`viba-interpreter.md`](viba-interpreter.md) | Running a module: `environ` in, `__ret__` out — the executable reading |
| [`viba-compliance.md`](viba-compliance.md) | Rules and witnesses as programs: judging, Prepare, replay |
| [`viba_builder.md`](viba_builder.md) | Writing .viba source from Python expressions |

## Modules

`viba/` is the package: the syntax layer, the type model and the judgment over it, and the
tools built on those.

| Module | Summary |
|--------|---------|
| `viba_ast/` | Node classes, chain canonicalization, unparse, visitors, `dump` |
| `type.py` | The Type model, the builtin names, `module_get_type` |
| `is_sub_type.py` | The subtype judgment (`<<`, units, coinductive cycles, `Any`) |
| `viba_type_descriptor.py` | The descriptor side: files, definitions, members, type expressions |
| `reflect.py` | The reflection protocol: addressing a design, reading a material |
| `serialize.py` | Writes a piece of material back out as viba source |
| `builder.py` | Writes .viba source from Python expressions — see `viba_builder.md` |
| `check_tag_and_inline.py` | The one-place check: one tag per product, inline chains end |
| `is_complete.py` | Whether a design can be reflected through |
| `interpret.py` | Runs a module: `environ` in, `__ret__` out — see `viba-interpreter.md` |
| `builtin.viba` | Builtin vocabulary visible from every module — the environment shapes among them |
| `compliance/` | Rules and witnesses as programs — see `viba-compliance.md` |

Two modules are implementation, not something a caller reaches for: `parser.py` (the PLY
grammar behind `viba_ast.parse`, with a self-test at the bottom) and `partial.py` (the
reduction `<<` goes through, used by the judgment).


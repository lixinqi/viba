# Viba

A DSL for defining types using algebraic operations — sum (`|`), product (`*`), and exponent (`<-`).

## Language Reference

### Syntax

```
Name[T, U, ...] := body
```

| Operator | Syntax | Meaning |
|----------|--------|---------|
| Assign | `Name := body` | Type definition |
| Sum | `A | B` | Either A or B |
| Product | `A * B` | Both A and B |
| Exponent | `B <- A` | Function from A to B |
| Generic | `Name[T]` | Parameterized type |
| Tag | `$label T` | Named field / variant |
| Void | `void` | Product identity (`A * void = A`) |
| Never | `never` | Sum identity (`A | never = A`) |
| Ellipsis | `...` | Open/variadic type |
| Tuple | `(A, B, C)` | Shorthand for `A * B * C` |
| Code block | `{ ... }` | Arbitrary text, supports nesting |

### Strings

- `"double quoted"` — standard string
- `'single quoted'` — single-quoted string
- `'''triple quoted'''` — preserves newlines and whitespace

### Comments

`# to end of line`

### Operator Precedence (low to high)

1. `|` (sum) — left-associative
2. `*` (product) — left-associative
3. `<-` (exponent) — right-associative

### Examples

```viba
# Standard ADTs
Option[T] := $some T | ()
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

## Modules

| Module | Summary |
|--------|---------|
| `parser.py` | Lexer + parser (PLY) — .viba source to dict AST |
| `type.py` | Typed AST dataclasses — dict to Type objects with round-tripping |
| `unparser.py` | Code generator — Type objects back to .viba source |
| `chain.py` | Flattens nested binary trees into Sum/Product/Exponent chains and back |
| `match.py` | Keyword-argument pattern matching over Type subclasses |
| `std_coding_style.py` | Validates definitions against Class/FuncDeclare/FuncImplement styles |

## Installation

```bash
pip install ply
python -m viba.parser   # run test suite
```

## Demo

```viba
# Optional value
Option[T] := $some T | ()

# Result with error
Result[T, E] := $ok T | $err E

# Linked list
List[T] := T * List[T] | ()

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

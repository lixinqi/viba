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
| Sum | `A \| B` | Either A or B |
| Product | `A * B` | Both A and B |
| Exponent | `B <- A` | Function from A to B |
| Generic | `Name[T]` | Parameterized type |
| Tag | `$label T` | Named field / variant |
| Void | `void` | Product identity (`A * void = A`) |
| Never | `never` | Sum identity (`A \| never = A`) |
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

1. `|` (sum) — left-associative
2. `*` (product) — left-associative
3. `<-` (exponent) — left-associative

All binary operators are left-associative.

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

`viba/` contains only the parser and the `ast` package — the parser's
grammar actions build `viba.ast.nodes` objects directly; there is no
intermediate representation.

| Module | Summary |
|--------|---------|
| `parser.py` | Lexer + parser (PLY) — .viba source straight to `viba.ast` nodes |
| `ast/nodes.py` | AST node classes (cf. `ast.AST`) |
| `ast/chain.py` | Flattens nested binary trees into Sum/Product/Exponent chains and back |
| `ast/unparse.py` | Code generator — nodes back to canonical .viba source |
| `ast/_match.py` | Keyword-argument pattern matching over node classes |
| `ast/__init__.py` | Public API: `parse`, `unparse`, `canonical`, `dump`, `walk`, visitors |

## Usage

```python
from viba import ast

tree = ast.parse("Option[T] := $some T | void")
print(ast.dump(tree))
print(ast.unparse(tree))   # canonical chain-style source
```

## Installation

```bash
pip install ply
python -m viba.parser                    # parser test suite
python -m viba.ast                       # ast round-trip checks
python tests/corpus/generate_corpus.py --check   # 120-file corpus round-trip
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

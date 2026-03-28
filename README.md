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

```
viba/
  parser.py          Lexer + parser (PLY)      — .viba source -> dict AST
  type.py            Typed AST dataclasses     — dict AST -> Type objects
  unparser.py        Code generator             — Type objects -> .viba source
  chain.py           Chain flattening           — nested binary <-> flat chain
  match.py           Pattern matching           — dispatch on Type subclass
  std_coding_style.py  Style checker           — validate coding conventions
```

### Pipeline

```
.viba source --parser.py--> dict AST --type.py--> Type objects --unparser.py--> .viba source
                                                 |
                                           chain.py (flatten / reconstruct)
                                           match.py (pattern dispatch)
                                           std_coding_style.py (validate)
```

### parser.py

Tokenizer and recursive-descent parser built on PLY. Produces a list of dicts (one per definition). Supports 114 test cases covering all language features.

### type.py

Dataclass-based AST with `from_dict` / `to_dict` round-tripping. `Type.from_dict(data)` dispatches to the correct subclass by `"node"` field.

### unparser.py

Converts Type AST back to formatted .viba source. Chain types render with aligned operators:

```
Pipeline[A, B, C] :=
  C
  <- A
  <- B
```

### chain.py

Flattens nested binary trees into flat chains and back:

- `SumType` <-> `SumChainType(elements=[...])`
- `ProductType` <-> `ProductChainType(elements=[...])`
- `ExponentType` <-> `ExponentChainType(result=..., args=[...])`

### match.py

Keyword-argument pattern matching over Type subclasses:

```python
viba_type_match(
    node,
    SumType=lambda s: s.left,
    TypeRefType=lambda r: r.name,
    _=lambda t: "other",
)
```

### std_coding_style.py

Validates definitions against three standard styles:

| Style | Rule |
|-------|------|
| `ClassDefine` | No exponent types in body |
| `FuncDeclare` | Exactly one exponent at top level |
| `FuncImplement` | Exponent chain as top-level body |

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

# Viba

A DSL for defining types using algebraic operations — sum (`|`), product (`*`), exponent (`<-`), and partial computation (`<<`).

A viba file is the definition side of a program: the types, and for every step
that has to be implemented a hint of what it is for — never the logic. A complete
run means an agent reading those hints and writing the logic behind them.

## Language Reference

### Syntax

`viba/parser.py` holds the grammar, and this is that grammar: one production per
line, checked against the parser whenever its self-test runs. A source that does
not fit it does not compile. `(empty)` matches nothing — an empty program, a
definition with no parameters, a type with no arguments.

```ebnf
program : statement_list | (empty)
statement_list : statement | statement statement_list
statement : definition | import_stmt
definition : type_definition | generic_definition
type_definition : CLASS_NAME ASSIGN partial_expr
generic_definition : CLASS_NAME LBRACKET CLASS_NAME type_param_list RBRACKET ASSIGN partial_expr
type_param_list : COMMA CLASS_NAME type_param_list | (empty)
import_stmt : IMPORT CLASS_NAME optional_alias
optional_alias : AS CLASS_NAME | (empty)
partial_expr : partial_expr APPLY_OP adt_expr | adt_expr
adt_expr : adt_expr SUM_OP product_expr | product_expr
product_expr : product_expr PROD_OP exponent_expr | exponent_expr
exponent_expr : exponent_expr EXP_OP unary_expr | unary_expr
unary_expr : TAGGED_CLASS_NAME type_app_expr | type_app_expr
type_app_expr : CLASS_NAME optional_type_args | primary_expr
optional_type_args : LBRACKET partial_expr adt_arg_list RBRACKET | LBRACKET RBRACKET | (empty)
adt_arg_list : COMMA partial_expr adt_arg_list | (empty)
primary_expr : CLASS_NAME | literal | NIL | NEVER | ANY | ELLIPSIS | LPAREN partial_expr RPAREN | LPAREN adt_expr_list RPAREN | CODE_BLOCK
adt_expr_list : partial_expr COMMA partial_expr | partial_expr COMMA adt_expr_list | (empty)
literal : FLOAT | INT | STRING | SINGLE_STRING | TRIPLE_STRING | BOOLEAN
```

The terminals it names:

| Terminal | Written as |
|----------|------------|
| `CLASS_NAME` | `Option`, `fx.GraphModule` — `\w+(\.\w+)*` |
| `TAGGED_CLASS_NAME` | `$x`, `$meta.id` — a `$` before a name |
| `INT`, `FLOAT` | `42`, `3.14`, `.5` |
| `STRING`, `SINGLE_STRING` | `"one line"`, `'one line'` — escapes yes, raw newline no |
| `TRIPLE_STRING` | `'''keeps newlines, spacing and quotes'''` |
| `BOOLEAN` | `true`, `false` |
| `NIL` | `nil`, `void`, `None` — one unit, three spellings |
| `NEVER`, `ANY`, `ELLIPSIS` | `never`, `Any`, `...` |
| `CODE_BLOCK` | `{ ... }` — text for the host to read; braces nest |
| `IMPORT`, `AS` | `import`, `as` |
| `ASSIGN` | `=` — a definition is written `=`, never `:=` |
| `SUM_OP`, `PROD_OP`, `EXP_OP`, `APPLY_OP` | `\|`, `*`, `<-`, `<<` |
| `LBRACKET`, `RBRACKET`, `LPAREN`, `RPAREN`, `COMMA` | `[`, `]`, `(`, `)`, `,` |

### Operators

| Operator | Form | Meaning |
|----------|------|---------|
| Assign | `Name = body` | Type definition |
| Sum | `A \| B` | Either A or B |
| Product | `A * B` | Both A and B |
| Exponent | `B <- A` | Function from A to B |
| Partial | `T << $a A` | Function T with that written argument given: `(B <- $a A) << $a A` is `B`; what is given must fit the slot (`A' <: A`) |
| Generic | `Name[T]` | Parameterized type |
| Tag | `$label T` | Named field / variant |
| Nil | `nil` | Product identity (`A * nil = A`); `void` and `None` are aliases |
| Never | `never` | Sum identity (`A \| never = A`), the bottom: it is a subtype of everything |
| Any | `Any` | The top: every type is its subtype, and only Any (or a type equal to it, e.g. `Any \| int`) is below it |
| Ellipsis | `...` | Open/variadic type |
| Tuple | `(A, B, C)` | Positional product (order matters); not sugar for the tagged `*` |
| Code block | `{ ... }` | Arbitrary text, supports nesting — a note on a design, or the hint a step's implementation is written from |
| Import | `import a.b [as c]` | Module reference (top level) |

### Strings

- `"double quoted"` — standard string
- `'single quoted'` — single-quoted string
- `'''triple quoted'''` — preserves newlines and whitespace

### Comments

`# to end of line`

### Operator Precedence (low to high)

The layering above is the precedence, from `program` down: partial computation is
loosest, then sum, product, exponent — and application and tagging bind tightest.

1. `<<` (partial computation) — left-associative, loosest
2. `|` (sum) — left-associative
3. `*` (product) — left-associative
4. `<-` (exponent) — left-associative

All binary operators are left-associative. `T << X` is not a constructor: it reduces a
written function on the spot, and giving every argument leaves the result itself.

### Examples

```viba
# Standard ADTs
Option[T] = $some T | nil
Result[T, E] = $ok T | $err E

# Function types
Map[A, B] = B <- A
Curried = C <- B <- A

# Struct (product of tagged fields)
MatchContext =
  Object
  * $match_result MatchResult
  * $target fx.GraphModule

# Open sum type
Variadic = A | B | ...

# Literals
Config = "fast" * 42 * 3.14

# Code block
Handler = {def forward(self, x): return x}
```

## Usage

### Reading a design

```python
from viba import viba_ast

tree = viba_ast.parse("Option[T] = $some T | nil")
print(viba_ast.dump(tree))
print(viba_ast.unparse(tree))   # canonical chain-style source
```

### Running a module

The same file is also a program: a module is a function whose input is `environ`
and whose answer is `__ret__`. A file with no `__ret__` is a design, not a
program, and running it is a `VibaProgramErr`.

```viba
# add_demo.viba
add =
	int
	<- $env Environment
	<- $a int
	<- $b int
	<- { add two integers }

__ret__ =
	add
	<< $env environ
	<< $a 999999
	<< $b 1
```

`{ add two integers }` is all the file says about that step's implementation: a
hint. Every step that has to be implemented gets a hint like it — a line about
what the step is for — and no logic, so a file on its own does not run: that half
is what an agent reads the hints to write. Where its functions are handed to the
run is `get_func`.

The host provides the environment — where snapshots go (`EnvironmentStorage`,
by default a temporary directory) and the implementations (`EnvironmentCompute`,
a `get_func(module_path, func_name)`) — and `interpret` reads a file and runs it.

```python
from viba.interpret import Environment, EnvironmentCompute, EnvironmentStorage, interpret

def get_func(module_path, func_name):
    if func_name == "add":
        return lambda env, a, b: a.value + b.value
    return None

environ = Environment(EnvironmentStorage("root"), EnvironmentCompute(get_func))

answer = interpret("add_demo.viba", environ)
print(answer)                    # Ok(VibaNode(root))
print(answer.ok_value.value)     # 1000000 — the number the host answered
```

- An executable function takes `$env Environment` as one of its slots, and every
  call gives it: `environ` is the builtin standing for the environment the run
  was handed. A function without that slot, or a call that leaves it out, is an
  `VibaProgramErr`.
- A written argument arrives at the host as a piece of material — the literal
  `999999` lands as a node, whose `.value` is the bare number — while the
  environment arrives as itself.
- `interpret` ships no library of its own: every implementation a run can reach is
  one `get_func` answered for, written from the hints the file carries.
- What comes back is `Ok(node)`, `VibaProgramErr(message)`, `UnderlyingVibaOpFailed`
  (`$underlying_viba_op_failed Failure`, when a step's implementation broke) or
  `NotMyDutyException` (`$not_my_duty_exception Duty`, when the run reached a step
  this host does not implement). Both of the last two name
  the `step`; the deferral also carries the `call` material, so the run can be written
  up as a work order and handed on — what [`roadmap.md`](roadmap.md) builds on.
  `node.value` is the answer when it landed on a literal; a product or a sum is walked
  with the node accessors of [`viba-reflect.md`](viba-reflect.md).

### Calling one module from another

```viba
import add_demo as demo

__ret__ = demo << (environ.sub_env << "add_demo")
```

A module is called with the environment it should run under: the name given to
`environ.sub_env` is what `get_func` sees as `module_path`, and the storage path
is the call's identity. So no two module calls in one run may share a path, and
saying it twice is a `VibaProgramErr` that spells the fix out:

```
VibaProgramErr("module 'add_demo' was handed the storage path 'root', which another module call
already used: give each module call a sub-environment of its own (environ.sub_env << ...)")
```

`environ.sub_env << "name"` hands back the same child whenever that name is asked
for, so calling one module twice means choosing two names; `environ.tmp_sub_env
<< ()` is for the calls that need no name, and hands out a fresh child every
time. Where an `import` is looked for is the environment's business: next to the
file that wrote it, then along `Environment`'s `viba_path` (directories, like
`PYTHONPATH`).

### Idempotence: answers have to replay

The one thing in an executable function that need not repeat itself is a host
function — it may read a clock or roll a die. So it, and not viba, keeps the run
repeatable: it snapshots its answer, and the next run of the same call replays it.

```python
import random
from viba.interpret import replayed

def roll(env, n):
    return replayed(env, lambda: random.randint(1, 10 ** 6), f"roll-{n.value}")
```

`replayed(env, compute, name)` reads `<cur storage path>/<name>.viba` under the
store root; finding nothing, it runs `compute()` and writes the answer. Snapshots
are serialized viba data, not pickle: a person can read them, and the type side
reads them as material. Two runs against one store therefore give one value and
walk the impure step once. It is the path that has to be stable: a `tmp_sub_env`
child is new on every call, so what hangs under it never replays.

The whole chapter — `get_file`, the typed reading of a module, the error list —
is [`viba-interpreter.md`](viba-interpreter.md).

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
Option[T] = $some T | nil

# Result with error
Result[T, E] = $ok T | $err E

# Linked list
List[T] = T * List[T] | nil

# Dictionary entry
Pair[K, V] = $key K * $value V

# HTTP handler
Handler = Response <- Request

# Parse pipeline
Parser = AST <- Tokens <- String

# 2D point
Point = $x float * $y float

# Color enum
Color = $red int | $green int | $blue int
```

## Docs

| Document | Subject |
|----------|---------|
| [`viba-reflect.md`](viba-reflect.md) | The reflection protocol: addressing a design, reading a material |
| [`viba-interpreter.md`](viba-interpreter.md) | Running a module: `environ` in, `__ret__` out — the executable reading |
| [`viba-compliance.md`](viba-compliance.md) | Rules and witnesses as programs: judging, Prepare, replay |
| [`viba_builder.md`](viba_builder.md) | Writing .viba source from Python expressions |
| [`roadmap.md`](roadmap.md) | The direction: one ontology, and execution handed across languages, nodes and agents |

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
| `builtin.viba` | Builtin vocabulary visible from every module — `Environment` among them |
| `compliance/` | Rules and witnesses as programs — see `viba-compliance.md` |

Two modules are implementation, not something a caller reaches for: `parser.py` (the PLY
grammar behind `viba_ast.parse`, with a self-test at the bottom that also checks the
Syntax section above spells that grammar out) and `partial.py` (the reduction `<<` goes
through, used by the judgment).


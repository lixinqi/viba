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
statement : definition | import_stmt | pattern_stmt
definition : type_definition | generic_definition
type_definition : CLASS_NAME ASSIGN partial_expr
generic_definition : CLASS_NAME LBRACKET CLASS_NAME type_param_list RBRACKET ASSIGN partial_expr
type_param_list : COMMA CLASS_NAME type_param_list | (empty)
import_stmt : IMPORT class_path optional_alias
pattern_stmt : PATTERN partial_expr
optional_alias : AS CLASS_NAME | (empty)
class_path : CLASS_NAME | class_path DOT CLASS_NAME
partial_expr : partial_expr APPLY_OP adt_expr | member_head APPLY_OP adt_expr | adt_expr
member_head : TAGGED_CLASS_NAME
adt_expr : adt_expr SUM_OP product_expr | product_expr
product_expr : product_expr PROD_OP exponent_expr | exponent_expr
exponent_expr : exponent_expr EXP_OP unary_expr | unary_expr
unary_expr : tag_path member_expr | member_expr
tag_path : TAGGED_CLASS_NAME | tag_path DOT CLASS_NAME
name_tail : DOT CLASS_NAME name_tail | (empty)
member_expr : member_expr DOT CLASS_NAME | type_app_expr
type_app_expr : CLASS_NAME name_tail optional_type_args | primary_expr
optional_type_args : LBRACKET partial_expr adt_arg_list RBRACKET | LBRACKET RBRACKET | (empty)
adt_arg_list : COMMA partial_expr adt_arg_list | (empty)
primary_expr : CLASS_NAME | literal | NIL | NEVER | ANY | ELLIPSIS | LPAREN partial_expr RPAREN | LPAREN adt_expr_list RPAREN | CODE_BLOCK
adt_expr_list : partial_expr COMMA partial_expr | partial_expr COMMA adt_expr_list | (empty)
literal : FLOAT | INT | STRING | SINGLE_STRING | TRIPLE_STRING | BOOLEAN
```

The terminals it names:

| Terminal | Written as |
|----------|------------|
| `CLASS_NAME` | `Option`, `GraphModule` — `\w+` (one segment; a `.` is its own token) |
| `TAGGED_CLASS_NAME` | `$x`, `$meta` — a `$` before a name (one segment) |
| `DOT` | `.` — reads a member: `a.b`, `$meta.id`, `g[T].value` |
| `INT`, `FLOAT` | `42`, `3.14`, `.5` |
| `STRING`, `SINGLE_STRING` | `"one line"`, `'one line'` — escapes yes, raw newline no |
| `TRIPLE_STRING` | `'''keeps newlines, spacing and quotes'''` |
| `BOOLEAN` | `true`, `false` |
| `NIL` | `nil`, `void`, `None` — one unit, three spellings |
| `NEVER`, `ANY`, `ELLIPSIS` | `never`, `Any`, `...` |
| `CODE_BLOCK` | `{ ... }` — text for the host to read; braces nest |
| `IMPORT`, `AS`, `PATTERN` | `import`, `as`, `pattern` |
| `ASSIGN` | `=` — a definition |
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
| Pattern | `Name[T]` | A **generic**: `Name` is a directory of files, one `pattern` line per parameter, and the decision over the written arguments picks one — [`viba-pattern.md`](viba-pattern.md) |
| Tag | `$label T` | Named field / variant |
| Nil | `nil` | Product identity (`A * nil = A`); `void`, `None` and `Object` are aliases — `Object` is the same unit, written at the head of a product laid out as a block |
| Never | `never` | Sum identity (`A \| never = A`), the bottom: it is a subtype of everything; `Oneof` is the same unit, written at the head of a sum laid out as a block |
| Any | `Any` | The top: every type is a subtype of it, and only Any (or a type equal to it, e.g. `Any \| int`) is below it |
| Ellipsis | `...` | Open/variadic type |
| Tuple | `(A, B, C)` | Positional product (order matters); not sugar for the tagged `*` |
| Code block | `{ ... }` | Arbitrary text, supports nesting — a note on a type, or the hint a step's implementation is written from |
| Import | `import a.b [as c]` | Module reference (top level) |

### Writing a definition

Writing one has conventions of its own — every field and argument tagged, a block's head written, a sum written with one branch is no sum, and how the builtin containers are used: [`viba-style.md`](viba-style.md), which
also says how to check what you wrote.

### Evaluation

Viba is **declarative**, not imperative: definitions are **bindings**, not statements, and a file
is not executed top to bottom. Evaluation is demand-driven — a binding is evaluated when it is
used, and only once (call-by-need, with memoization) — so written order is not evaluation order, a
definition may refer to one written after it (a forward reference), one module may not define the
same name twice, and a binding nobody uses is never evaluated. A binding used
while it is still being evaluated is a cyclic definition, reported rather than left to overflow
the stack.

That is one half of the strategy; the other half is at the call: arguments are call-by-value, and
only a function-typed parameter is non-strict — the call written there is handed over with its
environment and evaluated only if the callee asks for it. [`viba-interpreter.md`](viba-interpreter.md)
states both halves and names the cases that pin them; [`viba-style.md`](viba-style.md) §9 is where
writing such a parameter belongs.

### Pattern

A generic is a directory, and one of its files answers. The directory's basename is
the generic's name, `__generic__.viba` marks it, and every other `.viba` file in it is
named by its decision order — a number, read smallest first:

```
demo/is_base_type/__generic__.viba      # __generic__.viba
demo/is_base_type/100.viba              pattern bool | int | float | str
                                        value = true
demo/is_base_type/200.viba              pattern A
                                        value = false
```

```viba
import demo.is_base_type as is_base_type

Flag = is_base_type[bool].value         # true, from 100.viba
Other = is_base_type[list[int]].value   # false, from 200.viba
```

What the decision picks is **that file as a module** (module semantics): the caller reads a
member of it by name. A file that is not a function names its answer for what it is —
`type = A` when it answers a type, `value = true` when it answers a value, the way a C++
template says `::type` and `::value` — so `ret_type_of[int <- int].type` is `int` and
`is_base_type[bool].value` is `true`; a generic file writes no `__decl__` unless the file
itself is a function.

`pattern` writes one line per parameter, in written order. A bare `pattern A` with no
restriction takes the object itself; a known type restricts that argument (the argument must
fit it); a name the file never defines is a parameter, and what stands in the argument there
is extracted — `pattern list[A]` answers the element type with `type = A`. An object that is
a module is read as its `__decl__` for a structured pattern (`pattern A <- B`). A member that
is a function chain **is** the call the chain writes when it stands at a chain head —
`wrapper[inc].type << args.env << inc << $x 1` runs the function it was handed;
when that file also writes `__impl__`, the call is the file's own run, in a sub-environment
named by its decision order, and a name the enclosing decision bound still stands for the
argument written at the call site.
An argument may write a call of its own (`g[add << $a 2]`): it is read as the type that
call stands for, the chain left once the argument is given. A decision that finds no
file is a program error, not `never`. A tag may be written as a symbol string:
`__tagged__["a", T]` is `$a T`, `__tagged__["hello"] << persion` is `$hello << persion`,
and a pattern line may claim the symbol itself — `pattern __tagged__[name, T]` takes `"a"`
for `$a int`, so a design builds a tag out of what another one carried. The member a name
gives as a value is read with `$__getattr__` (`$__getattr__ << args << "name"` is
`args.name`). The whole rule, the pattern forms and the errors:
[`viba-pattern.md`](viba-pattern.md).

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
written function on the spot, and giving it every argument leaves you with the result itself.

### Examples

```viba
# Standard ADTs
Option[T] = $some T | nil
Result[T, E] = $ok T | $err E

# Function types
Map[A, B] = B <- $key A
Curried = C <- $b B <- $a A

# Struct (a product of tagged fields)
MatchContext =
  Object
  * $match_result MatchResult
  * $target fx.GraphModule

# Open sum type
Variadic = $a A | $b B | ...

# Literals
Config = $mode "fast" * $threads 42 * $ratio 3.14

# Code block
Handler = {def forward(self, x): return x}
```

## Usage

### Reading a type

```python
from viba import viba_ast

tree = viba_ast.parse("Option[T] = $some T | nil")
print(viba_ast.dump(tree))
print(viba_ast.unparse(tree))   # canonical chain-style source
```

### Running a module

The same file is a module and — when it says so — a function, and those are the only two
things it is. As a module it is a map of definitions: every top-level definition is a member of
it, read by name (`foo_module.Bar`, and the same way off a generic application: `g[T].value`,
`g[T].type`). As a function it is a declaration and a body: `__decl__` is the chain that says
what the module returns and which parameters it takes (the environment among them), `__impl__`
is what it returns, and a file that declares no `__decl__` is no function at all — calling it
is a program error. A file with no `__impl__` is a type, not a program, and running it raises
`VibaProgramErr`. One module may not define the same name twice.

A module that wants arguments declares them in `__decl__` and reads them back with
`args = __get_args__ << __decl__`: as a type that chain's parameters are a product
type, and as a computation `args` is the data the call was handed. Giving the
environment is what runs the call, and a call without one is a closure — a value
you can keep, pass on, or serialize and run later. Running a module evaluates
`__impl__` and, on demand, the bindings it uses: see [Evaluation](#evaluation)
above and [`viba-interpreter.md`](viba-interpreter.md) for the whole rule.

```viba
# add_demo.viba
__decl__ =
    void
  <- $env Env

args = __get_args__ << __decl__

add =
	int
	<- $env Env
	<- $a int
	<- $b int
	<- { add two integers }

__impl__ =
	add
	<< args.env
	<< $a 999999
	<< $b 1
```

`{ add two integers }` is all the file says about that step's implementation: a
hint. Every step that has to be implemented gets a hint like it — a line about
what the step is for — and no logic, so a file on its own does not run: that half—the logic—is what an agent writes by reading the hints. Where its functions are handed to the
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

- An executable function takes `$env Env` as one of its parameters, and every
  call gives it: `args.env` is the environment the run was handed, and `Env` is
  the builtin name of the environment's type (`Env = Environment`). A function
  without that parameter, or a call that leaves it out, is a
  `VibaProgramErr`.
- The environment is no answer: only a builtin function may declare `Env` as its
  result, and writing `Env` (or `Environment`) as the result of a module's
  `__decl__`, of a definition inside a module, or of a chain a product carries as
  a member refuses the file with a `VibaProgramErr` — the environment is the
  call's rule, not a value to hand back. (A `__impl__` may still be the
  environment: `args.env` is that object, it is simply declared `Any`.)
- A written argument arrives at the host as an instance — the literal
  `999999` lands as a node, whose `.value` is the bare number — while the
  environment arrives as itself.
- A slot written as a function type — `$get_v (T <- $env Env)` — is the
  one exception: the host is handed the written call and runs it with an
  environment it picks, so an argument nobody asks for is never computed and one
  asked for twice is computed once. `builtin.echo << $x v` is the builtin that
  turns a value already worked out into that form: it answers `v` for any
  environment.
- `interpret` ships no library of its own: every implementation a run can reach
  comes from a single `get_func` answer, written from the hints the file carries.
- What comes back is `Ok(node)`, `VibaProgramErr(message)`, `UnderlyingVibaOpFailed`
  (`$underlying_viba_op_failed Failure`, when a step's implementation broke) or
  `NotMyDutyException` (`$not_my_duty_exception Duty`, when the run reached a step
  this host does not implement). Both of the last two name
  the `step`; the deferral also carries the `call` instance, so the deferred step can be written
  up as a work order and handed on — what [`roadmap.md`](roadmap.md) builds on.
  `node.value` is the answer when it landed on a literal; a product or a sum is walked
  with the node accessors of [`viba-reflect.md`](viba-reflect.md).

### Calling one module from another

```viba
import add_demo as demo

__impl__ = demo << (args.env.sub_env << args.env << "add_demo")
```

A module is called with the environment it should run under, and then with its
arguments. The name given to `args.env.sub_env` is what `get_func` sees as
`module_path`, and the storage path is the call's identity — its address. Three things
can happen at one address: a call already running there is a cycle (`the storage path '...' is
already running a call`, with the fix spelled out); a call that already answered there **by the
same module** is that same sub-computation asked twice, and its answer is handed back; a call that
already answered there **by another module** is two calls squeezed into one address, and the host
could not tell them apart:

```
VibaProgramErr("module 'add_demo' was handed the storage path 'root', which another module call
already used: give each module call a sub-environment of its own (args.env.sub_env << args.env << ...)")
```

`args.env.sub_env << args.env << "name"` hands back the same child whenever that name is asked
for, so calling one module twice means choosing two names; `args.env.tmp_env
<< args.env` (also written `$tmp_env << args.env`) is for the calls that need
no name, and hands out a fresh child every time. Where an `import` is looked for is the environment's business: next to the
file that wrote it, then along `Environment`'s `viba_path` (directories, like
`PYTHONPATH`), and last in the builtin directory (`viba/`, where `builtin.viba`
and the package's own vocabulary lives) — so `import ycombinator` reaches the
builtin `ycombinator.Y` from anywhere.

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
reads them as instances. Two runs against one store therefore give one value and
walk the impure step once. It is the path that has to be stable: a `tmp_env`
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

# Linked list: a block, so the heads are written and every field is named
List[T] =
  Oneof
  | Object
    * $head T
    * $tail List[T]
  | nil

# Dictionary entry
Pair[K, V] = $key K * $value V

# HTTP handler
Handler = Response <- $request Request

# Parse pipeline
Parser = AST <- $tokens Tokens <- $source String

# 2D point
Point = $x float * $y float

# Color enum
Color = $red int | $green int | $blue int
```

## Docs

| Document | Subject |
|----------|---------|
| [`viba_tutorial.md`](viba_tutorial.md) | Learning the language: from one definition to a module that runs |
| [`viba-reflect.md`](viba-reflect.md) | The reflection protocol: addressing a type, reading an instance |
| [`viba-interpreter.md`](viba-interpreter.md) | Running a module: `__decl__` in, `__impl__` out — the executable reading, and the call-by-need evaluation strategy |
| [`viba-pattern.md`](viba-pattern.md) | A generic is a directory: `pattern`, the decision order, and what each layer reads |
| [`viba-compliance.md`](viba-compliance.md) | Rules and witnesses as programs: judging, Prepare, replay |
| [`viba_builder.md`](viba_builder.md) | Writing .viba source from Python expressions |
| [`viba-style.md`](viba-style.md) | Writing a definition: tags, heads, containers, and how to check what you wrote |
| [`roadmap.md`](roadmap.md) | The direction: one ontology, and execution handed across languages, nodes and agents |

## Modules

`viba/` is the package: the syntax layer, the type model and the judgment over it, and the
tools built on those.

| Module | Summary |
|--------|---------|
| `viba_ast/` | Node classes, chain canonicalization, unparse, visitors, `dump` |
| `type.py` | The Type model, the builtin names, `module_get_type` |
| `is_sub_type.py` | The subtype judgment (`<<`, units, coinductive cycles, `Any`) |
| `pattern.py` | A generic and its directory of patterns: pattern matching, the decision — see `viba-pattern.md` |
| `viba_type_descriptor.py` | The descriptor side: files, definitions, members, type expressions |
| `reflect.py` | The reflection protocol: addressing a type, reading an instance |
| `serialize.py` | Writes an instance back out as viba source |
| `builder.py` | Writes .viba source from Python expressions — see `viba_builder.md` |
| `check_tag_and_inline.py` | The one-place check: one tag per product, inline chains end |
| `is_complete.py` | Whether a type can be reflected through |
| `interpret.py` | Runs a module: `__decl__` in, `__impl__` out — see `viba-interpreter.md` |
| `builtin.viba` | Builtin vocabulary visible from every module — `Environment` and `Env` among them |
| `compliance/` | Rules and witnesses as programs — see `viba-compliance.md` |

Two modules are implementation, not something a caller reaches for: `parser.py` (the PLY
grammar behind `viba_ast.parse`, with a self-test at the bottom that also checks this file
spells that grammar out, and that every `.viba` sample here, in `viba-style.md` and in
`viba_tutorial.md` compiles) and `partial.py` (the reduction `<<` goes through, used by the judgment).


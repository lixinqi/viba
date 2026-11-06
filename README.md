# Viba

A DSL for defining types using algebraic operations — sum (`|`), product (`*`), exponent (`<-`), and partial computation (`<<`).

A viba file is the definition side of a program: the types, and for every step that
needs logic a hint of what that step is for — never the logic itself. One file is read
by reflection, by serialization and by the interpreter alike; a run gets the logic from
one `get_func`, which the caller hands in.

## Examples

A branch, a request and a handler:

```viba
# A branch: a sum of tagged options, and the condition picks one of them to run
Reply =
  Oneof
  | $found Response
  | $missing UserId

# A request: a product of fields, each under its own tag
Request =
  Object
  * $method str
  * $path str
  * $headers dict[str, str]

# A handler: what it takes, what it returns, and the one step the host implements
Handler =
    Response
  <- $env Env
  <- $request Request
  <- { check the request, then answer it }
```

Every form a definition can take, in the order of the operators: `|`, `*`, `<-`, `<<`.

```viba
# `|` — sum: the options are told apart by tag, or by the kind of the value
Option[T] = $some T | nil
Result[T, E] = $ok T | $err E
Color = $red int | $green int | $blue int

# an open sum takes the options a later file adds
Variadic = $a A | $b B | ...

# `*` — product: fields, each under its own tag
Point = $x float * $y float

# a block heads the block with `Object` and names every field
Reshard =
  Object
  * $plan WeightPlan
  * $cost Cost
  * $target fx.GraphModule

# `|` and `*` in block form: every head spelled out
List[T] =
  Oneof
  | Object
    * $head T
    * $tail List[T]
  | nil

# `<-` — exponent: a function, the result first, then each argument under its own tag
HttpHandler = Response <- $env Env <- $request Request
Curried = C <- $b B <- $a A
Parser = AST <- $tokens Tokens <- $source String

# `<<` — partial computation: one argument per step, and what is left is the result
Replied = handler << $env Env << $request request     # the type that call returns
first = items[0]                                     # `$get_item << items << $env nil << 0`
there = $in << items << $env nil << 3
how_many = $len << items << $env nil                 # elements; a dict is counted by its keys
keys = $keys << DictLiteral[("k", "v")] << $env nil  # a dict's keys, as a list[str]

# leaves: a literal stays on the value side, and a hint block is what the host fills
Config = $mode "fast" * $threads 42 * $ratio 3.14
Validate = UserId <- $raw str <- { reject an empty id, keep the rest }
```

Each block parses as it stands. `viba/core/viba-reference.md` names every form they
use, and `viba/exec/README.md` runs one of them.

A type needs no environment; only a program's `__decl__` takes one when it asks for it, and
the builtin member operators take one in their own chain. Running is one side of the
language; the other — types, the judgment, the data, the documents — is the wiki.

## Read or run

- **The language, the type model and the judgment** — the grammar, the conventions a definition
  follows, generics, the descriptors, serialization, the Python builder, and `is_sub_type`:
  [`viba/core/README.md`](viba/core/README.md).
- **Documents and JSON Schema** — a directory written as one Markdown document, `repomix`, JSON
  Schema, and the checks that keep a design well formed:
  [`viba/wiki/README.md`](viba/wiki/README.md).
- **Run a module** — a run is `__decl__` in, `__impl__` out, and the logic comes from the
  host: [`viba/exec/README.md`](viba/exec/README.md) (`interpret`, `Environment`).

## Installation

```bash
pip install ply
python -m viba.parser                    # parser test suite
python -m viba.viba_ast                  # ast round-trip checks
python tests/corpus/generate_corpus.py --check   # 130-file corpus round-trip
```


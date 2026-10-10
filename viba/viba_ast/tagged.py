"""`tagged`: a tag given as a symbol string.

A tag is an address (`$a`) and a design gives it, so a tag that only exists as
a string value has no place in a type. `tagged` is that place:

    tagged["a", T]      the tagged type `$a T`
    tagged["a"]         the member `$a`

The symbol is a string (`"a"`, or `"$a"` with the sigil), and it has
to be one: letters, digits and `_`, not starting with a digit. A string literal
is folded here, where the source is taken, so what the layers see is the tag
itself. A symbol that is a *name* — a parameter in a `pattern` line, or one a
decision bound — is folded where the design is taken, with that name in its
place (`viba/pattern.py`).
"""

from typing import List, Optional

from viba.viba_ast.nodes import (AST, Constant, Member, Module, Tagged, TypeApp)

TAGGED_NAME = "tagged"

# The builtin member that takes a member by a name given as a value: `$get_attr`.
GET_ATTR_TAG = "$get_attr"

# The builtin member that takes an element by an address given as a value:
# `$get_item` — an index for a list, a key for a dict.
GET_ITEM_TAG = "$get_item"

# The builtin member that asks whether a piece is in a container:
# `$in` — an element for a list, a set or a tuple, a key for a dict.
IN_TAG = "$in"

# The builtin member that counts a container: `$len` — its elements for a
# list, a set or a tuple, its keys for a dict.
LEN_TAG = "$len"

# The builtin member that gives a dict's keys: `$keys`, in the order the
# implementation keeps them in.
KEYS_TAG = "$keys"

# The five members above, as the one group they are: the members the interpreter
# answers itself (`viba/partial.py`, `viba/interpret.py`). A chain headed by one
# of them is the call below, and no other layer has to know which tag it is.
INTERPRETER_MEMBER_TAGS = (GET_ATTR_TAG, GET_ITEM_TAG, IN_TAG, LEN_TAG, KEYS_TAG)

# The parameter the five take first, spelled once: the environment, the way every
# call that runs takes it (`viba/builtin.viba`). It is required — a chain headed by
# one of the five is a call, so until it is given one that chain is a closure, the
# way every call without one is (`viba-interpreter.md`).
INTERPRETER_ENV_SLOT = "$env Env"

# The call each of the five stands for: that environment, then the member's own
# arguments. No answer of the five is computed from the environment it was given —
# each answers from its own arguments — which is why the `nil` the
# interpreter's own chains spell there (`xs[1]`) runs one where it stands
# (`viba-interpreter.md`).
INTERPRETER_MEMBER_CHAINS = {
    GET_ATTR_TAG: f"Any <- {INTERPRETER_ENV_SLOT} <- $value Any <- $name str",
    GET_ITEM_TAG: f"Any <- {INTERPRETER_ENV_SLOT} <- $container Any <- $address Any",
    IN_TAG: f"bool <- {INTERPRETER_ENV_SLOT} <- $container Any <- $piece Any",
    LEN_TAG: f"int <- {INTERPRETER_ENV_SLOT} <- $container Any",
    KEYS_TAG: f"list[str] <- {INTERPRETER_ENV_SLOT} <- $table Any",
}

_CHAINS = {}


def interpreter_member_chain(tag: str):
    """The call a chain headed by this member stands for, as source; None for any
    other tag.

    The chain is taken once and handed out as it is: no layer changes it, so one
    chain is one signature. It is written as a definition of the built-in
    vocabulary, so its names (`Env`, `Any`, `int`, `list[str]`) mean what they
    mean there (`viba/builtin.viba`).
    """
    source = INTERPRETER_MEMBER_CHAINS.get(tag)
    if source is None:
        return None
    chain = _CHAINS.get(tag)
    if chain is None:
        from viba.viba_ast import parse
        # A function chain in the source is what a module's `__decl__` is, so
        # that is how it is parsed (`viba/interpret.py`).
        program = parse(f"__decl__ =\n  {source}\n")
        chain = program.body[0].body
        _CHAINS[tag] = chain
    return chain


def symbol_of(text) -> Optional[str]:
    """The symbol a string in the source spells, or None when it spells none.

    `"a"` and `"$a"` both spell `a`: the sigil is how a tag stands in a
    type, and the symbol is the name without it.
    """
    if not isinstance(text, str):
        return None
    name = text[1:] if text.startswith("$") else text
    if not name:
        return None
    if not (name[0].isalpha() or name[0] == "_"):
        return None
    if not all(part.isalnum() or part == "_" for part in name):
        return None
    return name


def symbol_problem(text) -> Optional[str]:
    """Why this string spells no symbol for a tag, or None when it does."""
    if symbol_of(text) is None:
        return f"{text!r} is no symbol for a tag"
    return None


def tag_of(name: str) -> str:
    """The tag a symbol names: `a` is `$a`."""
    return "$" + name


def string_literal(node) -> bool:
    """Whether this piece is a string literal."""
    return isinstance(node, Constant) and isinstance(node.value, str)


def literal_symbol(node) -> Optional[str]:
    """The symbol a string literal in the source spells; None when it is no literal
    or spells no symbol."""
    if not string_literal(node):
        return None
    return symbol_of(node.value)


def tagged_node(symbol: str, arguments: List[AST]):
    """The tag a `tagged[symbol, ...]` in the source stands for.

    One argument is the member `$symbol` — the member a chain takes from the
    value it gives first (`tagged["hello"] << persion` is `$hello <<
    persion`). Two are the tagged type `$symbol T`.
    """
    if len(arguments) == 1:
        return Member(tag_of(symbol))
    return Tagged(tag_of(symbol), arguments[1])


def tagged_problem(arguments: List[AST]) -> Optional[str]:
    """Why this `tagged[...]` in the source is no tag, or None when it is one."""
    if len(arguments) not in (1, 2):
        return (f"{TAGGED_NAME} takes one argument (the symbol) or two (the symbol "
                f"and the type it marks), not {len(arguments)}")
    first = arguments[0]
    if isinstance(first, Constant):
        # A literal in the source is the whole symbol: a number, a truth value or a
        # string that spells no name is refused where it stands. A name is
        # another matter — which tag it spells is known where it is taken.
        return symbol_problem(first.value)
    return None


def fold_tagged_symbols(tree):
    """A parsed tree with every string symbol in the source folded into its tag.

    A `tagged[symbol, T]` whose symbol is a string literal is the tagged
    type `$symbol T`, so every layer takes a tag where the source had a
    string. The one-argument form, and every symbol that is a name, is left
    standing: a member does not stand alone (it is taken where a chain head is),
    and which tag a name spells is known where the design is taken (a `pattern`
    line binds it, a decision hands it over). An application that is no tag at
    all — the wrong number of arguments, or a string that spells no symbol — is
    refused here, where it stands.
    """
    if isinstance(tree, list):
        return [fold_tagged_symbols(node) for node in tree]
    if isinstance(tree, Module):
        return Module(fold_tagged_symbols(tree.body))
    if isinstance(tree, TypeApp) and tree.constructor == TAGGED_NAME:
        arguments = [fold_tagged_symbols(argument) for argument in tree.args]
        problem = tagged_problem(arguments)
        if problem is not None:
            raise SyntaxError(f"Viba parse error: {problem}")
        symbol = literal_symbol(arguments[0])
        if symbol is None or len(arguments) == 1:
            # A symbol that is a name is taken where the design is taken, and the
            # member a lone symbol names does not stand alone: both are left as
            # given (`viba/partial.py` takes them where a chain head is taken).
            return TypeApp(TAGGED_NAME, arguments)
        return tagged_node(symbol, arguments)
    if not isinstance(tree, AST):
        return tree
    for field in tree._fields:
        value = getattr(tree, field, None)
        if isinstance(value, list):
            setattr(tree, field, [fold_tagged_symbols(part) for part in value])
        elif isinstance(value, AST):
            setattr(tree, field, fold_tagged_symbols(value))
    return tree

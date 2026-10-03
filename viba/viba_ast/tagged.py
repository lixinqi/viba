"""`__tagged__`: a tag written as a symbol string.

A tag is an address (`$a`) and a design writes it, so a tag that only exists as
a string value has no place in a type. `__tagged__` is that place:

    __tagged__["a", T]      the tagged type `$a T`
    __tagged__["a"]         the member `$a`

The symbol is written as a string (`"a"`, or `"$a"` with the sigil), and it has
to be one: letters, digits and `_`, not starting with a digit. A string literal
is folded here, where the source is read, so what the layers see is the tag
itself. A symbol that is a *name* — a parameter in a `pattern` line, or one a
decision bound — is folded where the design is read, with that name in its
place (`viba/pattern.py`).
"""

from typing import List, Optional

from viba.viba_ast.nodes import (AST, Constant, Member, Module, Tagged, TypeApp)

TAGGED_NAME = "__tagged__"

# The builtin member that reads a member by a name given as a value: `$__getattr__`.
GETATTR_TAG = "$__getattr__"


def symbol_of(text) -> Optional[str]:
    """The symbol a written string spells, or None when it spells none.

    `"a"` and `"$a"` both spell `a`: the sigil is how a tag is written in a
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
    """Whether this written piece is a string literal."""
    return isinstance(node, Constant) and isinstance(node.value, str)


def literal_symbol(node) -> Optional[str]:
    """The symbol a written string literal spells; None when it is no literal
    or spells no symbol."""
    if not string_literal(node):
        return None
    return symbol_of(node.value)


def tagged_node(symbol: str, arguments: List[AST]):
    """The tag a written `__tagged__[symbol, ...]` stands for.

    One argument is the member `$symbol` — the member a chain takes from the
    value it gives first (`__tagged__["hello"] << persion` is `$hello <<
    persion`). Two are the tagged type `$symbol T`.
    """
    if len(arguments) == 1:
        return Member(tag_of(symbol))
    return Tagged(tag_of(symbol), arguments[1])


def tagged_problem(arguments: List[AST]) -> Optional[str]:
    """Why this written `__tagged__[...]` is no tag, or None when it is one."""
    if len(arguments) not in (1, 2):
        return (f"{TAGGED_NAME} takes one argument (the symbol) or two (the symbol "
                f"and the type it marks), not {len(arguments)}")
    first = arguments[0]
    if isinstance(first, Constant):
        # A written literal is the whole symbol: a number, a truth value or a
        # string that spells no name is refused where it is written. A name is
        # another matter — which tag it spells is known where it is read.
        return symbol_problem(first.value)
    return None


def fold_written(tree):
    """A parsed tree with every written string symbol folded into its tag.

    A `__tagged__[symbol, T]` whose symbol is a string literal is the tagged
    type `$symbol T`, so every layer reads a tag where the source wrote a
    string. The one-argument form, and every symbol that is a name, is left
    standing: a member does not stand alone (it is read where a chain head is),
    and which tag a name spells is known where the design is read (a `pattern`
    line binds it, a decision hands it over). An application that is no tag at
    all — the wrong number of arguments, or a string that spells no symbol — is
    refused here, where it is written.
    """
    if isinstance(tree, list):
        return [fold_written(node) for node in tree]
    if isinstance(tree, Module):
        return Module(fold_written(tree.body))
    if isinstance(tree, TypeApp) and tree.constructor == TAGGED_NAME:
        arguments = [fold_written(argument) for argument in tree.args]
        problem = tagged_problem(arguments)
        if problem is not None:
            raise SyntaxError(f"Viba parse error: {problem}")
        symbol = literal_symbol(arguments[0])
        if symbol is None or len(arguments) == 1:
            # A symbol that is a name is read where the design is read, and the
            # member a lone symbol names does not stand alone: both are left as
            # written (`viba/partial.py` reads them where a chain head is read).
            return TypeApp(TAGGED_NAME, arguments)
        return tagged_node(symbol, arguments)
    if not isinstance(tree, AST):
        return tree
    for field in tree._fields:
        value = getattr(tree, field, None)
        if isinstance(value, list):
            setattr(tree, field, [fold_written(part) for part in value])
        elif isinstance(value, AST):
            setattr(tree, field, fold_written(value))
    return tree

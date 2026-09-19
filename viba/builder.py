"""viba.builder — write Viba source with Python operators.

    import viba.builder

    vb = viba.builder.Builder("import store_core as sc")
    tag = viba.builder.tag

    vb.UserName = str

    vb.Optional[vb.T] = vb.T | None

    vb.List[vb.T] = (
        vb.Oneof
        | vb.Object * tag.head(vb.T) * tag.tail(vb.List[vb.T])
        | None
    )

    vb.latest_file[vb.Ctx] = (
        vb.sc.Result[vb.sc.FileState]
        ** tag.ctx(vb.Ctx)
        ** tag.file(vb.sc.FileId)
    )

    print(str(vb))

The operators are the language's operators:

    |            sum            A | B
    *            product        A * B
    **           exponent       A <- B, and the right side must be a tag
    tag.x(body)  a tagged field $x body
    tag(body)    a group kept whole
    vb.nil       nil            (None is nil too)
    vb.never     never
    ...          the ellipsis
    vb.Name      a written name, and the slot a definition lands in
    vb.Name[T] = body           Name[T] := body
    vb.Name = body              Name := body
    vb.Name[arg]                Name[arg], an application
    vb.Name[()]                 Name[], an application of nothing
    vb.a.b.Name                 a dotted name, e.g. an import's module
    add_import(vb, m, a)        import m as a

Python's own `str` / `int` / `float` / `bool` / `list` / `set` / `dict` are the
language's names, `int | str` (what Python itself makes of a union) is a sum,
`None` is `nil`, `...` is the ellipsis, and any other Python value is a literal. A Python `list` / `set` / `dict` is the container literal it
means — `[a, b]` is `ListLiteral[a, b]`, `{a, b}` is `SetLiteral[a, b]` (written
in sorted order, a set has none of its own) and `{a: 0}` is
`DictLiteral[(a, 0)]`, in the order given.

The helpers stand outside the builder, so that every `vb.<name>` is a
definition and never a method:

    viba.builder.code(text)                    { text }
    viba.builder.add_import(vb, module, alias) import module as alias
    viba.builder.comment(vb, text)             # text
    viba.builder.check(vb)                     read str(vb) back; raises otherwise

Starting from a file that is already there appends to it:

    vb = viba.builder.Builder(Path("store.viba").read_text())
    vb.Added = vb.Object * tag.x(vb.T)     # a new definition, after the old ones
    Path("store.viba").write_text(str(vb))

An exponent's right side is a tagged field and nothing else, so a slip shows up
right away: `A ** tag.count(vb.T)` is `A <- $count T`, `A ** tag(B ** tag.p(C))`
is `A <- (B <- $p C)`, and `A ** vb.T` is a TypeError — `**` is not a power
here.

Python has no `:=` for a subscript target, so a definition is always `=`:
`vb.Name = body` and `vb.Name[Params] = body` are `Name := body` and
`Name[Params] := body`.

Associativity: the language's `|` and `*` are left-associative and Python's are
too, so `A | B | C` grows the one chain and `A | (B | C)` keeps a branch — the
same shapes the parser reads back. Python's `**` is right-associative while
`<-` is left-associative, so `A ** B ** C` is read as the written chain
`A <- B <- C`; the nested group `A <- (B <- C)` needs `vb(...)`.

What is written is canonical: definitions come out in one style, chains as
chains, so `str(vb)` is stable — parse it and unparse it and nothing moves.
"""

from __future__ import annotations

import types
from typing import List, Optional

from viba import viba_ast

__all__ = ["Builder", "tag", "code", "add_import", "comment", "check"]


class _Expr:
    """One written type expression."""

    def to_ast(self):  # pragma: no cover - every subclass answers
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    # `|` and `*` grow the chain on the left, as the language does; a
    # parenthesized right operand lands as one element, which is a branch.
    def __or__(self, other):
        return _Sum([self, other])

    def __ror__(self, other):
        return _Sum([other, self])

    def __mul__(self, other):
        return _Product([self, other])

    def __rmul__(self, other):
        return _Product([other, self])

    # `<-` is left-associative in the language and `**` is right-associative in
    # Python: `A ** B ** C` arrives as `A ** (B ** C)` and _splice puts it back
    # into the written order A <- B <- C.
    def __pow__(self, other):
        return _Exponent(_splice([self], _power_argument(other)))

    def __rpow__(self, other):
        return _Exponent(_splice([other], _power_argument(self)))


class _Sum(_Expr):
    def __init__(self, elements):
        self.elements = [_wrap(e) for e in elements]

    def __or__(self, other):
        return _Sum(self.elements + [other])

    def to_ast(self):
        return viba_ast.SumChain([_ast(e) for e in self.elements])


class _Product(_Expr):
    def __init__(self, elements):
        self.elements = [_wrap(e) for e in elements]

    def __mul__(self, other):
        return _Product(self.elements + [other])

    def to_ast(self):
        return viba_ast.ProductChain([_ast(e) for e in self.elements])


class _Exponent(_Expr):
    def __init__(self, elements):
        self.elements = [_wrap(e) for e in elements]

    def __pow__(self, other):
        return _Exponent(_splice(self.elements, _power_argument(other)))

    def to_ast(self):
        return viba_ast.ExponentChain([_ast(e) for e in self.elements])


def _power_argument(value) -> _Expr:
    """The right side of `**`: a tagged field, a group, or a run of tags.

    `A ** B` means "A with a tagged field", not "A to the power B", so anything
    else is a slip and is refused here rather than written out.
    """
    if isinstance(value, (_Tagged, _Branch)):
        return value
    if isinstance(value, _Exponent):
        # A chain can only have been built by `**`, and every one of those
        # checked its own right side already.
        return value
    raise TypeError(
        f"the right side of ** is a tagged field — tag.name(body), or "
        f"tag(body) for a group — not {value!r}")


def _splice(elements, other) -> list:
    """The written chain: the run on the left, then a right `**` chain's own
    elements, so that A ** (B ** C) is the flat A <- B <- C."""
    if isinstance(other, _Exponent):
        return list(elements) + other.elements
    return list(elements) + [other]


class _Branch(_Expr):
    """A group kept whole: one element of the chain around it."""

    def __init__(self, inner):
        self.inner = _wrap(inner)

    def to_ast(self):
        return _ast(self.inner)


class _Tagged(_Expr):
    def __init__(self, name: str, body):
        self.name = name.lstrip("$")
        self.body = _wrap(body)

    def to_ast(self):
        return viba_ast.Tagged(f"${self.name}", _ast(self.body))


class _Tag:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"tag.{self.name}"

    def __call__(self, body) -> _Tagged:
        return _Tagged(self.name, body)


class _TagFactory:
    """`tag.head(T)` writes `$head T`; `tag(expr)` keeps a group whole."""

    def __getattr__(self, name: str) -> _Tag:
        if name.startswith("_"):
            raise AttributeError(name)
        return _Tag(name)

    def __call__(self, *args) -> _Expr:
        if len(args) == 1:
            return _Branch(args[0])
        if len(args) == 2:
            return _Tag(args[0])(args[1])
        raise TypeError("tag(body) keeps a group whole; tag(name, body) is a field")


tag = _TagFactory()


class _Apply(_Expr):
    def __init__(self, constructor, args):
        self.constructor = _path(constructor)
        self.args = [_wrap(a) for a in args]

    def to_ast(self):
        return viba_ast.TypeApp(self.constructor, [_ast(a) for a in self.args])


class _Literal(_Expr):
    def __init__(self, value):
        self.value = value

    def to_ast(self):
        return viba_ast.Constant(self.value)


class _Tuple(_Expr):
    def __init__(self, elements):
        self.elements = [_wrap(e) for e in elements]

    def to_ast(self):
        return viba_ast.Tuple([_ast(e) for e in self.elements])


class _Nil(_Expr):
    def to_ast(self):
        return viba_ast.Nil()


class _Ellipsis(_Expr):
    def to_ast(self):
        return viba_ast.Ellipsis()


class _Code(_Expr):
    def __init__(self, text: str):
        self.text = text

    def to_ast(self):
        return viba_ast.CodeBlock(self.text)


class _Name(_Expr):
    """A written name: a reference, an application head, a definition slot."""

    def __init__(self, path: str, owner: Optional["Builder"] = None):
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "owner", owner)

    def __repr__(self):
        return self.path

    def __setattr__(self, name: str, body) -> None:
        """`vb.a.b = body` is a slip: a definition's name is one name."""
        raise TypeError(
            f"a definition's name is one name: write vb.{name} = ..., not "
            f"vb.{self.path}.{name} = ...")

    def __getattr__(self, name: str) -> "_Name":
        if name.startswith("_"):
            raise AttributeError(name)
        return _Name(f"{self.path}.{name}", self.owner)

    def __getitem__(self, key) -> _Apply:
        return _Apply(self.path, _items(key))

    def __setitem__(self, key, body) -> None:
        """`vb.Name[T] = body` writes `Name[T] := body`."""
        if self.owner is None:
            raise TypeError(f"{self.path!r} is a name, not a definition slot")
        params = [_param(p) for p in _items(key)]
        if len(set(params)) != len(params):
            raise TypeError(f"the parameters of {self.path} repeat: {params}")
        definition = viba_ast.GenericDefinition(
            self.owner._free(_definition_name(self.path)), params, _ast(_wrap(body)))
        self.owner._add(definition)

    def to_ast(self):
        # `nil` and `never` are keywords: however they are written, what comes
        # out is the unit itself. The other keywords are slips here — Python
        # has a spelling for them — and are refused.
        if self.path in ("nil", "void", "None"):
            return viba_ast.Nil()
        if self.path == "never":
            return viba_ast.Never()
        if self.path in _SLIPS:
            raise TypeError(f"write {_SLIPS[self.path]}, not vb.{self.path}")
        return viba_ast.TypeRef(self.path)


def code(text: str) -> _Code:
    """`{ text }` — a code block, kept verbatim."""
    return _Code(text)


def add_import(vb: "Builder", module: str, alias: Optional[str] = None) -> "Builder":
    """`import module [as alias]` on that builder.

    An import stands above the definitions, so this only works on a file that
    has none yet: one built from existing text can only take new definitions.
    """
    if vb._existing:
        raise TypeError("this builder continues an existing file: an import "
                        "would have to stand above it")
    return vb._add(viba_ast.Import(module, alias))


def comment(vb: "Builder", text: str) -> "Builder":
    """A `# text` line, written where it stands between the definitions."""
    return vb._add(_Comment(text))


def check(vb: "Builder") -> "viba_ast.Module":
    """Read `str(vb)` back; raises when what was written is not Viba source."""
    try:
        return viba_ast.parse(str(vb))
    except SyntaxError as error:
        raise ValueError(f"what was written does not parse: {error}") from None


_BUILTIN_NAMES = {str: "str", int: "int", float: "float", bool: "bool",
                  list: "list", set: "set", dict: "dict"}

# The keywords that are not the language's own spelling of a value: writing one
# as a name is a slip, and Python has the spelling that was meant.
_SLIPS = {"true": "True", "false": "False"}


def _existing_names(source: str) -> frozenset:
    """The definition names a file already has: appending one is a redefinition."""
    if not source.strip():
        return frozenset()
    try:
        parsed = viba_ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"not Viba source: {error}") from None
    return frozenset(node.name for node in parsed.body
                     if isinstance(node, (viba_ast.TypeDefinition,
                                          viba_ast.GenericDefinition)))


def _definition_name(name: str) -> str:
    """A definition's name: one plain name, and not a keyword."""
    if isinstance(name, str) and name.isidentifier() and name not in _RESERVED:
        return name
    raise TypeError(f"{name!r} is not a definition name")


_RESERVED = ("true", "false", "nil", "void", "None", "never", "import", "as")


def _wrap(value) -> _Expr:
    """A Python value as an expression.

    The builtins are the language's names, `None` is `nil`, `...` is the
    ellipsis, a tuple is a tuple type, and any other value is a literal.
    """
    if isinstance(value, _Expr):
        return value
    if value is None or value is type(None):
        return _Nil()
    if isinstance(value, types.UnionType):
        # What Python's own `int | str` gives: a sum, branches and all.
        return _Sum([_wrap(arg) for arg in value.__args__])
    if value is Ellipsis:
        return _Ellipsis()
    if isinstance(value, type) and value in _BUILTIN_NAMES:
        return _Name(_BUILTIN_NAMES[value])
    if isinstance(value, _Tag):
        raise TypeError(f"{value!r} is a tag with no body: write {value!r}(body)")
    if isinstance(value, (bool, int, float, str)):
        return _Literal(value)
    if isinstance(value, tuple):
        return _Tuple(value)
    if isinstance(value, list):
        return _Apply("ListLiteral", value)
    if isinstance(value, set):
        # A set has no order of its own: write the members sorted, so the same
        # set always gives the same source.
        return _Apply("SetLiteral",
                      sorted(value, key=lambda member: viba_ast.unparse_type(_ast(member))))
    if isinstance(value, dict):
        return _Apply("DictLiteral", [_Tuple([key, item]) for key, item in value.items()])
    raise TypeError(f"cannot write {value!r} as a viba type expression")


def _ast(value) -> "viba_ast.AST":
    return _wrap(value).to_ast()


def _items(key) -> list:
    """The subscript as a list: `vb.X[a, b]` hands over a tuple."""
    return list(key) if isinstance(key, tuple) else [key]


def _param(value) -> str:
    """One generic parameter: a name."""
    if isinstance(value, _Name):
        return value.path
    if isinstance(value, str):
        return value
    raise TypeError(f"a generic parameter is a name, not {value!r}")


def _path(value) -> str:
    """The written name an application hangs off."""
    if isinstance(value, _Name):
        return value.path
    if isinstance(value, str):
        return value
    raise TypeError(f"an application needs a name, not {value!r}")


class _Comment:
    """A `# ...` line: not a node, kept as written."""

    def __init__(self, text: str):
        self.text = text.strip().lstrip("#").strip()

    @staticmethod
    def text(item: "_Comment") -> str:
        return f"# {item.text}" if item.text else "#"


class Builder:
    """One Viba file under construction: what it starts from, then definitions.

    `Builder()` starts an empty file, `Builder("import a.b as c")` starts one
    with that header, and `Builder(existing)` starts from a whole file that is
    already there — only new definitions are appended to it, and a name it
    already defines is refused. `str(vb)` gives the file: what it started from
    as written, then the new definitions canonically.
    """

    def __init__(self, existing: str = ""):
        object.__setattr__(self, "_existing", existing.strip())
        object.__setattr__(self, "_body", [])
        object.__setattr__(self, "_taken", _existing_names(existing))

    def __getattr__(self, name: str) -> _Name:
        if name.startswith("_"):
            raise AttributeError(name)
        return _Name(name, self)

    def __call__(self, expr) -> _Branch:
        """`vb(expr)` keeps a group whole: one element of the chain around it.

        Only an exponent needs it — `vb(A ** B ** C)` is `A <- (B <- C)`,
        where `A ** B ** C` alone is the chain `A <- B <- C` — a sum or a
        product gets the same grouping from plain parentheses.
        """
        return _Branch(expr)

    def __setattr__(self, name: str, value) -> None:
        """`vb.Name = body` writes `Name := body`."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._add(viba_ast.TypeDefinition(
            self._free(_definition_name(name)), _ast(_wrap(value))))

    def _free(self, name: str) -> str:
        """The name, unless the file this builder continues already has it."""
        if name in self._taken:
            raise TypeError(f"{name!r} is already defined: only new definitions "
                            f"can be appended")
        return name

    def _add(self, node) -> "Builder":
        self._body.append(node)
        return self

    def __str__(self) -> str:
        """The whole file: what it started from, then the new definitions."""
        body = list(self._body)
        # A file's imports stand first, however late they were added.
        imports = [n for n in body if isinstance(n, viba_ast.Import)]
        rest = [n for n in body if not isinstance(n, viba_ast.Import)]
        parts = [self._existing] if self._existing else []
        parts += [_Comment.text(item) if isinstance(item, _Comment)
                  else viba_ast.unparse(viba_ast.Module([item]))
                  for item in imports + rest]
        if not parts:
            return ""
        return "\n\n".join(parts) + "\n"

    def __repr__(self) -> str:
        return f"<Builder {len(self._body)} node(s)>"

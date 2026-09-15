# ast.py
# VIBA AST — a Python-ast-style node layer for Viba.
#
#   from viba import ast
#
#   tree = ast.parse("Option[T] := $some T | ()")
#   print(ast.dump(tree, indent=2))
#
# Design notes:
#   * Node shapes mirror the grammar (parser.py): binary Sum/Product/Exponent
#     trees. The flattened "chain" representation lives in chain.py and is
#     intentionally NOT part of this module.
#   * Every node class follows the CPython convention: a `_fields` tuple and
#     plain attributes, constructed positionally or by keyword.
#   * No lineno/col_offset yet — the dict AST produced by parser.py does not
#     carry source locations.

from collections import deque
from typing import Any, Iterator, List, Optional

from viba.parser import parser as _ply_parser
from viba.type import (
    Type,
    DefinitionType,
    SumType,
    ProductType,
    ExponentType,
    TaggedType,
    TypeAppType,
    TypeRefType,
    IdentityType,
    EllipsisType,
    LiteralType,
    CodeBlockType,
)


class AST:
    """Base class of all Viba AST nodes (cf. ast.AST)."""

    _fields: tuple = ()

    def __init__(self, *args: Any, **kwargs: Any):
        if len(args) > len(self._fields):
            raise TypeError(
                f"{self.__class__.__name__} expected at most "
                f"{len(self._fields)} positional arguments, got {len(args)}"
            )
        for name, value in zip(self._fields, args):
            setattr(self, name, value)
        for name in self._fields[len(args):]:
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
            else:
                setattr(self, name, None)
        if kwargs:
            raise TypeError(
                f"{self.__class__.__name__} got unexpected keyword "
                f"argument(s): {', '.join(kwargs)}"
            )


class Module(AST):
    """A whole .viba file: a sequence of definitions."""

    _fields = ("body",)


class Definition(AST):
    """name[T, U] := body"""

    _fields = ("name", "generic_params", "body")


class Sum(AST):
    """left | right"""

    _fields = ("left", "right")


class Product(AST):
    """left * right"""

    _fields = ("left", "right")


class Exponent(AST):
    """result <- argument  (i.e. argument -> result)"""

    _fields = ("result", "argument")


class Tagged(AST):
    """$tag Type"""

    _fields = ("tag", "type")


class TypeApp(AST):
    """Constructor[Arg, ...]"""

    _fields = ("constructor", "args")


class TypeRef(AST):
    """A named type reference."""

    _fields = ("name",)


class Constant(AST):
    """Literal int / float / str / bool value (cf. ast.Constant)."""

    _fields = ("value",)


class Void(AST):
    """void — the product identity."""

    _fields = ()


class Never(AST):
    """never — the sum identity."""

    _fields = ()


class Ellipsis(AST):
    """... — open/variadic marker."""

    _fields = ()


class CodeBlock(AST):
    """{ ... } — opaque code content, kept verbatim."""

    _fields = ("code",)


# ----------------------------------------------------------------------
# Parse: .viba source -> ast.Module
# ----------------------------------------------------------------------


def _from_dict(data: dict) -> AST:
    node = data["node"]
    if node == "Definition":
        return Definition(
            data["name"], data["generic_params"], _from_dict(data["body"])
        )
    if node == "SumType":
        return Sum(_from_dict(data["left"]), _from_dict(data["right"]))
    if node == "ProductType":
        return Product(_from_dict(data["left"]), _from_dict(data["right"]))
    if node == "ExponentType":
        return Exponent(_from_dict(data["result"]), _from_dict(data["argument"]))
    if node == "TaggedType":
        return Tagged(data["tag"], _from_dict(data["type"]))
    if node == "TypeApp":
        return TypeApp(data["constructor"], [_from_dict(a) for a in data["args"]])
    if node == "TypeRef":
        return TypeRef(data["name"])
    if node == "Identity":
        # parser emits {"node": "Identity", "type": "ProductIdentity"|"SumIdentity"}
        return Void() if data["type"] == "ProductIdentity" else Never()
    if node == "Ellipsis":
        return Ellipsis()
    if node == "Literal":
        return Constant(data["val"])
    if node == "CodeBlock":
        return CodeBlock(data["code"])
    raise ValueError(f"Unknown node type: {node}")


def parse(source: str) -> Module:
    """Parse Viba source into an ast.Module (cf. ast.parse)."""
    program = _ply_parser.parse(source) or []
    return Module([_from_dict(d) for d in program])


# ----------------------------------------------------------------------
# Conversion to viba.type and unparse (cf. ast.unparse)
# ----------------------------------------------------------------------


def _to_type(node: AST) -> Type:
    """Convert one viba.ast node to its viba.type counterpart."""
    if isinstance(node, Definition):
        return DefinitionType(
            node="Definition",
            name=node.name,
            generic_params=node.generic_params,
            body=_to_type(node.body),
        )
    if isinstance(node, Sum):
        return SumType(node="SumType", left=_to_type(node.left), right=_to_type(node.right))
    if isinstance(node, Product):
        return ProductType(node="ProductType", left=_to_type(node.left), right=_to_type(node.right))
    if isinstance(node, Exponent):
        return ExponentType(node="ExponentType", result=_to_type(node.result), argument=_to_type(node.argument))
    if isinstance(node, Tagged):
        return TaggedType(node="TaggedType", tag=node.tag, type=_to_type(node.type))
    if isinstance(node, TypeApp):
        return TypeAppType(node="TypeApp", constructor=node.constructor, args=[_to_type(a) for a in node.args])
    if isinstance(node, TypeRef):
        return TypeRefType(node="TypeRef", name=node.name)
    if isinstance(node, Constant):
        return LiteralType(node="Literal", val=node.value, val_type=type(node.value).__name__)
    if isinstance(node, Void):
        return IdentityType(node="Identity", type="ProductIdentity")
    if isinstance(node, Never):
        return IdentityType(node="Identity", type="SumIdentity")
    if isinstance(node, Ellipsis):
        return EllipsisType(node="Ellipsis")
    if isinstance(node, CodeBlock):
        return CodeBlockType(node="CodeBlock", code=node.code)
    raise TypeError(f"Cannot convert {node.__class__.__name__} to viba.type")


def to_type(tree: Module) -> List[Type]:
    """Convert an ast.Module to a list of viba.type definitions."""
    return [_to_type(defn) for defn in tree.body]


def unparse(tree: Module) -> str:
    """Convert an ast.Module back to Viba source (cf. ast.unparse)."""
    from viba.chain import convert_to_chain_style
    from viba.unparser import unparse as _unparse_program

    program = [convert_to_chain_style(t) for t in to_type(tree)]
    return _unparse_program(program)


# ----------------------------------------------------------------------
# Traversal (cf. ast.iter_child_nodes / ast.walk)
# ----------------------------------------------------------------------


def iter_child_nodes(node: AST) -> Iterator[AST]:
    """Yield all direct child nodes of `node`."""
    for name in node._fields:
        value = getattr(node, name, None)
        if isinstance(value, AST):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, AST):
                    yield item


def walk(node: AST) -> Iterator[AST]:
    """Breadth-first traversal of `node` and all its descendants."""
    queue = deque([node])
    while queue:
        current = queue.popleft()
        yield current
        queue.extend(iter_child_nodes(current))


# ----------------------------------------------------------------------
# Visitors (cf. ast.NodeVisitor / ast.NodeTransformer)
# ----------------------------------------------------------------------


class NodeVisitor:
    """Walk the AST calling visit_<ClassName> methods."""

    def visit(self, node: AST) -> Any:
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: AST) -> Any:
        for child in iter_child_nodes(node):
            self.visit(child)


class NodeTransformer(NodeVisitor):
    """Walk the AST and allow visit methods to replace or drop nodes."""

    def generic_visit(self, node: AST) -> AST:
        for field in node._fields:
            value = getattr(node, field, None)
            if isinstance(value, AST):
                setattr(node, field, self.visit(value))
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if not isinstance(item, AST):
                        new_list.append(item)
                        continue
                    new_item = self.visit(item)
                    if new_item is None:
                        continue
                    if not isinstance(new_item, AST):
                        raise TypeError(
                            f"{self.__class__.__name__}: visit method must "
                            f"return an AST node or None, got {type(new_item)}"
                        )
                    new_list.append(new_item)
                setattr(node, field, new_list)
        return node


# ----------------------------------------------------------------------
# dump (cf. ast.dump)
# ----------------------------------------------------------------------


def _dump_value(value: Any, annotate_fields: bool, indent: Optional[int], level: int) -> str:
    if isinstance(value, AST):
        return _dump_node(value, annotate_fields, indent, level)
    if isinstance(value, list):
        return "[" + ", ".join(
            _dump_value(item, annotate_fields, indent, level) for item in value
        ) + "]"
    return repr(value)


def _dump_node(node: AST, annotate_fields: bool, indent: Optional[int], level: int) -> str:
    cls = node.__class__.__name__
    fields = []
    for name in node._fields:
        value = getattr(node, name, None)
        rendered = _dump_value(value, annotate_fields, indent, level + 1)
        fields.append(f"{name}={rendered}" if annotate_fields else rendered)

    if indent is None:
        return f"{cls}({', '.join(fields)})"

    # Indented, one node per line
    pad = " " * (indent * (level + 1))
    closing = " " * (indent * level)
    if not fields:
        return f"{cls}()"
    sep = ",\n" + pad
    return f"{cls}(\n{pad}{sep.join(fields)},\n{closing})"


def dump(node: AST, annotate_fields: bool = True, *, indent: Optional[int] = None) -> str:
    """Return a string representation of the AST (cf. ast.dump)."""
    return _dump_node(node, annotate_fields, indent, 0)


# ----------------------------------------------------------------------
# Test run
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Parse every case in the parser test suite through viba.ast
    import ast as _py_ast

    count = 0
    tree_src = _py_ast.parse(open("viba/parser.py").read())
    for node in _py_ast.walk(tree_src):
        if isinstance(node, _py_ast.Assign) and getattr(node.targets[0], "id", "") == "test_cases":
            for tup in node.value.elts:
                if isinstance(tup, _py_ast.Tuple) and isinstance(tup.elts[0], _py_ast.Constant):
                    parse(tup.elts[0].value)
                    count += 1
    print(f"ast.parse OK on {count} suite cases")

    # 2. dump with indent
    demo = parse("Option[T] := $some T | ()")
    print(dump(demo, indent=2))

    # 3. NodeVisitor: count node kinds
    class KindCounter(NodeVisitor):
        def __init__(self):
            self.counts = {}

        def generic_visit(self, node):
            name = node.__class__.__name__
            self.counts[name] = self.counts.get(name, 0) + 1
            super().generic_visit(node)

    counter = KindCounter()
    counter.visit(parse('FinalBoss[In, Out] := ($res.val Out | $res.err never) <- $cfg.mode "fast" * In * 0.99'))
    print("node counts:", counter.counts)

    # 4. NodeTransformer: rename a TypeRef everywhere
    class RenameA(NodeTransformer):
        def visit_TypeRef(self, node):
            if node.name == "A":
                node.name = "Z"
            return node

    renamed = RenameA().visit(parse("T := A * (B <- A)"))
    print("renamed:", dump(renamed))

    # 5. Empty source parses to an empty Module
    empty = parse("")
    assert empty.body == []
    print("empty source OK")

    # 6. Round-trip on the ast layer: ast.unparse(ast.parse(X))
    strict = fixed = failed = 0
    for node in _py_ast.walk(tree_src):
        if isinstance(node, _py_ast.Assign) and getattr(node.targets[0], "id", "") == "test_cases":
            for tup in node.value.elts:
                if not (isinstance(tup, _py_ast.Tuple) and isinstance(tup.elts[0], _py_ast.Constant)):
                    continue
                X = tup.elts[0].value
                try:
                    out1 = unparse(parse(X))
                    out2 = unparse(parse(out1))
                    if out1 == out2:
                        fixed += 1
                    else:
                        failed += 1
                        print("NOT FIXED POINT:", repr(X))
                    if out1 == X:
                        strict += 1
                except Exception as e:
                    failed += 1
                    print("ROUNDTRIP ERROR:", repr(X), "->", type(e).__name__, e)
    print(f"round-trip: strict ===X on {strict}/{count}, fixed point {fixed}, failed {failed}")

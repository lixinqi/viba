"""viba.viba_ast 的验收：parse/unparse/dump、遍历、访问器、链式规范化。

这个包是 parser 之上的一层公共 API（cf. 标准库 ast），前面只有一份
`python -m viba.viba_ast` 的自测在跑，没有进套件。这里把它压一遍：

- 语料就是 `viba/parser.py` 里那份正例表（131 条）：每条都 parse 一遍，
  写到不动点，链式规范化再还原回来还是同一份源码；
- 遍历与访问器：BFS 次序、`visit_<Class>` 分派、替换、删除、坏返回值；
- 链式规范化：分组不丢（`A * (B * C)` 与 `A * B * C` 不同形）、空链的
  单位元、反向重建；
- `viba_type_match` 的分派与 strict 行为；节点构造的参数检查。

    python3 tests/test_viba_ast.py
"""

import ast as py_ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viba.viba_ast import (  # noqa: E402
    Any,
    CodeBlock,
    Constant,
    GenericDefinition,
    Ellipsis,
    Exponent,
    ExponentChain,
    Module,
    Never,
    Nil,
    NodeTransformer,
    NodeVisitor,
    Partial,
    Product,
    ProductChain,
    Sum,
    SumChain,
    Tagged,
    Tuple,
    TypeApp,
    TypeDefinition,
    TypeRef,
    canonical,
    convert_from_chain_style,
    convert_to_chain_style,
    dump,
    iter_child_nodes,
    parse,
    unparse,
    unparse_type,
    walk,
)
from viba.viba_ast._match import match_builder, viba_type_match  # noqa: E402

PASS = FAIL = 0


def check(label: str, ok: bool):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"FAIL: {label}")


def same(label: str, got, want):
    check(f"{label}: {got!r} != {want!r}", got == want)


def raises(label: str, want: str, func, *args, **kwargs):
    """`want` is a substring of the raised message."""
    try:
        func(*args, **kwargs)
    except Exception as exc:
        check(f"{label}: {type(exc).__name__}: {exc}",
              want in str(exc))
    else:
        check(f"{label}: did not raise, wanted {want!r}", False)


def body_of(source: str):
    return parse(source).body[0].body


def _parser_cases():
    """`viba/parser.py` 里那份正例表：131 条能编的源码。"""
    tree = py_ast.parse((ROOT / "viba" / "parser.py").read_text())
    for node in py_ast.walk(tree):
        if (isinstance(node, py_ast.Assign)
                and getattr(node.targets[0], "id", "") == "test_cases"):
            return [tup.elts[0].value for tup in node.value.elts
                    if isinstance(tup, py_ast.Tuple)
                    and isinstance(tup.elts[0], py_ast.Constant)]
    raise AssertionError("viba/parser.py has no test_cases")


# ----------------------------------------------------------------------
# parse / unparse / canonical / dump
# ----------------------------------------------------------------------


def run_module_cases():
    tree = parse("Option[T] = $some T | nil\nOther = int\n")
    check("parse answers a Module", isinstance(tree, Module))
    same("two definitions", [d.name for d in tree.body], ["Option", "Other"])
    check("the first is a generic definition",
          isinstance(tree.body[0], GenericDefinition)
          and tree.body[0].generic_params == ["T"])
    same("the body is a sum", type(tree.body[0].body).__name__, "Sum")
    same("parse of nothing", parse("").body, [])
    same("parse of only spaces", parse("   \n\n").body, [])

    # canonical: 链式规范化后的 Module，原 Module 不动
    plain = parse("X = A | B | C")
    same("before canonical, still binary", type(plain.body[0].body).__name__, "Sum")
    chained = canonical(plain)
    check("canonical answers chains", isinstance(chained.body[0].body, SumChain))
    same("canonical leaves the original alone", type(plain.body[0].body).__name__, "Sum")

    # unparse: 写出规范源码（不带结尾换行）；空 Module 写出空串
    same("unparse of an empty module", unparse(parse("")), "")
    same("unparse writes canonical chains",
         unparse(parse("X = A | B | C")), "X =\n  A\n  | B\n  | C")
    same("unparse of an import", unparse(parse("import a.b as c")), "import a.b as c")

    # unparse_type: 单个类型表达式，不带定义
    same("unparse_type on a node", unparse_type(parse("X = A * B").body[0].body),
         "A * B")
    same("unparse_type on a partial",
         unparse_type(Partial(TypeRef("F"), Tagged("$a", TypeRef("T")))),
         "F << $a T")


def run_unparse_type_cases():
    """unparse_type 直接吃二元/链节点：括号该加的地方加上。"""
    case = lambda source: unparse_type(parse(f"X = {source}").body[0].body)
    same("a sum inside a product keeps its parentheses", case("(A | B) * C"), "(A | B) * C")
    same("and so does a sum on the right", case("A * (B | C)"), "A * (B | C)")
    same("a sum inside a sum needs none", case("(A | B) | C"), "A | B | C")
    same("a sum inside an exponent keeps its parentheses",
         case("(A | B) <- C"), "(A | B) <- C")
    same("a product as an argument keeps its parentheses",
         case("A <- (B * C)"), "A <- (B * C)")
    same("an exponent on the left keeps its parentheses",
         case("A <- B <- C"), "(A <- B) <- C")
    same("and on the right too",
         case("A <- (B <- C)"), "A <- (B <- C)")
    same("a tagged leaf", unparse_type(Tagged("$a", Nil())), "$a nil")
    same("a code block goes back in braces", unparse_type(CodeBlock("x")), "{x}")
    same("a chain inside a binary product keeps its parentheses",
         unparse_type(Product(ProductChain([TypeRef("A"), TypeRef("B")]), TypeRef("C"))),
         "(  A\n  * B) * C")
    same("an empty sum chain is never", unparse_type(SumChain([])), "never")
    same("an empty product chain is nil", unparse_type(ProductChain([])), "nil")
    same("an empty exponent chain is never", unparse_type(ExponentChain([])), "never")
    same("a one-element chain is its element",
         unparse_type(SumChain([TypeRef("A")])), "A")
    from viba.viba_ast.chain import is_chain_type
    check("chain nodes are chains",
          all(is_chain_type(c) for c in (SumChain([]), ProductChain([]), ExponentChain([]))))
    check("a binary node is not a chain",
          not is_chain_type(Product(TypeRef("A"), TypeRef("B"))))

    same("a branch chain is parenthesized inside a product",
         unparse_type(ProductChain([TypeRef("A"), ProductChain([TypeRef("B")])])),
         "A\n* (B)")


def viba_code(text: str):
    """A CodeBlock whose content is `text` without the braces."""
    from viba.viba_ast import CodeBlock
    return CodeBlock(text[1:-1])

    # dump: 带字段名 / 不带 / 缩进 / 没有字段的节点
    node = parse("X = $a int").body[0]
    same("dump with fields", dump(node),
         "TypeDefinition(name='X', body=Tagged(tag='$a', type=TypeRef(name='int')))")
    same("dump without fields", dump(node, annotate_fields=False),
         "TypeDefinition('X', Tagged('$a', TypeRef('int')))")
    same("dump of a node with no fields", dump(Nil()), "Nil()")
    same("dump of a node with no fields, indented", dump(Nil(), indent=2), "Nil()")
    same("dump indented by one",
         dump(node, indent=1),
         "TypeDefinition(\n name='X',\n body=Tagged(\n   tag='$a',\n"
         "   type=TypeRef(\n     name='int',\n    ),\n  ),\n)")
    same("dump of a list field", dump(parse("X = (A, B)").body[0]),
         "TypeDefinition(name='X', body=Tuple(elements=[TypeRef(name='A'), "
         "TypeRef(name='B')]))")


def run_corpus_cases():
    """正例表：每条都 parse、写到不动点、链式规范化再还原。"""
    cases = _parser_cases()
    check(f"the positive table has cases ({len(cases)})", len(cases) > 100)

    broken = []
    for source in cases:
        try:
            once = unparse(parse(source))
            if unparse(parse(once)) != once:
                broken.append(source)
        except Exception as exc:               # noqa: BLE001 - report, do not stop
            broken.append(f"{source!r}: {type(exc).__name__}: {exc}")
    check(f"every case writes to a fixed point ({broken[:3]})", not broken)

    mismatched = []
    for source in cases:
        chained = canonical(parse(source))
        back = Module([convert_from_chain_style(d) for d in chained.body])
        if unparse(back) != unparse(chained):
            mismatched.append(source)
    check(f"chain style and back write the same source ({mismatched[:3]})", not mismatched)


# ----------------------------------------------------------------------
# 遍历与访问器
# ----------------------------------------------------------------------


def run_traversal_cases():
    tree = parse("X = A * B")
    same("walk is breadth-first",
         [node.__class__.__name__ for node in walk(tree)],
         ["Module", "TypeDefinition", "Product", "TypeRef", "TypeRef"])
    same("walk of a leaf is the leaf", [type(n).__name__ for n in walk(Nil())], ["Nil"])
    same("children of a name are none", list(iter_child_nodes(TypeRef("A"))), [])
    same("children of a literal are none", list(iter_child_nodes(Constant(1))), [])
    same("children of a tuple are its elements",
         [type(n).__name__ for n in iter_child_nodes(Tuple([TypeRef("A"), Nil()]))],
         ["TypeRef", "Nil"])
    same("children of an application are its arguments",
         [type(n).__name__ for n in iter_child_nodes(TypeApp("list", [TypeRef("A")]))],
         ["TypeRef"])


def run_visitor_cases():
    seen = []

    class Kinds(NodeVisitor):
        def generic_visit(self, node):
            seen.append(node.__class__.__name__)
            super().generic_visit(node)

    Kinds().visit(parse("X = A * B"))
    same("generic_visit sees every node", seen,
         ["Module", "TypeDefinition", "Product", "TypeRef", "TypeRef"])

    reached = []

    class Shallow(NodeVisitor):
        def visit_Product(self, node):
            reached.append("product")

        def visit_TypeRef(self, node):
            reached.append(node.name)

    Shallow().visit(parse("X = A * B"))
    same("a visit method may stop the recursion", reached, ["product"])

    replaced = []

    class Names(NodeVisitor):
        def visit_TypeRef(self, node):
            replaced.append(node.name)

    Names().visit(parse("X = $a (A * B) | C"))
    same("visit_<Class> dispatches by class name", replaced, ["A", "B", "C"])


def run_transformer_cases():
    class Rename(NodeTransformer):
        def visit_TypeRef(self, node):
            if node.name == "A":
                node.name = "Z"
            return node

    tree = Rename().visit(parse("X = A * (B <- A)"))
    same("a transformer rewrites in place",
         unparse(tree), unparse(parse("X = Z * (B <- Z)")))

    class DropDefs(NodeTransformer):
        def visit_TypeDefinition(self, node):
            return None if node.name == "Drop" else node

    tree = DropDefs().visit(parse("Keep = int\nDrop = str\n"))
    same("returning None drops a list item", [d.name for d in tree.body], ["Keep"])

    class DropArg(NodeTransformer):
        def visit_TypeApp(self, node):
            node.args = []
            return node

    tree = DropArg().visit(parse("X = list[A, B]"))
    same("a transformer may empty a list field", tree.body[0].body.args, [])

    class DropField(NodeTransformer):
        def visit_TypeRef(self, node):
            return None

    tree = DropField().visit(parse("X = A <- B"))
    check("returning None in a single-slot field sets it to None",
          tree.body[0].body.result is None and tree.body[0].body.argument is None)

    class Bad(NodeTransformer):
        def visit_TypeRef(self, node):
            return "not a node"

    raises("a visit method must answer a node or None", "must return an AST node",
           Bad().visit, parse("X = A * B"))

    class BadInList(NodeTransformer):
        def visit_TypeDefinition(self, node):
            return 7

    raises("and the same holds inside a list", "must return an AST node",
           BadInList().visit, parse("X = int\n"))

    class TouchNothing(NodeTransformer):
        def visit_TypeRef(self, node):
            return node

    tree = TouchNothing().visit(Module([Tuple([TypeRef("A"), "stray"])]))
    same("a list item that is not a node is kept", tree.body[0].elements[1], "stray")

    class TupleExtending(NodeTransformer):
        def visit_Nil(self, node):
            return TypeRef("nil")

    tree = TupleExtending().visit(parse("X = (A, nil)"))
    same("a transformer reaches tuple members",
         unparse(tree), "X =\n  (A, nil)")


# ----------------------------------------------------------------------
# 链式规范化
# ----------------------------------------------------------------------


def run_chain_cases():
    def chain(source):
        return convert_to_chain_style(body_of(source))

    flattened = chain("X = A * B * C")
    check("a written run is one chain", isinstance(flattened, ProductChain))
    same("with its elements in order",
         [type(e).__name__ for e in flattened.elements],
         ["TypeRef", "TypeRef", "TypeRef"])

    grouped = chain("X = A * (B * C)")
    same("grouping stays: two elements, the second a chain",
         [type(e).__name__ for e in grouped.elements],
         ["TypeRef", "ProductChain"])

    sums = chain("X = A | B | C")
    check("sum runs chain too", isinstance(sums, SumChain))

    exponents = chain("X = A <- B <- C")
    check("exponent runs chain in written order", isinstance(exponents, ExponentChain))
    same("result first, then the arguments",
         [type(e).__name__ for e in exponents.elements],
         ["TypeRef", "TypeRef", "TypeRef"])
    same("a parenthesised exponent is a branch",
         [type(e).__name__ for e in chain("X = A <- (B <- C)").elements],
         ["TypeRef", "ExponentChain"])

    tagged = chain("X = $a (A * B)")
    check("tags keep their body", isinstance(tagged, Tagged))
    check("and the body is chained", isinstance(tagged.type, ProductChain))
    applied = chain("X = list[A * B]")
    check("applications chain their arguments",
          isinstance(applied.args[0], ProductChain))
    tuples = chain("X = (A * B, C)")
    check("tuples chain their elements", isinstance(tuples.elements[0], ProductChain))
    partial = chain("X = (F <- $a A) << $a (B * C)")
    check("a partial chains both sides",
          isinstance(partial, Partial)
          and isinstance(partial.function, ExponentChain)
          and isinstance(partial.argument, Tagged)
          and isinstance(partial.argument.type, ProductChain))

    definition = convert_to_chain_style(parse("G[T] = A * B").body[0])
    check("definitions are chained through",
          isinstance(definition.body, ProductChain) and definition.generic_params == ["T"])

    check("a chain is left as it is",
          isinstance(convert_to_chain_style(ProductChain([TypeRef("A")])), ProductChain))
    check("Any is left as it is", isinstance(convert_to_chain_style(Any()), Any))
    check("a plain leaf is left as it is",
          isinstance(convert_to_chain_style(TypeRef("A")), TypeRef))
    check("and so is an ellipsis", isinstance(convert_to_chain_style(Ellipsis()), Ellipsis))

    # 反向：链回到左嵌套的二元形式
    same("an empty sum chain is never",
         type(convert_from_chain_style(SumChain([]))).__name__, "Never")
    same("an empty product chain is nil",
         type(convert_from_chain_style(ProductChain([]))).__name__, "Nil")
    same("an empty exponent chain is bottom",
         type(convert_from_chain_style(ExponentChain([]))).__name__, "Never")
    one = convert_from_chain_style(SumChain([TypeRef("A")]))
    check("a one-element chain is its element", isinstance(one, TypeRef))
    three = convert_from_chain_style(SumChain([TypeRef("A"), TypeRef("B"), TypeRef("C")]))
    check("three elements nest to the left",
          isinstance(three, Sum) and isinstance(three.left, Sum))
    check("and the order is the written one",
          isinstance(three.left.left, TypeRef) and three.left.left.name == "A"
          and three.right.name == "C")
    exponent = convert_from_chain_style(ExponentChain([TypeRef("A"), TypeRef("B")]))
    check("an exponent chain becomes an exponent",
          isinstance(exponent, Exponent) and isinstance(exponent.result, TypeRef)
          and isinstance(exponent.argument, TypeRef))
    nested = convert_from_chain_style(
        ProductChain([TypeRef("A"), ProductChain([TypeRef("B"), TypeRef("C")])]))
    check("a branch is rebuilt as a branch",
          isinstance(nested, Product) and isinstance(nested.right, Product))
    check("a non-chain node passes through",
          isinstance(convert_from_chain_style(TypeRef("A")), TypeRef))
    definition = convert_from_chain_style(
        TypeDefinition("X", ProductChain([TypeRef("A"), TypeRef("B")])))
    check("definitions are rebuilt through",
          isinstance(definition, TypeDefinition) and isinstance(definition.body, Product))


# ----------------------------------------------------------------------
# viba_type_match 与节点构造
# ----------------------------------------------------------------------


def run_match_cases():
    node = parse("X = A * B").body[0].body
    same("dispatch by class name",
         viba_type_match(node, Product=lambda p: "product", _=lambda n: "other"),
         "product")
    same("the default handler takes what is left",
         viba_type_match(TypeRef("A"), Product=lambda p: "product", _=lambda n: "other"),
         "other")
    same("anything with a handler of its own matches",
         viba_type_match(Nil(), Nil=lambda n: "nil", _=lambda n: "other"), "nil")
    raises("strict raises when nothing matches", "No handler matched",
           viba_type_match, TypeRef("A"))
    same("strict off answers None",
         viba_type_match(TypeRef("A"), strict=False), None)

    describe = match_builder(Product=lambda p: "a product", _=lambda n: "something")
    same("a built matcher answers", describe(node), "a product")
    same("and falls back", describe(Nil()), "something")
    strict_builder = match_builder(TypeRef=lambda r: r.name)
    same("a strict built matcher answers", strict_builder(TypeRef("A")), "A")
    raises("and raises on the rest", "No handler matched", strict_builder, Nil())


def run_node_cases():
    partial = Sum(TypeRef("A"))
    check("a missing field is None", partial.right is None and partial.left.name == "A")
    raised = Sum(TypeRef("A"), TypeRef("B"))
    check("positional fields land in order",
          raised.left.name == "A" and raised.right.name == "B")
    keyword = Sum(right=TypeRef("B"), left=TypeRef("A"))
    check("keyword fields land by name",
          keyword.left.name == "A" and keyword.right.name == "B")
    raises("too many positional arguments", "expected at most 2",
           Sum, TypeRef("A"), TypeRef("B"), TypeRef("C"))
    raises("an unknown keyword", "unexpected keyword", Sum, nope=1)


def run():
    run_module_cases()
    run_unparse_type_cases()
    run_corpus_cases()
    run_traversal_cases()
    run_visitor_cases()
    run_transformer_cases()
    run_chain_cases()
    run_match_cases()
    run_node_cases()
    print(f"viba_ast: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(run())

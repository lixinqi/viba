"""Self-test for viba.viba_ast: python -m viba.viba_ast"""

import ast as _py_ast

from viba.viba_ast import (
    parse,
    unparse,
    dump,
    walk,
    NodeVisitor,
    NodeTransformer,
    Module,
    Tuple,
    Product,
)

# 1. Parse every case in the parser test suite through viba.viba_ast
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
demo = parse("Option[T] = $some T | ()")
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
counter.visit(parse('FinalBoss[In, Out] = ($res.val Out | $res.err never) <- $cfg.mode "fast" * In * 0.99'))
print("node counts:", counter.counts)

# 4. NodeTransformer: rename a TypeRef everywhere
class RenameA(NodeTransformer):
    def visit_TypeRef(self, node):
        if node.name == "A":
            node.name = "Z"
        return node

renamed = RenameA().visit(parse("T = A * (B <- A)"))
print("renamed:", dump(renamed))

# 5. Empty source parses to an empty Module
empty = parse("")
assert empty.body == []
print("empty source OK")

# 5b. Tuple vs Product are distinct nodes
tup = parse("X = (A, B, C)").body[0].body
prod = parse("X = A * B * C").body[0].body
assert isinstance(tup, Tuple) and len(tup.elements) == 3, dump(tup)
assert isinstance(prod, Product), dump(prod)
nested = parse("X = (A, (B, C))").body[0].body
assert isinstance(nested.elements[1], Tuple), dump(nested)
print("tuple/product distinction OK")

# 5c. Import statements: parse, round-trip, alias
imp = parse("import fx.graph as g\nX = g.Tensor")
assert imp.body[0].module == "fx.graph" and imp.body[0].alias == "g", dump(imp)
assert unparse(parse("import a.b.c")) == "import a.b.c"
assert unparse(parse("import a.b as c")) == "import a.b as c"
print("import statements OK")

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

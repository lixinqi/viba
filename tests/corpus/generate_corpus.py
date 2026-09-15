#!/usr/bin/env python3
# generate_corpus.py
# Generate a corpus of .viba test files:
#   120 files (100 + 20 extra), ~100 lines each, indentation at least
#   3 levels deep.
# Deterministic: fixed seed. APPEND-ONLY: existing case_*.viba files are
# never rewritten — re-runs only create missing indices (the rng stream is
# still consumed for skipped indices, so new files stay deterministic).
# Verify with --check.

import random
import sys
from pathlib import Path

SEED = 20260915
N_FILES = 100
N_EXTRA = 20  # supplementary batch, drawn from an independent stream
N_IMPORT = 10  # import-statement batch, own stream as well
OUT_DIR = Path(__file__).parent

IND = " " * 4  # one indentation level

IMPORT_MODULES = [
    "numpy", "torch", "pandas", "fx.graph", "viba.std", "a.b.c",
    "torch.nn", "collections", "typing", "pathlib",
]

TYPEREFS = [
    "Alpha", "Beta", "Gamma", "Delta", "Omega", "Sigma", "Input", "Output",
    "Config", "State", "Error", "Handler", "Context", "Metric", "GraphModule",
    "fx.Graph", "torch.Tensor", "MatchResult",
]
GENERIC_PARAMS = ["T", "U", "V", "K", "E", "S", "In", "Out"]
TAGS = [
    "$name", "$idx", "$cfg", "$meta", "$data", "$next", "$src", "$dst",
    "$res", "$err", "$input", "$output", "$value", "$kind",
]
PATH_TAGS = ["$meta.id", "$cfg.mode", "$res.val", "$env.scope", "$a.b.c.d"]
DEF_THEMES = [
    "Option", "Result", "Pair", "List", "Map", "Handler", "Config", "State",
    "Node", "Edge", "Stream", "Batch", "Tensor", "Graph", "Plan", "Task",
    "Rule", "Check", "Report", "Slice",
]


# ----------------------------------------------------------------------
# Random expression builder (node = nested tuples)
# ----------------------------------------------------------------------


class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.counter = 0

    def name(self, theme):
        self.counter += 1
        return f"{theme}{self.counter}"

    def atom(self):
        r = self.rng
        choice = r.random()
        if choice < 0.30:
            return ("ref", r.choice(TYPEREFS))
        if choice < 0.40:
            return ("int", r.randint(0, 999))
        if choice < 0.48:
            return ("float", round(r.uniform(0, 10), r.randint(1, 3)))
        if choice < 0.55:
            return ("bool", r.choice([True, False]))
        if choice < 0.65:
            words = r.sample(["fast", "slow", "auto", "viba", "strict", "lazy"], r.randint(1, 3))
            return ("str", " ".join(words))
        if choice < 0.72:
            return ("tstr", r.choice(["line one\nline two", "keep\n  indent", "a\nb\nc"]))
        if choice < 0.78:
            return ("nil",)
        if choice < 0.84:
            return ("never",)
        if choice < 0.90:
            return ("puretag", r.choice(PATH_TAGS))
        if choice < 0.95:
            depth = r.randint(1, 3)
            inner = "x + 1"
            for _ in range(depth - 1):
                inner = f"fn({inner})"
            return ("code", inner)
        return ("ellipsis",)

    def expr(self, depth):
        r = self.rng
        if depth <= 0:
            return self.atom()
        roll = r.random()
        if roll < 0.18:
            return ("tag", r.choice(TAGS), self.expr(depth - 1))
        if roll < 0.38:
            n = r.randint(2, 4)
            return ("sum", [self.expr(depth - 1) for _ in range(n)])
        if roll < 0.58:
            n = r.randint(2, 4)
            return ("prod", [self.expr(depth - 1) for _ in range(n)])
        if roll < 0.72:
            n = r.randint(1, 3)
            args = [self.expr(depth - 1) for _ in range(n)]
            return ("exp", self.expr(depth - 1), args)
        if roll < 0.82:
            n = r.randint(1, 3)
            ctor = r.choice(["List", "Pair", "Option", "Result", "Map", "Box"])
            return ("app", ctor, [self.expr(depth - 1) for _ in range(n)])
        if roll < 0.90:
            n = r.randint(2, 4)
            return ("tuple", [self.expr(depth - 1) for _ in range(n)])
        return ("paren", self.expr(depth - 1))


# ----------------------------------------------------------------------
# Renderer: node -> indented source lines (level = indentation depth)
# ----------------------------------------------------------------------


def is_atom(node):
    return node[0] in (
        "ref", "int", "float", "bool", "str", "tstr", "nil", "never",
        "puretag", "code", "ellipsis",
    )


def render_atom(node):
    kind = node[0]
    if kind == "ref":
        return node[1]
    if kind == "int":
        return str(node[1])
    if kind == "float":
        return repr(node[1])
    if kind == "bool":
        return "true" if node[1] else "false"
    if kind == "str":
        return f"'{node[1]}'"
    if kind == "tstr":
        return f"'''{node[1]}'''"
    if kind == "nil":
        return "nil"
    if kind == "never":
        return "never"
    if kind == "puretag":
        # bare $tag no longer parses (PureTag was cut from the grammar);
        # a standalone tag atom is written with the unit body
        return f"{node[1]} ()"
    if kind == "code":
        return "{" + node[1] + "}"
    if kind == "ellipsis":
        return "..."
    raise AssertionError(node)


def _strip_parens(node):
    while node[0] == "paren":
        node = node[1]
    return node


def flatten_children(kind, children):
    """Splice same-operator children (and parens around them) inline.

    Chain-style canonical form is flat and left-associative, so a
    right-nested same-op group like `A | (B | C)` cannot survive
    unparse -> reparse with an identical AST. Flattening at render time
    keeps every generated file AST-stable under the corpus check.
    Parentheses are stripped to any depth before the operator check.
    """
    out = []
    for ch in children:
        core = _strip_parens(ch)
        if core[0] == kind:
            out.extend(flatten_children(kind, core[1]))
        else:
            out.append(ch)
    return out


def render(node, level):
    """Render node as source lines with `level` indentation levels.

    Viba is whitespace-insensitive: only parentheses define grouping. So
    every composite operand inside an operator chain is parenthesized,
    no matter how the lines are indented.
    """
    pad = IND * level
    kind = node[0]

    if is_atom(node):
        return [pad + render_atom(node)]

    if kind == "paren":
        inner = render(node[1], level + 1)
        return [pad + "("] + inner + [pad + ")"]

    if kind == "tag":
        body = node[2]
        if is_atom(body) and body[0] != "puretag":
            return [pad + f"{node[1]} {render_atom(body)}"]
        # a tagged body (e.g. $x ()) is a unary_expr, not a primary:
        # grammar requires parentheses around it
        inner = render(body, level + 1)
        return [pad + f"{node[1]} ("] + inner + [pad + ")"]

    if kind in ("sum", "prod"):
        op = "|" if kind == "sum" else "*"
        children = flatten_children(kind, node[1])
        lines = render_operand(children[0], level)
        for child in children[1:]:
            sub = render_operand(child, level + 1)
            lines.append(IND * (level + 1) + op + " " + sub[0].strip())
            lines.extend(sub[1:])
        return lines

    if kind == "exp":
        _, result, args = node
        lines = render_operand(result, level)
        for arg in reversed(args):  # textual order: rightmost arg is fed first
            sub = render_operand(arg, level + 1)
            lines.append(IND * (level + 1) + "<- " + sub[0].strip())
            lines.extend(sub[1:])
        return lines

    if kind == "app":
        _, ctor, args = node
        if all(is_atom(a) for a in args):
            return [pad + f"{ctor}[" + ", ".join(render_atom(a) for a in args) + "]"]
        lines = [pad + f"{ctor}["]
        for i, a in enumerate(args):
            sub = (
                [IND * (level + 1) + render_atom(a)]
                if is_atom(a)
                else render(a, level + 1)
            )
            if i < len(args) - 1:
                sub[-1] += ","
            lines.extend(sub)
        lines.append(pad + "]")
        return lines

    if kind == "tuple":
        children = node[1]
        lines = [pad + "("]
        for i, child in enumerate(children):
            sub = render(child, level + 1)
            if i < len(children) - 1:
                sub[-1] += ","
            lines.extend(sub)
        lines.append(pad + ")")
        return lines

    raise AssertionError(node)


def render_operand(node, level):
    """Render a chain operand: atoms bare, composites parenthesized."""
    pad = IND * level
    if is_atom(node):
        return render(node, level)
    inner = render(node, level + 1)
    return [pad + "("] + inner + [pad + ")"]


def render_definition(gen, theme, depth):
    """Build one (name, params, node) definition rendered as lines."""
    name = gen.name(theme)
    params = gen.rng.sample(GENERIC_PARAMS, gen.rng.randint(0, 2))
    node = gen.expr(depth)
    header = name + ("[" + ", ".join(params) + "]" if params else "") + " :="
    return [header] + render(node, 1)


def gen_assert_definition(gen):
    """The project's signature idiom: metric + Assert code block."""
    n = gen.name("Guarded")
    limit = gen.rng.randint(10, 500)
    metric = gen.rng.choice(["int", "float"])
    tag = gen.rng.choice(["$x", "$lines", "$ratio"])
    return [
        f"{n} :=",
        IND + f"{tag} {metric}",
        IND + f"* Assert[{{{tag[1:]} <= {limit}}}]",
        IND + f"* {gen.rng.choice(TYPEREFS)}",
    ]


def block_max_indent(block):
    return max(
        (len(l) - len(l.lstrip(" "))) for l in block if l.strip()
    )


def gen_import_header(rng):
    """1-3 import lines: `import a.b` or `import a.b as c`."""
    n = rng.randint(1, 3)
    modules = rng.sample(IMPORT_MODULES, n)
    lines = []
    for m in modules:
        if rng.random() < 0.5:
            lines.append(f"import {m} as {m.split('.')[-1]}")
        else:
            lines.append(f"import {m}")
    return lines


def gen_file(rng, index, with_imports=False):
    gen = Gen(rng)
    lines = [
        f"# corpus case {index:03d} — generated by generate_corpus.py (seed {SEED})",
        f"# themes: {', '.join(rng.sample(DEF_THEMES, 5))}",
        "",
    ]
    if with_imports:
        lines.extend(gen_import_header(rng))
        lines.append("")
    themes = [rng.choice(DEF_THEMES) for _ in range(12)]
    rng.shuffle(themes)
    first = True
    for theme in themes:
        depth = rng.randint(2, 4)
        block = render_definition(gen, theme, depth)
        # oversized blocks (deep trees blow up combinatorially): retry
        # shallower rather than silently dropping the theme
        while len(lines) + len(block) + 1 > 112 and depth > 1:
            depth -= 1
            block = render_definition(gen, theme, depth)
        if first:
            # guarantee >= 3 indentation levels somewhere in the file
            tries = 0
            while (
                block_max_indent(block) < 3 * len(IND)
                or len(lines) + len(block) + 1 > 112
            ) and tries < 10:
                depth = 4
                block = render_definition(gen, theme, depth)
                while len(lines) + len(block) + 1 > 112 and depth > 1:
                    depth -= 1
                    block = render_definition(gen, theme, depth)
                tries += 1
            first = False
        if len(lines) + len(block) + 1 > 112:
            break
        lines.extend(block)
        lines.append("")
        if len(lines) >= 96 and rng.random() < 0.5:
            break
    # guarantee one Assert-idiom definition per file
    if len(lines) < 108:
        lines.extend(gen_assert_definition(gen))
        lines.append("")
    # pad with comment filler if still short
    while len(lines) < 92:
        lines.insert(len(lines) - 1, f"# note: {rng.choice(TYPEREFS).lower()} placeholder")
    text = "\n".join(lines).rstrip() + "\n"
    # embedded newlines inside triple-quoted strings also count: drop
    # filler comments, then trailing theme definitions, if the physical
    # line count overshoots 120 (the Guarded assert block always stays)
    while text.count("\n") > 120:
        for j in range(len(lines) - 1, -1, -1):
            if lines[j].startswith("# note:"):
                del lines[j]
                break
        else:
            blocks = [i for i, l in enumerate(lines)
                      if l and not l[0].isspace() and ":=" in l
                      and not l.startswith("Guarded")]
            if not blocks:
                break
            del lines[blocks[-1]:]
            while lines and not lines[-1].strip():
                lines.pop()
        text = "\n".join(lines).rstrip() + "\n"
    return text


# ----------------------------------------------------------------------
# Verify a corpus file: parse, structural stability, fixed point
# ----------------------------------------------------------------------


def check_file(path, ast_mod):
    src = path.read_text()
    tree1 = ast_mod.parse(src)
    canon1 = ast_mod.unparse(tree1)
    tree2 = ast_mod.parse(canon1)
    canon2 = ast_mod.unparse(tree2)
    n_lines = src.count("\n")
    max_indent = max(
        (len(line) - len(line.lstrip(" "))) for line in src.splitlines() if line.strip()
    )
    ok = (
        ast_mod.dump(tree1) == ast_mod.dump(tree2)
        and canon1 == canon2
        and 88 <= n_lines <= 120
        and max_indent >= 3 * len(IND)
    )
    return ok, n_lines, max_indent, ast_mod.dump(tree1) == ast_mod.dump(tree2), canon1 == canon2


def main():
    rng = random.Random(SEED)
    rng_extra = random.Random(SEED + 777)  # extra batch: own stream, so
    # re-runs keep case_101..case_120 byte-identical
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = []
    created = 0
    for i in range(1, N_FILES + 1):
        path = OUT_DIR / f"case_{i:03d}.viba"
        text = gen_file(rng, i)  # stream consumed even when skipping,
        if not path.exists():    # so future appends stay deterministic
            path.write_text(text)
            created += 1
        stats.append(i)
    for i in range(N_FILES + 1, N_FILES + N_EXTRA + 1):
        path = OUT_DIR / f"case_{i:03d}.viba"
        text = gen_file(rng_extra, i)
        if not path.exists():
            path.write_text(text)
            created += 1
        stats.append(i)
    rng_import = random.Random(SEED + 1555)  # import batch: own stream
    for i in range(N_FILES + N_EXTRA + 1, N_FILES + N_EXTRA + N_IMPORT + 1):
        path = OUT_DIR / f"case_{i:03d}.viba"
        text = gen_file(rng_import, i, with_imports=True)
        if not path.exists():
            path.write_text(text)
            created += 1
        stats.append(i)

    if "--check" in sys.argv:
        sys.path.insert(0, str(OUT_DIR.parent.parent))
        from viba import ast as viba_ast

        bad = []
        line_counts = []
        indents = []
        for i in stats:
            path = OUT_DIR / f"case_{i:03d}.viba"
            try:
                ok, n, ind, dump_eq, canon_eq = check_file(path, viba_ast)
            except Exception as e:
                bad.append((i, f"{type(e).__name__}: {e}"))
                continue
            line_counts.append(n)
            indents.append(ind)
            if not ok:
                reasons = []
                if not dump_eq:
                    reasons.append("dump_neq")
                if not canon_eq:
                    reasons.append("canon_neq")
                if not 88 <= n <= 120:
                    reasons.append(f"lines={n}")
                if ind < 3 * len(IND):
                    reasons.append(f"indent={ind}")
                bad.append((i, " ".join(reasons)))
        print(f"checked {len(stats)} files")
        print(
            f"lines: min={min(line_counts)} max={max(line_counts)} avg={sum(line_counts)/len(line_counts):.1f}"
        )
        print(f"max indent: min={min(indents)} max={max(indents)}")
        if bad:
            print(f"FAILED {len(bad)}:")
            for i, msg in bad[:20]:
                print(f"  case_{i:03d}.viba: {msg}")
            sys.exit(1)
        print("all corpus files OK")
    else:
        print(f"created {created} new file(s), {len(stats)} present in {OUT_DIR}")


if __name__ == "__main__":
    main()

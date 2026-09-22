import os
import sys

if __name__ == "__main__" and __package__ is None:
    # `python3 viba/parser.py` puts viba/ (not the checkout root) on
    # sys.path, so `import viba` would fail; add the root first.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ply.lex as lex
import ply.yacc as yacc

from viba.viba_ast.nodes import (
    TypeDefinition,
    GenericDefinition,
    Import,
    Sum,
    Product,
    Exponent,
    Partial,
    Tagged,
    TypeApp,
    Tuple,
    TypeRef,
    Constant,
    Nil,
    Never,
    Any,
    Ellipsis,
    CodeBlock,
    AST,
)

# ================================================================= #
# 1. LEXER DEFINITIONS
# ================================================================= #

tokens = (
    "CLASS_NAME",
    "TAGGED_CLASS_NAME",
    "FLOAT",
    "INT",
    "STRING",
    "SINGLE_STRING",
    "TRIPLE_STRING",
    "BOOLEAN",
    "ASSIGN",  # :=
    "SUM_OP",  # |
    "PROD_OP",  # *
    "EXP_OP",  # <-
    "APPLY_OP",  # <<
    "IMPORT",  # import
    "AS",  # as
    "LBRACKET",  # [
    "RBRACKET",  # ]
    "LPAREN",  # (
    "RPAREN",  # )
    "LBRACE",  # {
    "RBRACE",  # }
    "COMMA",
    "NIL",
    "NEVER",
    "ANY",
    "ELLIPSIS",  # ...
    "CODE_BLOCK",  # { ... }
)

t_ASSIGN = r":="
t_SUM_OP = r"\|"
t_PROD_OP = r"\*"
# `<<` is listed first: PLY takes same-length tokens in definition order.
t_APPLY_OP = r"<<"  
t_EXP_OP = r"<-"
t_LBRACKET = r"\["
t_RBRACKET = r"\]"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_LBRACE = r"\{"
t_RBRACE = r"\}"
t_COMMA = r","
t_ELLIPSIS = r"\.\.\."


# Comment support: ignore everything from # to end of line
def t_COMMENT(t):
    r"\#.*"
    pass  # No return value means token is ignored


def t_BOOLEAN(t):
    r"true\b|false\b"
    t.value = t.value == "true"
    return t


def t_NIL(t):
    r"nil\b|void\b|None\b"
    t.value = "nil"  # canonical spelling; void/None are aliases
    return t


def t_NEVER(t):
    r"never\b"
    return t


def t_IMPORT(t):
    r"import\b"
    return t


def t_AS(t):
    r"as\b"
    return t


def t_FLOAT(t):
    r"(\d+\.\d*|\.\d+)"
    t.value = float(t.value)
    return t


def t_INT(t):
    r"\d+"
    t.value = int(t.value)
    return t


def t_STRING(t):
    r'"([^\\\n]|(\\.))*?"'
    t.value = t.value[1:-1]
    return t


def t_TRIPLE_STRING(t):
    r"'''[\s\S]*?'''"
    # Preserve the content exactly as-is (including newlines and spaces)
    # Just strip the opening and closing triple quotes
    t.value = t.value[3:-3]
    return t


def t_SINGLE_STRING(t):
    r"'([^\\\n]|(\\.))*?'"
    t.value = t.value[1:-1]
    return t


def t_ANY(t):
    r"Any\b"
    return t


def t_TAGGED_CLASS_NAME(t):
    r"\$\w+(\.\w+)*"
    return t


def t_CLASS_NAME(t):
    r"\w+(\.\w+)*"
    return t


def t_CODE_BLOCK(t):
    r'\{'
    depth = 1
    start = t.lexer.lexpos
    lexdata = t.lexer.lexdata
    while depth > 0:
        if start >= len(lexdata):
            print(f"Lexical Error: Unterminated code block at line {t.lexer.lineno}")
            t.lexer.skip(1)
            return
        c = lexdata[start]
        if c == '\n':
            t.lexer.lineno += 1
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        start += 1
    # Value is everything between the outer { and }
    t.value = CodeBlock(lexdata[t.lexer.lexpos : start - 1])
    t.lexer.lexpos = start
    return t


t_ignore = " \t"


def t_newline(t):
    r"\n+"
    t.lexer.lineno += len(t.value)


def t_error(t):
    print(f"Lexical Error: Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
    t.lexer.skip(1)


lexer = lex.lex()

# ================================================================= #
# 2. PARSER DEFINITIONS
# ================================================================= #


def p_program(p):
    """program : statement_list
    | epsilon"""
    p[0] = p[1] if p[1] else []


def p_statement_list(p):
    """statement_list : statement
    | statement statement_list"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]


def p_statement(p):
    """statement : definition
    | import_stmt"""
    p[0] = p[1]


def p_definition(p):
    """definition : type_definition
    | generic_definition"""
    p[0] = p[1]


# The builtin containers are a shape, not names: nothing may define one, and
# no generic may take one as a parameter. The check runs over the parsed
# definitions — raising inside a grammar action would be swallowed by PLY's
# error recovery, and the definition would vanish from the tree instead.
BUILTIN_TYPE_NAMES = {
    "list": "a builtin container",
    "set": "a builtin container",
    "dict": "a builtin container",
    "ListLiteral": "a builtin literal constructor",
    "SetLiteral": "a builtin literal constructor",
    "DictLiteral": "a builtin literal constructor",
}


def _reject_builtin_name(name: str, position: str) -> None:
    if name in BUILTIN_TYPE_NAMES:
        raise SyntaxError(
            f"Viba parse error: {name} is {BUILTIN_TYPE_NAMES[name]}; "
            f"it cannot be {position}")


def check_definition_names(definitions) -> None:
    """Refuse a builtin type name on the left of `:=`."""
    for node in definitions:
        if isinstance(node, TypeDefinition):
            _reject_builtin_name(node.name, "a definition name")
        elif isinstance(node, GenericDefinition):
            _reject_builtin_name(node.name, "a definition name")
            for param in node.generic_params or []:
                _reject_builtin_name(param, "a generic parameter")


def p_partial_expr(p):
    """partial_expr : partial_expr APPLY_OP adt_expr
    | adt_expr"""
    if len(p) == 4:
        p[0] = Partial(p[1], p[3])
    else:
        p[0] = p[1]


def p_type_definition(p):
    """type_definition : CLASS_NAME ASSIGN partial_expr"""
    p[0] = TypeDefinition(p[1], p[3])


def p_generic_definition(p):
    """generic_definition : CLASS_NAME LBRACKET CLASS_NAME type_param_list RBRACKET ASSIGN partial_expr"""
    p[0] = GenericDefinition(p[1], [p[3]] + p[4], p[7])


def p_import_stmt(p):
    """import_stmt : IMPORT CLASS_NAME optional_alias"""
    p[0] = Import(p[2], p[3])


def p_optional_alias(p):
    """optional_alias : AS CLASS_NAME
    | epsilon"""
    p[0] = p[2] if len(p) > 2 else None


def p_type_param_list(p):
    """type_param_list : COMMA CLASS_NAME type_param_list
    | epsilon"""
    if len(p) > 2:
        p[0] = [p[2]] + p[3]
    else:
        p[0] = []


# --- ADT Hierarchy ---


def p_adt_expr(p):
    """adt_expr : adt_expr SUM_OP product_expr
    | product_expr"""
    if len(p) == 4:
        p[0] = Sum(p[1], p[3])
    else:
        p[0] = p[1]


def p_product_expr(p):
    """product_expr : product_expr PROD_OP exponent_expr
    | exponent_expr"""
    if len(p) == 4:
        p[0] = Product(p[1], p[3])
    else:
        p[0] = p[1]


def p_exponent_expr(p):
    """exponent_expr : exponent_expr EXP_OP unary_expr
    | unary_expr"""
    if len(p) == 4:
        p[0] = Exponent(p[1], p[3])
    else:
        p[0] = p[1]


def p_unary_expr(p):
    """unary_expr : TAGGED_CLASS_NAME type_app_expr
    | type_app_expr"""
    if len(p) == 3:
        p[0] = Tagged(p[1], p[2])
    else:
        p[0] = p[1]


def p_type_app_expr(p):
    """type_app_expr : CLASS_NAME optional_type_args
    | primary_expr"""
    if len(p) == 3:
        if p[2] is not None:
            p[0] = TypeApp(p[1], p[2])
        else:
            p[0] = TypeRef(p[1])
    else:
        p[0] = p[1]


def p_optional_type_args(p):
    """optional_type_args : LBRACKET partial_expr adt_arg_list RBRACKET
    | LBRACKET RBRACKET
    | epsilon"""
    if len(p) == 5:
        p[0] = [p[2]] + p[3]
    elif len(p) == 3:
        p[0] = []
    else:
        p[0] = None


def p_adt_arg_list(p):
    """adt_arg_list : COMMA partial_expr adt_arg_list
    | epsilon"""
    if len(p) > 2:
        p[0] = [p[2]] + p[3]
    else:
        p[0] = []


# ================================================================= #
# PARSER RULES: TUPLE & PRIMARY EXPRESSIONS
# ================================================================= #


def p_adt_expr_list(p):
    """adt_expr_list : partial_expr COMMA partial_expr
    | partial_expr COMMA adt_expr_list"""
    # Flattens comma-separated expressions into a Python list
    if len(p) == 4 and not isinstance(p[3], list):
        p[0] = [p[1], p[3]]
    else:
        p[0] = [p[1]] + p[3]


def p_adt_expr_list_empty(p):
    "adt_expr_list :"
    # The empty production exists so () reaches the tuple rule as an
    # empty list: () is the EMPTY TUPLE (its own node, |()| = 1).
    p[0] = []


def p_primary_expr(p):
    """primary_expr : CLASS_NAME
    | literal
    | NIL
    | NEVER
    | ANY
    | ELLIPSIS
    | LPAREN partial_expr RPAREN
    | LPAREN adt_expr_list RPAREN
    | CODE_BLOCK"""
    # 1. Handle atomic units (Length 2)
    if len(p) == 2:
        val = p[1]
        if isinstance(val, AST):
            # Literal (Constant), CODE_BLOCK (CodeBlock): pass through
            p[0] = val
        elif val == "nil":
            p[0] = Nil()
        elif val == "never":
            p[0] = Never()
        elif val == "Any":
            p[0] = Any()
        elif val == "...":
            p[0] = Ellipsis()
        else:
            # Simple type reference
            p[0] = TypeRef(val)

    # 2. Parentheses: empty list () is the EMPTY TUPLE (its own node,
    # |()| = 1); a non-empty list is Tuple; a lone expr is grouping.
    else:
        content = p[2]
        if isinstance(content, list):
            # (A, B, C) is a Tuple: positional product, order matters,
            # and () is its nullary form. Neither is sugar for the
            # tagged product `*` — distinct nodes at every layer.
            p[0] = Tuple(content)
        else:
            # Standard grouping: ( adt_expr )
            p[0] = content


def p_literal(p):
    """literal : FLOAT
    | INT
    | STRING
    | SINGLE_STRING
    | TRIPLE_STRING
    | BOOLEAN"""
    p[0] = Constant(p[1])


def p_epsilon(p):
    "epsilon :"
    p[0] = None


def p_error(p):
    """Raise on a syntax error: callers must be able to tell that this
    source does not compile."""
    if p:
        raise SyntaxError(f"Viba parse error: unexpected {p.value!r} at line {p.lineno}")
    raise SyntaxError("Viba parse error: unexpected EOF")


parser = yacc.yacc()

# ================================================================= #
# 3. TEST RUN
# ================================================================= #

if __name__ == "__main__":
    test_cases = [
        # 1-5: Basic Algebraic Identities & Atomic Types
        ("IdentitySum := A | never", "Sum with identity zero"),
        ("IdentityProd := A * nil", "Product with identity unit"),
        ("UnitOnly := nil", "Pure unit type"),
        ("BottomOnly := never", "Pure bottom type"),
        ("Variadic := A | B | ...", "Open sum type with ellipsis"),
        # 6-10: Literals & Constants
        ("ConfigInt := 42", "Integer literal"),
        ("ConfigFloat := 3.1415", "Float literal"),
        ("ConfigBool := true * false", "Boolean literals in product"),
        ('ConfigStr := "viba_v1" * 1.0', "Mixed string and float"),
        ("ComplexLiteral := 0.5 * nil | never", "Mixed literals and identities"),
        # 10a-10e: String literals the unparser has to spell back out
        ("""QuotedStr := 'say "hi" now'""", "Single-quoted string with a double quote"),
        ('ApostropheStr := "it\'s fine"', "Double-quoted string with an apostrophe"),
        ('UnicodeStr := "中文🙂"', "Unicode string literal"),
        ("MultilineStr := '''one\ntwo'''", "Triple-quoted string spanning lines"),
        ("TrailingSlashStr := '''trail\\'''", "Triple-quoted string ending in a backslash"),
        # 10i-10j: the top type
        ("Top := Any", "The top type"),
        ("TopSum := Any | int * str", "The top inside a sum and a product"),
        # 10f-10h: partial computation, `<<`
        ("Partial := (A <- $b B <- $c C) << $b B",
         "Function with one argument given"),
        ("PartialAll := T << $c C << $b B",
         "Arguments given in the other order"),
        ("PartialNested := M << $b (P * Q)",
         "The given argument is a product"),
        ("PartialGrouped := M << (N << $a P)",
         "A `<<` given as the argument keeps its grouping"),
        # 11-15: Semantic Paths & Tagging
        ("SimpleTag := $target Output", "Basic tagged type"),
        ("NestedPath := $meta.id.hash STRING", "Nested semantic path ($a.b.c)"),
        ("TagChain := $src In * $dst Out", "Multiple tags in product"),
        ("DeepPath := $a.b.c.d.e INT", "Very deep semantic path"),
        (
            "TaggedParens := ($res Result <- $arg Input)",
            "Tagged exponent in parentheses",
        ),
        # 16-20: Exponents & Currying (Higher-order types)
        ("MapType := B <- A", "Simple function/exponent"),
        ("Curried := C <- B <- A", "Nested currying (Left-associative)"),
        (
            "ComplexExponent := (Out | Error) <- In * Config",
            "Product argument to sum result",
        ),
        (
            "CurriedParens := ($ret Ret <- ($p1 A) <- ($p2 B))",
            "Nested parenthesized currying",
        ),
        (
            "AutoEncoder := ($output Out <- $input In <- $intent Intent)",
            "AE style currying",
        ),
        # 21-25: Generics & Combinations
        ("List[T] := T * List[T] | nil", "Recursive generic list"),
        ("Pair[K, V] := K * V", "Multi-parameter generic"),
        ("Option[T] := T | nil", "Standard Option type"),
        ("Result[T, E] := $ok T | $err E", "Tagged result sum type"),
        ("HLSegment[T] := $data T * $next ...", "Generic with variadic tail"),
        # 26: The "Final Boss" case
        (
            'FinalBoss[In, Out] := ($res.val Out | $res.err never) <- $cfg.mode "fast" * In * 0.99',
            "Comprehensive stress test",
        ),
        ("EmptyTuple := ()", "Empty tuple: its own node, |()| = 1"),
        ("MixedNil := A | () | nil", "Mixing empty tuple and nil in sum"),
        ("VoidAlias := void", "void is an alias of nil"),
        ("NoneAlias := None", "None is an alias of nil"),
        ("AliasMix := A * void | None", "void and None alias mix"),
        ("AE_ReturnUnit := () <- Input", "Using () as return type"),
        ("IdentitySum := A | never", "Sum with identity zero"),
        ("IdentityProd := A * nil", "Product with identity unit"),
        ("Variadic := A | B | ...", "Open sum type with ellipsis"),
        ("ConfigInt := 42", "Integer literal"),
        ("ConfigFloat := 3.1415", "Float literal"),
        ("ConfigBool := true * false", "Boolean literals"),
        ('ConfigStr := "viba_v1" * 1.0', "Mixed string and float"),
        ("SimpleTag := $target Output", "Basic tagged type"),
        ("NestedPath := $meta.id.hash STRING", "Nested semantic path"),
        ("TagChain := $src In * $dst Out", "Multiple tags"),
        ("DeepPath := $a.b.c.d.e INT", "Deep semantic path"),
        ("TaggedParens := ($res Result <- $arg Input)", "Tagged exponent"),
        ("MapType := B <- A", "Simple exponent"),
        ("Curried := C <- B <- A", "Nested currying"),
        ("AutoEncoder := ($output Out <- $input In <- $intent Intent)", "AE currying"),
        ("List[T] := T * List[T] | ()", "Recursive list with ()"),
        ("Result[T, E] := $ok T | $err E", "Tagged result"),
        (
            'FinalBoss[In, Out] := ($res.val Out | $res.err never) <- $cfg.mode "fast" * In * 0.99',
            "Comprehensive test",
        ),
        # ====== TRIPLE-QUOTED STRING TESTS (20 cases) ======
        ("TripleSimple := '''a b c'''", "Simple triple-quoted string"),
        ("TripleSingleWord := '''hello'''", "Triple-quoted with single word"),
        ("TripleMultiLine := '''line1\nline2\nline3'''", "Triple-quoted with newlines"),
        ("TripleSpaces := '''  multiple   spaces  '''", "Triple-quoted with spaces"),
        ("TripleProduct := '''text1''' * '''text2'''", "Triple-quoted strings in product"),
        ("TripleSum := '''option1''' | '''option2'''", "Triple-quoted strings in sum"),
        ("TripleExponent := '''result''' <- '''input'''", "Triple-quoted in exponent"),
        ("TripleWithTags := $tag '''value'''", "Triple-quoted with semantic tag"),
        ("TripleTuple := ('''first''', '''second''')", "Triple-quoted in tuple"),
        ("TripleNested := ('''a''' * '''b''') | '''c'''", "Nested triple-quoted expressions"),
        ("TripleGeneric := List['''item''']", "Triple-quoted as generic argument"),
        ("TripleWithNil := '''data''' * nil", "Triple-quoted with nil identity"),
        ("TripleWithNever := '''text''' | never", "Triple-quoted with never identity"),
        ("TripleEllipsis := '''base''' | ...", "Triple-quoted with ellipsis"),
        ("TripleEllipsisTail := '''head''' * ...", "Triple-quoted product with ellipsis"),
        ("TripleComplex := ($res '''OK''' | $err '''Error''') <- '''input'''", "Complex triple-quoted expression"),
        ("TripleCurried := '''C''' <- '''B''' <- '''A'''", "Triple-quoted currying"),
        ("TripleRecursive := '''item''' * TripleRecursive | nil", "Recursive with triple-quoted"),
        ("TripleVariadic := '''a''' | '''b''' | '''c''' | ...", "Multiple triple-quoted sum with ellipsis"),
        ("TripleFinal := ('''x''' * '''y''', '''z''')", "Triple-quoted in nested tuple"),
        # ====== SINGLE-QUOTED STRING TESTS (20 cases) ======
        ("SingleSimple := 'a b c'", "Simple single-quoted string"),
        ("SingleWord := 'hello'", "Single word in single quotes"),
        ("SingleSpaces := '  multiple   spaces  '", "Single-quoted with spaces"),
        ("SingleProduct := 'text1' * 'text2'", "Single-quoted strings in product"),
        ("SingleSum := 'option1' | 'option2'", "Single-quoted strings in sum"),
        ("SingleExponent := 'result' <- 'input'", "Single-quoted in exponent"),
        ("SingleWithTags := $tag 'value'", "Single-quoted with semantic tag"),
        ("SingleTuple := ('first', 'second')", "Single-quoted in tuple"),
        ("SingleNested := ('a' * 'b') | 'c'", "Nested single-quoted expressions"),
        ("SingleGeneric := List['item']", "Single-quoted as generic argument"),
        ("SingleWithNil := 'data' * nil", "Single-quoted with nil identity"),
        ("SingleWithNever := 'text' | never", "Single-quoted with never identity"),
        ("SingleEllipsis := 'base' | ...", "Single-quoted with ellipsis"),
        ("SingleEllipsisTail := 'head' * ...", "Single-quoted product with ellipsis"),
        ("SingleComplex := ($res 'OK' | $err 'Error') <- 'input'", "Complex single-quoted expression"),
        ("SingleCurried := 'C' <- 'B' <- 'A'", "Single-quoted currying"),
        ("SingleRecursive := 'item' * SingleRecursive | nil", "Recursive with single-quoted"),
        ("SingleVariadic := 'a' | 'b' | 'c' | ...", "Multiple single-quoted sum with ellipsis"),
        ("SingleFinal := ('x' * 'y', 'z')", "Single-quoted in nested tuple"),
        # ====== MIXED STRING TYPE TESTS (10 cases) ======
        ("MixedSingleDouble := 'single' * \"double\"", "Single and double quotes together"),
        ("MixedDoubleSingle := \"double\" | 'single'", "Double and single quotes together"),
        ("MixedTripleSingle := '''triple''' * 'single'", "Triple and single quotes together"),
        ("MixedAll := 'one' * \"two\" | '''three'''", "All three string types together"),
        ("MixedWithTypes := 'text' * 42 * 3.14 * \"str\"", "Mixed with number literals"),
        ("MixedInTuple := ('first', \"second\", 'third')", "Mixed quotes in tuple"),
        ("MixedInSum := 'a' | \"b\" | 'c'", "Mixed quotes in sum"),
        ("MixedInExponent := 'result' <- (\"input1\" * 'input2')", "Mixed quotes in exponent"),
        ("MixedTagged := $tag 'val' * $other \"val2\"", "Mixed quotes with tags"),
        ("MixedComplex := ('''A''' | 'B') <- (\"x\" * 'y' * '''z''')", "Complex mixed quotes"),
        # ====== CODE BLOCK TESTS ======
        ("CodeBlockBasic := {hello}", "Simple code block"),
        ("CodeBlockWithSpaces := {  some text  }", "Code block with spaces"),
        ("CodeBlockMultiLine := {line1\nline2\nline3}", "Code block with newlines"),
        ("CodeBlockNested := {outer {inner} end}", "Nested code block"),
        ("CodeBlockDeepNested := {a {b {c {d} c} b} a}", "Deeply nested code block"),
        ("CodeBlockInProduct := {fn} * {config}", "Code blocks in product"),
        ("CodeBlockInSum := {opt1} | {opt2}", "Code blocks in sum"),
        ("CodeBlockInExponent := {result} <- {input}", "Code blocks in exponent"),
        ("CodeBlockTagged := $tag {value}", "Tagged code block"),
        ("CodeBlockMixed := {code} * nil | never", "Code block with identities"),
        ("CodeBlockWithTuple := ({a}, {b})", "Code blocks in tuple"),
        ("CodeBlockGeneric := List[{item}]", "Code block as generic arg"),
        ("EmptyApp := ListLiteral[]", "Application of no arguments"),
        ("EmptyAppInSum := ListLiteral[] | nil", "Empty application in a sum"),
        ("CodeBlockComplex := ({res {OK} | {err}} <- {inp})", "Complex code block expression"),
        # ====== IMPORT TESTS ======
        ("import numpy", "Plain import"),
        ("import fx.graph", "Dotted module import"),
        ("import torch as t", "Aliased import"),
        ("import a.b.c as abc", "Dotted aliased import"),
        (
            "import numpy\nimport torch as t\nTensor := t.Tensor * numpy.ndarray",
            "Imports before definitions",
        ),
    ]

    print(f"{'TEST CASE':<50} | {'STATUS'}")
    print("-" * 65)

    success_count = 0
    for code, desc in test_cases:
        try:
            parser.parse(code)
            print(f"{desc:<50} | SUCCESS")
            success_count += 1
        except Exception as e:
            print(f"{desc:<50} | FAILED: {e}")

    print("-" * 65)
    print(f"Passed {success_count}/{len(test_cases)} tests.")

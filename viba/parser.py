import ply.lex as lex
import ply.yacc as yacc
import json

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
    "LBRACKET",  # [
    "RBRACKET",  # ]
    "LPAREN",  # (
    "RPAREN",  # )
    "COMMA",
    "VOID",
    "NEVER",
    "ELLIPSIS",  # ...
)

t_ASSIGN = r":="
t_SUM_OP = r"\|"
t_PROD_OP = r"\*"
t_EXP_OP = r"<-"
t_LBRACKET = r"\["
t_RBRACKET = r"\]"
t_LPAREN = r"\("
t_RPAREN = r"\)"
t_COMMA = r","
t_ELLIPSIS = r"\.\.\."


# Comment support: ignore everything from # to end of line
def t_COMMENT(t):
    r"\#.*"
    pass  # No return value means token is ignored


def t_BOOLEAN(t):
    r"true|false"
    t.value = t.value == "true"
    return t


def t_VOID(t):
    r"void"
    return t


def t_NEVER(t):
    r"never"
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
    r'"([^\\\n]|(\\.))*?" '
    t.value = t.value[1:-1]
    return t


def t_TRIPLE_STRING(t):
    r"'''[\s\S]*?'''"
    # Preserve the content exactly as-is (including newlines and spaces)
    # Just strip the opening and closing triple quotes
    t.value = t.value[3:-3]
    return t


def t_SINGLE_STRING(t):
    r"'([^\\\n]|(\\.))*?' "
    t.value = t.value[1:-1]
    return t


def t_TAGGED_CLASS_NAME(t):
    r"\$\w+(\.\w+)*"
    return t


def t_CLASS_NAME(t):
    r"\w+(\.\w+)*"
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


# Fixed: Removed the Literal token ":=" and used ASSIGN token instead
def p_statement(p):
    """statement : CLASS_NAME optional_type_params ASSIGN adt_expr"""
    p[0] = {"node": "Definition", "name": p[1], "generic_params": p[2], "body": p[4]}


def p_optional_type_params(p):
    """optional_type_params : LBRACKET CLASS_NAME type_param_list RBRACKET
    | epsilon"""
    if len(p) > 2:
        p[0] = [p[2]] + p[3]
    else:
        p[0] = []


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
        p[0] = {"node": "SumType", "left": p[1], "right": p[3]}
    else:
        p[0] = p[1]


def p_product_expr(p):
    """product_expr : product_expr PROD_OP exponent_expr
    | exponent_expr"""
    if len(p) == 4:
        p[0] = {"node": "ProductType", "left": p[1], "right": p[3]}
    else:
        p[0] = p[1]


def p_exponent_expr(p):
    """exponent_expr : exponent_expr EXP_OP unary_expr
    | unary_expr"""
    if len(p) == 4:
        p[0] = {"node": "ExponentType", "result": p[1], "argument": p[3]}
    else:
        p[0] = p[1]


def p_unary_expr(p):
    """unary_expr : TAGGED_CLASS_NAME type_app_expr
    | type_app_expr"""
    if len(p) == 3:
        p[0] = {"node": "TaggedType", "tag": p[1], "type": p[2]}
    else:
        p[0] = p[1]


def p_type_app_expr(p):
    """type_app_expr : CLASS_NAME optional_type_args
    | primary_expr"""
    if len(p) == 3:
        if p[2] is not None:
            p[0] = {"node": "TypeApp", "constructor": p[1], "args": p[2]}
        else:
            p[0] = {"node": "TypeRef", "name": p[1]}
    else:
        p[0] = p[1]


def p_optional_type_args(p):
    """optional_type_args : LBRACKET adt_expr adt_arg_list RBRACKET
    | epsilon"""
    if len(p) > 2:
        p[0] = [p[2]] + p[3]
    else:
        p[0] = None


def p_adt_arg_list(p):
    """adt_arg_list : COMMA adt_expr adt_arg_list
    | epsilon"""
    if len(p) > 2:
        p[0] = [p[2]] + p[3]
    else:
        p[0] = []


# ================================================================= #
# PARSER RULES: TUPLE & PRIMARY EXPRESSIONS
# ================================================================= #


def p_adt_expr_list(p):
    """adt_expr_list : adt_expr COMMA adt_expr
    | adt_expr COMMA adt_expr_list"""
    # Flattens comma-separated expressions into a Python list
    if len(p) == 4 and not isinstance(p[3], list):
        p[0] = [p[1], p[3]]
    else:
        p[0] = [p[1]] + p[3]


def p_primary_expr(p):
    """primary_expr : CLASS_NAME
    | TAGGED_CLASS_NAME
    | literal
    | VOID
    | NEVER
    | ELLIPSIS
    | LPAREN adt_expr RPAREN
    | LPAREN adt_expr_list RPAREN
    | LPAREN RPAREN"""
    # 1. Handle atomic units (Length 2)
    if len(p) == 2:
        val = p[1]
        if isinstance(val, dict):
            p[0] = val
        elif val == "void":
            p[0] = {"node": "Identity", "type": "ProductIdentity"}
        elif val == "never":
            p[0] = {"node": "Identity", "type": "SumIdentity"}
        elif val == "...":
            p[0] = {"node": "Ellipsis"}
        elif str(val).startswith("$"):
            # Semantic tag as a standalone atom
            p[0] = {"node": "PureTag", "name": val, "path": val[1:].split(".")}
        else:
            # Simple type reference
            p[0] = {"node": "TypeRef", "name": val}

    # 2. Handle empty tuple () as alias for void (Length 3)
    elif len(p) == 3:
        p[0] = {"node": "Identity", "type": "ProductIdentity", "alias": "()"}

    # 3. Handle Parentheses or Tuples (Length 4)
    else:
        content = p[2]
        if isinstance(content, list):
            # Transform (A, B, C) into a nested ProductType tree
            # This ensures (A, B) is semantically identical to A * B
            def fold_to_product(lst):
                if len(lst) == 2:
                    return {"node": "ProductType", "left": lst[0], "right": lst[1]}
                return {
                    "node": "ProductType",
                    "left": lst[0],
                    "right": fold_to_product(lst[1:]),
                }

            p[0] = fold_to_product(content)
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
    p[0] = {"node": "Literal", "val": p[1], "val_type": type(p[1]).__name__}


def p_epsilon(p):
    "epsilon :"
    p[0] = None


def p_error(p):
    if p:
        print(f"Viba Parse Error: Unexpected '{p.value}' at line {p.lineno}")
    else:
        print("Viba Parse Error: Unexpected EOF")


parser = yacc.yacc()

# ================================================================= #
# 3. TEST RUN
# ================================================================= #

if __name__ == "__main__":
    test_cases = [
        # 1-5: Basic Algebraic Identities & Atomic Types
        ("IdentitySum := A | never", "Sum with identity zero"),
        ("IdentityProd := A * void", "Product with identity unit"),
        ("UnitOnly := void", "Pure unit type"),
        ("BottomOnly := never", "Pure bottom type"),
        ("Variadic := A | B | ...", "Open sum type with ellipsis"),
        # 6-10: Literals & Constants
        ("ConfigInt := 42", "Integer literal"),
        ("ConfigFloat := 3.1415", "Float literal"),
        ("ConfigBool := true * false", "Boolean literals in product"),
        ('ConfigStr := "viba_v1" * 1.0', "Mixed string and float"),
        ("ComplexLiteral := 0.5 * void | never", "Mixed literals and identities"),
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
        ("List[T] := T * List[T] | void", "Recursive generic list"),
        ("Pair[K, V] := K * V", "Multi-parameter generic"),
        ("Option[T] := T | void", "Standard Option type"),
        ("Result[T, E] := $ok T | $err E", "Tagged result sum type"),
        ("HLSegment[T] := $data T * $next ...", "Generic with variadic tail"),
        # 26: The "Final Boss" case
        (
            'FinalBoss[In, Out] := ($res.val Out | $res.err never) <- $cfg.mode "fast" * In * 0.99',
            "Comprehensive stress test",
        ),
        ("VoidAlias := ()", "Using () as primary"),
        ("MixedVoid := A | () | void", "Mixing () and void in sum"),
        ("AE_ReturnUnit := () <- Input", "Using () as return type"),
        ("IdentitySum := A | never", "Sum with identity zero"),
        ("IdentityProd := A * void", "Product with identity unit"),
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
        ("Tuple_Basic := (A, B)", "Binary tuple as product"),
        ("Tuple_Ternary := (A, B, C)", "Ternary tuple nested product"),
        ("Tuple_Tagged := ($input In, $config Conf)", "Tagged members in tuple"),
        ("Tuple_Complex := (Result <- Op, $meta Meta)", "Mixed expressions in tuple"),
        ("Nested_Parens := ((A | B), C)", "Nested grouping inside tuple"),
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
        ("TripleWithVoid := '''data''' * void", "Triple-quoted with void identity"),
        ("TripleWithNever := '''text''' | never", "Triple-quoted with never identity"),
        ("TripleEllipsis := '''base''' | ...", "Triple-quoted with ellipsis"),
        ("TripleEllipsisTail := '''head''' * ...", "Triple-quoted product with ellipsis"),
        ("TripleComplex := ($res '''OK''' | $err '''Error''') <- '''input'''", "Complex triple-quoted expression"),
        ("TripleCurried := '''C''' <- '''B''' <- '''A'''", "Triple-quoted currying"),
        ("TripleRecursive := '''item''' * TripleRecursive | void", "Recursive with triple-quoted"),
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
        ("SingleWithVoid := 'data' * void", "Single-quoted with void identity"),
        ("SingleWithNever := 'text' | never", "Single-quoted with never identity"),
        ("SingleEllipsis := 'base' | ...", "Single-quoted with ellipsis"),
        ("SingleEllipsisTail := 'head' * ...", "Single-quoted product with ellipsis"),
        ("SingleComplex := ($res 'OK' | $err 'Error') <- 'input'", "Complex single-quoted expression"),
        ("SingleCurried := 'C' <- 'B' <- 'A'", "Single-quoted currying"),
        ("SingleRecursive := 'item' * SingleRecursive | void", "Recursive with single-quoted"),
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

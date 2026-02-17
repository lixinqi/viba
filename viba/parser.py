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
    r"\"([^\\\n]|(\\.))*?\" "
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


def p_primary_expr(p):
    """primary_expr : CLASS_NAME
    | TAGGED_CLASS_NAME
    | literal
    | VOID
    | NEVER
    | ELLIPSIS
    | LPAREN adt_expr RPAREN
    | LPAREN RPAREN"""
    if len(p) == 2:
        if isinstance(p[1], dict):
            p[0] = p[1]
        elif p[1] == "void":
            p[0] = {"node": "Identity", "type": "ProductIdentity"}
        elif p[1] == "never":
            p[0] = {"node": "Identity", "type": "SumIdentity"}
        elif p[1] == "...":
            p[0] = {"node": "Ellipsis"}
        elif str(p[1]).startswith("$"):
            p[0] = {"node": "PureTag", "name": p[1], "path": p[1][1:].split(".")}
        else:
            p[0] = {"node": "TypeRef", "name": p[1]}
    elif len(p) == 3:
        # Handles () as an alias for void
        p[0] = {"node": "Identity", "type": "ProductIdentity", "alias": "()"}
    else:
        # Handles ( adt_expr )
        p[0] = p[2]


def p_literal(p):
    """literal : FLOAT
    | INT
    | STRING
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

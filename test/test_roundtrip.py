"""
VIBA Round-Trip Test Suite

Tests the unparser by verifying:
parser.parse(X) == parser.parse(unparser.unparse(parser.parse(X)))

50 test cases ranging from simple to complex.
"""

from viba.parser import parser
from viba.type import parse as type_parse
from viba.unparser import unparse
from viba.chain import convert_to_chain_style


def normalize_to_chain(ast):
    """Normalize AST to chain style for comparison."""
    if isinstance(ast, list):
        return [convert_to_chain_style(type_parse([item])[0]).to_dict() for item in ast]
    elif isinstance(ast, dict):
        # Already in dict form, convert to Type, then to chain, then back to dict
        type_obj = type_parse([ast])[0]
        return convert_to_chain_style(type_obj).to_dict()
    return ast


def ast_equal(ast1, ast2):
    """Recursively compare two AST structures (normalized to chain style)."""
    # Normalize both to chain style first
    ast1 = normalize_to_chain(ast1)
    ast2 = normalize_to_chain(ast2)

    if type(ast1) != type(ast2):
        return False

    if isinstance(ast1, dict):
        if set(ast1.keys()) != set(ast2.keys()):
            return False
        return all(ast_equal(ast1[k], ast2[k]) for k in ast1.keys())
    elif isinstance(ast1, list):
        if len(ast1) != len(ast2):
            return False
        return all(ast_equal(a, b) for a, b in zip(ast1, ast2))
    elif isinstance(ast1, (str, int, float, bool)):
        return ast1 == ast2
    elif ast1 is None:
        return ast2 is None
    else:
        # For objects, compare their to_dict() representation
        if hasattr(ast1, 'to_dict') and hasattr(ast2, 'to_dict'):
            return ast_equal(ast1.to_dict(), ast2.to_dict())
        return str(ast1) == str(ast2)


# ==============================================================================
# 100 VIBA Test Cases - Simple to Complex
# ==============================================================================

TEST_CASES = [
    # ========================================================================
    # 1-5: Basic Types and Identities
    # ========================================================================
    (
        "Simple := A",
        "1. Simple type reference"
    ),
    (
        "VoidType := void",
        "2. Void identity (product unit)"
    ),
    (
        "NeverType := never",
        "3. Never identity (sum zero)"
    ),
    (
        "VoidAlias := ()",
        "4. Empty tuple as void alias"
    ),
    (
        "EllipsisType := ...",
        "5. Open type ellipsis"
    ),

    # ========================================================================
    # 6-10: Basic Sum Types
    # ========================================================================
    (
        "BinarySum := A | B",
        "6. Binary sum type"
    ),
    (
        "TernarySum := A | B | C",
        "7. Ternary sum type"
    ),
    (
        "SumWithVoid := A | void",
        "8. Sum with void"
    ),
    (
        "SumWithNever := A | never",
        "9. Sum with never (identity)"
    ),
    (
        "LongSum := A | B | C | D | E",
        "10. Long sum chain"
    ),

    # ========================================================================
    # 11-15: Basic Product Types
    # ========================================================================
    (
        "BinaryProduct := A * B",
        "11. Binary product type"
    ),
    (
        "TernaryProduct := A * B * C",
        "12. Ternary product type"
    ),
    (
        "ProductWithVoid := A * void",
        "13. Product with void (identity)"
    ),
    (
        "ProductWithNever := A * never",
        "14. Product with never"
    ),
    (
        "LongProduct := A * B * C * D * E",
        "15. Long product chain"
    ),

    # ========================================================================
    # 16-20: Basic Exponent Types
    # ========================================================================
    (
        "SimpleFunc := B <- A",
        "16. Simple function type"
    ),
    (
        "CurriedFunc := C <- B <- A",
        "17. Curried function type"
    ),
    (
        "NestedFunc := (D | E) <- (A * B)",
        "18. Nested complex types in function"
    ),
    (
        "FuncWithVoid := void <- A",
        "19. Function returning void"
    ),
    (
        "FuncFromVoid := A <- void",
        "20. Function from void"
    ),

    # ========================================================================
    # 21-25: Tuples
    # ========================================================================
    (
        "Tuple := (A, B)",
        "21. Binary tuple"
    ),
    (
        "Triple := (A, B, C)",
        "22. Ternary tuple"
    ),
    (
        "Quad := (A, B, C, D)",
        "23. Quaternary tuple"
    ),
    (
        "TupleInSum := (A, B) | (C, D)",
        "24. Tuples in sum"
    ),
    (
        "TupleInProduct := (A, B) * (C, D)",
        "25. Tuples in product"
    ),

    # ========================================================================
    # 26-30: Tagged Types
    # ========================================================================
    (
        "SimpleTag := $ok A",
        "26. Simple tagged type"
    ),
    (
        "MultiTag := $ok A | $err B",
        "27. Multiple tags in sum"
    ),
    (
        "NestedTag := $meta.data.value C",
        "28. Nested tag path"
    ),
    (
        "TagInProduct := $input A * $output B",
        "29. Tags in product"
    ),
    (
        "TagInExponent := $result C <- $argument A",
        "30. Tags in exponent"
    ),

    # ========================================================================
    # 31-35: Type Applications (Generics)
    # ========================================================================
    (
        "SimpleApp := List[A]",
        "31. Simple type application"
    ),
    (
        "MultiApp := Map[A, B]",
        "32. Multiple type parameters"
    ),
    (
        "NestedApp := List[Map[A, B]]",
        "33. Nested type applications"
    ),
    (
        "AppInSum := List[A] | Set[B]",
        "34. Type applications in sum"
    ),
    (
        "AppInProduct := List[A] * Set[B]",
        "35. Type applications in product"
    ),

    # ========================================================================
    # 36-40: Literals
    # ========================================================================
    (
        "IntLit := 42",
        "36. Integer literal"
    ),
    (
        "FloatLit := 3.14159",
        "37. Float literal"
    ),
    (
        "StrLit := \"hello\"",
        "38. String literal"
    ),
    (
        "BoolLit := true",
        "39. Boolean literal (true)"
    ),
    (
        "BoolLit2 := false",
        "40. Boolean literal (false)"
    ),

    # ========================================================================
    # 41-45: Complex Combinations
    # ========================================================================
    (
        "Option := $some T | void",
        "41. Option type pattern"
    ),
    (
        "Result := $ok T | $err E",
        "42. Result type pattern"
    ),
    (
        "Either := $left A | $right B",
        "43. Either type pattern"
    ),
    (
        "Maybe := A | void",
        "44. Maybe type"
    ),
    (
        "Pair := A * B",
        "45. Pair type"
    ),

    # ========================================================================
    # 46-50: Final Boss - Very Complex
    # ========================================================================
    (
        "Complex1 := ($ok (A * B) | $err C) <- (D * E)",
        "46. Complex nested with tags and mixed ops"
    ),
    (
        "Complex2 := List[A] | Map[B, C] | Set[D]",
        "47. Multiple generic types in sum"
    ),
    (
        "Complex3 := ((A | B) * (C | D)) | ((E | F) * (G | H))",
        "48. Deeply nested sum and product"
    ),
    (
        "Complex4 := $res (List[A] | never) <- $req (B * void)",
        "49. Complex with tags, generics, and void"
    ),
    (
        "Complex5 := ($ok (A * (B | C)) | $err D) <- ($in E * $out F * 0.5)",
        "50. Ultimate complex test"
    ),

    # ========================================================================
    # 51-100: MORE COMPLICATED CASES
    # ========================================================================

    # 51-55: Deep nesting with all operators
    (
        "Deep1 := ((((A | B) * C) | D) * E)",
        "51. Deep nested mix of sum and product"
    ),
    (
        "Deep2 := (((A <- B) <- C) <- D)",
        "52. Deep nested exponent (left-associative currying)"
    ),
    (
        "Deep3 := ((A * (B | (C * D))) | (E * F))",
        "53. Complex alternation of operators"
    ),
    (
        "Deep4 := ((A | B) <- (C * D)) | ((E | F) <- (G * H))",
        "54. Nested exponents in sum"
    ),
    (
        "Deep5 := (((A | B) | C) * ((D | E) | F))",
        "55. Nested sums in product"
    ),

    # 56-60: Operator precedence edge cases
    (
        "Prec1 := A | B * C",
        "56. Product binds tighter than sum (need parens for B*C)"
    ),
    (
        "Prec2 := (A | B) * (C | D)",
        "57. Sums need parens in product context"
    ),
    (
        "Prec3 := A * B | C",
        "58. Product before sum without parens"
    ),
    (
        "Prec4 := A <- B | C",
        "59. Exponent with sum in argument"
    ),
    (
        "Prec5 := A <- B * C",
        "60. Exponent with product in argument"
    ),

    # 61-65: Complex generic scenarios
    (
        "Gen1 := List[Option[A]]",
        "61. Generic with nested generic"
    ),
    (
        "Gen2 := Map[Option[A], Result[B, E]]",
        "62. Multiple nested generics with sum types"
    ),
    (
        "Gen3 := List[Either[A, B]] * Map[C, D]",
        "63. Generics in product"
    ),
    (
        "Gen4 := ($ok List[A] | $err List[B]) <- Option[C]",
        "64. Generics in tagged types and exponent"
    ),
    (
        "Gen5 := List[Map[A, B]] | Map[List[A], List[B]]",
        "65. Deeply nested generic combinations"
    ),

    # 66-70: Deep tag paths
    (
        "Tag1 := $a.b.c.d.e.f A",
        "66. Very deep tag path (6 levels)"
    ),
    (
        "Tag2 := $meta.config.service.data.type B",
        "67. Realistic nested tag path"
    ),
    (
        "Tag3 := $ok ($some A) | $err ($none void)",
        "68. Tags with tagged inner types"
    ),
    (
        "Tag4 := ($src.input A * $dst.output B) | $error never",
        "69. Tags with product in sum"
    ),
    (
        "Tag5 := $result ($val A * $err E) <- $input I",
        "70. Tags with product and exponent"
    ),

    # 71-75: Multiple identity combinations
    (
        "Id1 := A | never | never",
        "71. Multiple never (should collapse to one)"
    ),
    (
        "Id2 := A * void * void",
        "72. Multiple void (should collapse to one)"
    ),
    (
        "Id3 := (A | never) * (B | never)",
        "73. Never in each sum within product"
    ),
    (
        "Id4 := (A * void) | (B * void)",
        "74. Void in each product within sum"
    ),
    (
        "Id5 := void | never",
        "75. Void and never together"
    ),

    # 76-80: Complex literal combinations
    (
        "Lit1 := 1 | 2 | 3",
        "76. Multiple integer literals in sum"
    ),
    (
        "Lit2 := 1.5 * 2.5 * 3.5",
        "77. Multiple float literals in product"
    ),
    (
        "Lit3 := \"a\" | \"b\" | \"c\"",
        "78. Multiple string literals in sum"
    ),
    (
        "Lit4 := true * false * true",
        "79. Multiple boolean literals in product"
    ),
    (
        "Lit5 := 42 * \"test\" * 3.14 * true",
        "80. Mixed literal types in product"
    ),

    # 81-85: Tuple edge cases
    (
        "Tup1 := ((A, B), (C, D))",
        "81. Nested tuples"
    ),
    (
        "Tup2 := (A, (B, C), D)",
        "82. Mixed nesting in tuple"
    ),
    (
        "Tup3 := (A | B) | (C | D)",
        "83. Tuples that look like sums (actually sums)"
    ),
    (
        "Tup4 := ((A, B) | C) * (D | (E, F))",
        "84. Mixed tuples and operators"
    ),
    (
        "Tup5 := (((A, B, C), D), E)",
        "85. Deeply nested tuples"
    ),

    # 86-90: Complex exponent chains
    (
        "Exp1 := (((D <- C) <- B) <- A)",
        "86. Left-associative exponent chain explicit parens"
    ),
    (
        "Exp2 := ((A <- B) <- (C | D))",
        "87. Exponent with sum in chain"
    ),
    (
        "Exp3 := ($res R <- $arg1 A) <- $arg2 B",
        "88. Tagged exponents in chain"
    ),
    (
        "Exp4 := (List[A] <- Option[B]) <- Map[C, D]",
        "89. Generics in exponent chain"
    ),
    (
        "Exp5 := ((A | B) <- (C * D)) <- (E | F)",
        "90. Complex types in exponent chain"
    ),

    # 91-95: Extreme nesting
    (
        "Ext1 := ((((A | B) * (C | D)) | (E * F)) * (G | H))",
        "91. Extreme alternation 4 levels deep"
    ),
    (
        "Ext2 := ((($ok A) | ($err B)) * ($ok C | $err D))",
        "92. Tagged types in deep nesting"
    ),
    (
        "Ext3 := ((A <- (B <- (C <- D))) | (E <- F))",
        "93. Exponent in exponent in sum"
    ),
    (
        "Ext4 := (((A * B) * C) * D) | (((E * F) * G) * H)",
        "94. Deep products in sum"
    ),
    (
        "Ext5 := ((A | (B | (C | D))) * (E | (F | G)))",
        "95. Deep sums in product"
    ),

    # 96-100: The ultimate stress tests
    (
        "Ult1 := (($ok ((A * B) | (C * D))) | ($err ((E * F) | (G * H)))) <- ($in (I | J) * $out (K | L))",
        "96. Ultimate test with all operators"
    ),
    (
        "Ult2 := List[Result[Option[A], E]] | Map[Option[B], Result[C, D]]",
        "97. Deep generic nesting with sum types"
    ),
    (
        "Ult3 := ($res.value (List[A] | Map[B, C]) | $res.error never) <- $req ((D * E) | void)",
        "98. Complex with nested tag paths and mixed ops"
    ),
    (
        "Ult4 := (((A | B) * (C | D)) | ((E | F) * (G | H))) <- (((I | J) * (K | L)) | ((M | N) * (O | P)))",
        "99. Massive nested structure"
    ),
    (
        "Ult5 := ($a ($b ($c (A | B) * (C | D)) | $e E) * $f (G | H)) <- $g ((I * J) | (K * L) | (M * N))",
        "100. The final boss - maximum complexity"
    ),

    # ========================================================================
    # 101-200: EVEN MORE COMPLICATED CASES
    # ========================================================================

    # 101-105: Extreme generic nesting
    (
        "EGen1 := Map[List[Option[A]], Result[Map[B, C], D]]",
        "101. Triple nested generics with Map"
    ),
    (
        "EGen2 := List[Map[A, List[Map[B, C]]]]",
        "102. Recursive generic nesting"
    ),
    (
        "EGen3 := ($ok List[Option[A]] | $err Map[B, E]) <- Option[Result[C, D]]",
        "103. Generics in tagged sum as exponent"
    ),
    (
        "EGen4 := Map[(A | B), (C * D)]",
        "104. Sum and product as generic args"
    ),
    (
        "EGen5 := List[($a A | $b B) * ($c C | $d D)]",
        "105. Tagged sum product in generic"
    ),

    # 106-110: Mixed literals with complex types
    (
        "MLit1 := 0 * A * 1 * B * 2",
        "106. Literals interleaved with types in product"
    ),
    (
        "MLit2 := (A * 42) | (B * 3.14) | (C * \"test\")",
        "107. Products with literals in sum"
    ),
    (
        "MLit3 := $config 42 * $name \"viba\" * $version 1.0",
        "108. Tagged literals in product"
    ),
    (
        "MLit4 := (true | false) <- (1 | 0)",
        "109. Literal sums in exponent"
    ),
    (
        "MLit5 := List[42] | Map[\"key\", true]",
        "110. Literals in generic args"
    ),

    # 111-115: Complex void/never scenarios
    (
        "VoidNev1 := (A | void) | (B | void)",
        "111. Multiple void in nested sums"
    ),
    (
        "VoidNev2 := (A * never) | (B * never)",
        "112. Multiple never in nested products"
    ),
    (
        "VoidNev3 := (void | A) * (never | B)",
        "113. Void and never in mixed positions"
    ),
    (
        "VoidNev4 := ((A | never) * (B | void)) | ((C * void) | D)",
        "114. Complex identity combinations"
    ),
    (
        "VoidNev5 := void <- never <- A",
        "115. Void and never in exponent chain"
    ),

    # 116-120: Tuple edge cases with operators
    (
        "TupOp1 := (A, B) | (C, D) | (E, F)",
        "116. Multiple tuples in sum"
    ),
    (
        "TupOp2 := (A, B) * (C, D) * (E, F)",
        "117. Multiple tuples in product"
    ),
    (
        "TupOp3 := ((A, B, C), (D, E, F))",
        "118. Nested multi-element tuples"
    ),
    (
        "TupOp4 := (A | B, C | D)",
        "119. Sums inside tuple elements"
    ),
    (
        "TupOp5 := (A <- B, C <- D)",
        "120. Exponents inside tuple elements"
    ),

    # 121-125: Deep tag path combinations
    (
        "DeepTag1 := $a.b.c (A | B) | $d.e.f (C | D)",
        "121. Deep tag paths with sums"
    ),
    (
        "DeepTag2 := $x.y ($p.q A * $r.s B)",
        "122. Deep tag paths with products"
    ),
    (
        "DeepTag3 := $a.b ($c.d ($e.f A))",
        "123. Triple nested deep tag paths"
    ),
    (
        "DeepTag4 := $meta.config.server.input.data A | $meta.config.server.output.data B",
        "124. Realistic deep paths in sum"
    ),
    (
        "DeepTag5 := $a ($b.c ($d.e.f A * B) | $g.h C)",
        "125. Deep paths with nesting"
    ),

    # 126-130: Exponent with complex operands
    (
        "ExpCmplx1 := ((A | B) <- (C | D)) <- (E | F)",
        "126. Sum in both exponent result and arg"
    ),
    (
        "ExpCmplx2 := ((A * B) <- (C * D)) <- (E * F)",
        "127. Product in both exponent result and arg"
    ),
    (
        "ExpCmplx3 := ($a A <- $b B) <- $c C",
        "128. Tagged types in exponent chain"
    ),
    (
        "ExpCmplx4 := (List[A] <- Map[B, C]) <- Option[D]",
        "129. Generics in exponent chain"
    ),
    (
        "ExpCmplx5 := ((A <- B) | (C <- D)) <- ((E <- F) | (G <- H))",
        "130. Exponent sums in exponent"
    ),

    # 131-135: Product-sum alternation patterns
    (
        "ProdSum1 := (A * (B | (C * D))) | E",
        "131. Product containing sum containing product"
    ),
    (
        "ProdSum2 := ((A * B) | C) * ((D | E) * F)",
        "132. Symmetric product-sum-product pattern"
    ),
    (
        "ProdSum3 := A | (B * (C | (D * (E | F))))",
        "133. Deep alternating product-sum"
    ),
    (
        "ProdSum4 := ((A | B) * C | D) * E",
        "134. Mixed precedence in product"
    ),
    (
        "ProdSum5 := (A * B | C * D) | (E * F | G * H)",
        "135. Products and sums at multiple levels"
    ),

    # 136-140: Generic with sum/product as args
    (
        "GenArg1 := Result[(A | B), (C | D)]",
        "136. Sum types as generic arguments"
    ),
    (
        "GenArg2 := Map[(A * B), (C * D)]",
        "137. Product types as generic arguments"
    ),
    (
        "GenArg3 := Option[(A | B) * (C | D)]",
        "138. Complex type as single generic arg"
    ),
    (
        "GenArg4 := List[($a A | $b B)]",
        "139. Tagged sum as generic arg"
    ),
    (
        "GenArg5 := Map[A <- B, C <- D]",
        "140. Exponent types as generic args"
    ),

    # 141-145: Ellipsis combinations
    (
        "Ellip1 := A | B | ...",
        "141. Basic open sum with ellipsis"
    ),
    (
        "Ellip2 := A | ... | B",
        "142. Ellipsis in middle of sum"
    ),
    (
        "Ellip3 := (A | ...) | (B | ...)",
        "143. Multiple open sums combined"
    ),
    (
        "Ellip4 := ($ok A | $err B | ...)",
        "144. Tagged open sum"
    ),
    (
        "Ellip5 := List[A | ...]",
        "145. Open sum in generic"
    ),

    # 146-150: Nested function types
    (
        "NestedFunc1 := ((A <- B) <- C) <- D",
        "146. Left-nested exponents"
    ),
    (
        "NestedFunc2 := (A <- (B <- (C <- D)))",
        "147. Right-nested exponents explicit"
    ),
    (
        "NestedFunc3 := (($a A <- $b B) <- $c C) <- $d D",
        "148. Left-nested tagged exponents"
    ),
    (
        "NestedFunc4 := (A <- (B <- C)) | (D <- (E <- F))",
        "149. Right-nested exponents in sum"
    ),
    (
        "NestedFunc5 := ((A | B) <- (C <- D)) <- ((E <- F) | G)",
        "150. Mixed nested exponents with sums"
    ),

    # 151-155: Identity edge cases
    (
        "IdEdge1 := () | ()",
        "151. Multiple empty tuples in sum"
    ),
    (
        "IdEdge2 := () * ()",
        "152. Multiple empty tuples in product"
    ),
    (
        "IdEdge3 := () <- ()",
        "153. Empty tuple as exponent operands"
    ),
    (
        "IdEdge4 := (A | ()) | (B | ())",
        "154. Empty tuple in nested sums"
    ),
    (
        "IdEdge5 := ((A * ()) * (B * ())) | ()",
        "155. Empty tuple in nested products"
    ),

    # 156-160: String literal edge cases
    (
        "StrEdge1 := \"a\" | \"b\" | \"c\" | \"d\"",
        "156. Multiple strings in sum"
    ),
    (
        "StrEdge2 := \"hello world\" * \"foo bar\"",
        "157. Strings with spaces in product"
    ),
    (
        "StrEdge3 := $msg \"important message\" | $err \"error occurred\"",
        "158. Tagged string literals"
    ),
    (
        "StrEdge4 := List[\"single\"] | Map[\"key\", \"value\"]",
        "159. String literals in generics"
    ),
    (
        "StrEdge5 := (\"a\" * \"b\") | (\"c\" * \"d\")",
        "160. String products in sum"
    ),

    # 161-165: Boolean combinations
    (
        "BoolComb1 := true | false | true | false",
        "161. Repeated booleans in sum"
    ),
    (
        "BoolComb2 := true * false * true * false",
        "162. Alternating booleans in product"
    ),
    (
        "BoolComb3 := ($on true | $off false)",
        "163. Tagged boolean literals"
    ),
    (
        "BoolComb4 := (true | A) * (false | B)",
        "164. Booleans with types in mixed ops"
    ),
    (
        "BoolComb5 := true <- false <- true",
        "165. Boolean exponent chain"
    ),

    # 166-170: Numeric literal combinations
    (
        "NumComb1 := 0 | 1 | 2 | 3 | 4 | 5",
        "166. Many integers in sum"
    ),
    (
        "NumComb2 := 0.1 * 0.2 * 0.3 * 0.4",
        "167. Float sequence in product"
    ),
    (
        "NumComb3 := ($int 42 | $float 3.14 | $str \"num\")",
        "168. Tagged numeric literals"
    ),
    (
        "NumComb4 := (1 | A) * (2.0 | B)",
        "169. Numbers with types in mixed ops"
    ),
    (
        "NumComb5 := 100 <- 200 <- 300",
        "170. Numeric exponent chain"
    ),

    # 171-175: Complex with all three operators
    (
        "ThreeOp1 := (A | B) <- (C * D)",
        "171. Sum and product in exponent"
    ),
    (
        "ThreeOp2 := ((A <- B) | C) * D",
        "172. Exponent and sum in product"
    ),
    (
        "ThreeOp3 := (A * B) | (C <- D)",
        "173. Product and exponent in sum"
    ),
    (
        "ThreeOp4 := ((A | B) <- (C * D)) | ((E <- F) * (G | H))",
        "174. All three operators nested"
    ),
    (
        "ThreeOp5 := ((A <- B) * (C | D)) <- ((E * F) | (G <- H))",
        "175. All three in exponent context"
    ),

    # 176-180: Real-world patterns
    (
        "Real1 := $success Result[Data, Error] | $pending void | $cancelled never",
        "176. Async operation result pattern"
    ),
    (
        "Real2 := ($request Request <- $response Response) | $error Error",
        "177. Request-response pattern"
    ),
    (
        "Real3 := $node (Value * $children List[Node])",
        "178. Tree node pattern"
    ),
    (
        "Real4 := $event ($click (X * Y) | $keypress Key | $scroll (DX * DY))",
        "179. Event union pattern"
    ),
    (
        "Real5 := $state ($idle void | $running Process | $finished Result | $failed Error)",
        "180. State machine pattern"
    ),

    # 181-185: Maximum depth tests
    (
        "MaxD1 := (((((A | B) | C) | D) | E) | F)",
        "181. Deep left-nested sum"
    ),
    (
        "MaxD2 := (((((A * B) * C) * D) * E) * F)",
        "182. Deep left-nested product"
    ),
    (
        "MaxD3 := (((((F <- E) <- D) <- C) <- B) <- A)",
        "183. Deep left-nested exponent"
    ),
    (
        "MaxD4 := (((((A, B), C), D), E), F)",
        "184. Deep left-nested tuple"
    ),
    (
        "MaxD5 := (((($a A | $b B) | $c C) | $d D) | $e E) | $f F",
        "185. Deep nested tagged sum"
    ),

    # 186-190: Symmetric patterns
    (
        "Sym1 := (A | B) | (B | A)",
        "186. Symmetric sum structure"
    ),
    (
        "Sym2 := (A * B) * (B * A)",
        "187. Symmetric product structure"
    ),
    (
        "Sym3 := (A <- B) | (B <- A)",
        "188. Symmetric exponent in sum"
    ),
    (
        "Sym4 := ((A | B) * (C | D)) | ((C | D) * (A | B))",
        "189. Complex symmetric pattern"
    ),
    (
        "Sym5 := ((A <- B) * (C <- D)) | ((D <- C) * (B <- A))",
        "190. Full symmetric with all operators"
    ),

    # 191-195: Pathological cases
    (
        "Path1 := A | (B | (C | (D | (E | F))))",
        "191. Right-nested sum"
    ),
    (
        "Path2 := A * (B * (C * (D * (E * F))))",
        "192. Right-nested product"
    ),
    (
        "Path3 := A | B * C | D * E | F",
        "193. Mixed precedence without parens"
    ),
    (
        "Path4 := (A * B | C) | (D | E * F)",
        "194. Asymmetric mixed precedence"
    ),
    (
        "Path5 := (((A | B) * C) | D) * (((E | F) * G) | H)",
        "195. Nested triple operator mix"
    ),

    # 196-200: Ultimate stress tests
    (
        "UltStress1 := ($a (($b (A | B) * $c (C | D)) | ($d E * $f F)) <- $g ((G | H) * (I | J)))",
        "196. Deep nesting with all features"
    ),
    (
        "UltStress2 := Map[List[Result[Option[A], E]], Map[B, List[Option[C]]]]",
        "197. Maximum generic nesting depth"
    ),
    (
        "UltStress3 := (($ok ($a A | $b B) | $err ($c C | $d D)) <- ($in (E * F) | $out (G * H))) | ($timeout never)",
        "198. All operators with tags and identities"
    ),
    (
        "UltStress4 := (((A <- (B <- (C <- D))) | (E <- F)) * ((G <- H) | (I <- (J <- K)))) <- ((L | M) * (N | O))",
        "199. Maximum exponent nesting with sums and products"
    ),
    (
        "UltStress5 := ($meta.config.data ($val (List[Map[A, B]] | Map[List[A], B]) | $err ($code 500 * $msg \"error\")) <- ($req.input (C * D * E) * $req.options (F | G | H)))",
        "200. The ultimate stress test - all features combined"
    ),

    # ========================================================================
    # 201-400: MOST COMPLICATED CASES
    # ========================================================================

    # 201-210: More extreme generic combinations
    (
        "UGen1 := Map[Map[Map[A, B], Map[C, D]], Map[E, F]]",
        "201. Triple nested Maps"
    ),
    (
        "UGen2 := List[List[List[List[Option[A]]]]]",
        "202. Quadruple nested Lists"
    ),
    (
        "UGen3 := Result[Result[Option[A], E], Result[B, F]]",
        "203. Nested Results with Options"
    ),
    (
        "UGen4 := Map[Either[A, B], Either[Option[C], Either[D, E]]]",
        "204. Complex generic with Either and Option"
    ),
    (
        "UGen5 := List[Map[Result[Option[A], B], Result[C, Option[D]]]]",
        "205. Maximum generic nesting depth"
    ),
    (
        "UGen6 := ($val A -> $err B) | ($ok C -> $fail D)",
        "206. Arrow-like patterns in sum (actually just types)"
    ),
    (
        "UGen7 := $ok (Result[List[A], E] | never) <- $err (Map[B, C] | void)",
        "207. Generics in tagged with exponents"
    ),
    (
        "UGen8 := Map[(A | B) * (C | D), (E * F) | (G * H)]",
        "208. Complex types in both generic args"
    ),
    (
        "UGen9 := List[($a A | $b B) * ($c C | $d D) * ($e E | $f F)]",
        "209. Tagged sums in products in generic"
    ),
    (
        "UGen10 := Map[Result[A, (B | C)], Option[(D * E) | F]]",
        "210. Sum/product as generic params"
    ),

    # 211-220: Literal in deeply nested structures
    (
        "LitNest1 := (((1 * 2) | 3) * (4 | (5 * 6))) | (7 * 8)",
        "211. Numbers in deep nested ops"
    ),
    (
        "LitNest2 :=\"a\" * (\"b\" | (\"c\" * \"d\")) | (\"e\" * \"f\")",
        "212. Strings in deep nested ops"
    ),
    (
        "LitNest3 := true * (false | (true * false)) | (true | false)",
        "213. Booleans in deep nested ops"
    ),
    (
        "LitNest4 := ($tag 42) | ($val 3.14)",
        "214. Tagged literals in sum"
    ),
    (
        "LitNest5 := List[$key 42] | Map[$val 3.14, $str \"test\"]",
        "215. Tagged literals in generics"
    ),
    (
        "LitNest6 := 42 <- 3.14 <- \"test\" <- true",
        "216. Mixed literal exponent chain"
    ),
    (
        "LitNest7 := (1 | A) * (2 | B) * (3 | C)",
        "217. Literals and types in product"
    ),
    (
        "LitNest8 := ($x (A | 1)) | ($y (B | 2.0))",
        "218. Tagged mixed sum with literals"
    ),
    (
        "LitNest9 := List[(A | 1) * (B | 2)]",
        "219. Mixed types in generic arg"
    ),
    (
        "LitNest10 := ((0 | A) | (1 | B)) <- ((2 | C) | (3 | D))",
        "220. Mixed sums in exponent"
    ),

    # 221-230: Void/Never in extreme positions
    (
        "VN1 := ((((void | void) | void) | void) | void)",
        "221. Many void in sum"
    ),
    (
        "VN2 := ((((never * never) * never) * never) * never)",
        "222. Many never in product"
    ),
    (
        "VN3 := ((void | never) * (never | void)) | ((void * never) | (never * void))",
        "223. Complex void/never combinations"
    ),
    (
        "VN4 := void <- never <- void <- never",
        "224. Void/never exponent chain"
    ),
    (
        "VN5 := ($a void | $b never) * ($c never | $d void)",
        "225. Tagged void/never in product"
    ),
    (
        "VN6 := List[void] | Map[never, void]",
        "226. Identities in generic args"
    ),
    (
        "VN7 := (void | A) * (never | B) * (void | C)",
        "227. Identities with types in product"
    ),
    (
        "VN8 := ((void * A) | (never * B)) <- (C | void)",
        "228. Identities in complex structures"
    ),
    (
        "VN9 := (((() | void) | never) | ())",
        "229. Empty tuple with identities"
    ),
    (
        "VN10 := (A | void) * (B | never) * (C | void)",
        "230. Types with identity alternations"
    ),

    # 231-240: Empty tuple edge cases
    (
        "ET1 := () * () * ()",
        "231. Multiple empty tuples in product"
    ),
    (
        "ET2 := () | () | ()",
        "232. Multiple empty tuples in sum"
    ),
    (
        "ET3 := (((), ()), ((), ()))",
        "233. Nested empty tuples"
    ),
    (
        "ET4 := ((), A) * (B, ())",
        "234. Empty tuple mixed with types"
    ),
    (
        "ET5 := List[()] | Map[(), ()]",
        "235. Empty tuple in generics"
    ),
    (
        "ET6 := $ok () | $err ()",
        "236. Tagged empty tuples in sum"
    ),
    (
        "ET7 := (() <- ()) <- ()",
        "237. Empty tuple exponent chain"
    ),
    (
        "ET8 := ((A | ()) | B) * ((C | ()) | D)",
        "238. Empty tuple in nested sums"
    ),
    (
        "ET9 := (() * A) | (() * B)",
        "239. Empty tuple in products in sum"
    ),
    (
        "ET10 := ($tag ()) * ($other ())",
        "240. Tagged empty tuples in product"
    ),

    # 241-250: Ellipsis edge cases
    (
        "EllExt1 := A | B | C | ... | D | E",
        "241. Ellipsis with elements on both sides"
    ),
    (
        "EllExt2 := (A | ...) * (B | ...)",
        "242. Ellipsis in product"
    ),
    (
        "EllExt3 := ($ok A | $err B | ...)",
        "243. Tagged open sum"
    ),
    (
        "EllExt4 := List[A | ...] | Map[B | ..., C]",
        "244. Ellipsis in nested generics"
    ),
    (
        "EllExt5 := ((A | ...) | B) | (C | ...)",
        "245. Multiple ellipsis in sum"
    ),
    (
        "EllExt6 := (A | ...) <- (B | ...)",
        "246. Ellipsis in exponent"
    ),
    (
        "EllExt7 := ($open ... | $closed (A | B))",
        "247. Tagged ellipsis with closed variant"
    ),
    (
        "EllExt8 := List[Map[A, ...]] | Set[...]",
        "248. Ellipsis in nested generic"
    ),
    (
        "EllExt9 := ... | A | B | C",
        "249. Ellipsis at start of sum"
    ),
    (
        "EllExt10 := (A | B) | (C | ...)",
        "250. Ellipsis in nested sum"
    ),

    # 251-260: Extreme exponent nesting
    (
        "ExpExt1 := (((((A <- B) <- C) <- D) <- E) <- F)",
        "251. Deep left-nested exponent"
    ),
    (
        "ExpExt2 := (A <- (B <- (C <- (D <- (E <- F)))))",
        "252. Deep right-nested exponent"
    ),
    (
        "ExpExt3 := ((A <- B) | (C <- D)) <- ((E <- F) | (G <- H))",
        "253. Exponents in sum as exponent"
    ),
    (
        "ExpExt4 := (((A | B) <- C) | D) <- ((E | F) <- G)",
        "254. Mixed exponents in exponent"
    ),
    (
        "ExpExt5 := (($a A <- $b B) <- $c C) <- ($d D <- $e E)",
        "255. Tagged deep left-nested exponent"
    ),
    (
        "ExpExt6 := ($a A <- ($b B <- ($c C <- $d D)))",
        "256. Tagged deep right-nested exponent"
    ),
    (
        "ExpExt7 := (List[A] <- Map[B, C]) <- Option[(D | E)]",
        "257. Generics in deep exponent"
    ),
    (
        "ExpExt8 := ((A <- B) * (C <- D)) | ((E <- F) * (G <- H))",
        "258. Products of exponents in sum"
    ),
    (
        "ExpExt9 := (((A | B) <- (C | D)) <- ((E | F) | (G | H)))",
        "259. Sums in deep exponent"
    ),
    (
        "ExpExt10 := (A <- (B <- C)) <- (D <- (E <- F))",
        "260. Mixed nesting in exponent"
    ),

    # 261-270: Tuple in all contexts
    (
        "TupAll1 := ((A, B), C, D)",
        "271. Tuple with tuple as first element"
    ),
    (
        "TupAll2 := (A, (B, C), D)",
        "272. Tuple with tuple as middle element"
    ),
    (
        "TupAll3 := (A, B, (C, D))",
        "273. Tuple with tuple as last element"
    ),
    (
        "TupAll4 := (((), ()), (A, B))",
        "274. Nested empty tuples with real tuples"
    ),
    (
        "TupAll5 := (($a A, $b B), ($c C, $d D))",
        "275. Tuples with tagged types"
    ),
    (
        "TupAll6 := (List[A], Map[B, C], Option[D])",
        "276. Tuple with generics"
    ),
    (
        "TupAll7 := ((1, 2), (3, 4), (5, 6))",
        "277. Tuple of tuples of literals"
    ),
    (
        "TupAll8 := ((A | B), (C | D), (E | F))",
        "278. Tuple with sums as elements"
    ),
    (
        "TupAll9 := ((A * B), (C * D), (E * F))",
        "279. Tuple with products as elements"
    ),
    (
        "TupAll10 := ((A <- B), (C <- D), (E <- F))",
        "280. Tuple with exponents as elements"
    ),

    # 271-280: Tagged extreme cases
    (
        "TagExt1 := $a.b.c.d.e.f.g A",
        "281. 7-level deep tag path"
    ),
    (
        "TagExt2 := $a ($b ($c ($d ($e ($f A)))))",
        "282. 6-level deep nested tagged types"
    ),
    (
        "TagExt3 := ($a ($b A | $c B)) | ($d ($e C | $f D))",
        "283. Tagged sums in tagged sum"
    ),
    (
        "TagExt4 := ($a ($b A * $c D)) * ($d ($e B * $f E))",
        "284. Tagged products in tagged product"
    ),
    (
        "TagExt5 := ($a ($b A | $c B | $d E | $f F | $g G | $h H | $i I))",
        "285. Tagged sum with many variants"
    ),
    (
        "TagExt6 := ($a A * $b B * $c C * $d D * $e E * $f F * $g G)",
        "286. Tagged product with many fields"
    ),
    (
        "TagExt7 := ($a ($b (A | B) | $c (C * D))) | ($e ($f (E | F) | $g (G * H)))",
        "287. Deep nested tagged complex"
    ),
    (
        "TagExt8 := $a.b.c (List[Map[A, B]] | Map[List[A], B])",
        "288. Deep tag with complex body"
    ),
    (
        "TagExt9 := ($x.y A) | ($z.w B) | ($p.q C) | ($r.s D)",
        "289. Many deep tags in sum"
    ),
    (
        "TagExt10 := ($a A) <- ($b B) <- ($c C) <- ($d D)",
        "290. Tagged exponent chain"
    ),

    # 281-290: All three operators combined
    (
        "AllOp1 := ((A | B) * (C | D)) | ((E | F) * (G | H))",
        "291. Sum of products of sums"
    ),
    (
        "AllOp2 := ((A * B) | (C * D)) * ((E * F) | (G * H))",
        "292. Product of sums of products"
    ),
    (
        "AllOp3 := (A | (B * C)) | (D | (E * F))",
        "293. Sums with products inside"
    ),
    (
        "AllOp4 := (A * (B | C)) * (D * (E | F))",
        "294. Products with sums inside"
    ),
    (
        "AllOp5 := ((A | B) <- (C * D)) | ((E * F) <- (G | H))",
        "295. Exponents with sum/product in exponent"
    ),
    (
        "AllOp6 := ((A * B) <- (C | D)) | ((E | F) <- (G * H))",
        "296. Exponents with product/sum in exponent"
    ),
    (
        "AllOp7 := ((A <- B) | (C <- D)) * ((E <- F) | (G <- H))",
        "297. Product of exponent sums"
    ),
    (
        "AllOp8 := ((A <- B) * (C <- D)) | ((E <- F) * (G <- H))",
        "298. Sum of exponent products"
    ),
    (
        "AllOp9 := ((A | B) * (C | D)) <- ((E | F) * (G | H))",
        "299. Exponent with product of sums"
    ),
    (
        "AllOp10 := (((A | B) <- C) | D) * (((E | F) <- G) | H)",
        "300. Maximum operator mixing"
    ),

    # 291-300: Real-world extreme patterns
    (
        "RealExt1 := $request (($path String * $query Map[String, String]) * $headers Map[String, String]) | $cancel void",
        "301. HTTP request pattern"
    ),
    (
        "RealExt2 := $response (($status Int * $body Option[JSON]) * $headers Map[String, String]) | $error Error",
        "302. HTTP response pattern"
    ),
    (
        "RealExt3 := $tree ($value Value * $left Option[Tree] * $right Option[Tree])",
        "303. Binary tree node pattern"
    ),
    (
        "RealExt4 := $node ($data Data * $children List[Node]) | $leaf Data",
        "304. Rose tree pattern"
    ),
    (
        "RealExt5 := $state ($idle void | $running Process | $paused Process | $finished Result | $failed Error | $cancelled never)",
        "305. Enhanced state machine pattern"
    ),
    (
        "RealExt6 := $event (($click (Int * Int) | $keypress Key | $scroll (Int * Int)) | $resize (Int * Int) | $focus Bool)",
        "306. UI event pattern"
    ),
    (
        "RealExt7 := ($success Result[Data, Error] | $failure Error) <- $input Request",
        "307. Async operation with input"
    ),
    (
        "RealExt8 := $config ($host String * $port Int * $timeout Int * $retries Int * $ssl Bool)",
        "308. Configuration pattern"
    ),
    (
        "RealExt9 := ($token String * $expires Int) | ($error Error)",
        "309. Authentication result pattern"
    ),
    (
        "RealExt10 := ($page Page | $redirect String | $error Error) <- $route Route",
        "310. Router handler pattern"
    ),

    # 301-310: Maximum nesting (8+ levels)
    (
        "MaxNest1 := (((((((A | B) | C) | D) | E) | F) | G) | H)",
        "311. 8-level left-nested sum"
    ),
    (
        "MaxNest2 := (((((((A * B) * C) * D) * E) * F) * G) * H)",
        "312. 8-level left-nested product"
    ),
    (
        "MaxNest3 := (((((((A <- B) <- C) <- D) <- E) <- F) <- G) <- H)",
        "313. 8-level left-nested exponent"
    ),
    (
        "MaxNest4 := (((((((((((A | B) | C) | D) | E) | F) | G) | H) | I) | J) | K) | L)",
        "314. 12-level left-nested sum"
    ),
    (
        "MaxNest5 := ((A | ((B | (C | (D | (E | (F | (G | H)))))))))",
        "315. 8-level right-nested sum"
    ),
    (
        "MaxNest6 := ((A * ((B * (C * (D * (E * (F * (G * H)))))))))",
        "316. 8-level right-nested product"
    ),
    (
        "MaxNest7 := (A <- (B <- (C <- (D <- (E <- (F <- (G <- H)))))))",
        "317. 8-level right-nested exponent"
    ),
    (
        "MaxNest8 := ((((A | B) * (C | D)) | (E | F)) * ((G | H) | (I | J)))",
        "318. 4-level alternating pattern"
    ),
    (
        "MaxNest9 := (((((A | B) * C) | D) * E) | F) * ((G | H) | I)",
        "319. Deep mixed nesting"
    ),
    (
        "MaxNest10 := ((($a A | $b B) * $c C) | $d D) * $e E",
        "320. Deep tagged nesting"
    ),

    # 311-320: Symmetric complex patterns
    (
        "SymCmplx1 := ((A | B) * (C | D)) | ((D | C) * (B | A))",
        "321. Reversible complex structure"
    ),
    (
        "SymCmplx2 := ((A <- B) | (C <- D)) | ((D <- C) | (B <- A))",
        "322. Reversible exponent sum"
    ),
    (
        "SymCmplx3 := (((A * B) | (C * D)) * ((E | F) | (G | H))) | (((H | G) | (F | E)) * ((D * C) | (B * A)))",
        "323. Fully reversible complex"
    ),
    (
        "SymCmplx4 := ((($a A | $b B) * ($c C | $d D)) | (($e E | $f F) * ($g G | $h H)))",
        "324. Tagged symmetric pattern"
    ),
    (
        "SymCmplx5 := (A | B | C) | (C | B | A)",
        "325. Palindrome sum pattern"
    ),
    (
        "SymCmplx6 := (A * B * C) * (C * B * A)",
        "326. Palindrome product pattern"
    ),
    (
        "SymCmplx7 := ($a A | $b B | $c C) | ($c C | $b B | $a A)",
        "327. Palindrome tagged sum"
    ),
    (
        "SymCmplx8 := ((A | B) | C) | ((B | A) | C)",
        "328. Nested palindrome sum"
    ),
    (
        "SymCmplx9 := ((A * B) * C) * ((B * A) * C)",
        "329. Nested palindrome product"
    ),
    (
        "SymCmplx10 := (($a A | $b B) | $c C) | (($b B | $a A) | $c C)",
        "330. Nested palindrome tagged"
    ),

    # 321-330: Pathological edge cases
    (
        "PathCmplx1 := (A | (B | (C | (D | (E | (F | (G | (H | I))))))))",
        "331. Extreme right-nested sum"
    ),
    (
        "PathCmplx2 := (A * (B * (C * (D * (E * (F * (G * (H * I))))))))",
        "332. Extreme right-nested product"
    ),
    (
        "PathCmplx3 := (A | B) * (C | D) | (E | F) * (G | H) | (I | J) * (K | L)",
        "333. Alternating pattern without parens"
    ),
    (
        "PathCmplx4 := ((A | B * C) | D) | ((E * F | G) | H)",
        "334. Ambiguous precedence patterns"
    ),
    (
        "PathCmplx5 := (A <- B | C) <- (D * E | F)",
        "335. Exponent with ambiguous arguments"
    ),
    (
        "PathCmplx6 := (((A | B) | C) | (D | E | F)) | ((G | H) | (I | J))",
        "336. Mixed sum lengths"
    ),
    (
        "PathCmplx7 := ((A * B) * (C * D)) * ((E * F * G) * (H * I))",
        "337. Mixed product lengths"
    ),
    (
        "PathCmplx8 := (A | (B * C) | D) * (E | (F * G) | H)",
        "338. Same pattern repeated"
    ),
    (
        "PathCmplx9 := ($a A | $b (B | C | D) | $e E) * ($f F | $g (G * H) | $i I)",
        "339. Tagged mixed patterns"
    ),
    (
        "PathCmplx10 := (A | B) <- (C * D) | (E | F) <- (G * H)",
        "340. Exponents in complex sum"
    ),

    # 331-340: Type app with complex bodies
    (
        "TypeApp1 := List[(A | B) * (C | D) | (E | F) * (G | H)]",
        "341. Complex sum of products in generic"
    ),
    (
        "TypeApp2 := Map[(A * B) | (C * D), (E * F) | (G * H)]",
        "342. Complex products in both generic args"
    ),
    (
        "TypeApp3 := Result[((A <- B) | (C <- D)), ((E <- F) | (G <- H))]",
        "343. Exponents in Result"
    ),
    (
        "TypeApp4 := Option[($a A | $b B) * ($c C | $d D)]",
        "344. Tagged sums in generic"
    ),
    (
        "TypeApp5 := Map[List[A | B], Set[C | D]]",
        "345. Generic with sums in nested generics"
    ),
    (
        "TypeApp6 := List[Map[(A | B) * (C | D), Option[(E | F)]]]",
        "346. Deep generic with complex types"
    ),
    (
        "TypeApp7 := Result[List[Option[A]], Map[B, Option[C]]]",
        "347. Generic with nested generics"
    ),
    (
        "TypeApp8 := Either[Result[A, B], Option[C] | D]",
        "348. Mixed complex types in Either"
    ),
    (
        "TypeApp9 := Map[($a A | $b B) <- C, D <- ($e E | $f F)]",
        "349. Exponents in generic args"
    ),
    (
        "TypeApp10 := List[Result[Option[A | B], C | D]]",
        "350. Sum in nested generic params"
    ),

    # 341-350: All features together
    (
        "AllFeat1 := ($ok (Result[Option[A], (B | C)] | never) | $err Map[(D * E), List[F]]) <- ($req (G | H) * $opt (I | J))",
        "351. All features combined #1"
    ),
    (
        "AllFeat2 := List[Map[($a A | $b B), ($c C | $d D)]] | Map[List[($e E | $f F)], Set[($g G | $h H)]]",
        "352. All features combined #2"
    ),
    (
        "AllFeat3 := ((($x (A | B) | $y (C | D)) * ($z (E | F) | $w (G | H))) | $error never) <- ($input (I * J | K) * $config (L | M | N))",
        "353. All features combined #3"
    ),
    (
        "AllFeat4 := Result[Map[Option[A | B], List[C | D]], Option[Map[E | F, List[G | H]]]]",
        "354. All features combined #4"
    ),
    (
        "AllFeat5 := ($state ($idle void | $running Process | $finished Result[Option[Data], Error])) <- ($event (($click Event | $keypress Key) | $timeout never))",
        "355. All features combined #5"
    ),
    (
        "AllFeat6 := Map[Result[Option[A], Option[B]], Map[Result[C, D], Option[E]]] | List[Result[Map[A, B], Map[C, D]]]",
        "356. All features combined #6"
    ),
    (
        "AllFeat7 := (($req Request | $error Error) <- $auth (String | never)) | ($cancel void)",
        "357. All features combined #7"
    ),
    (
        "AllFeat8 := List[Set[Map[Option[A], Result[B, C]]]]",
        "358. All features combined #8"
    ),
    (
        "AllFeat9 := ($ok ($a A | $b B | $c C) | $err ($d D | $e E)) <- ($in ($f F | $g G) * $out ($h H | $i I))",
        "359. All features combined #9"
    ),
    (
        "AllFeat10 := Result[Result[Option[A], B], Result[C, Option[D]]]",
        "360. All features combined #10"
    ),

    # 351-360: Extreme literal patterns
    (
        "LitExt1 := 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9",
        "361. Many integers in sum"
    ),
    (
        "LitExt2 := 0.1 * 0.2 * 0.3 * 0.4 * 0.5 * 0.6",
        "362. Many floats in product"
    ),
    (
        "LitExt3 := \"a\" | \"b\" | \"c\" | \"d\" | \"e\" | \"f\" | \"g\" | \"h\"",
        "363. Many strings in sum"
    ),
    (
        "LitExt4 := true * false * true * false * true * false",
        "364. Many booleans in product"
    ),
    (
        "LitExt5 := ($tag 0 | $label 1) * ($tag 2 | $label 3)",
        "365. Tagged integers in product"
    ),
    (
        "LitExt6 := List[0 | 1 | 2 | 3] | Map[\"a\", \"b\" | \"c\"]",
        "366. Literals in nested generics"
    ),
    (
        "LitExt7 := (1 | A) * (2 | B) * (3 | C) * (4 | D)",
        "367. Literals and types alternating"
    ),
    (
        "LitExt8 :=\"hello\" <- \"world\" <- \"test\"",
        "368. String exponent chain"
    ),
    (
        "LitExt9 := (true | false) | (1 | 0) | (\"a\" | \"b\")",
        "369. Different literal types in sum"
    ),
    (
        "LitExt10 := ($a 42 | $b 3.14) | ($c \"test\" | $d true)",
        "370. Mixed tagged literals in sum"
    ),

    # 361-370: Deep generic recursion simulation
    (
        "GenRec1 := List[A] | (A * List[A])",
        "371. Simulated recursive type (list)"
    ),
    (
        "GenRec2 := Option[A] | A",
        "372. Simulated recursive type (option)"
    ),
    (
        "GenRec3 := Map[A, B] | A | B",
        "373. Simulated recursive type (map)"
    ),
    (
        "GenRec4 := Result[A, B] | ($ok A) | ($err B)",
        "374. Simulated recursive result"
    ),
    (
        "GenRec5 := Either[A, B] | ($left A) | ($right B)",
        "375. Simulated recursive either"
    ),
    (
        "GenRec6 := ($nil void) | ($cons (A * List[A]))",
        "376. Classic linked list pattern"
    ),
    (
        "GenRec7 :=$leaf A | ($node (Tree[A] * Tree[A]))",
        "377. Binary tree pattern"
    ),
    (
        "GenRec8 := Maybe[A] | A | never",
        "378. Maybe with explicit identity"
    ),
    (
        "GenRec9 := $none void | $some A",
        "379. Standard option variant"
    ),
    (
        "GenRec10 := $empty void | $elem (A * List[A])",
        "380. List variant pattern"
    ),

    # 371-380: Bracket operator stress
    (
        "Brack1 := A[B] | C[D] | E[F]",
        "381. Multiple type apps in sum"
    ),
    (
        "Brack2 := A[B] * C[D] * E[F]",
        "382. Multiple type apps in product"
    ),
    (
        "Brack3 := A[B[C[D[E[F]]]]]",
        "383. Deeply nested type apps"
    ),
    (
        "Brack4 := List[Map[Set[A, B], Map[C, D]]]",
        "384. Mixed generics deep"
    ),
    (
        "Brack5 := Result[A, (B | C)] <- Option[(D * E) | F]",
        "385. Type apps in exponent with sums/products"
    ),
    (
        "Brack6 := (A[B] | C[D]) * (E[F] | G[H])",
        "386. Type apps in complex expressions"
    ),
    (
        "Brack7 := $tag A[B | C] | $other D[E | F]",
        "387. Type apps with tags"
    ),
    (
        "Brack8 := List[($a A | $b B)] | Map[($c C, $d D), ($e E | $f F)]",
        "388. Type apps with tagged in generics"
    ),
    (
        "Brack9 := ((A[B] | C[D]) | E[F]) <- ((G[H] | I[J]) | K[L])",
        "389. Type apps in complex exponent"
    ),
    (
        "Brack10 := Map[List[Result[Option[A], B]], Set[Map[C, Option[D]]]]",
        "390. Maximum generic nesting with all features"
    ),

    # 381-390: Tag + bracket combinations
    (
        "TagBrack1 := $ok Result[A, B] | $err Map[C, D]",
        "391. Tagged generics in sum"
    ),
    (
        "TagBrack2 := ($a Option[A] | $b List[B]) * ($c Map[C, D] | $d Set[E])",
        "392. Tagged generics in product"
    ),
    (
        "TagBrack3 := ($val A[B]) <- ($key C[D])",
        "393. Tagged generics in exponent"
    ),
    (
        "TagBrack4 := List[($a A | $b B)] | Map[$c C, $d D]",
        "394. Tags inside generics"
    ),
    (
        "TagBrack5 := ($a.b A[C] | $c.d B[D])",
        "395. Deep tags with generics"
    ),
    (
        "TagBrack6 := $meta List[Result[Option[A], B]]",
        "396. Tagged deeply nested generic"
    ),
    (
        "TagBrack7 := ($a A[B]) | ($b C[D]) | ($c E[F])",
        "397. Multiple tagged generics in sum"
    ),
    (
        "TagBrack8 := ($a A[B]) * ($b C[D]) * ($c E[F])",
        "398. Multiple tagged generics in product"
    ),
    (
        "TagBrack9 := (($a A[B] | $b C[D]) <- ($c E[F] | $d G[H]))",
        "399. Tagged generics in complex exponent"
    ),
    (
        "TagBrack10 := Map[Result[$a A, $b B], Option[$c C | $d D]]",
        "400. Tags everywhere"
    ),

    # 391-400: The ultimate stress tests
    (
        "Ult200_1 := (($a ($b (A | B) | $c (C * D)) * ($d (E | F) | $e (G * H))) | ($f ($g (I | J) | $h (K * L)) * ($i (M | N) | $j (O * P)))) <- ($req (($q (A | B) * $r (C | D)) | ($s (E | F) * $t (G * H))) * $opt (($u (I | J) | $v (K * L)) | ($w (M | N) | $x (O * P))))",
        "401. Ultimate stress test #1"
    ),
    (
        "Ult200_2 := List[Map[Result[Option[Set[A, B]], Option[List[C, D]]], Map[Result[Set[E, F], List[G, H]], Option[Result[I, J]]]]]",
        "402. Ultimate stress test #2 - max generic depth"
    ),
    (
        "Ult200_3 := ((($ok (Result[List[Option[A]], Map[B, C]] | never)) | ($err (Set[Option[D], List[E]] | never))) <- ($req ((A * B * C) | (D * E * F))) | $cancel void) | $timeout never",
        "403. Ultimate stress test #3"
    ),
    (
        "Ult200_4 := (((((((A | B) | C) | D) | E) | F) * (((G | H) | I) | J) | K) | (((L | M) | N) | O) | (((P | Q) | R) | S))",
        "404. Ultimate stress test #4 - deep sum mixing"
    ),
    (
        "Ult200_5 := ($meta.config.service ($data ($value (List[Map[A, B]] | Map[List[A], B]) | $error ($code (500 | 404 | 403) * $msg (\"error\" | \"not found\")))) <- ($req (($input (C * D * E * F) | $fallback void) * $options ((G | H) * (I | J) * (K | L)))))",
        "405. Ultimate stress test #5 - real-world complexity"
    ),
    (
        "Ult200_6 := (List[($a A | $b B | $c C | $d D | $e E)] | Map[($f F | $g G), ($h H | $i I | $j J)]) <- (Option[($k K | $l L)] | Set[($m M | $n N)])",
        "406. Ultimate stress test #6"
    ),
    (
        "Ult200_7 := (((A <- B) | (C <- D)) * ((E <- F) | (G <- H))) | (((I <- J) | (K <- L)) * ((M <- N) | (O <- P)))",
        "407. Ultimate stress test #7 - exponent products in sum"
    ),
    (
        "Ult200_8 := Map[Result[Option[(A | B) * (C | D)], Set[(E | F) | (G | H)]], List[Map[Option[(I | J)], Set[(K | L) | (M | N)]]]]",
        "408. Ultimate stress test #8 - nested complexity"
    ),
    (
        "Ult200_9 := ($ok ($val ($data (A | B | C) | $meta ($info ($x D * $y E * $z F)))) | $err ($msg ($code 404 * $reason \"not found\") | $timeout ($ms 5000 * $msg \"timeout\")))",
        "409. Ultimate stress test #9"
    ),
    (
        "Ult200_10 := (((($a A | $b B | $c C | $d D) * ($e E | $f F | $g G | $h H)) | ($i I | $j J | $k K | $l L)) <- (($m M | $n N | $o O | $p P) * ($q Q | $r R | $s S | $t T))) | ($u U | $v V | $w W)",
        "410. Ultimate stress test #10 - the final boss"
    ),
]

# ==============================================================================
# Test Runner
# ==============================================================================


def run_tests():
    """Run all round-trip tests."""
    print("=" * 80)
    print("VIBA Round-Trip Test Suite")
    print("Testing: parser.parse(X) == parser.parse(unparser.unparse(parser.parse(X)))")
    print("=" * 80)
    print()

    passed = 0
    failed = 0
    failed_cases = []

    for i, (code, description) in enumerate(TEST_CASES, 1):
        try:
            # Step 1: Parse original code
            original_ast = parser.parse(code)

            # Step 2: Convert to Type objects
            type_objs = type_parse(original_ast)

            # Step 3: Unparse back to code
            unparsed_code = unparse(type_objs)

            # Step 4: Parse the unparsed code
            reparsed_ast = parser.parse(unparsed_code)

            # Step 5: Compare ASTs
            if ast_equal(original_ast, reparsed_ast):
                passed += 1
                print(f"[{i:2d}] PASS {description:<45}")
            else:
                failed += 1
                failed_cases.append((i, code, description))
                print(f"[{i:2d}] FAIL {description:<45}")
                print(f"      Original:   {code}")
                print(f"      Unparsed:   {unparsed_code}")

        except Exception as e:
            failed += 1
            failed_cases.append((i, code, description))
            print(f"[{i:2d}] ERROR {description:<45} | {e}")

    # Summary
    print()
    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(TEST_CASES)} tests")
    print("=" * 80)

    if failed_cases:
        print()
        print("Failed cases:")
        for i, code, desc in failed_cases:
            print(f"  [{i}] {desc}: {code}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

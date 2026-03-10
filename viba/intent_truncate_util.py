from viba.parser import parser as viba_parser
from viba.type import parse as viba_type_parse, ExponentChainType, DefinitionType
from viba.chain import convert_to_chain_style
from viba.unparser import unparse
from viba.std_coding_style import (
    check_viba_std_coding_style,
    FuncImplementStdCodingStyle,
)


def _parse_to_chain_statements(raw_viba_code: str) -> list:
    """Parse raw viba code into a list of chain-style Type objects."""
    ast_dicts = viba_parser.parse(raw_viba_code)
    if not ast_dicts:
        return []
    types = viba_type_parse(ast_dicts)
    return [convert_to_chain_style(t) for t in types]


def _is_func_impl(vibe_statement) -> bool:
    """Check if a chain-style Type is a FuncImplementStdCodingStyle definition."""
    try:
        result = check_viba_std_coding_style(vibe_statement)
        return result.success and isinstance(result.style, FuncImplementStdCodingStyle)
    except Exception:
        return False


def _get_exponent_chain(vibe_statement) -> list:
    """Get the exponent chain elements [result, arg0, arg1, ...] from a func impl definition."""
    body = vibe_statement.body
    if isinstance(body, ExponentChainType):
        return [body.result] + body.args
    return []


def _unparse_chain_slice(vibe_statement, chain_elements: list) -> str:
    """Create a new DefinitionType with a sliced exponent chain and unparse it."""
    if len(chain_elements) == 0:
        return ""
    if len(chain_elements) == 1:
        new_body = chain_elements[0]
    else:
        new_body = ExponentChainType(chain_elements[0], *chain_elements[1:])
    new_def = DefinitionType(
        node="Definition",
        name=vibe_statement.name,
        generic_params=vibe_statement.generic_params,
        body=new_body,
    )
    return unparse([new_def])


def get_all_truncated_vibe_code(
    raw_viba_code: str,
    num_parts: int = 5,
) -> tuple[str, list[str]]:
    """
    Generate progressively truncated vibe code from raw viba source.

    For each statement in the parsed code:
      - If it is NOT a function implementation (class/data definition):
        its full unparsed content is included in both intent_base and every truncation level.
      - If it IS a function implementation:
        only the declaration header (result + first argument) goes into intent_base,
        and each truncation level reveals progressively more of the exponent chain body.

    :param raw_viba_code: Raw viba source code string.
    :param num_parts: Number of truncation levels to generate.
    :return: (intent_base, list of truncated_intents)
    """
    parts = list(range(num_parts))

    # Parse into chain-style statements
    statements = _parse_to_chain_statements(raw_viba_code)
    if not statements:
        return ("", [""] * num_parts)

    # Classify each statement
    func_impl_flags = [_is_func_impl(stmt) for stmt in statements]

    # Build intent_base segments
    base_segments = []
    for i, stmt in enumerate(statements):
        if func_impl_flags[i]:
            # Func impl: header = exponent_chain[0:2] (result + first arg)
            chain = _get_exponent_chain(stmt)
            header = chain[0:2]
            base_segments.append(_unparse_chain_slice(stmt, header))
        else:
            # Class/data: full unparse
            base_segments.append(unparse([stmt]))

    intent_base = "\n".join(base_segments)

    # Build truncated intents for each part
    truncated_intents = []
    for part in parts:
        segments = []
        for i, stmt in enumerate(statements):
            if func_impl_flags[i]:
                chain = _get_exponent_chain(stmt)
                remaining = len(chain) - 2
                # Progressively reveal body: from header to full chain
                end = 2 + int((part + 1) * max(0, remaining) / num_parts)
                end = min(end, len(chain))
                segments.append(_unparse_chain_slice(stmt, chain[0:end]))
            else:
                segments.append(unparse([stmt]))
        truncated_intents.append("\n".join(segments))

    return (intent_base, truncated_intents)


if __name__ == "__main__":
    import unittest

    class TestIsFuncImpl(unittest.TestCase):

        def test_func_impl_detected(self):
            stmts = _parse_to_chain_statements("get_foo := $ret int <- $x str <- $y float")
            self.assertTrue(_is_func_impl(stmts[0]))

        def test_class_def_not_func(self):
            stmts = _parse_to_chain_statements("Person := $name str * $age int")
            self.assertFalse(_is_func_impl(stmts[0]))

        def test_invalid_viba(self):
            stmts = _parse_to_chain_statements("not valid viba !!!")
            self.assertEqual(stmts, [])

    class TestGetAllTruncatedVibeCode(unittest.TestCase):

        def test_single_class(self):
            code = "Person := $name str * $age int"
            intent_base, truncated = get_all_truncated_vibe_code(code, num_parts=3)
            self.assertIn("Person", intent_base)
            self.assertIn("$name", intent_base)
            self.assertEqual(len(truncated), 3)
            for t in truncated:
                self.assertIn("Person", t)
                self.assertIn("$name", t)

        def test_single_func(self):
            code = "get_foo := $ret int <- $x str <- $y float <- $z bool <- $w int"
            intent_base, truncated = get_all_truncated_vibe_code(code, num_parts=3)
            # Base: header = chain[0:2] = result + innermost arg
            self.assertIn("get_foo", intent_base)
            self.assertIn("$ret", intent_base)
            self.assertIn("$w", intent_base)
            # Header should not have outermost args
            self.assertNotIn("$x", intent_base)
            self.assertEqual(len(truncated), 3)
            # Progressive: last truncation has more content than first
            self.assertGreater(len(truncated[-1]), len(truncated[0]))
            # Last truncation should have all args
            self.assertIn("$x", truncated[-1])

        def test_mixed_class_and_func(self):
            code = "Person := $name str * $age int\ngreet := $ret str <- $name str <- $age int <- $greeting str"
            intent_base, truncated = get_all_truncated_vibe_code(code, num_parts=3)
            self.assertIn("Person", intent_base)
            self.assertIn("greet", intent_base)
            self.assertEqual(len(truncated), 3)
            for t in truncated:
                self.assertIn("Person", t)
                self.assertIn("greet", t)

        def test_progressive_truncation(self):
            # 5-element chain: result + 4 args (stored in reverse syntactic order)
            code = "compute := $ret int <- $a int <- $b int <- $c int <- $d int"
            intent_base, truncated = get_all_truncated_vibe_code(code, num_parts=3)
            # Header = chain[0:2] = result + innermost arg ($d)
            self.assertIn("$ret", intent_base)
            self.assertIn("$d", intent_base)
            self.assertNotIn("$a", intent_base)
            # Last truncation should have all args including outermost ($a)
            self.assertIn("$a", truncated[-1])
            # Lengths should be non-decreasing
            lengths = [len(t) for t in truncated]
            for i in range(len(lengths) - 1):
                self.assertLessEqual(lengths[i], lengths[i + 1])

        def test_empty_code(self):
            intent_base, truncated = get_all_truncated_vibe_code("", num_parts=3)
            self.assertEqual(intent_base, "")
            self.assertEqual(len(truncated), 3)
            for t in truncated:
                self.assertEqual(t, "")

        def test_short_func(self):
            # Only 2 chain elements (result + 1 arg) = header only
            code = "f := $ret int <- $x str"
            intent_base, truncated = get_all_truncated_vibe_code(code, num_parts=3)
            self.assertIn("$ret", intent_base)
            self.assertIn("$x", intent_base)

    unittest.main()

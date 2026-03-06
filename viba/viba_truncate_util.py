from viba.parser import parser as viba_parser
from viba.type import parse as viba_type_parse
from viba.chain import convert_to_chain_style
from viba.unparser import unparse
from viba.std_coding_style import (
    check_viba_std_coding_style,
    FuncImplementStdCodingStyle,
)


def _is_func_impl(vibe_segments: list[str]) -> bool:
    """Check if vibe_segments represents a FuncImplementStdCodingStyle definition."""
    code = "\n".join(vibe_segments)
    try:
        ast_dicts = viba_parser.parse(code)
        if not ast_dicts:
            return False
        types = viba_type_parse(ast_dicts)
        if not types:
            return False
        chained = convert_to_chain_style(types[0])
        result = check_viba_std_coding_style(chained)
        return result.success and isinstance(result.style, FuncImplementStdCodingStyle)
    except Exception:
        return False


def remove_all_comments(source_viba_code: str) -> str:
    """Remove all comments from viba code by round-tripping through parse → unparse."""
    try:
        ast_dicts = viba_parser.parse(source_viba_code)
        if not ast_dicts:
            return source_viba_code
        types = viba_type_parse(ast_dicts)
        return unparse(types)
    except Exception:
        return source_viba_code


def get_all_truncated_vibe_code(
    vibe_segments_list: list[list[str]],
    num_parts: int = 5,
) -> tuple[str, list[str]]:
    """
    Generate progressively truncated vibe code from a list of vibe segments.

    For each vibe_segments in vibe_segments_list:
      - If it is NOT a function implementation (class/data definition):
        its full content is always included (in both intent_base and every truncation level).
      - If it IS a function implementation:
        only the signature (vibe_segments[0]) goes into intent_base,
        and each truncation level reveals progressively more body segments.

    :param vibe_segments_list: List of vibe segments, each being a list of strings.
    :param num_parts: Number of truncation levels to generate.
    :return: (intent_base, list of truncated_intents)
    """
    parts = list(range(num_parts))

    # Classify each segment group
    func_impl_flags = [_is_func_impl(segs) for segs in vibe_segments_list]

    # Build intent_base: for non-func use full content, for func use only signature
    base_segments = []
    for i, vibe_segments in enumerate(vibe_segments_list):
        if func_impl_flags[i]:
            base_segments.append(vibe_segments[0])
        else:
            base_segments.append("\n".join(vibe_segments))

    intent_base = "\n".join(base_segments)

    # Build truncated intents for each part
    truncated_intents = []
    for part in parts:
        segments = []
        for i, vibe_segments in enumerate(vibe_segments_list):
            if func_impl_flags[i]:
                # Progressively reveal: vibe_segments[0:part+1]
                segments.append("\n".join(vibe_segments[0 : part + 1]))
            else:
                # Non-func: always full content
                segments.append("\n".join(vibe_segments))
        truncated_intents.append(remove_all_comments("\n".join(segments)))

    return (intent_base, truncated_intents)


if __name__ == "__main__":
    import unittest

    class TestRemoveAllComments(unittest.TestCase):

        def test_strips_comments(self):
            code = "Foo := $a int # this is a comment"
            result = remove_all_comments(code)
            self.assertNotIn("#", result)
            self.assertIn("Foo", result)

        def test_invalid_code_passthrough(self):
            code = "not valid !!!"
            self.assertEqual(remove_all_comments(code), code)

        def test_empty_string(self):
            self.assertEqual(remove_all_comments(""), "")

    class TestIsFuncImpl(unittest.TestCase):

        def test_func_impl_detected(self):
            # Curried exponent chain → FuncImplementStdCodingStyle
            segs = ["get_foo := $ret int <- $x str <- $y float"]
            self.assertTrue(_is_func_impl(segs))

        def test_class_def_not_func(self):
            # Product type with no exponent → ClassDefineStdCodingStyle
            segs = ["Person := $name str * $age int"]
            self.assertFalse(_is_func_impl(segs))

        def test_empty_segments(self):
            self.assertFalse(_is_func_impl([]))

        def test_invalid_viba(self):
            self.assertFalse(_is_func_impl(["not valid viba !!!"]))

    class TestGetAllTruncatedVibeCode(unittest.TestCase):

        def test_single_class_segment(self):
            segs_list = [["Person := $name str * $age int"]]
            intent_base, truncated = get_all_truncated_vibe_code(segs_list, num_parts=3)
            self.assertEqual(intent_base, "Person := $name str * $age int")
            # Non-func: every truncation level has the same content (unparsed form)
            self.assertEqual(len(truncated), 3)
            for t in truncated:
                self.assertIn("Person", t)
                self.assertIn("$name", t)

        def test_single_func_segment(self):
            segs_list = [
                [
                    "get_foo := $ret int <- $x str <- $y float",
                    "<- $z bool",
                    "<- $w int",
                ]
            ]
            intent_base, truncated = get_all_truncated_vibe_code(segs_list, num_parts=3)
            # Base: only signature
            self.assertEqual(intent_base, "get_foo := $ret int <- $x str <- $y float")
            self.assertEqual(len(truncated), 3)
            # part 0: signature only → parsed and unparsed without comments
            self.assertIn("get_foo", truncated[0])
            # part 1: signature + one more segment
            self.assertIn("get_foo", truncated[1])
            # part 2: all segments
            self.assertIn("get_foo", truncated[2])

        def test_mixed_class_and_func(self):
            segs_list = [
                ["Person := $name str * $age int"],  # class
                [
                    "greet := $ret str <- $name str <- $age int",
                    "<- $greeting str",
                    "<- $lang str",
                ],  # func
            ]
            intent_base, truncated = get_all_truncated_vibe_code(segs_list, num_parts=3)
            # Base: class full + func signature
            self.assertIn("Person", intent_base)
            self.assertIn("greet", intent_base)
            self.assertEqual(len(truncated), 3)
            # All truncation levels contain both definitions
            for t in truncated:
                self.assertIn("Person", t)
                self.assertIn("greet", t)

        def test_num_parts_exceeds_segments(self):
            segs_list = [
                ["compute := $ret int <- $x int", "<- $y float"]
            ]
            intent_base, truncated = get_all_truncated_vibe_code(segs_list, num_parts=5)
            self.assertEqual(len(truncated), 5)
            # part 3 and 4 should be same as full (slice beyond end is safe)
            self.assertEqual(truncated[3], truncated[4])

        def test_empty_segments_list(self):
            intent_base, truncated = get_all_truncated_vibe_code([], num_parts=3)
            self.assertEqual(intent_base, "")
            self.assertEqual(len(truncated), 3)
            for t in truncated:
                self.assertEqual(t, "")

    unittest.main()

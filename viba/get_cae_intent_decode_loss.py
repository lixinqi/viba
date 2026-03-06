import Levenshtein

def get_cae_intent_decode_loss(input_str: str, pairs: list[tuple[str, int]], beta: float = 1.0) -> float:
    """
    Compute the average decoding loss based on intent lengths and Levenshtein distances.

    :param input_str: The original input string x.
    :param pairs: List of (output_str, intent_len) pairs.
    :param beta: Weight coefficient for intent length.
    :return: Average of (beta * intent_len + levenshtein(x, output)) / (len(x) + 1)
    """
    n = len(pairs)
    if n == 0:
        return 0.0

    x_len = len(input_str)
    denominator = x_len + 1  # l(x) + 1

    total = 0.0
    for output_str, intent_len in pairs:
        d = Levenshtein.distance(input_str, output_str)
        total += (beta * intent_len + d) / denominator

    return total / n


if __name__ == "__main__":
    import unittest

    class TestCaeIntentDecodeLoss(unittest.TestCase):

        def test_empty_pairs(self):
            self.assertEqual(get_cae_intent_decode_loss("hello", []), 0.0)

        def test_single_pair_identical(self):
            result = get_cae_intent_decode_loss("hello", [("hello", 0)], beta=1.0)
            self.assertEqual(result, 0.0)

        def test_single_pair_with_intent_len(self):
            result = get_cae_intent_decode_loss("hi", [("hi", 3)], beta=2.0)
            self.assertAlmostEqual(result, 2.0, places=6)

        def test_single_pair_with_edit_distance(self):
            result = get_cae_intent_decode_loss("kitten", [("sitting", 0)], beta=1.0)
            expected = Levenshtein.distance("kitten", "sitting") / (len("kitten") + 1)
            self.assertAlmostEqual(result, expected, places=6)

        def test_multiple_pairs(self):
            pairs = [
                ("hello", 1),
                ("helo", 2),
                ("hallo", 3)
            ]
            result = get_cae_intent_decode_loss("hello", pairs, beta=1.0)
            # compute expected dynamically
            x_len = len("hello")
            denom = x_len + 1
            terms = []
            for out, ilen in pairs:
                d = Levenshtein.distance("hello", out)
                terms.append((1.0 * ilen + d) / denom)
            expected = sum(terms) / len(terms)
            self.assertAlmostEqual(result, expected, places=6)

        def test_beta_scaling(self):
            pairs = [("test", 5)]
            result_beta1 = get_cae_intent_decode_loss("test", pairs, beta=1.0)
            result_beta2 = get_cae_intent_decode_loss("test", pairs, beta=2.0)
            self.assertEqual(result_beta1, 1.0)
            self.assertEqual(result_beta2, 2.0)

        def test_zero_length_input(self):
            result = get_cae_intent_decode_loss("", [("a", 2)], beta=1.0)
            self.assertEqual(result, 3.0)

        def test_zero_length_output(self):
            result = get_cae_intent_decode_loss("abc", [("", 0)], beta=1.0)
            self.assertEqual(result, 0.75)

        def test_large_intent_len(self):
            result = get_cae_intent_decode_loss("x", [("x", 1000)], beta=1.0)
            self.assertEqual(result, 500.0)

        def test_mixed_pairs_precision(self):
            pairs = [("same", 0), ("different", 100)]
            result = get_cae_intent_decode_loss("same", pairs, beta=0.1)
            # compute expected dynamically
            x_len = len("same")
            denom = x_len + 1
            terms = []
            for out, ilen in pairs:
                d = Levenshtein.distance("same", out)
                terms.append((0.1 * ilen + d) / denom)
            expected = sum(terms) / len(terms)
            self.assertAlmostEqual(result, expected, places=6)

        def test_negative_beta(self):
            pairs = [("hello", 2)]
            result = get_cae_intent_decode_loss("hello", pairs, beta=-1.0)
            expected = (-2 + Levenshtein.distance("hello", "hello")) / (len("hello") + 1)
            self.assertAlmostEqual(result, expected, places=6)

    unittest.main()
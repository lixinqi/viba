from viba.get_cae_intent_decode_loss import get_cae_intent_decode_loss
from viba.get_cae_intent_norm_loss import get_cae_intent_norm_loss


def get_cae_loss(
    input_str: str,
    pairs: list[tuple[str, int]],
    intent_base_len: int,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """
    Compute the CAE (Code AutoEncoder) loss.

    :param input_str: The original input string.
    :param pairs: List of (output_str, intent_len) pairs.
    :param intent_base_len: Base length of the intent.
    :param alpha: Weight coefficient for norm_loss.
    :param beta: Weight coefficient passed to decode_loss.
    :return: decode_loss + alpha * norm_loss
    """
    decode_loss = get_cae_intent_decode_loss(input_str, pairs, beta)

    raw_intent_lens = [intent_len for _, intent_len in pairs]
    norm_loss = get_cae_intent_norm_loss(intent_base_len, raw_intent_lens)

    return decode_loss + alpha * norm_loss


if __name__ == "__main__":
    import unittest

    class TestGetCaeLoss(unittest.TestCase):

        def test_empty_pairs(self):
            result = get_cae_loss("hello", [], intent_base_len=0)
            self.assertEqual(result, 0.0)

        def test_single_pair_identical(self):
            # intent_len must be > intent_base_len per norm_loss assertion
            result = get_cae_loss("hello", [("hello", 1)], intent_base_len=0)
            decode = get_cae_intent_decode_loss("hello", [("hello", 1)], 1.0)
            # single element → norm_loss = 0
            self.assertAlmostEqual(result, decode, places=6)

        def test_alpha_zero_ignores_norm(self):
            pairs = [("hello", 10), ("hello", 20), ("hello", 30)]
            result = get_cae_loss("hello", pairs, intent_base_len=0, alpha=0.0, beta=1.0)
            expected = get_cae_intent_decode_loss("hello", pairs, 1.0)
            self.assertAlmostEqual(result, expected, places=6)

        def test_beta_zero_ignores_decode_intent(self):
            pairs = [("hello", 10), ("helo", 20), ("hallo", 30)]
            result = get_cae_loss("hello", pairs, intent_base_len=0, alpha=1.0, beta=0.0)
            decode = get_cae_intent_decode_loss("hello", pairs, 0.0)
            norm = get_cae_intent_norm_loss(0, [10, 20, 30])
            self.assertAlmostEqual(result, decode + norm, places=6)

        def test_default_alpha_beta(self):
            pairs = [("ab", 5), ("abc", 10), ("abcd", 20)]
            result = get_cae_loss("abc", pairs, intent_base_len=3)
            decode = get_cae_intent_decode_loss("abc", pairs, 1.0)
            norm = get_cae_intent_norm_loss(3, [5, 10, 20])
            self.assertAlmostEqual(result, decode + norm, places=6)

        def test_alpha_scaling(self):
            pairs = [("abc", 10), ("abcd", 20), ("abcde", 40)]
            r1 = get_cae_loss("abc", pairs, intent_base_len=5, alpha=1.0)
            r2 = get_cae_loss("abc", pairs, intent_base_len=5, alpha=2.0)
            decode = get_cae_intent_decode_loss("abc", pairs, 1.0)
            norm = get_cae_intent_norm_loss(5, [10, 20, 40])
            self.assertAlmostEqual(r1, decode + norm, places=6)
            self.assertAlmostEqual(r2, decode + 2.0 * norm, places=6)

        def test_uniform_intents_norm_zero(self):
            pairs = [("x", 10), ("xx", 20), ("xxx", 30)]
            result = get_cae_loss("x", pairs, intent_base_len=0, alpha=1.0, beta=1.0)
            decode = get_cae_intent_decode_loss("x", pairs, 1.0)
            self.assertAlmostEqual(result, decode, places=6)

        def test_components_add_up(self):
            pairs = [("ab", 6), ("abc", 15), ("a", 30)]
            decode = get_cae_intent_decode_loss("abc", pairs, 2.0)
            norm = get_cae_intent_norm_loss(3, [6, 15, 30])
            result = get_cae_loss("abc", pairs, intent_base_len=3, alpha=0.5, beta=2.0)
            self.assertAlmostEqual(result, decode + 0.5 * norm, places=6)

    unittest.main()

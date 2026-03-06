import math

def get_cae_intent_norm_loss(intent_base_len: int, raw_intent_lens: list[int], epsilon: float = 1e-12) -> float:
    """
    Compute the normalized intent score based on the distribution of intent lengths.

    The score measures how uneven the changes in intent lengths are across different truncation ratios.
    A value of 0 indicates perfectly uniform differences (all changes are identical), while 1 indicates
    maximal disparity.

    :param intent_base_len: Length of the full intent (used as the base for difference calculation)
    :param raw_intent_lens: List of intent lengths for various truncation ratios
    :param epsilon: Small value to avoid division by zero
    :return: A float between 0 and 1
    """
    # Enforce that every raw intent length is strictly greater than the base length
    for l in raw_intent_lens:
        assert l > intent_base_len, f"raw_intent_lens element {l} must be greater than intent_base_len {intent_base_len}"

    n = len(raw_intent_lens)
    if n <= 1:
        # With zero or one difference, disparity is undefined; return 0 (perfect uniformity)
        return 0.0

    # 1. Construct the full length sequence and compute adjacent differences
    intent_lens = [intent_base_len] + raw_intent_lens
    diffs = [intent_lens[i+1] - intent_lens[i] for i in range(n)]

    diff_sum = sum(diffs)

    # 2. Compute probability distribution (handle the special case where all diffs are zero)
    if diff_sum == 0:
        # All diffs are zero → uniform distribution
        probs = [1.0 / n] * n
    else:
        probs = [d / (diff_sum + epsilon) for d in diffs]

    # 3. Calculate information entropy (using natural log)
    entropy = -sum(p * math.log(p) for p in probs if p > 0)

    # 4. Final score: 1 - normalized entropy
    #    Normalized entropy = entropy / log(n), so score = 1 - entropy / log(n)
    score = 1 - entropy / math.log(n)
    return score
if __name__ == "__main__":
    # Test cases for get_cae_intent_norm_score (all raw lengths > base length)
    print("Testing get_cae_intent_norm_score...\n")

    # Test 1: Uniform differences (all diffs equal) → score 0
    base1 = 100
    raw1 = [110, 120, 130]          # diffs: [10, 10, 10]
    score1 = get_cae_intent_norm_score(base1, raw1)
    print(f"Test 1: base={base1}, raw={raw1} -> score={score1:.6f}  (expected ≈ 0.0)")
    assert abs(score1) < 1e-9, "Uniform differences should yield score 0"

    # Test 2: Maximal disparity (one diff dominates) → score very close to 1
    base2 = 100
    raw2 = [101, 102, 1000000]      # diffs: [1, 1, 999900] → probability ~ [1e-6, 1e-6, 0.999998]
    score2 = get_cae_intent_norm_score(base2, raw2)
    print(f"Test 2: base={base2}, raw={raw2} -> score={score2:.6f}  (expected > 0.99)")
    assert score2 > 0.999, "Highly uneven diffs should yield score extremely close to 1"

    # Test 3: Mixed diffs – moderate disparity
    base3 = 50
    raw3 = [60, 70, 80, 100]        # diffs: [10, 10, 10, 20]
    score3 = get_cae_intent_norm_score(base3, raw3)
    print(f"Test 3: base={base3}, raw={raw3} -> score={score3:.6f}  (expected between 0 and 1)")

    # Test 4: All diffs zero cannot occur due to assertion (raw > base)
    # So omitted.

    # Test 5: Single element in raw list → n=1 → return 0 (edge case)
    base5 = 100
    raw5 = [110]                     # only one diff: [10]
    score5 = get_cae_intent_norm_score(base5, raw5)
    print(f"Test 5: base={base5}, raw={raw5} -> score={score5:.6f}  (expected 0.0 by definition)")
    assert score5 == 0.0, "Single diff should return 0"

    # Test 6: Empty raw list → n=0 → return 0
    base6 = 100
    raw6 = []
    score6 = get_cae_intent_norm_score(base6, raw6)
    print(f"Test 6: base={base6}, raw={raw6} -> score={score6:.6f}  (expected 0.0)")
    assert score6 == 0.0, "Empty list should return 0"

    # Test 7: Very small epsilon effect (optional)
    base7 = 100
    raw7 = [110, 120, 1000000]
    score7a = get_cae_intent_norm_score(base7, raw7, epsilon=1e-12)
    score7b = get_cae_intent_norm_score(base7, raw7, epsilon=1e-6)
    print(f"Test 7: epsilon variation -> score7a={score7a:.6f}, score7b={score7b:.6f} (should be nearly identical)")

    # Test 8: Assertion failure demonstration (uncomment to see error)
    # base_bad = 100
    # raw_bad = [90, 120, 130]  # 90 is not > 100
    # score_bad = get_cae_intent_norm_score(base_bad, raw_bad)

    print("\nAll tests passed (assertions satisfied).")
import math

def get_cae_intent_norm_loss(intent_base_len: int, raw_intent_lens: list[int], epsilon: float = 1e-12) -> float:
    """
    Compute the normalized intent norm loss based on the distribution of log-scaled
    intent length differences.

    The loss measures how uneven the changes in intent lengths are across different
    truncation levels. A value of 0 indicates perfectly uniform log-differences
    (all changes are identical in log scale), while 1 indicates maximal disparity.

    :param intent_base_len: Length of the full intent (used as the base for difference calculation)
    :param raw_intent_lens: List of intent lengths for various truncation levels
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

    # 1. Construct the full length sequence and compute log-scaled adjacent differences
    intent_lens = [intent_base_len, *raw_intent_lens]
    log_diffs = [math.log(intent_lens[i+1] - intent_lens[i] + 1) for i in range(n)]

    diff_sum = sum(log_diffs)

    # 2. Compute probability distribution
    if diff_sum == 0:
        # All log_diffs are zero → uniform distribution
        probabilities = [1.0 / n] * n
    else:
        probabilities = [d / (diff_sum + epsilon) for d in log_diffs]

    # 3. Calculate information entropy (using natural log)
    normalized_entropy = -sum(p * math.log(p) for p in probabilities if p > 0)

    # 4. Final score: 1 - normalized entropy / log(n)
    return 1 - normalized_entropy / math.log(n)


if __name__ == "__main__":
    print("Testing get_cae_intent_norm_loss...\n")

    # Test 1: Uniform differences (all diffs equal) → loss 0
    base1 = 100
    raw1 = [110, 120, 130]          # diffs: [10, 10, 10], log_diffs: [log(11), log(11), log(11)]
    loss1 = get_cae_intent_norm_loss(base1, raw1)
    print(f"Test 1: base={base1}, raw={raw1} -> loss={loss1:.6f}  (expected ~ 0.0)")
    assert abs(loss1) < 1e-9, "Uniform differences should yield loss 0"

    # Test 2: Maximal disparity (one diff dominates) → loss close to 1
    base2 = 100
    raw2 = [101, 102, 1000000]      # diffs: [1, 1, 999898], log_diffs: [log(2), log(2), log(999899)]
    loss2 = get_cae_intent_norm_loss(base2, raw2)
    print(f"Test 2: base={base2}, raw={raw2} -> loss={loss2:.6f}  (expected > 0.5)")
    assert loss2 > 0.5, "Highly uneven diffs should yield high loss"

    # Test 3: Mixed diffs – moderate disparity
    base3 = 50
    raw3 = [60, 70, 80, 100]        # diffs: [10, 10, 10, 20]
    loss3 = get_cae_intent_norm_loss(base3, raw3)
    print(f"Test 3: base={base3}, raw={raw3} -> loss={loss3:.6f}  (expected between 0 and 1)")
    assert 0.0 < loss3 < 1.0, "Mixed diffs should yield moderate loss"

    # Test 4: Log scaling reduces disparity compared to raw diffs
    # With raw diffs [1, 1, 999898], log diffs are [log(2), log(2), log(999899)] ≈ [0.69, 0.69, 13.8]
    # This is far less extreme than the raw ratio, so loss should be < 1.0
    assert loss2 < 1.0, "Log scaling should reduce extreme disparity"
    print(f"Test 4: log scaling reduces disparity: loss2={loss2:.6f} < 1.0")

    # Test 5: Single element in raw list → n=1 → return 0 (edge case)
    base5 = 100
    raw5 = [110]
    loss5 = get_cae_intent_norm_loss(base5, raw5)
    print(f"Test 5: base={base5}, raw={raw5} -> loss={loss5:.6f}  (expected 0.0 by definition)")
    assert loss5 == 0.0, "Single diff should return 0"

    # Test 6: Empty raw list → n=0 → return 0
    base6 = 100
    raw6 = []
    loss6 = get_cae_intent_norm_loss(base6, raw6)
    print(f"Test 6: base={base6}, raw={raw6} -> loss={loss6:.6f}  (expected 0.0)")
    assert loss6 == 0.0, "Empty list should return 0"

    # Test 7: Epsilon variation
    base7 = 100
    raw7 = [110, 120, 1000000]
    loss7a = get_cae_intent_norm_loss(base7, raw7, epsilon=1e-12)
    loss7b = get_cae_intent_norm_loss(base7, raw7, epsilon=1e-6)
    print(f"Test 7: epsilon variation -> loss7a={loss7a:.6f}, loss7b={loss7b:.6f} (should be nearly identical)")
    assert abs(loss7a - loss7b) < 0.01, "Epsilon should have minimal effect"

    # Test 8: Assertion failure (uncomment to see error)
    # base_bad = 100
    # raw_bad = [90, 120, 130]  # 90 is not > 100
    # get_cae_intent_norm_loss(base_bad, raw_bad)

    print("\nAll tests passed.")
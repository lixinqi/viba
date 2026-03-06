"""
VIBA CAE Intent Validator Test Suite

Tests the CaeIntentValidator with 50 test cases:
- All exponent chains are in short list form
- 30 exponent chain patterns
- 20 complex type definitions without exponent types
"""

from viba.cae_intent_validator import CaeIntentValidator, validate_cae_intent, validate_from_string


# Test data - 50 test cases
TEST_CASES = [
    # =================================================================
    # 1-10: Simple exponent chains (all in short list form)
    # =================================================================
    {
        "name": "Single exponent",
        "input": [["F := B <- A"]],
        "expect_success": True,
    },
    {
        "name": "Double exponent",
        "input": [["F := C", "<- B", "<- A"]],
        "expect_success": True,
    },
    {
        "name": "Triple exponent",
        "input": [["F := D", "<- C", "<- B", "<- A"]],
        "expect_success": True,
    },
    {
        "name": "Four-element exponent",
        "input": [["F := E", "<- D", "<- C", "<- B", "<- A"]],
        "expect_success": True,
    },
    {
        "name": "Five-element exponent",
        "input": [["F := F", "<- E", "<- D", "<- C", "<- B", "<- A"]],
        "expect_success": True,
    },
    {
        "name": "Exponent with tagged input",
        "input": [["F := Result", "<- $x A", "<- $y B"]],
        "expect_success": True,
    },
    {
        "name": "Exponent with multiple tagged inputs",
        "input": [["F := Result", "<- $a A", "<- $b B", "<- $c C"]],
        "expect_success": True,
    },
    {
        "name": "Exponent with void return",
        "input": [["F := void", "<- A", "<- B"]],
        "expect_success": True,
    },
    {
        "name": "Exponent returning sum type",
        "input": [["F := (A | B)", "<- Input"]],
        "expect_success": True,
    },
    {
        "name": "Exponent returning product type",
        "input": [["F := (X * Y)", "<- E", "<- D", "<- C", "<- B", "<- A"]],
        "expect_success": True,
    },

    # =================================================================
    # 11-20: Long exponent chains (short list form)
    # =================================================================
    {
        "name": "Six-element exponent chain",
        "input": [["F := G", "<- F", "<- E", "<- D", "<- C", "<- B", "<- A"]],
        "expect_success": True,
    },
    {
        "name": "Seven-element exponent chain",
        "input": [["F := H", "<- G", "<- F", "<- E", "<- D", "<- C", "<- B", "<- A"]],
        "expect_success": True,
    },
    {
        "name": "Eight-element exponent chain",
        "input": [["F := I", "<- H", "<- G", "<- F", "<- E", "<- D", "<- C", "<- B", "<- A"]],
        "expect_success": True,
    },
    {
        "name": "Nine-element exponent chain",
        "input": [["F := J", "<- I", "<- H", "<- G", "<- F", "<- E", "<- D", "<- C", "<- B", "<- A"]],
        "expect_success": True,
    },
    {
        "name": "Ten-element exponent chain",
        "input": [["F := K", "<- J", "<- I", "<- H", "<- G", "<- F", "<- E", "<- D", "<- C", "<- B", "<- A"]],
        "expect_success": True,
    },
    {
        "name": "Exponent chain with type apps",
        "input": [["F := List[Result]", "<- List[A]", "<- List[B]"]],
        "expect_success": True,
    },
    {
        "name": "Curried generic chain",
        "input": [["F := List[Z]", "<- List[Y]", "<- List[X]"]],
        "expect_success": True,
    },
    {
        "name": "Exponent chain with all tagged",
        "input": [["F := R", "<- $a A", "<- $b B", "<- $c C"]],
        "expect_success": True,
    },
    {
        "name": "Exponent nested in product",
        "input": [["F := (A <- X) * (B <- Y)"]],
        "expect_success": False,  # Nested exponents in product not standard coding style
    },
    {
        "name": "Parenthesis mismatch",
        "input": [["C := (List[B]", "<- (B <- A))", "<- List[A]"]],
        "expect_success": False,
    },
    {
        "name": "Map function",
        "input": [["Map := List[B]", "<- (B <- A)", "<- List[A]"]],
        "expect_success": True,
    },

    # =================================================================
    # 21-30: Cumulative exponent chain building
    # =================================================================
    {
        "name": "Cumulative step 1",
        "input": [["Stage1 := B <- A"]],
        "expect_success": True,
    },
    {
        "name": "Cumulative step 2",
        "input": [["Stage1 := B <- A"], ["Stage2 := C <- Stage1"]],
        "expect_success": True,
    },
    {
        "name": "Cumulative step 3",
        "input": [["Stage1 := B <- A"], ["Stage2 := C <- Stage1"], ["Stage3 := D <- Stage2"]],
        "expect_success": True,
    },
    {
        "name": "Cumulative step 4",
        "input": [["S1 := B <- A"], ["S2 := C <- S1"], ["S3 := D <- S2"], ["S4 := E <- S3"]],
        "expect_success": True,
    },
    {
        "name": "Cumulative step 5",
        "input": [["S1 := B <- A"], ["S2 := C <- S1"], ["S3 := D <- S2"], ["S4 := E <- S3"], ["S5 := F <- S4"]],
        "expect_success": True,
    },
    {
        "name": "Cumulative step 6",
        "input": [["S1 := B <- A"], ["S2 := C <- S1"], ["S3 := D <- S2"], ["S4 := E <- S3"], ["S5 := F <- S4"], ["S6 := G <- S5"]],
        "expect_success": True,
    },
    {
        "name": "Cumulative step 7",
        "input": [["S1 := B <- A"], ["S2 := C <- S1"], ["S3 := D <- S2"], ["S4 := E <- S3"], ["S5 := F <- S4"], ["S6 := G <- S5"], ["S7 := H <- S6"]],
        "expect_success": True,
    },
    {
        "name": "Cumulative step 8",
        "input": [["S1 := B <- A"], ["S2 := C <- S1"], ["S3 := D <- S2"], ["S4 := E <- S3"], ["S5 := F <- S4"], ["S6 := G <- S5"], ["S7 := H <- S6"], ["S8 := I <- S7"]],
        "expect_success": True,
    },
    {
        "name": "Cumulative step 9",
        "input": [["S1 := B <- A"], ["S2 := C <- S1"], ["S3 := D <- S2"], ["S4 := E <- S3"], ["S5 := F <- S4"], ["S6 := G <- S5"], ["S7 := H <- S6"], ["S8 := I <- S7"], ["S9 := J <- S8"]],
        "expect_success": True,
    },
    {
        "name": "Cumulative step 10",
        "input": [["S1 := B <- A"], ["S2 := C <- S1"], ["S3 := D <- S2"], ["S4 := E <- S3"], ["S5 := F <- S4"], ["S6 := G <- S5"], ["S7 := H <- S6"], ["S8 := I <- S7"], ["S9 := J <- S8"], ["S10 := K <- S9"]],
        "expect_success": True,
    },

    # =================================================================
    # 31-50: Complex type definitions without exponent types
    # =================================================================
    {
        "name": "Product type with three fields",
        "input": [["Person := $name String * $age Int * $active Bool"]],
        "expect_success": True,
    },
    {
        "name": "Product type with many fields",
        "input": [["Config := $host String * $port Int * $timeout Int * $retry Int * $debug Bool"]],
        "expect_success": True,
    },
    {
        "name": "Nested product type",
        "input": [["Address := $city String * $coords (Lat * Lon)"]],
        "expect_success": True,
    },
    {
        "name": "Deep nested product",
        "input": [["Complex := ((A * B) * C) * (D * (E * F))"]],
        "expect_success": True,
    },
    {
        "name": "Sum type with three variants",
        "input": [["Color := $red Int | $green Int | $blue Int"]],
        "expect_success": True,
    },
    {
        "name": "Sum type with many variants",
        "input": [["Status := $pending | $running | $completed | $failed | $cancelled"]],
        "expect_success": True,
    },
    {
        "name": "Nested sum type",
        "input": [["Either := (A | B) | (C | D)"]],
        "expect_success": True,
    },
    {
        "name": "Option-like sum type",
        "input": [["Option := $some T | void"]],
        "expect_success": True,
    },
    {
        "name": "Result-like sum type",
        "input": [["Result := $ok T | $err E"]],
        "expect_success": True,
    },
    {
        "name": "Recursive list definition",
        "input": [["List := $cons (Item * List) | void"]],
        "expect_success": True,
    },
    {
        "name": "Binary tree definition",
        "input": [["Tree := $leaf T | $node (Tree * Tree)"]],
        "expect_success": True,
    },
    {
        "name": "Expression definition",
        "input": [["Expr := $lit Int | $add (Expr * Expr) | $mul (Expr * Expr)"]],
        "expect_success": True,
    },
    {
        "name": "Product with type apps",
        "input": [["Pair := $first List[A] * $second Map[B, C]"]],
        "expect_success": True,
    },
    {
        "name": "Sum with type apps",
        "input": [["Response := $success List[A] | $error String"]],
        "expect_success": True,
    },
    {
        "name": "Mixed tagged and untagged product",
        "input": [["Mixed := A * $label String * B * $id Int"]],
        "expect_success": True,
    },
    {
        "name": "Product with literals",
        "input": [["Constants := 42 * 3.14 * 'text' * true"]],
        "expect_success": True,
    },
    {
        "name": "Sum with literals",
        "input": [["Literal := 42 | 3.14 | 'text' | true | void"]],
        "expect_success": True,
    },
    {
        "name": "State monad style product",
        "input": [["StateResult := Result * State"]],
        "expect_success": True,
    },
    {
        "name": "Maybe chain sum",
        "input": [["Maybe := $just A | $just B | void"]],
        "expect_success": True,
    },
    {
        "name": "Complex nested structure",
        "input": [["Complex := ($ok (List[A] * B)) | ($err (C * String))"]],
        "expect_success": True,
    },
]


# =================================================================
# Tests
# =================================================================


def test_all_cases():
    """Run all 50 test cases."""
    validator = CaeIntentValidator()

    for i, case in enumerate(TEST_CASES, 1):
        result = validator(case["input"])

        if case["expect_success"]:
            assert result.success, (
                f"Test {i} '{case['name']}' failed: {result.error_msg}\n"
                f"Input: {case['input']}"
            )
        else:
            assert not result.success, (
                f"Test {i} '{case['name']}' should have failed but succeeded"
            )


# =================================================================
# Convenience function tests
# =================================================================


def test_validate_from_string_simple():
    """Test validate_from_string with simple input."""
    result = validate_from_string("F := B <- A")
    assert result.success, f"Failed: {result.error_msg}"


def test_validate_from_string_chain():
    """Test validate_from_string with chain."""
    result = validate_from_string("Pipeline := D, <- C, <- B, <- A")
    assert result.success, f"Failed: {result.error_msg}"


def test_validate_cae_intent_convenience():
    """Test the validate_cae_intent convenience function."""
    result = validate_cae_intent([["F := B <- A"]])
    assert result.success, f"Failed: {result.error_msg}"


# =================================================================
# Run tests
# =================================================================


if __name__ == "__main__":
    print("Running CAE Intent Validator Test Suite...\n")
    print("=" * 70)

    passed = 0
    failed = 0

    for i, case in enumerate(TEST_CASES, 1):
        validator = CaeIntentValidator()
        result = validator(case["input"])

        if result.success == case["expect_success"]:
            print(f"✓ Test {i}: {case['name']}")
            passed += 1
        else:
            print(f"✗ Test {i}: {case['name']}")
            print(f"  Expected: {case['expect_success']}, Got: {result.success}")
            if result.error_msg:
                print(f"  Error: {result.error_msg}")
            failed += 1

    print("=" * 70)
    print(f"Passed: {passed}/{len(TEST_CASES)}")
    print(f"Failed: {failed}/{len(TEST_CASES)}")

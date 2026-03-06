#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

PASS=0
FAIL=0

assert_eq() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $desc"
        echo "  expected: $expected"
        echo "  actual:   $actual"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    local desc="$1" needle="$2" haystack="$3"
    if echo "$haystack" | grep -q "$needle"; then
        echo "PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $desc"
        echo "  expected to contain: $needle"
        echo "  actual: $haystack"
        FAIL=$((FAIL + 1))
    fi
}

# Test 1: class-only input
echo '[["Person := $name str * $age int"]]' > "$TMPDIR/class_only.json"
OUT=$(python -m viba.get_truncated_intents "$TMPDIR/class_only.json" 2>/dev/null)
INTENT_BASE=$(echo "$OUT" | python -c "import sys,json; print(json.load(sys.stdin)['intent_base'])")
NUM_TRUNCATED=$(echo "$OUT" | python -c "import sys,json; print(len(json.load(sys.stdin)['truncated_intents']))")
assert_eq "class-only: intent_base contains Person" "Person := \$name str * \$age int" "$INTENT_BASE"
assert_eq "class-only: 5 truncation levels (default)" "5" "$NUM_TRUNCATED"

# Test 2: func-impl input
echo '[["get_foo := $ret int <- $x str <- $y float", "<- $z bool"]]' > "$TMPDIR/func.json"
OUT=$(python -m viba.get_truncated_intents "$TMPDIR/func.json" 2>/dev/null)
INTENT_BASE=$(echo "$OUT" | python -c "import sys,json; print(json.load(sys.stdin)['intent_base'])")
assert_eq "func: intent_base is signature only" "get_foo := \$ret int <- \$x str <- \$y float" "$INTENT_BASE"

# Test 3: mixed class + func
cat > "$TMPDIR/mixed.json" <<'EOF'
[
  ["Color := $r int * $g int * $b int"],
  ["render := $ret str <- $color Color <- $fmt str", "<- $quality float"]
]
EOF
OUT=$(python -m viba.get_truncated_intents "$TMPDIR/mixed.json" 2>/dev/null)
INTENT_BASE=$(echo "$OUT" | python -c "import sys,json; print(json.load(sys.stdin)['intent_base'])")
assert_contains "mixed: intent_base has Color" "Color" "$INTENT_BASE"
assert_contains "mixed: intent_base has render" "render" "$INTENT_BASE"
T0=$(echo "$OUT" | python -c "import sys,json; print(json.load(sys.stdin)['truncated_intents'][0])")
T4=$(echo "$OUT" | python -c "import sys,json; print(json.load(sys.stdin)['truncated_intents'][4])")
assert_contains "mixed: truncated[0] has Color" "Color" "$T0"
assert_contains "mixed: truncated[4] has quality" "quality" "$T4"

# Test 4: empty input
echo '[]' > "$TMPDIR/empty.json"
OUT=$(python -m viba.get_truncated_intents "$TMPDIR/empty.json" 2>/dev/null)
INTENT_BASE=$(echo "$OUT" | python -c "import sys,json; print(json.load(sys.stdin)['intent_base'])")
assert_eq "empty: intent_base is empty" "" "$INTENT_BASE"

# Test 5: output is valid JSON
echo '[["A := B"]]' > "$TMPDIR/json_check.json"
if python -m viba.get_truncated_intents "$TMPDIR/json_check.json" 2>/dev/null | python -m json.tool > /dev/null 2>&1; then
    echo "PASS: output is valid JSON"
    PASS=$((PASS + 1))
else
    echo "FAIL: output is not valid JSON"
    FAIL=$((FAIL + 1))
fi

echo "---"
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

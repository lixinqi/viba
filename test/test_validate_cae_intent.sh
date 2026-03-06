#!/bin/bash
set -euo pipefail

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

PASS=0
FAIL=0

assert_exit() {
    local desc="$1" expected_exit="$2" actual_exit="$3"
    if [ "$expected_exit" = "$actual_exit" ]; then
        echo "PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $desc"
        echo "  expected exit: $expected_exit"
        echo "  actual exit:   $actual_exit"
        FAIL=$((FAIL + 1))
    fi
}

assert_stderr_contains() {
    local desc="$1" needle="$2" stderr_file="$3"
    if grep -q "$needle" "$stderr_file"; then
        echo "PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $desc"
        echo "  expected stderr to contain: $needle"
        echo "  actual stderr: $(cat "$stderr_file")"
        FAIL=$((FAIL + 1))
    fi
}

# Test 1: valid class definition
echo '[["Person := $name str * $age int"]]' > "$TMPDIR/t1.json"
python -m viba.validate_cae_intent "$TMPDIR/t1.json" 2>/dev/null; RC=$?
assert_exit "valid class definition" 0 "$RC"

# Test 2: valid function declaration
echo '[["ParseInt := Int <- String"]]' > "$TMPDIR/t2.json"
python -m viba.validate_cae_intent "$TMPDIR/t2.json" 2>/dev/null; RC=$?
assert_exit "valid function declaration" 0 "$RC"

# Test 3: valid function implementation (exponent chain)
echo '[["CurriedFunc := C <- B <- A"]]' > "$TMPDIR/t3.json"
python -m viba.validate_cae_intent "$TMPDIR/t3.json" 2>/dev/null; RC=$?
assert_exit "valid function implementation" 0 "$RC"

# Test 4: valid sum type
echo '[["Color := $red Int | $green Int | $blue Int"]]' > "$TMPDIR/t4.json"
python -m viba.validate_cae_intent "$TMPDIR/t4.json" 2>/dev/null; RC=$?
assert_exit "valid sum type" 0 "$RC"

# Test 5: valid tagged result type
echo '[["Result := $ok T | $err E"]]' > "$TMPDIR/t5.json"
python -m viba.validate_cae_intent "$TMPDIR/t5.json" 2>/dev/null; RC=$?
assert_exit "valid tagged result type" 0 "$RC"

# Test 6: multiple valid statements
cat > "$TMPDIR/t6.json" <<'EOF'
[["Enc := Encoded <- Code"], ["Dec := Code <- Encoded"], ["AE := Enc * Dec"]]
EOF
python -m viba.validate_cae_intent "$TMPDIR/t6.json" 2>/dev/null; RC=$?
assert_exit "multiple valid statements" 0 "$RC"

# Test 7: cumulative multi-line statement
cat > "$TMPDIR/t7.json" <<'EOF'
[["A := Int", "B := String", "C := Bool"]]
EOF
python -m viba.validate_cae_intent "$TMPDIR/t7.json" 2>/dev/null; RC=$?
assert_exit "cumulative multi-line statement" 0 "$RC"

# Test 8: invalid syntax → exit 1
echo '[["not valid code"]]' > "$TMPDIR/t8.json"
RC=0; python -m viba.validate_cae_intent "$TMPDIR/t8.json" 2>"$TMPDIR/t8.err" || RC=$?
assert_exit "invalid syntax exits 1" 1 "$RC"
assert_stderr_contains "invalid syntax has error msg" "error" "$TMPDIR/t8.err"

# Test 9: empty segments list → exit 0
echo '[]' > "$TMPDIR/t9.json"
python -m viba.validate_cae_intent "$TMPDIR/t9.json" 2>/dev/null; RC=$?
assert_exit "empty list succeeds" 0 "$RC"

# Test 10: valid option type
echo '[["Option := $some T | void"]]' > "$TMPDIR/t10.json"
python -m viba.validate_cae_intent "$TMPDIR/t10.json" 2>/dev/null; RC=$?
assert_exit "valid option type" 0 "$RC"

# Test 11: valid with literals
echo '[["Config := $version 1.0 * $name str"]]' > "$TMPDIR/t11.json"
python -m viba.validate_cae_intent "$TMPDIR/t11.json" 2>/dev/null; RC=$?
assert_exit "valid with literals" 0 "$RC"

# Test 12: valid exponent with tags
echo '[["Handler := $ret Response <- $req Request <- $ctx Context"]]' > "$TMPDIR/t12.json"
python -m viba.validate_cae_intent "$TMPDIR/t12.json" 2>/dev/null; RC=$?
assert_exit "valid exponent with tags" 0 "$RC"

echo "---"
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]

# cae_intent_validator.py
# Code Autoencoder Intent Validator
# Auto-generated from cae_intent_validator.viba

from viba.parser import parser
from viba.type import parse as type_parse
from viba.chain import convert_to_chain_style
from viba.std_coding_style import check_viba_std_coding_style
from dataclasses import dataclass
from typing import Optional


# =================================================================
# Result type
# =================================================================


@dataclass
class ValidationResult:
    """Result of CAE intent validation."""

    success: bool
    error_msg: Optional[str] = None


# =================================================================
# CAE Intent Validator
# =================================================================


class CaeIntentValidator:
    """
    CaeIntentValidator implemented according to the Viba DSL description.

    - __init__: no arguments, returns void (None).
    - __call__: receives list[list[str]], each inner list is a raw statement block.
                Returns None on success, otherwise returns an error message string.
    """

    def __init__(self):
        """Initialize the CAE intent validator."""
        pass

    def __call__(self, raw_statements_list):
        """
        Validate CAE intent from raw VIBA statements.

        Parameters:
            raw_statements_list: list[list[str]], each inner list is a raw statement

        Returns:
            None if all statements pass validation,
            str error message if any validation fails.
        """
        # Inline step: generate cumulative statements
        cumulative_statements = []
        for raw_statement in raw_statements_list:
            # Generate cumulative statements for this raw_statement
            # Corresponds to: "\n".join(raw_statment[0:i+1] for i in range(len(raw_statent)))
            cum_for_one = [
                "\n".join(raw_statement[: i + 1]) for i in range(len(raw_statement))
            ]
            cumulative_statements.extend(cum_for_one)

        # Parse and check each cumulative statement
        for cum_str in cumulative_statements:
            # Parse chain: parser.parse -> type.parse -> chain.convert_to_chain_style
            parsed = parser.parse(cum_str)
            if parsed is None:
                # Invalid parse result - return error
                return ValidationResult(success=False, error_msg=f"Parse error: {cum_str}")
            typed = type_parse(parsed)
            if typed is None or len(typed) == 0:
                # Invalid type parse result - return error
                return ValidationResult(success=False, error_msg=f"Type parse error: {cum_str}")

            # Convert each type to chain style and check
            for t in typed:
                viba_code = convert_to_chain_style(t)

                # Check standard coding style
                check_result = check_viba_std_coding_style(viba_code)

                if not check_result.success:
                    # Return error message (corresponds to $error_msg str)
                    return ValidationResult(success=False, error_msg=check_result.message)

        # All succeeded (corresponds to $success void)
        return ValidationResult(success=True, error_msg=None)


# =================================================================
# Convenience functions
# =================================================================


def validate_cae_intent(raw_viba_code):
    """
    Convenience function to validate CAE intent.

    Parameters:
        raw_viba_code: list[list[str]] - raw VIBA statements

    Returns:
        None on success, error message string on failure
    """
    validator = CaeIntentValidator()
    return validator(raw_viba_code)


def validate_from_string(viba_code):
    """
    Validate CAE intent from a single VIBA code string.

    Parameters:
        viba_code: VIBA code as a single string

    Returns:
        None on success, error message string on failure
    """
    # Split into lines and wrap as list of statements
    lines = [line.strip() for line in viba_code.strip().split("\n") if line.strip()]
    raw_viba_code = [[line] for line in lines]

    validator = CaeIntentValidator()
    return validator(raw_viba_code)


# =================================================================
# Example usage and tests
# =================================================================


if __name__ == "__main__":
    print("Testing CAE Intent Validator...\n")

    # Example 1: Valid class definition (Product type)
    print("1. Valid class definition (Product type):")
    valid_class = [
        ["Person := $name String * $age Int * $active Bool"],
    ]
    result = validate_cae_intent(valid_class)
    if result is None:
        print("  ✓ Success")
    else:
        print(f"  ✗ Failed: {result}")

    # Example 2: Valid function declaration (Exponent type)
    print("\n2. Valid function declaration (Exponent type):")
    valid_func = [
        ["ParseInt := Int <- String"],
    ]
    result = validate_cae_intent(valid_func)
    if result is None:
        print("  ✓ Success")
    else:
        print(f"  ✗ Failed: {result}")

    # Example 3: Valid function implementation (Exponent chain)
    print("\n3. Valid function implementation (Exponent chain):")
    valid_impl = [
        ["CurriedFunc := C <- B <- A"],
    ]
    result = validate_cae_intent(valid_impl)
    if result is None:
        print("  ✓ Success")
    else:
        print(f"  ✗ Failed: {result}")

    # Example 4: Valid sum type (Class definition)
    print("\n4. Valid sum type (Class definition):")
    valid_sum = [
        ["Color := $red Int | $green Int | $blue Int"],
    ]
    result = validate_cae_intent(valid_sum)
    if result is None:
        print("  ✓ Success")
    else:
        print(f"  ✗ Failed: {result}")

    # Example 5: Cumulative statements (multi-line within one statement)
    print("\n5. Cumulative statements (single statement with multiple lines):")
    cumulative = [
        ["A := Int", "B := String", "C := Bool"],
    ]
    result = validate_cae_intent(cumulative)
    if result is None:
        print("  ✓ Success")
    else:
        print(f"  ✗ Failed: {result}")

    # Example 6: Using validate_from_string
    print("\n6. Using validate_from_string:")
    viba_string = """
    Option := $some T | void
    """
    result = validate_from_string(viba_string)
    if result is None:
        print("  ✓ Success")
    else:
        print(f"  ✗ Failed: {result}")

    # Example 7: Multi-stage pipeline with cumulative per-statement validation
    print("\n7. Multi-stage pipeline (cumulative across statements):")
    pipeline = [
        ["Enc := Encoded <- Code"],
        ["Dec := Code <- Encoded"],
        ["AE := Enc * Dec"],
    ]
    result = validate_cae_intent(pipeline)
    if result is None:
        print("  ✓ Success")
    else:
        print(f"  ✗ Failed: {result}")

    print("\nAll tests completed.")

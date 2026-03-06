import argparse
import json
import sys

from viba.cae_intent_validator import CaeIntentValidator


def main():
    parser = argparse.ArgumentParser(description="Validate CAE intent segments.")
    parser.add_argument("file", help="Path to JSON file containing list of intent segments (list[list[str]])")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        intent_segments: list[list[str]] = json.load(f)

    validator = CaeIntentValidator()
    result = validator(intent_segments)

    if result.success:
        sys.exit(0)
    else:
        print(result.error_msg, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

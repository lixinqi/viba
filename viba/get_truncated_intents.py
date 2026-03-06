import argparse
import json
import sys

from viba.intent_truncate_util import get_all_truncated_vibe_code


def main():
    parser = argparse.ArgumentParser(description="Generate truncated intents from vibe segments.")
    parser.add_argument("file", help="Path to JSON file containing list of intent segments (list[list[str]])")
    args = parser.parse_args()

    with open(args.file, "r") as f:
        vibe_segments_list: list[list[str]] = json.load(f)

    intent_base, truncated_intents = get_all_truncated_vibe_code(vibe_segments_list)

    result = {
        "intent_base": intent_base,
        "truncated_intents": truncated_intents,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

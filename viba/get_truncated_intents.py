import argparse
import json

from viba.intent_truncate_util import get_all_truncated_vibe_code


def main():
    parser = argparse.ArgumentParser(description="Generate truncated intents from viba code.")
    parser.add_argument("file", help="Path to a file containing viba source code")
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        viba_code: str = f.read()

    intent_base, truncated_intents = get_all_truncated_vibe_code(viba_code)

    result = {
        "intent_base": intent_base,
        "truncated_intents": truncated_intents,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

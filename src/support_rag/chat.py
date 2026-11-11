"""Ask the offline support assistant one question."""

import argparse
import sys

from support_rag.pipeline import VARIANTS, build_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the Sableboard help assistant.")
    parser.add_argument("question", nargs="?", help="Customer question. Read stdin when omitted.")
    parser.add_argument("--variant", choices=VARIANTS, default="improved")
    args = parser.parse_args(argv)
    question = args.question if args.question is not None else sys.stdin.read().strip()
    if not question:
        print("Pass a question as an argument.", file=sys.stderr)
        return 2
    answer = build_pipeline(args.variant).answer(question)
    print(answer.response)
    if answer.guardrail:
        print(f"\nGuardrail: {answer.guardrail}")
    print("\nSources:")
    for item in answer.retrieved:
        print(f"- {item.chunk.doc_id} ({item.chunk.title})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse

from sentence_transformers import SentenceTransformer

from .semantic import DEFAULT_EMBEDDING_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download the embedding model used by semantic inference.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SentenceTransformer(args.embedding_model)
    print(f"Downloaded embedding model: {args.embedding_model}")


if __name__ == "__main__":
    main()

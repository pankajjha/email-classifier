from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .labels import fasttext_label, normalize_label
from .text import clean_text, join_email_fields


def training_text(record: dict, include_body: bool) -> str:
    if include_body:
        return join_email_fields(record)

    parts = []
    for field in ("subject", "from", "to", "cc", "snippet"):
        if record.get(field):
            parts.append(f"{field}: {record[field]}")
    return clean_text(" ".join(parts))


def to_fasttext_line(record: dict, include_body: bool) -> str | None:
    label = normalize_label(str(record.get("label", "")))
    text = training_text(record, include_body)
    if not label or not text:
        return None
    return f"{fasttext_label(label)} {text}"


def read_records(paths: list[Path]) -> list[dict]:
    records = []
    seen = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                dedupe_key = (
                    record.get("account"),
                    record.get("message_id"),
                    hashlib.sha1(join_email_fields(record).encode("utf-8")).hexdigest(),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                records.append(record)
    return records


def stratified_split(lines: list[str], valid_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    by_label: dict[str, list[str]] = {}
    for line in lines:
        label = line.split(" ", 1)[0]
        by_label.setdefault(label, []).append(line)

    rng = random.Random(seed)
    train: list[str] = []
    valid: list[str] = []
    for label_lines in by_label.values():
        rng.shuffle(label_lines)
        valid_count = max(1, round(len(label_lines) * valid_ratio)) if len(label_lines) > 1 else 0
        valid.extend(label_lines[:valid_count])
        train.extend(label_lines[valid_count:])

    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def oversample_minority_labels(lines: list[str], min_per_label: int, seed: int) -> list[str]:
    if min_per_label <= 0:
        return lines

    by_label: dict[str, list[str]] = {}
    for line in lines:
        label = line.split(" ", 1)[0]
        by_label.setdefault(label, []).append(line)

    rng = random.Random(seed)
    balanced = list(lines)
    for label_lines in by_label.values():
        if len(label_lines) >= min_per_label:
            continue
        needed = min_per_label - len(label_lines)
        balanced.extend(rng.choice(label_lines) for _ in range(needed))

    rng.shuffle(balanced)
    return balanced


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert exported Gmail JSONL into FastText files.")
    parser.add_argument("inputs", nargs="+", help="One or more exported JSONL files.")
    parser.add_argument("--train-output", default="data/processed/train.txt")
    parser.add_argument("--valid-output", default="data/processed/valid.txt")
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--min-train-per-label", type=int, default=0, help="Oversample train labels up to this count.")
    parser.add_argument("--include-body", action="store_true", help="Include full exported body text. Default uses headers and snippet only.")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records([Path(path) for path in args.inputs])
    lines = [line for record in records if (line := to_fasttext_line(record, args.include_body))]
    train, valid = stratified_split(lines, args.valid_ratio, args.seed)
    train = oversample_minority_labels(train, args.min_train_per_label, args.seed)
    write_lines(Path(args.train_output), train)
    write_lines(Path(args.valid_output), valid)

    counts = Counter(line.split(" ", 1)[0] for line in lines)
    print(f"Prepared {len(train)} train and {len(valid)} validation examples")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()

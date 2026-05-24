from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from .text import strip_reply_noise


TOKEN_RE = re.compile(r"[a-z0-9_+#.-]+")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def sender_domain(record: dict) -> str:
    sender = str(record.get("from", "")).lower()
    match = re.search(r"@([a-z0-9.-]+\.[a-z]{2,})", sender)
    return match.group(1) if match else sender[:80]


def normalize_for_similarity(record: dict, max_chars: int) -> str:
    text = " ".join(
        part
        for part in (
            f"subject: {record.get('subject', '')}",
            f"from: {sender_domain(record)}",
            f"snippet: {record.get('snippet', '')}",
            strip_reply_noise(record.get("text", ""), max_chars=max_chars),
        )
        if part.strip()
    ).lower()

    replacements = (
        (r"https?://\S+", " URL "),
        (r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", " EMAIL "),
        (r"\b[a-z]{1,10}-\d+\b", " ISSUE "),
        (r"\b#[0-9]+\b", " NUM "),
        (r"\b\d{1,2}[:/.-]\d{1,2}([:/.-]\d{2,4})?\b", " DATE "),
        (r"\b\d{4,}\b", " NUM "),
        (r"\b\d+([,.]\d+)?\b", " NUM "),
        (r"\b(rs|inr|usd|eur|gbp)\s*\.?\s*num\b", " MONEY "),
        (r"\b[a-f0-9]{8,}\b", " ID "),
    )
    for pattern, value in replacements:
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)

    tokens = TOKEN_RE.findall(text)
    compact = []
    previous = None
    for token in tokens:
        if token == previous and token in {"num", "date", "url", "email", "id"}:
            continue
        compact.append(token)
        previous = token
    return " ".join(compact)


def shingles(text: str, width: int = 3) -> list[str]:
    tokens = text.split()
    if len(tokens) <= width:
        return tokens
    return [" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]


def simhash(text: str) -> int:
    vector = [0] * 64
    features = shingles(text)
    if not features:
        return 0
    for feature in features:
        digest = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if digest & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def exact_template_key(record: dict, normalized: str) -> str:
    subject = str(record.get("subject", "")).lower()
    subject = re.sub(r"\b[a-z]{1,10}-\d+\b", "ISSUE", subject, flags=re.IGNORECASE)
    subject = re.sub(r"\b#?\d+\b", "NUM", subject)
    subject = re.sub(r"\s+", " ", subject).strip()[:160]
    body_prefix = " ".join(normalized.split()[:60])
    digest = hashlib.blake2s(body_prefix.encode("utf-8"), digest_size=8).hexdigest()
    return f"{sender_domain(record)}|{subject}|{digest}"


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def group_records(records: list[dict], max_chars: int, hamming_threshold: int) -> tuple[list[dict], list[dict]]:
    normalized = [normalize_for_similarity(record, max_chars) for record in records]
    hashes = [simhash(text) for text in normalized]
    uf = UnionFind(len(records))

    exact: dict[str, int] = {}
    buckets: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = exact_template_key(record, normalized[index])
        if key in exact:
            uf.union(exact[key], index)
        else:
            exact[key] = index

        domain = sender_domain(record)
        for band in range(4):
            bucket = (domain, band, (hashes[index] >> (band * 16)) & 0xFFFF)
            for candidate_index in buckets[bucket]:
                if hamming(hashes[index], hashes[candidate_index]) <= hamming_threshold:
                    uf.union(candidate_index, index)
            buckets[bucket].append(index)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[uf.find(index)].append(index)

    representatives = []
    assignments = []
    for group_number, indexes in enumerate(sorted(grouped.values(), key=lambda value: (-len(value), value[0])), start=1):
        group_id = f"group_{group_number:05d}"
        representative_index = min(indexes, key=lambda index: len(normalized[index]) or 999999)
        representative = {
            **records[representative_index],
            "template_group_id": group_id,
            "template_group_size": len(indexes),
            "template_group_message_ids": [records[index].get("message_id") for index in indexes],
            "similarity_text": normalized[representative_index],
        }
        representatives.append(representative)
        for index in indexes:
            assignments.append(
                {
                    "template_group_id": group_id,
                    "message_id": records[index].get("message_id"),
                    "representative_message_id": records[representative_index].get("message_id"),
                    "group_size": len(indexes),
                }
            )

    return representatives, assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group similar exported emails before OpenAI labeling.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--representatives-output", required=True)
    parser.add_argument("--assignments-output", required=True)
    parser.add_argument("--max-body-chars", type=int, default=3000)
    parser.add_argument("--hamming-threshold", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(Path(args.input))
    representatives, assignments = group_records(records, args.max_body_chars, args.hamming_threshold)
    write_jsonl(Path(args.representatives_output), representatives)
    write_jsonl(Path(args.assignments_output), assignments)
    print(
        json.dumps(
            {
                "input_records": len(records),
                "representatives": len(representatives),
                "assignments": len(assignments),
                "largest_group": max((record["template_group_size"] for record in representatives), default=0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

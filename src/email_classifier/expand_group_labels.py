from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def expand(raw_path: Path, assignments_path: Path, labeled_representatives_path: Path, output_path: Path) -> None:
    raw_by_id = {record.get("message_id"): record for record in read_jsonl(raw_path)}
    assignments = read_jsonl(assignments_path)
    reps_by_group = {
        record.get("template_group_id"): record
        for record in read_jsonl(labeled_representatives_path)
        if record.get("template_group_id")
    }

    output = []
    missing_groups = set()
    for assignment in assignments:
        group_id = assignment.get("template_group_id")
        raw = raw_by_id.get(assignment.get("message_id"))
        rep = reps_by_group.get(group_id)
        if not raw:
            continue
        if not rep:
            missing_groups.add(group_id)
            continue
        label = rep.get("label") or rep.get("openai_label") or ""
        output.append(
            {
                **raw,
                "label": label,
                "openai_label": rep.get("openai_label", label),
                "openai_confidence": rep.get("openai_confidence"),
                "openai_reason": rep.get("openai_reason"),
                "review_status": rep.get("review_status", "propagated"),
                "template_group_id": group_id,
                "template_group_size": assignment.get("group_size"),
                "representative_message_id": assignment.get("representative_message_id"),
                "label_source": "template_group",
            }
        )

    write_jsonl(output_path, output)
    print(
        json.dumps(
            {
                "raw_records": len(raw_by_id),
                "labeled_records": len(output),
                "missing_groups": len(missing_groups),
                "output": str(output_path),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand representative labels back to all grouped emails.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--labeled-representatives", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expand(
        raw_path=Path(args.raw),
        assignments_path=Path(args.assignments),
        labeled_representatives_path=Path(args.labeled_representatives),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

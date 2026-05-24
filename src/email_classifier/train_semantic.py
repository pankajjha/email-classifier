from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from .labels import normalize_label
from .semantic import DEFAULT_EMBEDDING_MODEL
from .text import clean_text, strip_reply_noise


def read_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    seen: set[tuple[str | None, str | None]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                label = normalize_label(str(record.get("label", "")))
                if not label:
                    continue
                key = (record.get("account"), record.get("message_id"))
                if key in seen:
                    continue
                seen.add(key)
                record["label"] = label
                records.append(record)
    return records


def semantic_text(record: dict, body_chars: int) -> str:
    parts = []
    for field in ("subject", "from", "to", "cc", "snippet"):
        if record.get(field):
            parts.append(f"{field}: {record[field]}")

    if body_chars > 0 and record.get("text"):
        body = strip_reply_noise(record.get("text"), max_chars=body_chars)
        if body:
            parts.append(f"body: {body}")

    return clean_text(" ".join(parts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local semantic email classifier.")
    parser.add_argument("inputs", nargs="+", help="One or more labeled JSONL files.")
    parser.add_argument("--model-output", default="models/semantic_classifier.joblib")
    parser.add_argument("--metrics-output", default="models/semantic_classifier_metrics.json")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--body-chars", type=int, default=1200, help="Newest body chars to include. Use 0 for headers/snippet only.")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records([Path(path) for path in args.inputs])
    examples = [
        (text, record["label"])
        for record in records
        if (text := semantic_text(record, body_chars=args.body_chars))
    ]
    if not examples:
        raise SystemExit("No labeled examples found")

    texts = [text for text, _label in examples]
    labels = [label for _text, label in examples]
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        texts,
        labels,
        test_size=args.valid_ratio,
        random_state=args.seed,
        stratify=labels,
    )

    encoder = SentenceTransformer(args.embedding_model)
    train_embeddings = encoder.encode(train_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    valid_embeddings = encoder.encode(valid_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    classifier = LogisticRegression(
        max_iter=args.max_iter,
        class_weight=None if args.class_weight == "none" else "balanced",
        solver="lbfgs",
    )
    classifier.fit(train_embeddings, train_labels)
    predictions = classifier.predict(valid_embeddings)

    labels_sorted = sorted(set(labels))
    report = classification_report(valid_labels, predictions, labels=labels_sorted, output_dict=True, zero_division=0)
    metrics = {
        "embedding_model": args.embedding_model,
        "body_chars": args.body_chars,
        "class_weight": args.class_weight,
        "examples": len(examples),
        "train_examples": len(train_texts),
        "valid_examples": len(valid_texts),
        "accuracy": accuracy_score(valid_labels, predictions),
        "label_counts": dict(sorted(Counter(labels).items())),
        "classification_report": report,
    }

    output_path = Path(args.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "labels": list(classifier.classes_),
            "embedding_model": args.embedding_model,
            "body_chars": args.body_chars,
        },
        output_path,
    )

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()

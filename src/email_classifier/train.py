from __future__ import annotations

import argparse
import json
from pathlib import Path

import fasttext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a FastText email classifier.")
    parser.add_argument("--train", default="data/processed/train.txt")
    parser.add_argument("--valid", default="data/processed/valid.txt")
    parser.add_argument("--model-output", default="models/email_classifier.bin")
    parser.add_argument("--quantized-output", default="models/email_classifier.ftz")
    parser.add_argument("--metrics-output", default="models/metrics.json")
    parser.add_argument("--epoch", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.6)
    parser.add_argument("--word-ngrams", type=int, default=2)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--loss", default="softmax", choices=["softmax", "ova", "hs", "ns"])
    parser.add_argument("--no-quantize", action="store_true", help="Skip saving a compact quantized model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = fasttext.train_supervised(
        input=args.train,
        epoch=args.epoch,
        lr=args.lr,
        wordNgrams=args.word_ngrams,
        minCount=args.min_count,
        dim=args.dim,
        loss=args.loss,
        verbose=2,
    )

    output_path = Path(args.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))

    if not args.no_quantize:
        quantized_path = Path(args.quantized_output)
        quantized_path.parent.mkdir(parents=True, exist_ok=True)
        model.quantize(input=args.train, retrain=True, qnorm=True, cutoff=100000)
        model.save_model(str(quantized_path))

    metrics = {}
    valid_path = Path(args.valid)
    if valid_path.exists() and valid_path.stat().st_size > 0:
        examples, precision, recall = model.test(str(valid_path))
        metrics = {"examples": examples, "precision_at_1": precision, "recall_at_1": recall}
        Path(args.metrics_output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, indent=2))

    print(f"Saved model to {output_path}")
    if not args.no_quantize:
        print(f"Saved quantized model to {args.quantized_output}")


if __name__ == "__main__":
    main()

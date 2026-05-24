from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .text import strip_reply_noise


LABELS = (
    "urgent",
    "action_needed",
    "follow_up",
    "awaiting_reply",
    "meeting",
    "fyi",
    "done",
    "payment",
    "newsletter",
    "marketing",
    "misc",
)


SYSTEM_PROMPT = """You label emails for Pankaj's Gmail classifier.
Pick exactly one label.

Label definitions:
- urgent: needs immediate attention, has a hard deadline, escalation, outage, customer-blocking issue, or serious risk.
- action_needed: Pankaj needs to act, approve, reply, review, send, provide information, or make a decision.
- follow_up: sender is checking in/reminding, or the email should be chased later but is not urgent.
- awaiting_reply: Pankaj has already responded or sent something and is waiting for someone else's response/confirmation.
- meeting: calendar invite, scheduling, rescheduling, agenda, call, Zoom, Meet, or meeting logistics.
- fyi: informational update; no clear action needed.
- done: task is completed, resolved, fixed, closed, or acknowledged as finished.
- payment: invoice, receipt, refund, payment status, bank transfer, UTR, charge, subscription billing, or money movement.
- newsletter: recurring digest, newsletter, article roundup, automated product update.
- marketing: offer, promotion, sales pitch, trial, discount, outreach from vendor.
- misc: none of the above is clear.

Prefer action_needed over fyi when the sender explicitly asks Pankaj to do something.
Prefer follow_up when the main intent is checking/reminding.
Prefer urgent only when immediacy or risk is explicit.
Return strict JSON only."""


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def existing_message_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for record in read_jsonl(path):
        if record.get("message_id"):
            ids.add(record["message_id"])
    return ids


def write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_email(record: dict, max_body_chars: int) -> dict:
    return {
        "subject": record.get("subject", ""),
        "from": record.get("from", ""),
        "to": record.get("to", ""),
        "date": record.get("date", ""),
        "snippet": record.get("snippet", ""),
        "body": strip_reply_noise(record.get("text", ""), max_chars=max_body_chars),
        "gmail_label_names": record.get("gmail_label_names", []),
    }


def request_body(record: dict, model: str, max_body_chars: int) -> dict:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string", "enum": list(LABELS)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["label", "confidence", "reason"],
    }
    return {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(compact_email(record, max_body_chars), ensure_ascii=False),
                    }
                ],
            },
        ],
        "text": {"format": {"type": "json_schema", "name": "email_label", "strict": True, "schema": schema}},
        "max_output_tokens": 120,
    }


def parse_response(response: dict) -> dict:
    text = response.get("output_text")
    if not text:
        for item in response.get("output", []):
            for part in item.get("content", []):
                if part.get("refusal"):
                    raise RuntimeError(f"OpenAI refusal: {part['refusal']}")
                if isinstance(part.get("text"), str):
                    text = part["text"]
                    break
            if text:
                break
    if not text:
        raise RuntimeError("OpenAI response did not contain output text")
    parsed = json.loads(text)
    if parsed.get("label") not in LABELS:
        raise RuntimeError(f"Invalid label returned: {parsed.get('label')}")
    return parsed


def call_openai(api_key: str, body: dict, timeout: int, retries: int) -> dict:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_error: Exception | None = None
    context = ssl.create_default_context()
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_error}")


def label_file(
    input_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
    max_body_chars: int,
    min_confidence: float,
    limit: int | None,
    sleep_seconds: float,
    timeout: int,
    retries: int,
    workers: int,
) -> None:
    records = read_jsonl(input_path)
    done = existing_message_ids(output_path)
    pending = []
    skipped = 0
    for record in records:
        if limit is not None and len(pending) >= limit:
            break
        message_id = record.get("message_id")
        if message_id in done:
            skipped += 1
            continue
        pending.append(record)

    def label_record(record: dict) -> dict:
        result = parse_response(call_openai(api_key, request_body(record, model, max_body_chars), timeout, retries))
        labeled = {
            **record,
            "label": result["label"] if result["confidence"] >= min_confidence else "",
            "openai_label": result["label"],
            "openai_confidence": result["confidence"],
            "openai_reason": result["reason"],
            "review_status": "auto_accepted" if result["confidence"] >= min_confidence else "needs_review",
            "training_text": " ".join(
                part
                for part in (
                    f"subject: {record.get('subject', '')}",
                    f"from: {record.get('from', '')}",
                    f"snippet: {record.get('snippet', '')}",
                    strip_reply_noise(record.get("text", ""), max_chars=max_body_chars),
                )
                if part.strip()
            ),
        }
        return labeled

    processed = 0
    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    futures = [executor.submit(label_record, record) for record in pending]
    try:
        for future in as_completed(futures):
            write_jsonl(output_path, future.result())
            processed += 1
            if processed % 25 == 0:
                print(f"labeled={processed} skipped_existing={skipped}")
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        print(f"Interrupted after labeling {processed} records; output is resumable at {output_path}.")
        raise
    else:
        executor.shutdown(wait=True)
        if sleep_seconds:
            time.sleep(sleep_seconds)
    print(f"Labeled {processed} records into {output_path}; skipped {skipped} existing records.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label exported Gmail JSONL with OpenAI for offline training.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-body-chars", type=int, default=3000)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("Set OPENAI_API_KEY or pass --api-key.")
    return args


def main() -> None:
    args = parse_args()
    label_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        api_key=args.api_key,
        model=args.model,
        max_body_chars=args.max_body_chars,
        min_confidence=args.min_confidence,
        limit=args.limit,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
        retries=args.retries,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

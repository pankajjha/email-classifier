from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


AUTOMATED_LOCAL_WORDS = (
    "no-reply",
    "noreply",
    "do-not-reply",
    "donotreply",
    "notification",
    "notifications",
    "notify",
    "mailer",
    "mailer-daemon",
    "postmaster",
    "automated",
    "automation",
    "bot",
    "alerts",
    "alert",
    "updates",
    "digest",
    "newsletter",
    "support",
    "team",
    "hello",
    "info",
    "admin",
    "system",
    "billing",
    "invoice",
    "receipts",
    "receipt",
    "security",
    "calendar",
    "reminder",
    "news",
    "marketing",
)

AUTOMATED_NAME_MARKERS = (
    "github",
    "gitlab",
    "bitbucket",
    "dependabot",
    "tl;dv",
    "tldv",
    "jira",
    "confluence",
    "slack",
    "calendar",
    "calendly",
    "notion",
    "linear",
    "figma",
    "zoom",
    "hubspot",
    "razorpay",
    "stripe",
    "sentry",
    "datadog",
    "cloudflare",
    "aws",
    "amazon web services",
    "google",
    "workspace",
    "drive",
    "docs",
    "sheets",
    "newsletter",
    "digest",
    "notification",
)

AUTOMATED_DOMAIN_MARKERS = (
    "github",
    "tldv",
    "tl.dv",
    "slack",
    "atlassian",
    "jira",
    "linear",
    "notion",
    "calendly",
    "sendgrid",
    "mailgun",
    "amazonses",
    "mailchimp",
    "hubspot",
    "sentry",
    "figma",
    "vercel",
    "cloudflare",
    "postmark",
    "mandrill",
)


def extract_email(value: str | None) -> str:
    match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", value or "", re.IGNORECASE)
    return match.group(0).lower() if match else ""


def display_name(value: str | None) -> str:
    raw = value or ""
    email = extract_email(raw)
    name = raw.replace(f"<{email}>", "").replace(email, "").strip().strip('"').strip()
    return re.sub(r"\s+", " ", name)


def local_part_reasons(local_part: str) -> list[str]:
    normalized = re.sub(r"[._+]+", "-", local_part.lower())
    reasons = []
    for word in AUTOMATED_LOCAL_WORDS:
        if normalized == word or word in normalized:
            reasons.append(f"local:{word}")
            break
    return reasons


def marker_reason(value: str, markers: tuple[str, ...], prefix: str) -> list[str]:
    lowered = value.lower()
    for marker in markers:
        if marker in lowered:
            return [f"{prefix}:{marker}"]
    return []


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract likely automated sender blacklist candidates from exported Gmail JSONL.")
    parser.add_argument("input", help="Exported Gmail JSONL, for example data/raw/work_all.jsonl.")
    parser.add_argument("--output", default="data/processed/sender_blacklist_candidates.json")
    parser.add_argument("--min-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(Path(args.input))
    sender_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    names_by_sender: dict[str, Counter[str]] = defaultdict(Counter)
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, list[str]] = defaultdict(list)

    for record in records:
        sender = extract_email(record.get("from"))
        if not sender:
            continue
        local_part, domain = sender.split("@", 1)
        sender_counts[sender] += 1
        domain_counts[domain] += 1
        name = display_name(record.get("from"))
        if name:
            names_by_sender[sender][name] += 1
        label_counts[sender][record.get("label") or "unknown"] += 1
        if len(samples[sender]) < 3:
            samples[sender].append(str(record.get("subject") or "")[:140])

    exact_candidates = []
    for sender, count in sender_counts.items():
        local_part, domain = sender.split("@", 1)
        name = names_by_sender[sender].most_common(1)[0][0] if names_by_sender[sender] else ""
        reasons = [
            *local_part_reasons(local_part),
            *marker_reason(name, AUTOMATED_NAME_MARKERS, "name"),
            *marker_reason(domain, AUTOMATED_DOMAIN_MARKERS, "domain"),
        ]
        if not reasons:
            continue
        strong_local = any(reason.startswith("local:") for reason in reasons)
        strong_name = any(reason.startswith("name:") for reason in reasons)
        if count < args.min_count and not strong_local and not strong_name:
            continue
        exact_candidates.append(
            {
                "email": sender,
                "count": count,
                "display_name": name,
                "reasons": reasons,
                "labels": dict(label_counts[sender].most_common()),
                "sample_subjects": samples[sender],
            }
        )

    exact_candidates.sort(key=lambda item: (-item["count"], item["email"]))
    domain_candidates = [
        {"domain": domain, "count": count}
        for domain, count in domain_counts.items()
        if count >= args.min_count and marker_reason(domain, AUTOMATED_DOMAIN_MARKERS, "domain")
    ]
    domain_candidates.sort(key=lambda item: (-item["count"], item["domain"]))

    output = {
        "source": str(args.input),
        "records": len(records),
        "unique_senders": len(sender_counts),
        "unique_domains": len(domain_counts),
        "exact_sender_candidates": exact_candidates,
        "domain_candidates": domain_candidates,
        "top_domains": [{"domain": domain, "count": count} for domain, count in domain_counts.most_common(50)],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Records: {output['records']}")
    print(f"Unique senders: {output['unique_senders']}")
    print(f"Exact sender candidates: {len(exact_candidates)}")
    print(f"Domain candidates: {len(domain_candidates)}")


if __name__ == "__main__":
    main()

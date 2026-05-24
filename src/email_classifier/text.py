from __future__ import annotations

import html
import re


HEADER_FIELDS = ("subject", "from", "to", "cc")

QUOTE_MARKERS = (
    r"\nOn .+ wrote:\s*$",
    r"\nFrom:\s.+\nSent:\s.+\nTo:\s.+",
    r"\n-{2,}\s*Original Message\s*-{2,}",
    r"\n_{5,}",
)

SIGNATURE_MARKERS = (
    r"\n--\s*$",
    r"\nRegards,?\s*$",
    r"\nThanks,?\s*$",
    r"\nBest,?\s*$",
)


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = html.unescape(value)
    value = re.sub(r"<(script|style).*?</\1>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"https?://\S+", " URL ", value)
    value = re.sub(r"\S+@\S+", " EMAIL ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def strip_reply_noise(value: str | None, max_chars: int = 3000) -> str:
    """Keep the newest human-written portion of an email for classification."""
    if not value:
        return ""

    text = html.unescape(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<(script|style).*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)

    for marker in QUOTE_MARKERS:
        match = re.search(marker, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            text = text[: match.start()]
            break

    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith(">"):
            continue
        lines.append(stripped)

    text = "\n".join(lines)
    for marker in SIGNATURE_MARKERS:
        match = re.search(marker, text, flags=re.IGNORECASE | re.MULTILINE)
        if match and match.start() > 80:
            text = text[: match.start()]
            break

    text = clean_text(text)
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def join_email_fields(record: dict) -> str:
    parts = []
    for field in HEADER_FIELDS:
        if record.get(field):
            parts.append(f"{field}: {record[field]}")
    if record.get("snippet"):
        parts.append(f"snippet: {record['snippet']}")
    if record.get("text"):
        parts.append(record["text"])
    return clean_text(" ".join(parts))

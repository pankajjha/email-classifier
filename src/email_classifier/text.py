from __future__ import annotations

import html
import re


HEADER_FIELDS = ("subject", "from", "to", "cc")


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

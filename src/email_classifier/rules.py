from __future__ import annotations

from .text import clean_text


RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("urgent", ("urgent", "asap", "immediately", "critical", "production issue", "blocked")),
    ("follow_up", ("follow up", "following up", "gentle reminder", "checking in")),
    ("awaiting_reply", ("awaiting your reply", "waiting for your response", "waiting for your confirmation")),
    ("fyi", ("fyi", "for your information", "no action needed")),
    ("action_needed", ("action required", "please approve", "need your approval", "needs your approval", "please review", "please send")),
    ("meeting", ("calendar invite", "meeting invitation", "meeting", "google meet", "zoom", "agenda", "rescheduled")),
    ("payment", ("invoice", "payment", "receipt", "paid", "refund", "transaction", "amount credited", "utr")),
    ("newsletter", ("newsletter", "unsubscribe", "digest")),
    ("marketing", ("limited time", "offer", "discount", "promotion", "sale", "upgrade today")),
    ("done", ("completed", "resolved", "done", "fixed", "closing this out")),
)


def classify_by_rules(text: str) -> str | None:
    normalized = clean_text(text).lower()
    for label, phrases in RULES:
        if any(phrase in normalized for phrase in phrases):
            return label
    return None

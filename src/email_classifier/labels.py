from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_LABELS = {
    "urgent": ["urgent"],
    "action_needed": ["action needed", "action_needed"],
    "follow_up": ["follow up", "follow_up"],
    "awaiting_reply": ["awaiting reply", "awaiting_reply"],
    "meeting": ["meeting"],
    "fyi": ["fyi"],
    "done": ["done"],
    "payment": ["payment"],
    "newsletter": ["newsletter"],
    "marketing": ["marketing"],
    "misc": ["misc"],
}


def normalize_label(label: str) -> str:
    label = label.strip().lower().replace("&", " and ")
    label = re.sub(r"[^a-z0-9]+", "_", label)
    return re.sub(r"_+", "_", label).strip("_")


def load_label_config(path: str | Path | None = None) -> dict[str, list[str]]:
    if not path:
        return DEFAULT_LABELS

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    labels: dict[str, list[str]] = {}
    for canonical, aliases in raw.items():
        canonical_name = normalize_label(canonical)
        values = aliases if isinstance(aliases, list) else [aliases]
        labels[canonical_name] = [str(value) for value in values]
    return labels


def fasttext_label(label: str) -> str:
    return f"__label__{normalize_label(label)}"


def strip_fasttext_prefix(label: str) -> str:
    return label.removeprefix("__label__")

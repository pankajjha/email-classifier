from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .labels import normalize_label


@dataclass(frozen=True)
class SklearnPrediction:
    label: str
    confidence: float
    candidates: list[dict[str, float | str]]


class SklearnEmailClassifier:
    def __init__(self, *, classifier_path: str | Path, top_k: int = 3) -> None:
        artifact = joblib.load(classifier_path)
        self.pipeline = artifact["pipeline"]
        self.labels = [normalize_label(label) for label in artifact["labels"]]
        self.body_chars = int(artifact.get("body_chars", 0))
        self.top_k = top_k

    def predict(self, text: str) -> SklearnPrediction:
        probabilities = self._predict_probabilities([text])
        order = np.argsort(probabilities)[::-1][: self.top_k]
        candidates = [
            {"label": self.labels[index], "confidence": float(probabilities[index])}
            for index in order
        ]
        top = candidates[0]
        return SklearnPrediction(
            label=str(top["label"]),
            confidence=float(top["confidence"]),
            candidates=candidates,
        )

    def _predict_probabilities(self, texts: list[str]) -> np.ndarray:
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(texts)[0]

        if hasattr(self.pipeline, "decision_function"):
            scores = np.asarray(self.pipeline.decision_function(texts)[0], dtype=float)
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            return exp_scores / exp_scores.sum()

        label = normalize_label(str(self.pipeline.predict(texts)[0]))
        probabilities = np.full(len(self.labels), 0.0)
        probabilities[self.labels.index(label)] = 1.0
        return probabilities

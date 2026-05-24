from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from .labels import normalize_label


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class SemanticPrediction:
    label: str
    confidence: float
    candidates: list[dict[str, float | str]]


class SemanticEmailClassifier:
    def __init__(
        self,
        *,
        classifier_path: str | Path,
        embedding_model: str | None = None,
        top_k: int = 3,
    ) -> None:
        artifact = joblib.load(classifier_path)
        self.classifier = artifact["classifier"]
        self.labels = [normalize_label(label) for label in artifact["labels"]]
        self.embedding_model_name = embedding_model or artifact.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
        self.body_chars = int(artifact.get("body_chars", 1200))
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.top_k = top_k

    def predict(self, text: str) -> SemanticPrediction:
        embedding = self.embedding_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        probabilities = self._predict_probabilities(embedding)
        order = np.argsort(probabilities)[::-1][: self.top_k]
        candidates = [
            {"label": self.labels[index], "confidence": float(probabilities[index])}
            for index in order
        ]
        top = candidates[0]
        return SemanticPrediction(
            label=str(top["label"]),
            confidence=float(top["confidence"]),
            candidates=candidates,
        )

    def _predict_probabilities(self, embedding: Any) -> np.ndarray:
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(embedding)[0]

        if hasattr(self.classifier, "decision_function"):
            scores = np.asarray(self.classifier.decision_function(embedding)[0], dtype=float)
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            return exp_scores / exp_scores.sum()

        label = normalize_label(str(self.classifier.predict(embedding)[0]))
        probabilities = np.full(len(self.labels), 0.0)
        probabilities[self.labels.index(label)] = 1.0
        return probabilities

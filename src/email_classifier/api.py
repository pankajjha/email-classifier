from __future__ import annotations

import os
from functools import lru_cache

import fasttext
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .labels import strip_fasttext_prefix
from .rules import classify_by_rules
from .text import clean_text


load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/email_classifier.bin")
SEMANTIC_MODEL_PATH = os.getenv("SEMANTIC_MODEL_PATH", "models/semantic_classifier.joblib")
SKLEARN_MODEL_PATH = os.getenv("SKLEARN_MODEL_PATH", "models/sklearn_classifier.joblib")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
CLASSIFIER_BACKEND = os.getenv("CLASSIFIER_BACKEND", "fasttext").strip().lower()
API_KEY = os.getenv("CLASSIFIER_API_KEY", "")
USE_RULES = os.getenv("CLASSIFIER_USE_RULES", "1" if CLASSIFIER_BACKEND == "fasttext" else "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

app = FastAPI(title="Email Classifier", version="0.1.0")


TEST_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Email Classifier</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d9e0ea;
      --primary: #1769e0;
      --primary-dark: #0f56bd;
      --danger: #b42318;
      --success-bg: #e9f7ef;
      --success: #157347;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    main {
      width: min(1040px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }

    h1 {
      margin: 0 0 20px;
      font-size: 28px;
      font-weight: 700;
      line-height: 1.2;
      letter-spacing: 0;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 20px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    label {
      display: block;
      margin-bottom: 7px;
      color: #344054;
      font-size: 13px;
      font-weight: 650;
    }

    input,
    textarea {
      width: 100%;
      border: 1px solid #c9d3df;
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      font: inherit;
      font-size: 15px;
      letter-spacing: 0;
      outline: none;
      transition: border-color 140ms ease, box-shadow 140ms ease;
    }

    input {
      height: 42px;
      padding: 0 12px;
    }

    textarea {
      min-height: 310px;
      margin-top: 14px;
      padding: 12px;
      line-height: 1.45;
      resize: vertical;
    }

    input:focus,
    textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(23, 105, 224, 0.14);
    }

    .actions {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 14px;
    }

    button {
      height: 42px;
      min-width: 124px;
      border: 0;
      border-radius: 6px;
      background: var(--primary);
      color: #ffffff;
      font: inherit;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0;
      cursor: pointer;
    }

    button:hover {
      background: var(--primary-dark);
    }

    button:disabled {
      cursor: wait;
      opacity: 0.72;
    }

    .status {
      min-height: 20px;
      color: var(--muted);
      font-size: 14px;
    }

    .result {
      display: grid;
      gap: 16px;
    }

    .label-output {
      min-height: 82px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      background: var(--success-bg);
      color: var(--success);
      font-size: 30px;
      font-weight: 800;
      letter-spacing: 0;
      text-align: center;
      overflow-wrap: anywhere;
    }

    .confidence {
      color: var(--muted);
      font-size: 14px;
    }

    .candidates {
      display: grid;
      gap: 10px;
    }

    .candidate {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      font-size: 14px;
    }

    .bar {
      grid-column: 1 / -1;
      height: 7px;
      overflow: hidden;
      border-radius: 999px;
      background: #edf1f6;
    }

    .fill {
      height: 100%;
      width: 0;
      border-radius: inherit;
      background: var(--primary);
    }

    .error {
      color: var(--danger);
    }

    @media (max-width: 820px) {
      main {
        width: min(100% - 24px, 680px);
        padding: 20px 0;
      }

      .layout,
      .grid {
        grid-template-columns: 1fr;
      }

      .label-output {
        font-size: 26px;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Email Classifier</h1>
    <div class="layout">
      <section class="panel">
        <div class="grid">
          <div>
            <label for="subject">Subject</label>
            <input id="subject" autocomplete="off">
          </div>
          <div>
            <label for="sender">From</label>
            <input id="sender" autocomplete="off">
          </div>
          <div>
            <label for="snippet">Snippet</label>
            <input id="snippet" autocomplete="off">
          </div>
          <div>
            <label for="apiKey">API key</label>
            <input id="apiKey" autocomplete="off" type="password">
          </div>
        </div>
        <label for="body" style="margin-top: 14px;">Email text</label>
        <textarea id="body" autofocus></textarea>
        <div class="actions">
          <button id="classifyButton" type="button">Classify</button>
          <div id="status" class="status"></div>
        </div>
      </section>

      <aside class="panel result">
        <div>
          <label>Label</label>
          <div id="labelOutput" class="label-output">-</div>
        </div>
        <div id="confidence" class="confidence"></div>
        <div id="candidates" class="candidates"></div>
      </aside>
    </div>
  </main>

  <script>
    const fields = {
      subject: document.getElementById("subject"),
      sender: document.getElementById("sender"),
      snippet: document.getElementById("snippet"),
      body: document.getElementById("body"),
      apiKey: document.getElementById("apiKey")
    };
    const button = document.getElementById("classifyButton");
    const statusEl = document.getElementById("status");
    const labelOutput = document.getElementById("labelOutput");
    const confidenceEl = document.getElementById("confidence");
    const candidatesEl = document.getElementById("candidates");

    function formatPercent(value) {
      return `${Math.round(value * 1000) / 10}%`;
    }

    function renderResult(result) {
      labelOutput.textContent = result.label || "-";
      confidenceEl.textContent = result.confidence == null ? "" : `Confidence ${formatPercent(result.confidence)}`;
      candidatesEl.innerHTML = "";

      for (const candidate of result.candidates || []) {
        const row = document.createElement("div");
        row.className = "candidate";
        const name = document.createElement("strong");
        name.textContent = candidate.label;
        const score = document.createElement("span");
        score.textContent = formatPercent(candidate.confidence);
        const bar = document.createElement("div");
        bar.className = "bar";
        const fill = document.createElement("div");
        fill.className = "fill";
        fill.style.width = `${Math.max(0, Math.min(100, candidate.confidence * 100))}%`;
        bar.appendChild(fill);
        row.append(name, score, bar);
        candidatesEl.appendChild(row);
      }
    }

    async function classifyEmail() {
      const payload = {
        subject: fields.subject.value,
        from: fields.sender.value,
        snippet: fields.snippet.value,
        body: fields.body.value
      };

      statusEl.textContent = "";
      statusEl.className = "status";
      button.disabled = true;
      button.textContent = "Classifying";

      try {
        const headers = { "Content-Type": "application/json" };
        if (fields.apiKey.value.trim()) {
          headers["X-API-Key"] = fields.apiKey.value.trim();
        }

        const response = await fetch("/classify", {
          method: "POST",
          headers,
          body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Classification failed");
        }
        renderResult(data);
      } catch (error) {
        statusEl.textContent = error.message;
        statusEl.className = "status error";
      } finally {
        button.disabled = false;
        button.textContent = "Classify";
      }
    }

    button.addEventListener("click", classifyEmail);
    fields.body.addEventListener("keydown", (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        classifyEmail();
      }
    });
  </script>
</body>
</html>
"""


class EmailInput(BaseModel):
    subject: str = ""
    sender: str = Field(default="", alias="from")
    to: str = ""
    snippet: str = ""
    body: str = ""


class Candidate(BaseModel):
    label: str
    confidence: float


class Classification(BaseModel):
    label: str
    confidence: float
    candidates: list[Candidate]


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@lru_cache(maxsize=1)
def load_fasttext_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")
    return fasttext.load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def load_semantic_model():
    if not os.path.exists(SEMANTIC_MODEL_PATH):
        raise RuntimeError(f"Semantic model not found: {SEMANTIC_MODEL_PATH}")
    from .semantic import SemanticEmailClassifier

    return SemanticEmailClassifier(
        classifier_path=SEMANTIC_MODEL_PATH,
        embedding_model=EMBEDDING_MODEL or None,
        top_k=3,
    )


@lru_cache(maxsize=1)
def load_sklearn_model():
    if not os.path.exists(SKLEARN_MODEL_PATH):
        raise RuntimeError(f"Sklearn model not found: {SKLEARN_MODEL_PATH}")
    from .sklearn_classifier import SklearnEmailClassifier

    return SklearnEmailClassifier(classifier_path=SKLEARN_MODEL_PATH, top_k=3)


def load_model():
    if CLASSIFIER_BACKEND == "fasttext":
        return load_fasttext_model()
    if CLASSIFIER_BACKEND == "sklearn":
        return load_sklearn_model()
    if CLASSIFIER_BACKEND == "semantic":
        return load_semantic_model()
    raise RuntimeError(f"Unsupported classifier backend: {CLASSIFIER_BACKEND}")


def email_to_text(payload: EmailInput, body_chars: int | None = None) -> str:
    body = payload.body
    if body_chars is not None:
        if body_chars > 0:
            from .text import strip_reply_noise

            body = strip_reply_noise(body, max_chars=body_chars)
        else:
            body = ""

    return clean_text(
        " ".join(
            [
                f"subject: {payload.subject}",
                f"from: {payload.sender}",
                f"to: {payload.to}",
                f"snippet: {payload.snippet}",
                f"body: {body}" if body else "",
            ]
        )
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    load_model()
    model_path_by_backend = {
        "fasttext": MODEL_PATH,
        "sklearn": SKLEARN_MODEL_PATH,
        "semantic": SEMANTIC_MODEL_PATH,
    }
    model_path = model_path_by_backend.get(CLASSIFIER_BACKEND, MODEL_PATH)
    return {"status": "ready", "backend": CLASSIFIER_BACKEND, "model": model_path}


@app.get("/", response_class=HTMLResponse)
def test_page() -> str:
    return TEST_PAGE


@app.post("/classify", response_model=Classification, dependencies=[Depends(require_api_key)])
def classify(payload: EmailInput) -> Classification:
    model = load_model()
    body_chars = getattr(model, "body_chars", None) if CLASSIFIER_BACKEND in {"semantic", "sklearn"} else None
    text = email_to_text(payload, body_chars=body_chars)
    if not text:
        raise HTTPException(status_code=400, detail="Empty email text")

    if USE_RULES:
        rule_label = classify_by_rules(text)
        if rule_label:
            return Classification(
                label=rule_label,
                confidence=0.99,
                candidates=[Candidate(label=rule_label, confidence=0.99)],
            )

    if CLASSIFIER_BACKEND in {"semantic", "sklearn"}:
        prediction = model.predict(text)
        return Classification(
            label=prediction.label,
            confidence=prediction.confidence,
            candidates=[Candidate(**candidate) for candidate in prediction.candidates],
        )

    labels, scores = model.predict(text, k=3)
    candidates = [
        Candidate(label=strip_fasttext_prefix(label), confidence=float(score))
        for label, score in zip(labels, scores)
    ]
    top = candidates[0]
    return Classification(label=top.label, confidence=top.confidence, candidates=candidates)

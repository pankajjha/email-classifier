FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/email_classifier.ftz \
    SKLEARN_MODEL_PATH=/app/models/sklearn_classifier.joblib \
    CLASSIFIER_BACKEND=sklearn \
    CLASSIFIER_USE_RULES=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.inference.txt ./
COPY src ./src
COPY models/email_classifier.ftz ./models/email_classifier.ftz
COPY models/sklearn_classifier.joblib ./models/sklearn_classifier.joblib

RUN pip install --no-cache-dir -r requirements.inference.txt

CMD ["sh", "-c", "uvicorn email_classifier.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

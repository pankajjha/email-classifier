# Email Classifier

Small local pipeline for training a label classifier from Gmail messages that are already labelled by Dimension, Gmail, manual review, or the one-time OpenAI bootstrap flow.

Target labels:

- `urgent`
- `action_needed`
- `follow_up`
- `awaiting_reply`
- `meeting`
- `fyi`
- `done`
- `payment`
- `newsletter`
- `marketing`
- `misc`

## Architecture

1. Export already-labelled historical Gmail messages from both accounts.
2. Convert exported JSONL into local training data.
3. Train a small local classifier.
4. Deploy only the `/classify` API and compact model artifact on Render or a VPS.
5. n8n sends new unlabelled email fields to `/classify`, then applies the returned label in Gmail.

The current Render backend is `sklearn`: word/character TF-IDF features plus Logistic Regression. FastText remains available as a fallback backend, and the sentence-transformer semantic backend is available for experiments but is not the default because it underperformed on the current validation set.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

For the VPS/inference service only, install the smaller dependency set:

```bash
pip install -r requirements.inference.txt
```

Create a Google Cloud OAuth Desktop client and download it as `credentials.json`. The exporter uses the read-only Gmail scope.

## Export Training Data

Run this once per Gmail account. Use a different token file per account.

```bash
python -m email_classifier.gmail_export \
  --account office \
  --credentials credentials.json \
  --token .tokens/office.json \
  --output data/raw/office.jsonl \
  --labels config/labels.json

python -m email_classifier.gmail_export \
  --account personal \
  --credentials credentials.json \
  --token .tokens/personal.json \
  --output data/raw/personal.jsonl \
  --labels config/labels.json
```

For a quick test export:

```bash
python -m email_classifier.gmail_export \
  --account office \
  --credentials credentials.json \
  --token .tokens/office.json \
  --output data/raw/office_sample.jsonl \
  --labels config/labels.json \
  --max-per-label 10
```

### Export With OAuth Playground Tokens

For a temporary backfill, OAuth Playground is fine. Use the scope:

```text
https://www.googleapis.com/auth/gmail.readonly
```

In OAuth Playground:

1. Open Settings.
2. Keep access type as `Offline`.
3. For tokens that last beyond 24 hours, enable `Use your own OAuth credentials` and enter your Google Cloud OAuth web client ID/secret.
4. Authorize `https://www.googleapis.com/auth/gmail.readonly`.
5. Exchange the authorization code for tokens.

Then export using environment variables. Do not paste these tokens into chat.

```bash
export GOOGLE_ACCESS_TOKEN='access-token-from-playground'
export GOOGLE_REFRESH_TOKEN='refresh-token-from-playground'
export GOOGLE_CLIENT_ID='your-oauth-client-id'
export GOOGLE_CLIENT_SECRET='your-oauth-client-secret'

python -m email_classifier.gmail_export \
  --account office \
  --output data/raw/office.jsonl \
  --labels config/labels.json
```

For a very short one-shot export, only `GOOGLE_ACCESS_TOKEN` is required. If it expires during export, set the refresh token plus client credentials and rerun.

If your Gmail labels are nested or named differently, edit `config/labels.json`. Each canonical label can have multiple aliases.

## Bootstrap Labels With OpenAI

For a one-time quality pass, export all work-account emails, ask OpenAI to assign one canonical label, review uncertain labels locally, then train FastText from the reviewed JSONL.

Export all Gmail messages. The exporter appends to the output file and skips existing message IDs, so reruns are safe if an OAuth Playground token expires:

```bash
export GOOGLE_ACCESS_TOKEN='short-lived-token'

python -m email_classifier.gmail_export_all \
  --account work \
  --output data/raw/work_all.jsonl \
  --query=-in:chats
```

Group similar messages before sending anything to OpenAI. This avoids paying to label repeated templates such as invoices, Jira, GitHub, reports, and alert emails:

```bash
python -m email_classifier.cluster_similar \
  --input data/raw/work_all.jsonl \
  --representatives-output data/processed/work_representatives.jsonl \
  --assignments-output data/processed/work_group_assignments.jsonl
```

Optional but recommended: manually label the largest representative groups first. The UI shows how many emails each representative covers:

```bash
python -m email_classifier.review_ui \
  --input data/processed/work_representatives.jsonl \
  --output data/raw/work_representatives_reviewed.jsonl \
  --port 8088
```

Label the remaining representative emails offline with OpenAI. This writes `label` only when model confidence is at or above the threshold; lower-confidence records are marked `needs_review`:

```bash
export OPENAI_API_KEY='openai-api-key'

python -m email_classifier.openai_label \
  --input data/processed/work_representatives.jsonl \
  --output data/raw/work_representatives_openai_labeled.jsonl \
  --min-confidence 0.85 \
  --workers 8
```

Run the local review UI to correct OpenAI labels and write reviewed representative records:

```bash
python -m email_classifier.review_ui \
  --input data/raw/work_representatives_openai_labeled.jsonl \
  --output data/raw/work_representatives_reviewed.jsonl \
  --port 8088
```

Expand representative labels back to every exported email, then train the lightweight sklearn classifier:

```bash
python -m email_classifier.expand_group_labels \
  --raw data/raw/work_all.jsonl \
  --assignments data/processed/work_group_assignments.jsonl \
  --labeled-representatives data/raw/work_representatives_reviewed.jsonl \
  --output data/raw/work_reviewed.jsonl
```

```bash
python -m email_classifier.train_sklearn \
  data/raw/work_reviewed.jsonl \
  --model-output models/sklearn_classifier.joblib \
  --metrics-output models/sklearn_classifier_metrics.json \
  --body-chars 300
```

The deployable sklearn model is around 5 MB. The current work-account candidate measured `0.8902305159165752` validation accuracy, with `action_needed` recall around `0.797` and `urgent` recall around `0.769`.

FastText can still be trained as a fallback:

```bash
python -m email_classifier.prepare_fasttext \
  data/raw/work_reviewed.jsonl \
  --train-output data/processed/work_train.txt \
  --valid-output data/processed/work_valid.txt \
  --min-train-per-label 50

python -m email_classifier.train \
  --train data/processed/work_train.txt \
  --valid data/processed/work_valid.txt \
  --model-output models/email_classifier.bin \
  --quantized-output models/email_classifier.ftz
```

## Prepare FastText Files

```bash
python -m email_classifier.prepare_fasttext \
  data/raw/office.jsonl data/raw/personal.jsonl \
  --train-output data/processed/train.txt \
  --valid-output data/processed/valid.txt \
  --min-train-per-label 50
```

FastText expects one training example per line:

```text
__label__urgent subject: Example email body...
```

## Train

```bash
python -m email_classifier.train \
  --train data/processed/train.txt \
  --valid data/processed/valid.txt \
  --model-output models/email_classifier.bin \
  --quantized-output models/email_classifier.ftz
```

The defaults are intentionally small. If labels are imbalanced, export more examples or manually review noisy labels before tuning hyperparameters.

The project pins `numpy<2` because the current FastText Python wrapper can fail during prediction with NumPy 2.x.

Training uses headers and Gmail snippets by default because full Gmail bodies often contain quoted threads, signatures, and footer text. Add `--include-body` only if validation improves with full body text.

The API can apply a small keyword guard before model prediction when `CLASSIFIER_USE_RULES=true`. The Render sklearn deployment sets `CLASSIFIER_USE_RULES=false` so classification is driven by the trained model instead of case-by-case phrase overrides.

## Smoke Test

The repo includes a tiny synthetic fixture:

```bash
python -m email_classifier.prepare_fasttext \
  examples/sample_emails.jsonl \
  --train-output data/processed/sample_train.txt \
  --valid-output data/processed/sample_valid.txt

python -m email_classifier.train \
  --train data/processed/sample_train.txt \
  --valid data/processed/sample_valid.txt \
  --model-output models/sample_classifier.bin \
  --epoch 5 \
  --dim 16
```

## Run API

```bash
export CLASSIFIER_BACKEND=sklearn
export SKLEARN_MODEL_PATH=models/sklearn_classifier.joblib
export CLASSIFIER_API_KEY=change-me
uvicorn email_classifier.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/` for a small test page where you can paste email text and see the predicted label.

Example request from n8n:

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change-me' \
  -d '{
    "subject": "Can you approve this today?",
    "from": "person@example.com",
    "snippet": "Need your approval before EOD",
    "body": "Please approve this today so we can proceed."
  }'
```

Example response:

```json
{
  "label": "action_needed",
  "confidence": 0.82,
  "candidates": [
    {"label": "action_needed", "confidence": 0.82},
    {"label": "urgent", "confidence": 0.11},
    {"label": "follow_up", "confidence": 0.04}
  ]
}
```

## Deployment Notes

Train locally, then deploy only:

- `src/email_classifier`
- `requirements.inference.txt`
- `.env`
- `models/sklearn_classifier.joblib`
- `models/email_classifier.ftz`

For a tiny VPS, run with one worker:

```bash
uvicorn email_classifier.api:app --host 0.0.0.0 --port 8000 --workers 1
```

Keep the Gmail OAuth/export tooling off the VPS if n8n already handles new email ingestion.

## Render Deploy

The repo includes:

- `Dockerfile`
- `render.yaml`
- `.dockerignore`

Push the repo to GitHub, then create a Render Blueprint from the repo. Render will build the Docker image and run:

```bash
uvicorn email_classifier.api:app --host 0.0.0.0 --port $PORT --workers 1
```

The deploy uses `models/sklearn_classifier.joblib` by default. `models/email_classifier.ftz` is still copied as a fallback model. Raw email exports and training files are ignored.

After deploy, open:

```text
https://your-service.onrender.com/
```

Use the generated `CLASSIFIER_API_KEY` from Render's environment variables in the page's API key field, or clear `CLASSIFIER_API_KEY` for public testing.

Use `/ready` as the n8n preflight endpoint. It loads the model and returns only after the classifier is actually ready:

```text
https://your-service.onrender.com/ready
```

For the Gmail workflow, put this before the real `/classify` request:

1. Gmail Trigger.
2. HTTP Request: `GET /ready`, retry 10 times, wait 10 seconds between tries, timeout 120 seconds.
3. Get Gmail Labels, with execute once enabled.
4. Prepare Classifier Payload.
5. HTTP Request: `POST /classify`, retry 5 times, wait 10 seconds between tries, timeout 120 seconds.
6. Map Label ID.
7. Add Predicted Gmail Label.

This avoids sending real emails to the classifier while Render is still returning its "Application loading" HTML page.

For extra reliability, create a separate n8n keep-alive workflow:

1. Schedule Trigger every 10 minutes.
2. HTTP Request: `GET /ready`, retry 3 times, timeout 60 seconds.

The preflight step is still required because free hosting can sleep or restart despite keep-alives.

## Vercel Note

Vercel can run FastAPI on Python Functions, but this project is a better fit for Render or a VPS because it uses a native FastText package and loads a model file. Use Vercel only if you are comfortable debugging Python serverless packaging and cold starts.

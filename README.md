# Email FastText Classifier

Small local pipeline for training a label classifier from Gmail messages that are already labelled by Dimension or Gmail.

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
2. Convert exported JSONL into FastText supervised training files.
3. Train a small FastText model locally.
4. Deploy only the tiny `/classify` API and `models/email_classifier.bin` on the VPS.
5. n8n sends new unlabelled email fields to `/classify`, then applies the returned label in Gmail.

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

The API also applies a small keyword guard before FastText for obvious labels such as `payment`, `meeting`, `newsletter`, `urgent`, and `follow_up`. This is intentional because the current exported dataset has no examples for `follow_up`/`misc` and very few examples for some other labels.

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
export MODEL_PATH=models/email_classifier.ftz
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
- `requirements.txt`
- `requirements.inference.txt`
- `.env`
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

The deploy uses `models/email_classifier.ftz`, not the large `.bin` model. Raw email exports and training files are ignored.

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

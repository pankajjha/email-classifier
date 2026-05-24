# Email Classifier Handover

Last updated: 2026-05-24

## Current State

This repo contains the FastAPI Gmail classifier service used by n8n.

Render service:

- Base URL: `https://email-classifier-wdbu.onrender.com`
- Health endpoint: `GET /health`
- Readiness endpoint: `GET /ready`
- Classifier endpoint: `POST /classify`
- Latest pushed commit: `421928c Add sklearn email classifier backend`

The `/ready` endpoint returns after the configured model is loaded:

```json
{"status":"ready","backend":"sklearn","model":"/app/models/sklearn_classifier.joblib"}
```

## Offline Relabeling / Retraining Status

On 2026-05-24, a new offline training-data pipeline was added:

```text
Gmail all-mail export
-> strip reply/signature/template noise
-> group similar emails
-> manually review largest representative groups
-> OpenAI-label remaining representatives
-> expand representative labels to all messages
-> prepare FastText train/validation
-> train candidate model
```

Work account progress:

- Exported `6,070` work emails to ignored local file `data/raw/work_all.jsonl`.
- Grouped them into `2,276` representative/template groups.
- Pankaj manually reviewed `101` representative groups, covering `3,681` emails.
- OpenAI labeled the remaining `2,175` representative groups.
- Expanded labels back to all `6,070` exported emails in `data/raw/work_reviewed.jsonl`.
- Prepared `5,258` training examples and `911` validation examples.
- Trained candidate model `models/email_classifier_work_candidate.ftz`.
- Candidate validation precision/recall: `0.8759604829857299`.
- Current old production model on the same validation split: `0.6784`.
- Promoted candidate locally by copying it to `models/email_classifier.ftz`.
- Not deployed to Render yet unless a later session says otherwise.

Notes:

- Raw exports and generated training data live under ignored `data/raw/` and `data/processed/`; do not commit mailbox data.
- `models/sklearn_classifier.joblib` is the current primary deployable artifact.
- `models/email_classifier.ftz` remains a FastText fallback artifact.
- `marketing` and `misc` remain underrepresented in validation, so do not overinterpret those per-label scores.

Classifier backend update on 2026-05-24:

- Added `CLASSIFIER_BACKEND=fasttext|sklearn|semantic`.
- Render is configured for `CLASSIFIER_BACKEND=sklearn` and `CLASSIFIER_USE_RULES=false`.
- The deployable sklearn model is `models/sklearn_classifier.joblib`.
- It uses word/character TF-IDF plus Logistic Regression with the newest 300 body characters included.
- Validation accuracy: `0.8902305159165752`.
- Important validation recalls:
  - `action_needed`: `0.7970297029702971`
  - `urgent`: `0.7692307692307693`
  - `follow_up`: `0.375`
- A sentence-transformer semantic candidate was tested but not promoted because it underperformed (`0.7376509330406147` with 1200 body chars, `0.7903402854006586` with headers/snippet only).

## Important Security Note

API keys were pasted during setup. Do not copy them into docs or future prompts. Rotate these after confirming the workflows are stable:

- n8n API key
- classifier API key used by n8n
- any OpenAI API key used for draft generation

## Active n8n Workflows

Personal workflow:

- URL: `https://n8n.pankajjha.me/workflow/f1fB2D0v4UD5mjy4`
- ID: `f1fB2D0v4UD5mjy4`
- Name: `Personal Gmail Classifier`
- Status after update: active

Work workflow:

- URL: `https://n8n.pankajjha.me/workflow/4dA4s3KTBnLCpFEy`
- ID: `4dA4s3KTBnLCpFEy`
- Name: `Work Gmail Classifier`
- Status after update: active
- Latest workflow version after draft gating update: `14f83aa5-79da-40d9-a6fc-fbb3ee8b8123`

Personal workflow still has this shape:

```text
Gmail Trigger
-> Wait For Classifier Ready
-> Get Gmail Labels
-> Prepare Classifier Payload
-> Classify Email
-> Map Label ID
-> Add Predicted Gmail Label
```

Work workflow now has auto-draft enabled:

```text
Gmail Trigger
-> Wait For Classifier Ready
-> Ensure Draft Ready Label
-> Get Gmail Labels
-> Prepare Classifier Payload
-> Classify Email
-> Map Label ID
   -> Add Predicted Gmail Label
   -> IF shouldAutoDraft is true
      -> Prepare Draft Reply Payload
      -> Generate Draft Reply
      -> Parse Draft Reply
      -> Create Gmail Draft Reply
      -> Keep Draft Context
      -> Add Draft Ready Thread Label
```

Reliability settings applied:

- `Wait For Classifier Ready`
  - HTTP method: `GET`
  - URL: `https://email-classifier-wdbu.onrender.com/ready`
  - `executeOnce`: true
  - retry on fail: true
  - max tries: 10
  - wait between tries: 10000 ms
  - timeout: 120000 ms
- `Get Gmail Labels`
  - `executeOnce`: true
- `Classify Email`
  - retry on fail: true
  - max tries: 5
  - wait between tries: 10000 ms
  - timeout: 120000 ms

Reason for this change: Render returns an "Application loading" HTML page during cold starts. The readiness gate wakes the service before real email classification runs.

## Current Labels

Canonical classifier labels:

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

Gmail label mapping used in the workflows:

- `urgent` -> `1: urgent`
- `action_needed` -> `2: action needed`
- `follow_up` -> `3: follow up`
- `awaiting_reply` -> `4: awaiting reply`
- `meeting` -> `5: meeting`
- `fyi` -> `6: fyi`
- `done` -> `7: done`
- `payment` -> `8: payment`
- `newsletter` -> `9: newsletter`
- `marketing` -> `10: marketing`
- `misc` -> `misc`

## Known Prior Issues Fixed

1. `Map Label ID` emitted one item even when many emails arrived.
   - Cause: node logic was not batch-safe.
   - Fixed previously by using all input items and preserving per-message data.

2. `Add Predicted Gmail Label` showed `{{$json.gmailLabelId}}` as undefined.
   - Cause: output item from map step did not consistently carry `gmailLabelId`.
   - Fixed previously in the map node.

3. Gmail `labelId not found`.
   - Cause: classifier predicted labels that did not yet exist in Gmail.
   - Fixed by creating missing Gmail labels and mapping classifier names to actual Gmail label IDs.

4. Render "Application loading" HTML broke `Classify Email`.
   - Cause: Render cold start.
   - Fixed by adding `/ready` in the API and `Wait For Classifier Ready` in both n8n workflows.

## Next Feature: Auto Draft Replies

Goal: for emails classified as `urgent` or `action_needed`, create a Gmail draft reply automatically. Do not auto-send.

Status on 2026-05-23:

- Implemented on Work Gmail Classifier only.
- Draft trigger labels are `urgent`, `action_needed`, and `follow_up`.
- Not yet implemented on Personal Gmail Classifier.
- Created an n8n encrypted credential named `OpenAI API`; do not put the OpenAI key directly in workflow JSON.
- Uses OpenAI Responses API with `gpt-4.1-mini` and strict JSON schema output.
- Draft-ready label is `AI/draft ready`.
- The workflow creates Gmail drafts only, never sends replies.
- After a draft is created, it labels the Gmail thread with `AI/draft ready`.
- Dedupe is in `Prepare Classifier Payload`: it skips messages/threads already carrying `AI/draft ready`, already classified messages, and Gmail system messages labelled `DRAFT`, `SENT`, `TRASH`, or `SPAM`.
- OpenAI structured-output smoke test returned HTTP 200 and valid JSON in `output[].content[].text`.

Draft gating update on 2026-05-24:

- Work workflow drafts only when `shouldAutoDraft` is true.
- `shouldAutoDraft` is computed in `Map Label ID`.
- Required draft conditions:
  - classifier label is `urgent`, `action_needed`, or `follow_up`
  - original `To` header directly contains `pankaj.jha@ambak.com`
  - sender email can be extracted
  - sender is not `pankaj.jha@ambak.com`
  - sender does not match deterministic automated-mailbox checks
- Automated-mailbox checks block local parts and markers such as `no-reply`, `noreply`, `notification`, `mailer`, `bot`, `alerts`, `digest`, `newsletter`, plus domains/markers for GitHub, tl;dv/tldv, Slack, Jira/Atlassian, Linear, Notion, Calendly, Google notifications, SendGrid, Mailgun, Amazon SES, and Mailchimp.
- `Prepare Draft Reply Payload` rechecks `shouldAutoDraft` and throws before OpenAI if a future connection mistake routes a blocked item into the draft branch.
- Skipped reasons are visible in execution data as `draftBlockReason`, for example `not_directly_to_owned_mailbox:pankaj.jha@ambak.com` or `automated_sender_domain:github.com`.

Sender blacklist mining update on 2026-05-24:

- Added `email_classifier.extract_sender_blacklist` to mine likely automated senders from ignored local Gmail exports.
- Work mailbox mining found `6,070` records, `214` unique senders, and `65` exact automated/mailbox sender candidates.
- Generated blacklist candidate file: `data/processed/work_sender_blacklist_candidates.json` (ignored; do not commit because it contains private sender addresses).
- Work n8n workflow now includes an exact automated-sender blacklist copied from that candidate file.
- Domain checks were narrowed to machine-style domains from the export, and broad root domain blocking was avoided where a domain can plausibly send human email.
- App markers such as GitHub are checked against sender/header metadata, not email subject text, so a human email about GitHub is less likely to be blocked.

Recommended v1 design:

```text
Existing classifier workflow
-> IF classifierLabel is urgent, action_needed, or follow_up
-> Generate Draft Reply with OpenAI
-> Create Gmail Draft Reply
-> Add label "AI/draft ready" or "drafted"
```

Strong guardrails:

- Never send automatically.
- Only create drafts for `urgent`, `action_needed`, and `follow_up`.
- Skip emails already labelled `drafted` or `AI/draft ready`.
- Keep a human approval step in Gmail.
- Keep drafts short and professional.
- Do not make commitments, quote prices, approve requests, confirm payments, or promise timelines unless the email already contains the exact approved answer.
- If information is missing, draft a reply asking for the missing detail.
- For sensitive/legal/financial/payment disputes, draft a cautious acknowledgement only.

Suggested OpenAI model strategy:

- Start with a mini/small model for low cost and good enough quality.
- Use structured JSON output so n8n can safely parse `subject`, `body`, `needsReview`, and `reason`.
- Keep max output low, around 250-400 words.
- Truncate long email bodies before sending to OpenAI.

Suggested draft output schema:

```json
{
  "replySubject": "string",
  "replyBody": "string",
  "needsReview": true,
  "reason": "string"
}
```

Suggested prompt shape:

```text
You draft Gmail replies for Pankaj. Create a concise, professional reply draft.

Rules:
- Do not send the email.
- Do not invent facts.
- Do not promise delivery dates, discounts, approvals, refunds, or payments.
- If the sender asks for an action and context is missing, ask for the missing detail.
- If the email is urgent, acknowledge urgency and say we will review/respond.
- Keep tone calm, direct, and useful.
- Output only JSON matching the schema.

Email:
Subject: {{subject}}
From: {{from}}
Label: {{classifierLabel}}
Snippet: {{snippet}}
Body:
{{body}}
```

Drafting workflow detail to implement next:

1. Add an IF node after `Map Label ID` or after `Add Predicted Gmail Label`.
2. True branch condition:
   - `$json.classifierLabel` is `urgent` OR `action_needed`.
3. Add OpenAI HTTP/API node:
   - Generate structured JSON draft.
4. Add Gmail node:
   - Create draft reply, ideally linked to the original thread.
5. Add Gmail label:
   - `AI/draft ready` or `drafted`.
6. Add dedupe:
   - If original email/thread already has draft-ready label, skip.

Open question for next session:

- Should auto-draft run on both personal and work accounts, or only work first?
- Preferred draft label name: `AI/draft ready`, `drafted`, or something else?
- Should the draft mention "I" or "we" for work emails?

## Next Session Prompt

Copy this into a fresh Codex session:

```text
We are in /Users/pankajjha/development/pankaj/ml-classifer.

Read docs/HANDOVER_AUTO_DRAFT.md first. Continue from there.

Goal: implement auto draft replies in n8n for Gmail emails classified as urgent, action_needed, or follow_up. Do not auto-send. Create Gmail draft replies only, then label the thread/email as draft-ready so duplicates are skipped.

Current classifier service is live on Render:
- https://email-classifier-wdbu.onrender.com/ready
- https://email-classifier-wdbu.onrender.com/classify

Active n8n workflows:
- Personal Gmail Classifier: f1fB2D0v4UD5mjy4
- Work Gmail Classifier: 4dA4s3KTBnLCpFEy

Both workflows already include Wait For Classifier Ready before classification.

I have an OpenAI API key and want the easiest, cheapest reliable approach. Prefer a small/mini OpenAI model with structured JSON output. Use n8n API if I provide the key. Never echo secrets. First inspect both workflows, then patch the workflow(s) to add:
1. IF classifierLabel is urgent, action_needed, or follow_up
2. OpenAI draft generation
3. Gmail create draft reply
4. Apply draft-ready label
5. Dedupe so a thread does not get multiple drafts

Start with the work workflow unless I say otherwise.
```

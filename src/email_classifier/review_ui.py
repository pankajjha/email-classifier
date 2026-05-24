from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .openai_label import LABELS
from .text import strip_reply_noise


APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Email Label Review</title>
  <style>
    :root { color-scheme: light; --bg:#f7f8fb; --panel:#fff; --line:#d9e1ea; --text:#172033; --muted:#667085; --primary:#1769e0; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 20px 0 28px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 14px; }
    h1 { margin: 0; font-size: 22px; letter-spacing: 0; }
    .status { color: var(--muted); font-size: 14px; }
    .layout { display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 16px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .list { display: grid; gap: 8px; max-height: calc(100vh - 110px); overflow: auto; }
    .item { border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 9px; cursor: pointer; text-align: left; }
    .item.active { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(23,105,224,.12); }
    .item-title { font-size: 13px; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .item-meta { margin-top: 4px; color: var(--muted); font-size: 12px; }
    .message { display: grid; gap: 12px; }
    .meta { display: grid; gap: 6px; font-size: 14px; }
    .meta b { display: inline-block; min-width: 62px; }
    .body { min-height: 310px; max-height: 42vh; overflow: auto; white-space: pre-wrap; line-height: 1.45; border: 1px solid var(--line); border-radius: 6px; padding: 12px; background: #fbfcfe; }
    .labels { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); gap: 8px; }
    button { border: 1px solid var(--line); border-radius: 6px; background: #fff; min-height: 38px; padding: 0 10px; font: inherit; cursor: pointer; }
    button.primary { background: var(--primary); border-color: var(--primary); color: white; font-weight: 700; }
    button.selected { border-color: var(--primary); box-shadow: 0 0 0 2px rgba(23,105,224,.12); font-weight: 700; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .reason { color: var(--muted); font-size: 13px; }
    @media (max-width: 820px) { .layout { grid-template-columns: 1fr; } .list { max-height: 260px; } }
  </style>
</head>
<body>
<main>
  <header>
    <h1>Email Label Review</h1>
    <div id="status" class="status"></div>
  </header>
  <div class="layout">
    <section class="panel"><div id="list" class="list"></div></section>
    <section class="panel message">
      <div class="meta" id="meta"></div>
      <div class="reason" id="reason"></div>
      <div class="body" id="body"></div>
      <div class="labels" id="labels"></div>
      <div class="actions">
        <button id="save" class="primary">Save</button>
        <button id="skip">Skip</button>
        <button id="acceptedOnly">Export Accepted</button>
      </div>
    </section>
  </div>
</main>
<script>
const labels = __LABELS__;
let records = [];
let index = 0;
let selected = "";

const el = id => document.getElementById(id);
const escapeText = value => (value || "").toString();

async function load() {
  const res = await fetch("/api/records");
  records = await res.json();
  renderList();
  selectFirstPending();
}

function selectFirstPending() {
  const pending = records.findIndex(r => r.review_status !== "reviewed");
  index = pending >= 0 ? pending : 0;
  renderCurrent();
}

function renderList() {
  el("status").textContent = `${records.filter(r => r.review_status === "reviewed").length}/${records.length} reviewed`;
  el("list").innerHTML = "";
  records.forEach((record, i) => {
    const button = document.createElement("button");
    button.className = `item ${i === index ? "active" : ""}`;
    button.innerHTML = `<div class="item-title"></div><div class="item-meta"></div>`;
    button.querySelector(".item-title").textContent = record.subject || "(no subject)";
    const group = record.template_group_size ? ` · ${record.template_group_size} emails` : "";
    button.querySelector(".item-meta").textContent = `${record.label || record.openai_label || "unlabeled"} · ${record.review_status || ""}${group}`;
    button.onclick = () => { index = i; renderCurrent(); };
    el("list").appendChild(button);
  });
}

function renderCurrent() {
  const record = records[index];
  if (!record) return;
  selected = record.label || record.openai_label || "misc";
  el("meta").innerHTML = "";
  for (const [name, value] of [["Subject", record.subject], ["From", record.from], ["Date", record.date], ["Group", record.template_group_size ? `${record.template_group_size} emails` : ""], ["Current", record.openai_label || record.label]]) {
    const div = document.createElement("div");
    const b = document.createElement("b");
    b.textContent = `${name}:`;
    div.appendChild(b);
    div.append(document.createTextNode(value || ""));
    el("meta").appendChild(div);
  }
  el("reason").textContent = `Confidence ${Math.round((record.openai_confidence || 0) * 100)}% · ${record.openai_reason || ""}`;
  el("body").textContent = [record.snippet, record.display_text || record.training_text || record.text || ""].filter(Boolean).join("\\n\\n");
  renderLabelButtons();
  renderList();
}

function renderLabelButtons() {
  el("labels").innerHTML = "";
  labels.forEach(label => {
    const button = document.createElement("button");
    button.textContent = label;
    button.className = label === selected ? "selected" : "";
    button.onclick = () => { selected = label; renderLabelButtons(); };
    el("labels").appendChild(button);
  });
}

async function save(status = "reviewed") {
  const record = records[index];
  await fetch(`/api/records/${encodeURIComponent(record.message_id)}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({label: selected, review_status: status}),
  });
  record.label = selected;
  record.review_status = status;
  const next = records.findIndex((r, i) => i > index && r.review_status !== "reviewed");
  index = next >= 0 ? next : Math.min(index + 1, records.length - 1);
  renderCurrent();
}

el("save").onclick = () => save("reviewed");
el("skip").onclick = () => { index = Math.min(index + 1, records.length - 1); renderCurrent(); };
el("acceptedOnly").onclick = () => window.open("/api/export", "_blank");
load();
</script>
</body>
</html>"""


class ReviewUpdate(BaseModel):
    label: str
    review_status: str = "reviewed"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def create_app(input_path: Path, output_path: Path) -> FastAPI:
    app = FastAPI(title="Email Label Review")

    def load_records() -> list[dict]:
        reviewed_by_id = {record.get("message_id"): record for record in read_jsonl(output_path)}
        merged = []
        for record in read_jsonl(input_path):
            message_id = record.get("message_id")
            current = {**record, **reviewed_by_id.get(message_id, {})}
            current["display_text"] = strip_reply_noise(current.get("training_text") or current.get("text"), max_chars=4000)
            merged.append(current)
        return merged

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return APP_HTML.replace("__LABELS__", json.dumps(list(LABELS)))

    @app.get("/api/records")
    def records() -> list[dict]:
        return load_records()

    @app.post("/api/records/{message_id}")
    def update_record(message_id: str, update: ReviewUpdate) -> dict:
        if update.label not in LABELS:
            raise HTTPException(status_code=400, detail="Invalid label")
        records = load_records()
        matched = None
        for record in records:
            if record.get("message_id") == message_id:
                matched = record
                break
        if not matched:
            raise HTTPException(status_code=404, detail="Message not found")

        reviewed = {record.get("message_id"): record for record in read_jsonl(output_path)}
        reviewed[message_id] = {
            **matched,
            "label": update.label,
            "review_status": update.review_status,
            "review_source": "manual_ui",
        }
        write_jsonl(output_path, list(reviewed.values()))
        return {"ok": True}

    @app.get("/api/export")
    def export_reviewed() -> dict:
        reviewed = [record for record in read_jsonl(output_path) if record.get("label") in LABELS]
        write_jsonl(output_path, reviewed)
        return {"path": str(output_path), "records": len(reviewed)}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local UI to review OpenAI-labeled emails.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8088")))
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    app = create_app(Path(args.input), Path(args.output))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

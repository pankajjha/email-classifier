from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Iterable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .labels import load_label_config, normalize_label
from .text import clean_text


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


def get_service(
    credentials_path: Path,
    token_path: Path | None,
    access_token: str | None = None,
    refresh_token: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
):
    if access_token:
        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri=TOKEN_URI,
            client_id=client_id,
            client_secret=client_secret,
            scopes=SCOPES,
        )
        if refresh_token and client_id and client_secret:
            creds.refresh(Request())
        return build("gmail", "v1", credentials=creds)

    creds = None
    if token_path and token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        if token_path:
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")
    return build("gmail", "v1", credentials=creds)


def list_gmail_labels(service) -> dict[str, str]:
    response = service.users().labels().list(userId="me").execute()
    return {label["name"].lower(): label["id"] for label in response.get("labels", [])}


def resolve_label_ids(service, label_config: dict[str, list[str]]) -> dict[str, str]:
    gmail_labels = list_gmail_labels(service)
    resolved = {}
    for canonical, aliases in label_config.items():
        candidates = [canonical, *aliases]
        for candidate in candidates:
            label_id = gmail_labels.get(candidate.lower())
            if label_id:
                resolved[canonical] = label_id
                break
    return resolved


def iter_message_ids(service, label_id: str, max_results: int | None = None) -> Iterable[str]:
    seen = 0
    page_token = None
    while True:
        page_size = min(500, max_results - seen) if max_results else 500
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=[label_id],
                maxResults=page_size,
                pageToken=page_token,
                includeSpamTrash=False,
            )
            .execute()
        )
        for message in response.get("messages", []):
            yield message["id"]
            seen += 1
            if max_results and seen >= max_results:
                return
        page_token = response.get("nextPageToken")
        if not page_token:
            return


def header_value(payload: dict, name: str) -> str:
    for header in payload.get("headers", []):
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def decode_body(data: str | None) -> str:
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except Exception:
        return ""


def extract_parts(payload: dict) -> tuple[list[str], list[str]]:
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part: dict) -> None:
        mime_type = part.get("mimeType", "")
        body_data = part.get("body", {}).get("data")
        if mime_type == "text/plain":
            plain_parts.append(decode_body(body_data))
        elif mime_type == "text/html":
            html_parts.append(decode_body(body_data))
        for child in part.get("parts", []) or []:
            walk(child)

    walk(payload)
    return plain_parts, html_parts


def get_message_record(service, message_id: str, account: str, label: str) -> dict:
    message = (
        service.users()
        .messages()
        .get(userId="me", id=message_id, format="full")
        .execute()
    )
    payload = message.get("payload", {})
    plain_parts, html_parts = extract_parts(payload)
    text = "\n".join(plain_parts or html_parts)
    return {
        "account": account,
        "label": normalize_label(label),
        "message_id": message.get("id"),
        "thread_id": message.get("threadId"),
        "internal_date": message.get("internalDate"),
        "subject": header_value(payload, "Subject"),
        "from": header_value(payload, "From"),
        "to": header_value(payload, "To"),
        "cc": header_value(payload, "Cc"),
        "date": header_value(payload, "Date"),
        "snippet": clean_text(message.get("snippet", "")),
        "text": clean_text(text),
    }


def export_account(
    account: str,
    credentials_path: Path,
    token_path: Path | None,
    output_path: Path,
    label_config_path: Path | None,
    max_per_label: int | None,
    access_token: str | None = None,
    refresh_token: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> None:
    service = get_service(
        credentials_path=credentials_path,
        token_path=token_path,
        access_token=access_token,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
    )
    label_config = load_label_config(label_config_path)
    resolved = resolve_label_ids(service, label_config)
    missing = sorted(set(label_config) - set(resolved))
    if missing:
        print(f"Missing labels in Gmail for {account}: {', '.join(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as file:
        for label, label_id in resolved.items():
            count = 0
            for message_id in iter_message_ids(service, label_id, max_per_label):
                record = get_message_record(service, message_id, account, label)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                written += 1
            print(f"{account}: exported {count} messages for {label}")
    print(f"Wrote {written} records to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Gmail messages already tagged with target labels.")
    parser.add_argument("--account", required=True, help="Stable account name, e.g. office or personal.")
    parser.add_argument("--credentials", default="credentials.json", help="OAuth client JSON from Google Cloud.")
    parser.add_argument("--token", help="Per-account OAuth token path for local browser OAuth flow.")
    parser.add_argument("--access-token", default=os.getenv("GOOGLE_ACCESS_TOKEN"), help="Short-lived access token from OAuth Playground.")
    parser.add_argument("--refresh-token", default=os.getenv("GOOGLE_REFRESH_TOKEN"), help="Refresh token from OAuth Playground.")
    parser.add_argument("--client-id", default=os.getenv("GOOGLE_CLIENT_ID"), help="OAuth client ID for refreshing an OAuth Playground refresh token.")
    parser.add_argument("--client-secret", default=os.getenv("GOOGLE_CLIENT_SECRET"), help="OAuth client secret for refreshing an OAuth Playground refresh token.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--labels", default="config/labels.json", help="Canonical labels and Gmail aliases.")
    parser.add_argument("--max-per-label", type=int, default=None, help="Limit per label for test exports.")
    args = parser.parse_args()
    if not args.token and not args.access_token:
        parser.error("Provide either --token for local OAuth or --access-token from OAuth Playground.")
    return args


def main() -> None:
    args = parse_args()
    export_account(
        account=args.account,
        credentials_path=Path(args.credentials),
        token_path=Path(args.token) if args.token else None,
        output_path=Path(args.output),
        label_config_path=Path(args.labels) if args.labels else None,
        max_per_label=args.max_per_label,
        access_token=args.access_token,
        refresh_token=args.refresh_token,
        client_id=args.client_id,
        client_secret=args.client_secret,
    )


if __name__ == "__main__":
    main()

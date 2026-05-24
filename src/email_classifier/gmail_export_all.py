from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

from .gmail_export import (
    get_message_record,
    get_service,
    list_gmail_labels,
)


def iter_all_message_ids(
    service,
    query: str | None = None,
    max_messages: int | None = None,
    include_spam_trash: bool = False,
) -> Iterable[str]:
    seen = 0
    page_token = None
    while True:
        page_size = min(500, max_messages - seen) if max_messages else 500
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                maxResults=page_size,
                pageToken=page_token,
                includeSpamTrash=include_spam_trash,
                q=query,
            )
            .execute()
        )
        for message in response.get("messages", []):
            yield message["id"]
            seen += 1
            if max_messages and seen >= max_messages:
                return
        page_token = response.get("nextPageToken")
        if not page_token:
            return


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("message_id"):
                ids.add(record["message_id"])
    return ids


def label_maps(service) -> tuple[dict[str, str], dict[str, str]]:
    response = service.users().labels().list(userId="me").execute()
    name_to_id = {label["name"]: label["id"] for label in response.get("labels", [])}
    id_to_name = {label_id: name for name, label_id in name_to_id.items()}
    return name_to_id, id_to_name


def export_all(
    account: str,
    output_path: Path,
    credentials_path: Path,
    token_path: Path | None,
    access_token: str | None,
    refresh_token: str | None,
    client_id: str | None,
    client_secret: str | None,
    query: str | None,
    max_messages: int | None,
    include_spam_trash: bool,
    sleep_seconds: float,
) -> None:
    service = get_service(
        credentials_path=credentials_path,
        token_path=token_path,
        access_token=access_token,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
    )
    _, id_to_name = label_maps(service)
    existing_ids = read_existing_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0
    with output_path.open("a", encoding="utf-8") as file:
        for message_id in iter_all_message_ids(
            service,
            query=query,
            max_messages=max_messages,
            include_spam_trash=include_spam_trash,
        ):
            if message_id in existing_ids:
                skipped += 1
                continue
            record = get_message_record(service, message_id, account=account, label="")
            label_ids = record.pop("label_ids", [])
            record["gmail_label_ids"] = label_ids
            record["gmail_label_names"] = [id_to_name.get(label_id, label_id) for label_id in label_ids]
            record["label"] = ""
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
            if exported % 100 == 0:
                print(f"exported={exported} skipped_existing={skipped}")
            if sleep_seconds:
                time.sleep(sleep_seconds)

    print(f"Wrote {exported} new records to {output_path}; skipped {skipped} existing records.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all Gmail messages for offline labeling.")
    parser.add_argument("--account", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--credentials", default="credentials.json")
    parser.add_argument("--token")
    parser.add_argument("--access-token", default=os.getenv("GOOGLE_ACCESS_TOKEN"))
    parser.add_argument("--refresh-token", default=os.getenv("GOOGLE_REFRESH_TOKEN"))
    parser.add_argument("--client-id", default=os.getenv("GOOGLE_CLIENT_ID"))
    parser.add_argument("--client-secret", default=os.getenv("GOOGLE_CLIENT_SECRET"))
    parser.add_argument("--query", default="-in:chats", help="Gmail search query. Default excludes chats.")
    parser.add_argument("--max-messages", type=int)
    parser.add_argument("--include-spam-trash", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()
    if not args.token and not args.access_token:
        parser.error("Provide either --token or --access-token.")
    return args


def main() -> None:
    args = parse_args()
    export_all(
        account=args.account,
        output_path=Path(args.output),
        credentials_path=Path(args.credentials),
        token_path=Path(args.token) if args.token else None,
        access_token=args.access_token,
        refresh_token=args.refresh_token,
        client_id=args.client_id,
        client_secret=args.client_secret,
        query=args.query,
        max_messages=args.max_messages,
        include_spam_trash=args.include_spam_trash,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()

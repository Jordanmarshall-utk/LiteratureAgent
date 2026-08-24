#!/usr/bin/env python
"""Organize the actual Google Drive source folder from a triage manifest.

Default behavior is dry-run: it writes a move plan but does not modify Drive.
Use --apply to create destination folders and move files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive"]


@dataclass
class DriveMove:
    paper_slug: str
    file_id: str
    file_name: str
    current_parent_id: str
    destination_folder_name: str
    destination_folder_id: str
    recommended_folder: str
    confidence: str
    action: str
    reason: str


def load_drive_service(token_file: Path, client_secrets: Path | None = None):
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from google.auth.exceptions import RefreshError
    except Exception as exc:  # pragma: no cover - dependency message path
        raise SystemExit(
            "Google API packages are required. Install google-api-python-client, "
            "google-auth, and google-auth-oauthlib."
        ) from exc

    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_file.write_text(creds.to_json(), encoding="utf-8")
        except RefreshError as exc:
            print(f"[WARN] Existing Google token could not refresh: {exc}")
            print("[INFO] Re-authenticating with Drive organize scope.")
            creds = None
    if not creds or not creds.valid:
        if not client_secrets or not client_secrets.exists():
            raise SystemExit(
                f"Token is missing/invalid and client secrets were not found: {client_secrets}"
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
        creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")
    return build("drive", "v3", credentials=creds)


def load_processed_manifest(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    folder_id = str(data.get("folder_id") or "").strip()
    processed = data.get("processed_file_ids") or {}
    if not folder_id:
        raise SystemExit(f"Manifest does not contain folder_id: {path}")
    if not isinstance(processed, dict) or not processed:
        raise SystemExit(f"Manifest does not contain processed_file_ids: {path}")
    return folder_id, processed


def ensure_drive_folder(service, parent_id: str, folder_name: str, apply: bool) -> str:
    query = (
        f"'{parent_id}' in parents and "
        f"mimeType = '{DRIVE_FOLDER_MIME}' and "
        f"name = '{folder_name.replace(chr(39), chr(92) + chr(39))}' and trashed = false"
    )
    resp = (
        service.files()
        .list(
            q=query,
            fields="files(id,name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = resp.get("files") or []
    if files:
        return files[0]["id"]
    if not apply:
        return f"DRY_RUN_CREATE:{folder_name}"
    meta = {"name": folder_name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]}
    created = (
        service.files()
        .create(body=meta, fields="id", supportsAllDrives=True)
        .execute()
    )
    return created["id"]


def move_drive_file(service, file_id: str, source_parent: str, dest_parent: str) -> None:
    service.files().update(
        fileId=file_id,
        addParents=dest_parent,
        removeParents=source_parent,
        fields="id, parents",
        supportsAllDrives=True,
    ).execute()


def destination_name(recommended_folder: str) -> str:
    if recommended_folder == "01_device_stability_extraction":
        return "LiteratureAgent_device_stability_extraction"
    return "LiteratureAgent_materials_reviews_reference"


def build_moves(
    triage: pd.DataFrame,
    processed: dict[str, dict[str, Any]],
    parent_id: str,
    dest_ids: dict[str, str],
) -> list[DriveMove]:
    by_slug: dict[str, tuple[str, dict[str, Any]]] = {}
    for file_id, info in processed.items():
        slug = str(info.get("paper_slug") or "")
        if slug:
            by_slug[slug] = (file_id, info)

    rows: list[DriveMove] = []
    for _, row in triage.iterrows():
        slug = str(row.get("paper_slug") or "")
        recommended = str(row.get("recommended_folder") or "")
        if not slug or recommended not in {
            "01_device_stability_extraction",
            "02_materials_reviews_reference",
        }:
            continue
        if slug not in by_slug:
            rows.append(
                DriveMove(
                    paper_slug=slug,
                    file_id="",
                    file_name="",
                    current_parent_id=parent_id,
                    destination_folder_name=destination_name(recommended),
                    destination_folder_id=dest_ids.get(destination_name(recommended), ""),
                    recommended_folder=recommended,
                    confidence=str(row.get("confidence") or ""),
                    action="skip_no_drive_file_id",
                    reason=str(row.get("reason") or ""),
                )
            )
            continue
        file_id, info = by_slug[slug]
        dest_folder = destination_name(recommended)
        rows.append(
            DriveMove(
                paper_slug=slug,
                file_id=file_id,
                file_name=str(info.get("name") or ""),
                current_parent_id=parent_id,
                destination_folder_name=dest_folder,
                destination_folder_id=dest_ids.get(dest_folder, ""),
                recommended_folder=recommended,
                confidence=str(row.get("confidence") or ""),
                action="move",
                reason=str(row.get("reason") or ""),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triage-manifest", required=True, type=Path)
    parser.add_argument("--drive-processed-manifest", required=True, type=Path)
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path("secrets/google_drive_token.json"),
        help="OAuth token with Drive scope.",
    )
    parser.add_argument(
        "--client-secrets",
        type=Path,
        default=Path("secrets/google_drive_oauth_client.json"),
        help="OAuth client secrets, used only if the token must be regenerated.",
    )
    parser.add_argument("--out-plan", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Actually create folders and move files in Google Drive.")
    args = parser.parse_args()

    triage = pd.read_csv(args.triage_manifest, low_memory=False)
    parent_id, processed = load_processed_manifest(args.drive_processed_manifest)

    service = load_drive_service(args.token_file, args.client_secrets)
    folder_names = [
        "LiteratureAgent_device_stability_extraction",
        "LiteratureAgent_materials_reviews_reference",
    ]
    dest_ids = {
        name: ensure_drive_folder(service, parent_id, name, args.apply)
        for name in folder_names
    }
    moves = build_moves(triage, processed, parent_id, dest_ids)

    if args.apply:
        for move in moves:
            if move.action != "move":
                continue
            if move.destination_folder_id.startswith("DRY_RUN_CREATE:"):
                raise SystemExit("Internal error: destination folder was not created in apply mode.")
            move_drive_file(service, move.file_id, parent_id, move.destination_folder_id)

    args.out_plan.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([m.__dict__ for m in moves]).to_csv(args.out_plan, index=False, encoding="utf-8-sig")

    move_count = sum(1 for m in moves if m.action == "move")
    skipped = len(moves) - move_count
    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"Google Drive organization {mode}")
    print(f"Parent folder: {parent_id}")
    print(f"Move candidates: {move_count}")
    print(f"Skipped/no ID: {skipped}")
    print(f"Plan CSV: {args.out_plan}")
    print("Destination folders:")
    for name, folder_id in dest_ids.items():
        print(f"  {name}: {folder_id}")


if __name__ == "__main__":
    main()

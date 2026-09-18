#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from auth import (  # pyright: ignore[reportMissingImports]
    BASE_BACKUP_FOLDER,
    SHAREPOINT_DRIVE_ID,
    _console_prompt_callback,
    get_graph_client,
)
from download_graph import _list_children  # pyright: ignore[reportMissingImports]


REQUESTED_BACKUPS: dict[str, str] = {
    "andrew": "onedrive_backups/ajj/2026-05-12/fluoro-r2_round2_andrew.json",
    "jenna": "onedrive_backups/Jenna Nova 1/2026-05-19/fluoro-r2_round2_jenna.json",
    "paris": "onedrive_backups/root/2026-05-13/fluoro-r2_round2_paris.json",
    "scott": "onedrive_backups/SAB/2026-05-19/fluoro-r2_round2_scott.json",
    "sonia": "onedrive_backups/tylerjenkins/2026-05-13/fluoro-r2_round2_sonia.json",
    "tyler": "onedrive_backups/tylerjenkins/2026-05-15/fluoro-r2_round2_tyler.json",
}

DEFAULT_DEST = PROJECT_ROOT / "data" / "round2_multi_reviewer_review_2026-06-01"
INSTALL_IMAGE_ROOT = PROJECT_ROOT.parent / "data"
LOCAL_FALLBACK_DIR = INSTALL_IMAGE_ROOT / "fluoro_images_round_1"


def _image_has_annotation_content(record: dict[str, Any]) -> bool:
    annotations = record.get("annotations", {}) or {}
    if not isinstance(annotations, dict):
        return False

    for raw in annotations.values():
        if isinstance(raw, dict):
            value = raw.get("value")
            if value is not None:
                return True
            if raw.get("flag", False) or str(raw.get("note", "")).strip():
                return True
        elif raw is not None:
            return True

    return False


def _count_annotation_values(record: dict[str, Any]) -> int:
    annotations = record.get("annotations", {}) or {}
    if not isinstance(annotations, dict):
        return 0

    total = 0
    for raw in annotations.values():
        if isinstance(raw, dict):
            value = raw.get("value")
        else:
            value = raw
        if value is not None:
            total += 1
    return total


def _annotation_counts(data: dict[str, Any]) -> dict[str, int]:
    images = data.get("images", []) or []
    if not isinstance(images, list):
        images = []

    image_records = [record for record in images if isinstance(record, dict)]
    return {
        "assigned_images": len(image_records),
        "annotated_images": sum(
            1 for record in image_records if _image_has_annotation_content(record)
        ),
        "annotated_landmarks": sum(_count_annotation_values(record) for record in image_records),
    }


def _counts_for_json_path(json_path: Path) -> dict[str, int]:
    return _annotation_counts(json.loads(json_path.read_text(encoding="utf-8")))


def _print_count_report(rows: Iterable[dict[str, Any]]) -> None:
    print("\nRound 2 annotation counts:")
    print(f"{'reviewer':<10} {'annotated':>9} {'assigned':>9} {'landmarks':>10} source")
    for row in rows:
        print(
            f"{row['reviewer']:<10} "
            f"{row['annotated_images']:>9} "
            f"{row['assigned_images']:>9} "
            f"{row['annotated_landmarks']:>10} "
            f"{row.get('source', '')}"
        )


def _remote_candidates(requested_path: str) -> list[str]:
    requested = requested_path.strip("/")
    candidates = [requested]

    if requested.startswith("onedrive_backups/"):
        tail = requested.split("/", 1)[1]
        candidates.append(f"{BASE_BACKUP_FOLDER}/{tail}")
        candidates.append(f"{BASE_BACKUP_FOLDER}/{requested}")
    elif not requested.startswith(f"{BASE_BACKUP_FOLDER}/"):
        candidates.append(f"{BASE_BACKUP_FOLDER}/{requested}")

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


async def _download_remote_file(client: Any, requested_path: str) -> tuple[bytes, str]:
    errors: list[str] = []
    for remote_path in _remote_candidates(requested_path):
        item_path = f"root:/{remote_path}:"
        try:
            data = await (
                client.drives.by_drive_id(SHAREPOINT_DRIVE_ID)
                .items.by_drive_item_id(item_path)
                .content.get()
            )
            return data, remote_path
        except Exception as exc:
            errors.append(f"{remote_path}: {exc}")

    raise RuntimeError("; ".join(errors))


async def _download_drive_item_bytes(client: Any, item_id: str) -> bytes:
    return await (
        client.drives.by_drive_id(SHAREPOINT_DRIVE_ID)
        .items.by_drive_item_id(item_id)
        .content.get()
    )


def _item_modified_iso(item: Any) -> str:
    value = getattr(item, "last_modified_date_time", None)
    if value is None:
        return ""
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


async def _find_best_round2_backup(
    client: Any, reviewer: str
) -> tuple[bytes, str, dict[str, Any]]:
    expected_name = f"fluoro-r2_round2_{reviewer}.json".lower()
    root_path = f"root:/{BASE_BACKUP_FOLDER}:"
    user_folders = await _list_children(client, SHAREPOINT_DRIVE_ID, root_path)

    best: dict[str, Any] | None = None
    best_data: bytes | None = None
    inspected = 0

    for user_folder in user_folders:
        if not getattr(user_folder, "folder", None):
            continue

        user_name = user_folder.name
        user_path = f"root:/{BASE_BACKUP_FOLDER}/{user_name}:"
        try:
            date_folders = await _list_children(client, SHAREPOINT_DRIVE_ID, user_path)
        except Exception:
            continue

        for date_folder in date_folders:
            if not getattr(date_folder, "folder", None):
                continue

            date_name = date_folder.name
            folder_path = f"{BASE_BACKUP_FOLDER}/{user_name}/{date_name}"
            item_path = f"root:/{folder_path}:"
            try:
                children = await _list_children(client, SHAREPOINT_DRIVE_ID, item_path)
            except Exception:
                continue

            for item in children:
                if not getattr(item, "file", None):
                    continue
                if item.name.lower() != expected_name:
                    continue

                inspected += 1
                remote_path = f"{folder_path}/{item.name}"
                try:
                    data = await _download_drive_item_bytes(client, item.id)
                    parsed = json.loads(data.decode("utf-8"))
                    counts = _annotation_counts(parsed)
                except Exception as exc:
                    print(f"{reviewer}: skipped unreadable {remote_path}: {exc}")
                    continue

                modified = _item_modified_iso(item)
                candidate = {
                    "reviewer": reviewer,
                    "resolved_remote_path": remote_path,
                    "modified": modified,
                    "date_folder": date_name,
                    "inspected_candidates": inspected,
                    **counts,
                }

                if best is None:
                    best = candidate
                    best_data = data
                    continue

                candidate_key = (
                    candidate["annotated_images"],
                    candidate["annotated_landmarks"],
                    candidate["modified"],
                    candidate["date_folder"],
                )
                best_key = (
                    best["annotated_images"],
                    best["annotated_landmarks"],
                    best["modified"],
                    best["date_folder"],
                )
                if candidate_key > best_key:
                    best = candidate
                    best_data = data

    if best is None or best_data is None:
        raise FileNotFoundError(f"No OneDrive round 2 file found for {reviewer}")

    return best_data, str(best["resolved_remote_path"]), best


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _copy_local_fallback(reviewer: str, dest_path: Path) -> str:
    src = LOCAL_FALLBACK_DIR / f"fluoro-r2_round2_{reviewer}.json"
    if not src.exists():
        raise FileNotFoundError(f"Local fallback missing: {src}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_path)
    return str(src)


def _target_for_image_ref(image_ref: str) -> Path:
    ref = Path(image_ref.replace("\\", "/"))
    if ref.is_absolute():
        return ref
    candidate = INSTALL_IMAGE_ROOT / ref
    if candidate.exists():
        return candidate
    return LOCAL_FALLBACK_DIR / ref.name


def _create_tiff_symlinks(dest_root: Path, json_paths: list[Path]) -> dict[str, Any]:
    image_refs: set[str] = set()
    for json_path in json_paths:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for record in data.get("images", []):
            if isinstance(record, dict) and "image_path" in record:
                image_refs.add(str(record["image_path"]).replace("\\", "/"))

    created = 0
    existing = 0
    missing: list[str] = []
    conflicts: list[str] = []

    for image_ref in sorted(image_refs):
        target = _target_for_image_ref(image_ref)
        if not target.exists():
            missing.append(f"{image_ref} -> {target}")
            continue

        ref_path = Path(image_ref)
        link_path = dest_root / (ref_path.name if ref_path.is_absolute() else ref_path)
        link_path.parent.mkdir(parents=True, exist_ok=True)

        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink():
                if link_path.resolve() == target.resolve():
                    existing += 1
                    continue
                link_path.unlink()
            else:
                conflicts.append(str(link_path))
                continue

        os.symlink(target, link_path)
        created += 1

    return {
        "created": created,
        "existing": existing,
        "missing": missing,
        "conflicts": conflicts,
        "total_refs": len(image_refs),
    }


async def _repull(args: argparse.Namespace) -> int:
    dest_root = Path(args.dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.local_only:
        client = get_graph_client(prompt_callback_fn=_console_prompt_callback)

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "destination": str(dest_root),
        "search_most_recent": bool(args.search_most_recent),
        "sources": [],
    }
    json_paths: list[Path] = []
    count_rows: list[dict[str, Any]] = []

    for reviewer, requested_path in REQUESTED_BACKUPS.items():
        dest_path = dest_root / f"fluoro-r2_round2_{reviewer}.json"
        source_entry: dict[str, Any] = {
            "reviewer": reviewer,
            "requested_path": requested_path,
            "destination": str(dest_path),
        }

        try:
            if args.local_only:
                copied_from = _copy_local_fallback(reviewer, dest_path)
                source_entry["local_fallback"] = copied_from
                print(f"{reviewer}: copied local fallback")
            else:
                assert client is not None
                if args.search_most_recent:
                    data, resolved_remote, best = await _find_best_round2_backup(
                        client, reviewer
                    )
                    source_entry["search_result"] = best
                else:
                    data, resolved_remote = await _download_remote_file(
                        client, requested_path
                    )
                _write_bytes_atomic(dest_path, data)
                source_entry["resolved_remote_path"] = resolved_remote
                print(f"{reviewer}: downloaded {resolved_remote}")
        except Exception as exc:
            if not args.allow_local_fallback:
                raise
            copied_from = _copy_local_fallback(reviewer, dest_path)
            source_entry["download_error"] = str(exc)
            source_entry["local_fallback"] = copied_from
            print(f"{reviewer}: download failed; copied local fallback")

        counts = _counts_for_json_path(dest_path)
        source_entry["counts"] = counts
        count_rows.append(
            {
                "reviewer": reviewer,
                "source": source_entry.get("resolved_remote_path")
                or source_entry.get("local_fallback", ""),
                **counts,
            }
        )
        manifest["sources"].append(source_entry)
        json_paths.append(dest_path)

    manifest["counts"] = count_rows
    manifest["tiff_symlinks"] = _create_tiff_symlinks(dest_root, json_paths)
    manifest_path = dest_root / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _print_count_report(count_rows)
    print(f"Wrote {manifest_path}")
    print(
        "TIFF symlinks: "
        f"{manifest['tiff_symlinks']['created']} created, "
        f"{manifest['tiff_symlinks']['existing']} existing, "
        f"{len(manifest['tiff_symlinks']['missing'])} missing"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repull the round-two multi-reviewer JSON set and link TIFFs."
    )
    parser.add_argument("--dest", default=str(DEFAULT_DEST))
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Copy from the local install-data cache instead of OneDrive.",
    )
    parser.add_argument(
        "--allow-local-fallback",
        action="store_true",
        help="Use the local install-data cache if a OneDrive download fails.",
    )
    parser.add_argument(
        "--search-most-recent",
        "--search_most_recent",
        dest="search_most_recent",
        action="store_true",
        help=(
            "Search all OneDrive backup folders for each reviewer's round-two file "
            "and choose the candidate with the most annotated images, breaking ties "
            "by annotated landmark count and recency."
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_repull(args))


if __name__ == "__main__":
    raise SystemExit(main())

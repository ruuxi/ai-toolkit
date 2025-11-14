import os
import shutil
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config

from .config import get_settings


class SyncError(Exception):
    """Raised when the worker fails to mirror a dataset."""


def _build_s3_client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        region_name=settings.r2_region,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
    )


def _normalize_prefix(prefix: str) -> str:
    prefix = prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    return prefix


def sync_dataset(
    *,
    dataset_id: str,
    bucket: str,
    prefix: str,
    overwrite: bool = False,
) -> dict:
    settings = get_settings()
    client = _build_s3_client()
    dataset_root = Path(settings.dataset_root)
    local_path = dataset_root / dataset_id
    marker_file = local_path / ".sync_complete"
    if marker_file.exists() and not overwrite:
        return {
            "datasetId": dataset_id,
            "localPath": str(local_path),
            "alreadySynced": True,
            "bytesWritten": 0,
            "objectCount": 0,
        }

    if local_path.exists() and overwrite:
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    prefix = _normalize_prefix(prefix)
    paginator = client.get_paginator("list_objects_v2")
    bytes_written = 0
    object_count = 0

    temp_marker = local_path / ".sync_in_progress"
    temp_marker.write_text("syncing\n")

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                rel_key = key[len(prefix) :].lstrip("/")
                if not rel_key:
                    continue
                dest = local_path / rel_key
                dest.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, key, str(dest))
                bytes_written += obj.get("Size", 0)
                object_count += 1

        marker_file.write_text("complete\n")
    except Exception as exc:  # pragma: no cover - defensive logging
        raise SyncError(str(exc)) from exc
    finally:
        if temp_marker.exists():
            temp_marker.unlink(missing_ok=True)

    return {
        "datasetId": dataset_id,
        "localPath": str(local_path),
        "alreadySynced": False,
        "bytesWritten": bytes_written,
        "objectCount": object_count,
    }


def dataset_status(dataset_id: str) -> Optional[str]:
    settings = get_settings()
    local_path = Path(settings.dataset_root) / dataset_id
    marker = local_path / ".sync_complete"
    if marker.exists():
        return str(local_path)
    return None


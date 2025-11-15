import os
import sys
import tempfile
import json
from typing import Any, Dict, List, Optional

import runpod
from dotenv import load_dotenv

# Load environment variables from a local .env file if present.
load_dotenv()

# Match run.py behaviour: enable hf-transfer and disable telemetry.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "YES"

# Ensure local project imports resolve no matter where the handler runs.
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Optional debug flag, same as run.py.
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    import torch

    torch.autograd.set_detect_anomaly(True)

try:
    from toolkit.accelerator import get_accelerator
except ImportError:  # Allows lightweight dry-run testing without full deps.
    get_accelerator = None  # type: ignore[assignment]

try:
    from toolkit.print import print_acc, setup_log_to_file
except ImportError:
    def print_acc(message: str) -> None:  # type: ignore[no-redef]
        print(message)

    def setup_log_to_file(_: str) -> None:  # type: ignore[no-redef]
        print("Warning: logging to file is unavailable without toolkit dependencies.")


def _build_accelerator():
    if get_accelerator is None:
        class _SimpleAccelerator:
            is_main_process = True

        print("Warning: accelerate not available, using dummy accelerator (dry-run only).")
        return _SimpleAccelerator()
    return get_accelerator()


accelerator = _build_accelerator()


def _print_end_message(jobs_completed: int, jobs_failed: int) -> None:
    """Mirror the CLI summary output for consistency."""
    if not accelerator.is_main_process:
        return

    failure_string = (
        f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    )
    completed_string = (
        f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"
    )

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def _normalize_config_file_list(raw_value: Any) -> List[str]:
    if raw_value is None:
        raise ValueError("`config_file_list` is required in the job input.")
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, list) and all(isinstance(item, str) for item in raw_value):
        if len(raw_value) == 0:
            raise ValueError("`config_file_list` cannot be empty.")
        return raw_value
    raise ValueError("`config_file_list` must be a string or list of strings.")


def _sync_from_r2(dataset_id: str, bucket: str, prefix: str) -> str:
    """
    Download dataset from R2 to local storage.
    Returns the local path where the dataset is stored.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        raise RuntimeError("boto3 is required for R2 sync but not installed")

    endpoint_url = os.environ.get("R2_ENDPOINT")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    region = os.environ.get("R2_REGION", "auto")

    if not endpoint_url or not access_key or not secret_key:
        raise RuntimeError("R2_ENDPOINT, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY must be set")

    # Create local dataset directory
    local_root = os.environ.get("DATASET_ROOT", "/workspace/datasets")
    local_path = os.path.join(local_root, dataset_id)
    os.makedirs(local_path, exist_ok=True)

    # Check if already synced
    marker_file = os.path.join(local_path, ".sync_complete")
    if os.path.exists(marker_file):
        print_acc(f"Dataset {dataset_id} already synced at {local_path}")
        return local_path

    print_acc(f"Syncing dataset {dataset_id} from R2 bucket {bucket}, prefix {prefix}")

    # Initialize S3 client for R2
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )

    # List and download objects
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        object_count = 0
        bytes_written = 0

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # Skip if it's just the prefix directory marker
                if key == prefix or key.endswith("/"):
                    continue

                # Compute relative path and local file path
                rel_path = key[len(prefix):].lstrip("/")
                if not rel_path:
                    continue

                local_file = os.path.join(local_path, rel_path)
                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                # Download object
                s3_client.download_file(bucket, key, local_file)
                object_count += 1
                bytes_written += obj.get("Size", 0)

        # Write completion marker
        with open(marker_file, "w") as f:
            f.write(f"Synced {object_count} objects ({bytes_written} bytes) at {os.path.getmtime(local_path)}\n")

        print_acc(f"Synced {object_count} objects ({bytes_written} bytes) to {local_path}")
        return local_path

    except ClientError as e:
        raise RuntimeError(f"R2 sync failed: {e}") from e


def run_jobs(
    config_file_list: List[str],
    name: str | None = None,
    recover: bool = False,
    log_file: str | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Closest serverless analogue to the CLI execution loop in run.py."""
    if log_file:
        setup_log_to_file(log_file)

    jobs_completed = 0
    jobs_failed = 0
    responses: List[Dict[str, Any]] = []

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    get_job_fn = None
    if not dry_run:
        from toolkit.job import get_job as _get_job

        get_job_fn = _get_job

    for config_file in config_file_list:
        current_summary: Dict[str, Any] = {
            "config_file": config_file,
            "status": "pending",
            "error": None,
        }
        job = None
        if dry_run:
            current_summary["status"] = "skipped"
            current_summary["note"] = "Dry run: job was not executed."
            responses.append(current_summary)
            continue

        try:
            if get_job_fn is None:
                raise RuntimeError("get_job is unavailable. Is dry_run disabled without dependencies installed?")
            job = get_job_fn(config_file, name)
            job.run()
            job.cleanup()
            jobs_completed += 1
            current_summary["status"] = "completed"
        except KeyboardInterrupt as err:
            current_summary["status"] = "stopped"
            current_summary["error"] = str(err)
            jobs_failed += 1
            if job is not None:
                try:
                    job.process[0].on_error(err)  # type: ignore[attr-defined]
                except Exception as err2:  # pragma: no cover - best effort logging
                    print_acc(f"Error running on_error: {err2}")
            if not recover:
                _print_end_message(jobs_completed, jobs_failed)
                raise
        except Exception as err:
            current_summary["status"] = "failed"
            current_summary["error"] = str(err)
            jobs_failed += 1
            print_acc(f"Error running job: {err}")
            if job is not None:
                try:
                    job.process[0].on_error(err)  # type: ignore[attr-defined]
                except Exception as err2:  # pragma: no cover - best effort logging
                    print_acc(f"Error running on_error: {err2}")
            if not recover:
                _print_end_message(jobs_completed, jobs_failed)
                raise
        responses.append(current_summary)

    _print_end_message(jobs_completed, jobs_failed)

    return {
        "results": responses,
        "jobs_completed": jobs_completed,
        "jobs_failed": jobs_failed,
        "refresh_worker": jobs_failed > 0,
    }


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod Serverless handler entry point.
    
    Supports two modes:
    1. Legacy: config_file_list (paths to YAML configs)
    2. Direct: job_config (inline config dict) + r2_dataset (bucket/prefix to sync)
    """
    input_payload = event.get("input") or {}
    
    # Check for direct job_config mode (new trainer integration)
    job_config = input_payload.get("job_config")
    r2_dataset = input_payload.get("r2_dataset")
    
    if job_config and r2_dataset:
        # Direct mode: sync from R2 and run inline config
        dataset_id = r2_dataset.get("datasetId")
        bucket = r2_dataset.get("bucket")
        prefix = r2_dataset.get("prefix")
        
        if not dataset_id or not bucket or not prefix:
            return {
                "error": "r2_dataset must include datasetId, bucket, and prefix",
                "request_id": event.get("id"),
            }
        
        try:
            # Sync dataset from R2
            local_path = _sync_from_r2(dataset_id, bucket, prefix)
            
            # Update job config with the synced local path
            if "config" in job_config and "process" in job_config["config"]:
                for process_item in job_config["config"]["process"]:
                    if "datasets" in process_item:
                        for ds in process_item["datasets"]:
                            # Update with actual local path
                            if "dataset_path" in ds:
                                ds["dataset_path"] = local_path
                            if "folder_path" in ds:
                                ds["folder_path"] = local_path
            
            # Write config to temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, dir="/tmp"
            ) as f:
                json.dump(job_config, f, indent=2)
                temp_config_path = f.name
            
            # Run the job with the temp config
            name = input_payload.get("name")
            recover = bool(input_payload.get("recover", False))
            log_file = input_payload.get("log")
            
            response = run_jobs(
                config_file_list=[temp_config_path],
                name=name,
                recover=recover,
                log_file=log_file,
                dry_run=False,
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_config_path)
            except:
                pass
            
            response["request_id"] = event.get("id")
            response["dataset_synced"] = {
                "datasetId": dataset_id,
                "localPath": local_path,
            }
            return response
            
        except Exception as e:
            return {
                "error": str(e),
                "request_id": event.get("id"),
            }
    
    # Legacy mode: use config_file_list
    config_file_list = _normalize_config_file_list(input_payload.get("config_file_list"))
    name = input_payload.get("name")
    recover = bool(input_payload.get("recover", False))
    log_file = input_payload.get("log")
    dry_run = bool(input_payload.get("dry_run", False))

    response = run_jobs(
        config_file_list=config_file_list,
        name=name,
        recover=recover,
        log_file=log_file,
        dry_run=dry_run,
    )
    response["request_id"] = event.get("id")
    return response


runpod.serverless.start({"handler": handler})


import os
import sys
from typing import Any, Dict, List

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
    """RunPod Serverless handler entry point."""
    input_payload = event.get("input") or {}

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


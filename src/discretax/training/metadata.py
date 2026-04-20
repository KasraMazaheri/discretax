"""Runtime metadata capture for experiment executions."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import jax


def _format_timestamp(value: datetime) -> str:
    """Format a UTC timestamp using a stable ISO-8601 representation."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _package_version(name: str) -> str | None:
    """Return an installed package version when available."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _run_git_command(repository_root: Path, args: list[str]) -> str | None:
    """Run a git command and return stripped stdout when successful."""
    result = subprocess.run(
        ["git", *args],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_metadata(repository_root: Path) -> dict[str, Any]:
    """Collect git metadata for the current repository state."""
    commit = _run_git_command(repository_root, ["rev-parse", "HEAD"])
    branch = _run_git_command(repository_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    status_output = _run_git_command(repository_root, ["status", "--short"])
    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status_output),
    }


def _device_metadata() -> list[dict[str, Any]]:
    """Collect visible JAX device metadata."""
    devices: list[dict[str, Any]] = []
    for device in jax.devices():
        devices.append(
            {
                "id": getattr(device, "id", None),
                "platform": getattr(device, "platform", None),
                "device_kind": getattr(device, "device_kind", None),
                "process_index": getattr(device, "process_index", None),
            }
        )
    return devices


def _environment_metadata() -> dict[str, str]:
    """Collect a small set of runtime environment variables."""
    variable_names = (
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "WANDB_MODE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    return {name: value for name in variable_names if (value := os.environ.get(name)) is not None}


def collect_run_metadata(
    *,
    output_dir: Path,
    mode: str,
    started_at: datetime,
    resume_from: str | Path | None = None,
    restored_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Collect run metadata for a single experiment invocation."""
    repository_root = Path(__file__).resolve().parents[3]
    devices = _device_metadata()
    return {
        "mode": mode,
        "status": "running",
        "output_dir": str(output_dir),
        "started_at": _format_timestamp(started_at),
        "ended_at": None,
        "duration_seconds": None,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "resume_from": str(resume_from) if resume_from is not None else None,
        "restored_checkpoint": (
            str(restored_checkpoint) if restored_checkpoint is not None else None
        ),
        "git": _git_metadata(repository_root),
        "jax": {
            "version": _package_version("jax"),
            "jaxlib_version": _package_version("jaxlib"),
            "default_backend": jax.default_backend(),
            "device_count": len(devices),
            "devices": devices,
        },
        "environment": _environment_metadata(),
    }


def finalize_run_metadata(
    metadata: dict[str, Any],
    *,
    ended_at: datetime,
    status: str,
    final_step: int,
    best_val_metric: float | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return finalized run metadata for a completed invocation."""
    finalized = deepcopy(metadata)
    finalized["status"] = status
    finalized["ended_at"] = _format_timestamp(ended_at)
    started_at = datetime.fromisoformat(metadata["started_at"].replace("Z", "+00:00"))
    finalized["duration_seconds"] = (ended_at - started_at).total_seconds()
    finalized["final_step"] = final_step
    if best_val_metric is not None:
        finalized["best_val_metric"] = best_val_metric
    if error is not None:
        finalized["error"] = error
    return finalized


def tracker_runtime_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract a concise runtime summary payload for external trackers."""
    return {
        "run_mode": metadata["mode"],
        "hostname": metadata["hostname"],
        "git_commit": metadata["git"]["commit"],
        "git_branch": metadata["git"]["branch"],
        "git_dirty": metadata["git"]["is_dirty"],
        "jax_version": metadata["jax"]["version"],
        "jaxlib_version": metadata["jax"]["jaxlib_version"],
        "jax_backend": metadata["jax"]["default_backend"],
        "jax_device_count": metadata["jax"]["device_count"],
    }

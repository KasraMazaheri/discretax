"""Launch the default strong CIFAR-10 LinOSS sweep for one family on one node."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

FAMILY_SWEEP_CONFIGS = {
    "im": "configs/sweeps/cifar10_linoss_heads_proj_strong.yaml",
    "damped": "configs/sweeps/cifar10_linoss_heads_proj_strong_damped.yaml",
}


def _parse_args() -> argparse.Namespace:
    """Parse launcher arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=sorted(FAMILY_SWEEP_CONFIGS),
        required=True,
        help="Which LinOSS family to launch on this node.",
    )
    parser.add_argument(
        "--sweep-config",
        default=None,
        help="Optional explicit sweep config path. Defaults to the selected family config.",
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids to use on this node.",
    )
    parser.add_argument(
        "--slots-per-gpu",
        type=int,
        default=1,
        help="Concurrent jobs to allow per GPU. Default is 1 for these long CIFAR runs.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for launcher logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated commands without executing them.",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Launch detached tmux sessions and return immediately.",
    )
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return loaded


def _normalize_value(value: Any) -> str:
    """Encode an override value as a CLI-safe YAML scalar."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def _short_key(dotted_key: str) -> str:
    """Shorten a dotted config key for run-name generation."""
    aliases = {
        "trainer.seed": "seed",
        "model.backbone.kwargs.num_heads": "heads",
        "model.backbone.kwargs.use_head_output_projection": "proj",
        "optimizer.learning_rate": "lr",
        "model.backbone.kwargs.state_dim": "state",
        "model.hidden_dim": "hidden",
        "model.backbone.kwargs.num_blocks": "blocks",
    }
    return aliases.get(dotted_key, dotted_key.split(".")[-1])


def _run_name(prefix: str, include_keys: list[str], overrides: dict[str, Any]) -> str:
    """Construct a stable run name from selected config keys."""
    parts = [prefix]
    for key in include_keys:
        if key in overrides:
            parts.append(f"{_short_key(key)}{_normalize_value(overrides[key])}")
    return "-".join(parts)


def _build_jobs(sweep_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize the grid sweep into concrete override dictionaries."""
    fixed = dict(sweep_config.get("fixed", {}))
    grid = sweep_config.get("grid", {})
    grid_keys = list(grid)
    grid_values = [grid[key] for key in grid_keys]
    run_name_config = sweep_config["run_name"]
    prefix = str(run_name_config["prefix"])
    include_keys = list(run_name_config["include_keys"])

    jobs: list[dict[str, Any]] = []
    for values in itertools.product(*grid_values):
        overrides = dict(fixed)
        overrides.update(zip(grid_keys, values, strict=True))
        run_name = _run_name(prefix, include_keys, overrides)
        overrides["name"] = run_name
        overrides["wandb.run_name"] = run_name

        tags = list(overrides.get("wandb.tags", []))
        tags.extend(
            [
                f"state{overrides['model.backbone.kwargs.state_dim']}",
                f"heads{overrides['model.backbone.kwargs.num_heads']}",
                f"proj{_normalize_value(overrides['model.backbone.kwargs.use_head_output_projection'])}",
                f"seed{overrides['trainer.seed']}",
            ]
        )
        overrides["wandb.tags"] = tags
        jobs.append(overrides)
    return jobs


def _command(base_config: str, overrides: dict[str, Any]) -> list[str]:
    """Build the discretax-train CLI command for one job."""
    cmd = ["uv", "run", "discretax-train", "--config", base_config]
    for key, value in overrides.items():
        cmd.extend(["--set", f"{key}={_normalize_value(value)}"])
    return cmd


def _tmux_session_name(job_name: str, index: int) -> str:
    """Build a tmux-safe session name for a launched job."""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", job_name).strip("-")
    slug = slug[:48] if slug else "run"
    return f"dx-cifar-{index:02d}-{slug}"


def _gpu_uuid_map() -> dict[str, str]:
    """Map visible GPU indices to UUIDs."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    mapping: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid = (part.strip() for part in line.split(",", 1))
        mapping[index] = uuid
    return mapping


def _gpu_process_counts(gpu_uuids: dict[str, str]) -> dict[str, int]:
    """Count active compute processes per GPU."""
    counts = dict.fromkeys(gpu_uuids, 0)
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid_to_index = {uuid: index for index, uuid in gpu_uuids.items()}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        gpu_uuid = line.split(",", 1)[0].strip()
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is not None:
            counts[gpu_index] += 1
    return counts


def _wait_for_existing_slot(gpu: str, limit: int, gpu_uuids: dict[str, str]) -> None:
    """Wait until a GPU has fewer than `limit` active compute processes."""
    while True:
        counts = _gpu_process_counts(gpu_uuids)
        if counts.get(gpu, 0) < limit:
            return
        time.sleep(5)


def _tmux_launch_command(
    *,
    gpu: str,
    gpu_uuid: str,
    slots_per_gpu: int,
    cmd: list[str],
    log_path: Path,
) -> str:
    """Build the shell command executed inside a detached tmux session."""
    quoted_cmd = shlex.join(cmd)
    quoted_log = shlex.quote(str(log_path))
    query_command = (
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory "
        "--format=csv,noheader,nounits"
    )
    count_command = (
        f"{query_command} | "
        f"""awk -F', ' '$1 == "{gpu_uuid}" {{count++}} END {{print count + 0}}'"""
    )
    return "bash -lc " + shlex.quote(
        f"""
set -euo pipefail
while true; do
  current_jobs=$({count_command})
  if [ "$current_jobs" -lt "{slots_per_gpu}" ]; then
    break
  fi
  echo "[wait] gpu={gpu} current_jobs=$current_jobs limit={slots_per_gpu}" >> {quoted_log}
  sleep 30
done
cd {shlex.quote(str(Path.cwd()))}
export CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}
exec {quoted_cmd} >> {quoted_log} 2>&1
""".strip()
    )


def _write_manifest(
    *,
    log_dir: Path,
    sweep_name: str,
    launched_jobs: list[dict[str, Any]],
) -> Path:
    """Persist detached launch details for later inspection."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    manifest_path = log_dir / f"{timestamp}-{sweep_name}-launch.json"
    payload = {
        "launched_at": datetime.now(UTC).isoformat(),
        "jobs": launched_jobs,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _wait_for_slot(active: list[tuple[subprocess.Popen[Any], str]], gpu: str, limit: int) -> None:
    """Block until a GPU has a free process slot."""
    while True:
        active[:] = [(proc, assigned_gpu) for proc, assigned_gpu in active if proc.poll() is None]
        used = sum(1 for _, assigned_gpu in active if assigned_gpu == gpu)
        if used < limit:
            return
        time.sleep(5)


def _launch_jobs(
    jobs: list[dict[str, Any]],
    *,
    base_config: str,
    sweep_name: str,
    gpus: list[str],
    slots_per_gpu: int,
    log_dir: Path,
    dry_run: bool,
    detach: bool,
) -> int:
    """Launch jobs across GPUs and return the aggregate exit code."""
    log_dir.mkdir(parents=True, exist_ok=True)
    active: list[tuple[subprocess.Popen[Any], str]] = []
    launched_jobs: list[dict[str, Any]] = []
    gpu_uuids = _gpu_uuid_map()

    for index, job in enumerate(jobs):
        gpu = gpus[index % len(gpus)]
        cmd = _command(base_config, job)
        log_path = log_dir / f"{job['name']}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

        if dry_run:
            print(f"CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)} > {log_path} 2>&1")
            continue

        session_name = _tmux_session_name(job["name"], index)
        if detach:
            tmux_command = _tmux_launch_command(
                gpu=gpu,
                gpu_uuid=gpu_uuids[gpu],
                slots_per_gpu=slots_per_gpu,
                cmd=cmd,
                log_path=log_path,
            )
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session_name, tmux_command],
                check=True,
            )
            proc = None
        else:
            _wait_for_existing_slot(gpu, slots_per_gpu, gpu_uuids)
            _wait_for_slot(active, gpu, slots_per_gpu)
            log_handle = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=Path.cwd(),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=False,
            )
            log_handle.close()

        launched_jobs.append(
            {
                "name": job["name"],
                "gpu": gpu,
                "pid": proc.pid if proc is not None else None,
                "tmux_session": session_name,
                "log_path": str(log_path),
                "command": cmd,
            }
        )

        if detach:
            print(f"[detach] gpu={gpu} session={session_name} name={job['name']}")
        else:
            assert proc is not None
            active.append((proc, gpu))
            print(f"[launch] gpu={gpu} pid={proc.pid} name={job['name']}")

    if dry_run:
        return 0

    if detach:
        manifest_path = _write_manifest(
            log_dir=log_dir,
            sweep_name=sweep_name,
            launched_jobs=launched_jobs,
        )
        print(f"[detach] wrote launch manifest to {manifest_path}")
        return 0

    exit_code = 0
    for proc, gpu in active:
        return_code = proc.wait()
        if return_code != 0:
            exit_code = return_code
            print(f"[fail] gpu={gpu} pid={proc.pid} exit={return_code}", file=sys.stderr)
    return exit_code


def main() -> None:
    """Launch one default strong CIFAR family sweep."""
    args = _parse_args()
    sweep_config_path = args.sweep_config or FAMILY_SWEEP_CONFIGS[args.family]
    sweep_config = _load_yaml(sweep_config_path)
    base_config = str(sweep_config["base_config"])
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU id must be provided via --gpus")

    jobs = _build_jobs(sweep_config)
    print(f"Launching {len(jobs)} {args.family} jobs from {sweep_config_path}")
    exit_code = _launch_jobs(
        jobs,
        base_config=base_config,
        sweep_name=Path(sweep_config_path).stem,
        gpus=gpus,
        slots_per_gpu=args.slots_per_gpu,
        log_dir=Path(args.log_dir),
        dry_run=args.dry_run,
        detach=args.detach,
    )
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

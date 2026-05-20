"""Launch a YAML-defined experiment sweep on one node with centralized scheduling.

Key properties:
- The launcher itself stays alive and manages the queue.
- At most len(gpus) * slots_per_gpu jobs are active at once.
- Each running job gets its own tmux session.
- No per-tmux waiting loop, so no admission race.
- The launcher polls tmux + nvidia-smi periodically and backfills freed slots.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-config",
        required=True,
        help="Path to the sweep YAML config.",
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
        help="Maximum concurrent jobs per GPU.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help="Scheduler polling interval in seconds.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for launcher/job logs.",
    )
    parser.add_argument(
        "--launch-name",
        default=None,
        help="Optional manifest name override. Defaults to the sweep-config stem.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands without executing anything.",
    )
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_default_path(reference: str, current_path: Path) -> Path:
    default_path = Path(reference)
    if not default_path.suffix:
        default_path = default_path.with_suffix(".yaml")
    if not default_path.is_absolute():
        default_path = (current_path.parent / default_path).resolve()
    if not default_path.exists():
        raise FileNotFoundError(
            f"Default config {reference!r} resolved to missing path {default_path}"
        )
    return default_path


def _resolve_defaults(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = visited or set()
    if path in seen:
        raise ValueError(f"Detected recursive config defaults while resolving {path}")
    seen.add(path)

    raw_config = _load_yaml(path)
    defaults = raw_config.pop("defaults", [])
    if not isinstance(defaults, list):
        raise ValueError(f"'defaults' in {path} must be a list of config paths")

    merged: dict[str, Any] = {}
    for reference in defaults:
        if not isinstance(reference, str):
            raise ValueError(f"Defaults entries in {path} must be strings, got {reference!r}")
        merged = _deep_merge(
            merged,
            _resolve_defaults(_resolve_default_path(reference, path), seen),
        )

    seen.remove(path)
    return _deep_merge(merged, raw_config)


def _normalize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Avoid scientific notation because the config override loader currently
        # round-trips values like `1e-05` as strings instead of floats.
        text = format(value, ".15f").rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def _short_key(dotted_key: str) -> str:
    aliases = {
        "trainer.seed": "seed",
        "dataset.name": "dataset",
        "dataset.params.include_time": "time",
        "model.backbone.kwargs.discretization": "disc",
        "model.backbone.kwargs.num_heads": "heads",
        "model.backbone.kwargs.use_head_output_projection": "proj",
        "optimizer.learning_rate": "lr",
        "model.backbone.kwargs.state_dim": "state",
        "model.hidden_dim": "hidden",
        "model.backbone.kwargs.num_blocks": "blocks",
    }
    return aliases.get(dotted_key, dotted_key.split(".")[-1])


def _run_name(prefix: str, include_keys: list[str], overrides: dict[str, Any]) -> str:
    parts = [prefix]
    for key in include_keys:
        if key in overrides:
            parts.append(f"{_short_key(key)}{_normalize_value(overrides[key])}")
    return "-".join(parts)


def _matches_pattern(overrides: dict[str, Any], pattern: dict[str, Any]) -> bool:
    return all(overrides.get(key) == value for key, value in pattern.items())


def _merge_tags(*tag_sources: list[Any]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for source in tag_sources:
        for tag in source:
            tag_text = str(tag)
            if tag_text in seen:
                continue
            seen.add(tag_text)
            merged.append(tag_text)
    return merged


def _append_auto_tags(overrides: dict[str, Any], *, base_tags: list[str]) -> None:
    tags = _merge_tags(base_tags, list(overrides.get("wandb.tags", [])))
    for key, prefix in (
        ("dataset.name", "dataset"),
        ("model.hidden_dim", "hidden"),
        ("model.backbone.kwargs.state_dim", "state"),
        ("model.backbone.kwargs.num_heads", "heads"),
        ("model.backbone.kwargs.use_head_output_projection", "proj"),
        ("trainer.seed", "seed"),
    ):
        if key in overrides:
            tags = _merge_tags(tags, [f"{prefix}{_normalize_value(overrides[key])}"])
    overrides["wandb.tags"] = tags


def _apply_copied_overrides(
    overrides: dict[str, Any],
    copy_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = dict(overrides)
    for rule in copy_rules:
        if not isinstance(rule, dict):
            raise ValueError("copy_overrides entries must be mappings")
        source = rule.get("source")
        targets = rule.get("targets")
        if not isinstance(source, str):
            raise ValueError("copy_overrides.source must be a dotted key string")
        if source not in updated:
            continue
        if not isinstance(targets, list) or not all(isinstance(target, str) for target in targets):
            raise ValueError("copy_overrides.targets must be a list of dotted key strings")
        for target in targets:
            updated[target] = updated[source]
    return updated


def _build_jobs(sweep_config: dict[str, Any]) -> list[dict[str, Any]]:
    base_config_path = Path(str(sweep_config["base_config"]))
    resolved_base_config = _resolve_defaults(base_config_path)
    base_wandb_tags = list((resolved_base_config.get("wandb", {}) or {}).get("tags", []))

    fixed = dict(sweep_config.get("fixed", {}))
    exclude_patterns = [dict(pattern) for pattern in sweep_config.get("exclude", [])]
    copy_rules = list(sweep_config.get("copy_overrides", []))
    run_name_config = sweep_config["run_name"]
    prefix = str(run_name_config["prefix"])
    include_keys = list(run_name_config["include_keys"])

    explicit_jobs = sweep_config.get("jobs")
    if explicit_jobs is not None:
        job_grid = sweep_config.get("job_grid", {})
        job_grid_keys = list(job_grid)
        job_grid_values = [job_grid[key] for key in job_grid_keys]
        job_grid_product = list(itertools.product(*job_grid_values)) if job_grid_keys else [()]
        job_grid_order = str(sweep_config.get("job_grid_order", "inner"))

        jobs: list[dict[str, Any]] = []
        if job_grid_order not in {"inner", "outer"}:
            raise ValueError("job_grid_order must be either 'inner' or 'outer'")
        outer_iter = (
            (
                (job_overrides, grid_values)
                for grid_values in job_grid_product
                for job_overrides in explicit_jobs
            )
            if job_grid_order == "outer"
            else (
                (job_overrides, grid_values)
                for job_overrides in explicit_jobs
                for grid_values in job_grid_product
            )
        )
        for job_overrides, grid_values in outer_iter:
            overrides = dict(fixed)
            overrides.update(dict(job_overrides))
            overrides.update(zip(job_grid_keys, grid_values, strict=True))
            overrides = _apply_copied_overrides(overrides, copy_rules)
            if any(_matches_pattern(overrides, pattern) for pattern in exclude_patterns):
                continue
            run_name = _run_name(prefix, include_keys, overrides)
            overrides["name"] = run_name
            overrides["wandb.run_name"] = run_name
            _append_auto_tags(overrides, base_tags=base_wandb_tags)
            jobs.append(overrides)
        return jobs

    grid = sweep_config.get("grid", {})
    grid_keys = list(grid)
    grid_values = [grid[key] for key in grid_keys]

    jobs: list[dict[str, Any]] = []
    for values in itertools.product(*grid_values):
        overrides = dict(fixed)
        overrides.update(zip(grid_keys, values, strict=True))
        overrides = _apply_copied_overrides(overrides, copy_rules)
        if any(_matches_pattern(overrides, pattern) for pattern in exclude_patterns):
            continue
        run_name = _run_name(prefix, include_keys, overrides)
        overrides["name"] = run_name
        overrides["wandb.run_name"] = run_name
        _append_auto_tags(overrides, base_tags=base_wandb_tags)
        jobs.append(overrides)

    return jobs


def _command(base_config: str, overrides: dict[str, Any]) -> list[str]:
    cmd = ["uv", "run", "discretax-train", "--config", base_config]
    for key, value in overrides.items():
        cmd.extend(["--set", f"{key}={_normalize_value(value)}"])
    return cmd


def _tmux_session_name(job_name: str, index: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", job_name).strip("-")
    slug = slug[:48] if slug else "run"
    return f"dx-sweep-{index:04d}-{slug}"


def _gpu_uuid_map() -> dict[str, str]:
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
    """Count active compute processes per GPU.

    This counts all compute processes visible to nvidia-smi, not only jobs launched
    by this scheduler. That is desirable for admission control.
    """
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


def _tmux_session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _list_tmux_sessions() -> set[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _tmux_run_command(*, gpu: str, cmd: list[str], log_path: Path) -> str:
    quoted_cmd = shlex.join(cmd)
    quoted_log = shlex.quote(str(log_path))
    quoted_cwd = shlex.quote(str(Path.cwd()))
    return "bash -lc " + shlex.quote(
        f"""
set -euo pipefail
cd {quoted_cwd}
export CUDA_VISIBLE_DEVICES={shlex.quote(gpu)}
exec {quoted_cmd} >> {quoted_log} 2>&1
""".strip()
    )


def _launch_tmux_job(
    *,
    session_name: str,
    gpu: str,
    command: list[str],
    log_path: Path,
) -> None:
    tmux_command = _tmux_run_command(gpu=gpu, cmd=command, log_path=log_path)
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, tmux_command],
        check=True,
    )


def _write_manifest(
    *,
    log_dir: Path,
    sweep_name: str,
    launched_jobs: list[dict[str, Any]],
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    manifest_path = log_dir / f"{timestamp}-{sweep_name}-launch.json"
    payload = {
        "launched_at": datetime.now(UTC).isoformat(),
        "jobs": launched_jobs,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _available_slots(
    *,
    gpus: list[str],
    slots_per_gpu: int,
    gpu_process_counts: dict[str, int],
    active_jobs: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Compute conservative free slots.

    We reserve a slot immediately when we launch a tmux session, even before
    nvidia-smi shows the process. That prevents the scheduler from launching
    multiple jobs onto the same GPU during the CUDA-init window.
    """
    reserved_by_scheduler = collections.Counter(job["gpu"] for job in active_jobs.values())

    available: dict[str, int] = {}
    for gpu in gpus:
        observed = gpu_process_counts.get(gpu, 0)
        reserved = reserved_by_scheduler.get(gpu, 0)
        used = max(observed, reserved)
        available[gpu] = max(0, slots_per_gpu - used)
    return available


def _pick_next_gpu(
    gpus: list[str], available: dict[str, int], start_index: int
) -> tuple[str | None, int]:
    for offset in range(len(gpus)):
        idx = (start_index + offset) % len(gpus)
        gpu = gpus[idx]
        if available[gpu] > 0:
            return gpu, (idx + 1) % len(gpus)
    return None, start_index


def main() -> int:  # noqa: C901
    """Run the sweep scheduler."""
    args = _parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    sweep_config_path = Path(args.sweep_config)
    sweep_config = _load_yaml(sweep_config_path)
    base_config = str(sweep_config["base_config"])
    jobs = _build_jobs(sweep_config)

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("At least one GPU id must be provided")
    if args.slots_per_gpu < 1:
        raise ValueError("--slots-per-gpu must be >= 1")
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be >= 1")

    gpu_uuids = _gpu_uuid_map()
    missing = [gpu for gpu in gpus if gpu not in gpu_uuids]
    if missing:
        raise ValueError(f"GPU ids {missing} are not visible via nvidia-smi")

    sweep_name = args.launch_name or sweep_config_path.stem
    total_capacity = len(gpus) * args.slots_per_gpu
    print(f"Loaded {len(jobs)} jobs from {sweep_config_path}")
    print(
        f"Scheduler capacity: {len(gpus)} GPUs x {args.slots_per_gpu} slots = "
        f"{total_capacity} active tmux jobs"
    )

    if args.dry_run:
        for index, overrides in enumerate(jobs):
            job_name = str(overrides["name"])
            command = _command(base_config, overrides)
            log_path = log_dir / f"{job_name.replace('-', '_')}.log"
            session_name = _tmux_session_name(job_name, index)
            print(f"[dry-run] session={session_name} cmd={shlex.join(command)} log={log_path}")
        return 0

    pending: collections.deque[tuple[int, dict[str, Any]]] = collections.deque(enumerate(jobs))
    active: dict[str, dict[str, Any]] = {}
    launched_jobs: list[dict[str, Any]] = []
    rr_index = 0

    try:
        while pending or active:
            existing_sessions = _list_tmux_sessions()

            # Reap finished tmux jobs.
            finished_sessions = [name for name in list(active) if name not in existing_sessions]
            for session_name in finished_sessions:
                meta = active.pop(session_name)
                print(
                    f"[done] session={session_name} gpu={meta['gpu']} job={meta['job_name']} "
                    f"log={meta['log_path']}"
                )

            gpu_counts = _gpu_process_counts(gpu_uuids)
            available = _available_slots(
                gpus=gpus,
                slots_per_gpu=args.slots_per_gpu,
                gpu_process_counts=gpu_counts,
                active_jobs=active,
            )

            launched_this_round = 0
            while pending:
                gpu, rr_index = _pick_next_gpu(gpus, available, rr_index)
                if gpu is None:
                    break

                job_index, overrides = pending.popleft()
                job_name = str(overrides["name"])
                command = _command(base_config, overrides)
                log_path = log_dir / f"{job_name.replace('-', '_')}.log"
                session_name = _tmux_session_name(job_name, job_index)

                if _tmux_session_exists(session_name):
                    raise RuntimeError(
                        f"Refusing to reuse existing tmux session name {session_name!r}. "
                        "Kill it manually or change the naming scheme."
                    )

                _launch_tmux_job(
                    session_name=session_name,
                    gpu=gpu,
                    command=command,
                    log_path=log_path,
                )

                meta = {
                    "session_name": session_name,
                    "gpu": gpu,
                    "job_name": job_name,
                    "command": command,
                    "log_path": str(log_path),
                    "launched_at": datetime.now(UTC).isoformat(),
                }
                active[session_name] = meta
                launched_jobs.append(meta)
                available[gpu] -= 1
                launched_this_round += 1

                print(f"[launch] session={session_name} gpu={gpu} job={job_name}")

            print(
                "[status] "
                f"active={len(active)} pending={len(pending)} "
                f"gpu_counts={gpu_counts} free_slots={available}"
            )

            if pending or active:
                time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        print("\nInterrupted. Existing tmux jobs continue running.", file=sys.stderr)
        if active:
            print("Active tmux sessions:", file=sys.stderr)
            for session_name, meta in active.items():
                print(
                    f"  {session_name}: gpu={meta['gpu']} job={meta['job_name']} "
                    f"log={meta['log_path']}",
                    file=sys.stderr,
                )
        return 130

    manifest = _write_manifest(
        log_dir=log_dir,
        sweep_name=sweep_name,
        launched_jobs=launched_jobs,
    )
    print(f"Completed sweep scheduling. Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

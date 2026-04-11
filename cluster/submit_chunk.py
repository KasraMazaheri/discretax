"""Submit a contiguous chunk of an already-generated sweep.

Use this when a sweep is too large to submit all at once. Generate the sweep
once with generate_sweep.py, then submit it in batches with this script.

Examples:
--------
    # Submit jobs 0-269 (first chunk of 270)
    python cluster/submit_chunk.py --name my-sweep --start 0 --count 270

    # Or equivalently using chunk index
    python cluster/submit_chunk.py --name my-sweep --chunk 0 --chunk-size 270

    # Submit the next chunk
    python cluster/submit_chunk.py --name my-sweep --chunk 1 --chunk-size 270

    # List chunks without submitting
    python cluster/submit_chunk.py --name my-sweep --chunk-size 270 --list
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a chunk of an already-generated discretax sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--name", required=True, metavar="SWEEP_NAME", help="Sweep name.")
    parser.add_argument(
        "--sweeps-dir", default="sweeps", help="Root sweeps directory (default: sweeps/)."
    )
    parser.add_argument(
        "--submit-file",
        default="cluster/sweep.sub",
        help="HTCondor submit file template (default: cluster/sweep.sub).",
    )

    range_group = parser.add_argument_group("range (explicit start/count)")
    range_group.add_argument("--start", type=int, help="First job index to submit.")
    range_group.add_argument("--count", type=int, help="Number of jobs to submit.")

    chunk_group = parser.add_argument_group("chunk (index-based)")
    chunk_group.add_argument("--chunk", type=int, help="Zero-based chunk index.")
    chunk_group.add_argument("--chunk-size", type=int, help="Jobs per chunk.")

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all chunks and their job ranges, then exit.",
    )
    return parser.parse_args()


def _resolve_range(args: argparse.Namespace, total: int) -> tuple[int, int]:
    """Return (start, end) exclusive from parsed arguments."""
    if args.start is not None and args.count is not None:
        start = args.start
        end = min(start + args.count, total)
    elif args.chunk is not None and args.chunk_size is not None:
        start = args.chunk * args.chunk_size
        end = min(start + args.chunk_size, total)
    else:
        raise ValueError("Specify either --start and --count, or --chunk and --chunk-size.")
    if start >= total:
        raise ValueError(f"--start {start} is beyond the sweep size ({total} jobs).")
    return start, end


def _count_jobs(sweep_dir: Path) -> int:
    """Count config files in the sweep directory."""
    configs = list((sweep_dir / "configs").glob("config_*.txt"))
    if not configs:
        raise FileNotFoundError(f"No config files found in {sweep_dir / 'configs'}")
    return len(configs)


def main() -> None:
    """Entry point: submit or list a chunk of jobs from a generated sweep."""
    args = _parse_arguments()

    sweep_dir = Path(args.sweeps_dir) / args.name
    if not sweep_dir.exists():
        raise FileNotFoundError(
            f"Sweep directory {sweep_dir} not found. Run generate_sweep.py first."
        )

    total = _count_jobs(sweep_dir)

    if args.list:
        chunk_size = args.chunk_size
        if chunk_size is None:
            raise ValueError("--chunk-size is required with --list.")
        num_chunks = math.ceil(total / chunk_size)
        print(f"Sweep: {args.name}  |  {total} jobs  |  {num_chunks} chunks of {chunk_size}")
        for i in range(num_chunks):
            s = i * chunk_size
            e = min(s + chunk_size, total)
            print(f"  Chunk {i:3d}: jobs {s:5d}–{e - 1:5d}  ({e - s} jobs)")
        return

    start, end = _resolve_range(args, total)
    job_ids = list(range(start, end))

    submit_template = Path(args.submit_file)
    if not submit_template.exists():
        raise FileNotFoundError(f"Submit template {submit_template} not found.")

    submit_content = submit_template.read_text()
    submit_content = submit_content.replace("$(SWEEP_NAME)", args.name)
    submit_content = submit_content.replace("$(JOB_IDS)", " ".join(str(j) for j in job_ids))

    out_path = submit_template.parent / f"sweep.{args.name}.jobs-{start}-{end - 1}.sub.generated"
    out_path.write_text(submit_content)

    print(f"Sweep:  {args.name}  ({total} jobs total)")
    print(f"Chunk:  jobs {start}–{end - 1}  ({len(job_ids)} jobs)")
    print(f"Output: {out_path}")
    print("\nTo submit:")
    print(f"  condor_submit {out_path}")


if __name__ == "__main__":
    main()

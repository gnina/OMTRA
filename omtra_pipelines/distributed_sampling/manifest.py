"""Work splitting, manifest I/O, and status tracking for the distributed sampling pipeline."""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any


def read_system_indices(csv_path: Path) -> List[int]:
    """Read system indices from a single-line CSV of comma-separated ints."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        line = next(reader)
        return [int(idx.strip()) for idx in line]


def chunk_indices(indices: List[int], chunk_size: int) -> List[List[int]]:
    """Split indices into chunks. Last chunk may be smaller."""
    return [indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)]


def write_chunk_csv(chunk: List[int], output_file: Path) -> None:
    """Write a chunk of indices as a single-line CSV."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(chunk)


def build_manifest(config: dict, output_dir: Path) -> dict:
    """Load indices, chunk them, write chunk CSVs, and build a manifest dict.

    Args:
        config: Parsed pipeline config dict.
        output_dir: Root output directory for the pipeline run.

    Returns:
        Manifest dict with meta, sampling_tasks, and metrics_tasks.
    """
    # Resolve system indices
    if "sys_idx_file" in config and config["sys_idx_file"] is not None:
        indices = read_system_indices(Path(config["sys_idx_file"]))
    else:
        start = config.get("dataset_start_idx", 0)
        n = config["n_systems"]
        indices = list(range(start, start + n))

    chunk_size = config["sampling_chunk_size"]
    chunks = chunk_indices(indices, chunk_size)

    # Write chunk CSVs
    chunks_dir = output_dir / "work" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    sampling_tasks = {}
    metrics_tasks = {}

    n_replicates = config.get("n_replicates", 1)

    for i, chunk in enumerate(chunks):
        chunk_file = chunks_dir / f"chunk_{i:03d}.csv"
        write_chunk_csv(chunk, chunk_file)

        sampling_tasks[str(i)] = {
            "chunk_id": i,
            "chunk_file": str(chunk_file),
            "n_systems": len(chunk),
            "output_dir": str(output_dir / "samples" / f"chunk_{i}"),
        }

        metrics_tasks[str(i)] = {
            "chunk_id": i,
            "n_systems": len(chunk),
            "samples_dir": str(output_dir / "samples" / f"chunk_{i}"),
            "output_dir": str(output_dir / "metrics" / f"chunk_{i}"),
        }

    manifest = {
        "meta": {
            "mode": "dataset",
            "checkpoint": config["checkpoint"],
            "task": config["task"],
            "dataset": config["dataset"],
            "split": config.get("split", "test"),
            "n_replicates": n_replicates,
            "n_systems": len(indices),
            "n_chunks": len(chunks),
            "sampling_chunk_size": chunk_size,
            "output_dir": str(output_dir),
        },
        "sampling_tasks": sampling_tasks,
        "metrics_tasks": metrics_tasks,
    }

    return manifest


def chunk_replicates(n_total: int, chunk_size: int) -> List[int]:
    """Return list of per-chunk replicate counts. Last chunk may be smaller."""
    chunks = []
    remaining = n_total
    while remaining > 0:
        chunks.append(min(chunk_size, remaining))
        remaining -= chunk_size
    return chunks


def build_manifest_cli(config: dict, output_dir: Path) -> dict:
    """Build manifest for CLI mode (file-based, single system, chunked replicates).

    Args:
        config: Parsed pipeline config dict.
        output_dir: Root output directory for the pipeline run.

    Returns:
        Manifest dict with meta, sampling_tasks, and metrics_tasks.
    """
    n_total = config["n_replicates_total"]
    per_chunk = config["replicates_per_chunk"]
    chunks = chunk_replicates(n_total, per_chunk)

    sampling_tasks = {}
    metrics_tasks = {}

    for i, n_reps in enumerate(chunks):
        chunk_output = str(output_dir / "samples" / f"chunk_{i}")

        sampling_tasks[str(i)] = {
            "chunk_id": i,
            "n_replicates": n_reps,
            "output_dir": chunk_output,
        }

        metrics_tasks[str(i)] = {
            "chunk_id": i,
            "samples_dir": chunk_output,
            "n_replicates": n_reps,
            "output_dir": str(output_dir / "metrics" / f"chunk_{i}"),
        }

    manifest = {
        "meta": {
            "mode": "cli",
            "task": config["task"],
            "checkpoint": config.get("checkpoint"),
            "n_replicates_total": n_total,
            "replicates_per_chunk": per_chunk,
            "n_chunks": len(chunks),
            "output_dir": str(output_dir),
            "input_files": config.get("input_files", {}),
        },
        "sampling_tasks": sampling_tasks,
        "metrics_tasks": metrics_tasks,
    }

    return manifest


def write_manifest(manifest: dict, path: Path) -> None:
    """Write manifest dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: Path) -> dict:
    """Load manifest dict from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def get_incomplete_tasks(manifest: dict, stage: str) -> List[int]:
    """Return task IDs whose status files do not exist.

    Args:
        manifest: The pipeline manifest dict.
        stage: One of 'sampling' or 'metrics'.

    Returns:
        List of integer task IDs that have not yet completed.
    """
    output_dir = Path(manifest["meta"]["output_dir"])
    status_dir = output_dir / "status"

    if stage == "sampling":
        tasks = manifest["sampling_tasks"]
    elif stage == "metrics":
        tasks = manifest["metrics_tasks"]
    else:
        raise ValueError(f"Unknown stage: {stage}")

    incomplete = []
    for task_id_str in tasks:
        status_file = status_dir / f"{stage}_{task_id_str}.done"
        if not status_file.exists():
            incomplete.append(int(task_id_str))

    return sorted(incomplete)

#!/usr/bin/env python3
"""Aggregate per-chunk metrics and sys_info into combined result files."""

import argparse
import json
from pathlib import Path

import pandas as pd

from omtra_pipelines.distributed_sampling.manifest import load_manifest


def merge_sdf_files(manifest: dict, results_dir: Path) -> None:
    """Concatenate gen_ligands.sdf from all chunks into a single file (CLI mode)."""
    sdf_parts = []
    for task_id, task_info in sorted(
        manifest["sampling_tasks"].items(), key=lambda x: int(x[0])
    ):
        sdf_file = Path(task_info["output_dir"]) / "sys_0_gt" / "gen_ligands.sdf"
        if sdf_file.exists():
            sdf_parts.append(sdf_file.read_text())
        else:
            print(f"Warning: missing gen_ligands.sdf for chunk {task_id}: {sdf_file}")

    if sdf_parts:
        merged_path = results_dir / "gen_ligands_all.sdf"
        merged_path.write_text("".join(sdf_parts))
        print(f"Merged {len(sdf_parts)} SDF files to {merged_path}")
    else:
        print("No gen_ligands.sdf files found to merge.")


def aggregate(manifest_path: Path) -> None:
    manifest = load_manifest(manifest_path)
    output_dir = Path(manifest["meta"]["output_dir"])
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    mode = manifest["meta"].get("mode", "dataset")

    # Collect eval_metrics.csv from each metrics chunk
    metrics_dfs = []
    for task_id, task_info in sorted(manifest["metrics_tasks"].items(), key=lambda x: int(x[0])):
        metrics_file = Path(task_info["output_dir"]) / "eval_metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            df["chunk_id"] = int(task_id)
            metrics_dfs.append(df)
        else:
            print(f"Warning: missing metrics for chunk {task_id}: {metrics_file}")

    if metrics_dfs:
        metrics_all = pd.concat(metrics_dfs, ignore_index=True)
        metrics_all.to_csv(results_dir / "eval_metrics_all.csv", index=False)
        print(f"Wrote {len(metrics_all)} rows to {results_dir / 'eval_metrics_all.csv'}")
    else:
        print("No metrics files found.")
        metrics_all = pd.DataFrame()

    # Collect sys_info.csv from each sampling chunk
    sys_info_dfs = []
    for task_id, task_info in sorted(manifest["sampling_tasks"].items(), key=lambda x: int(x[0])):
        sys_info_file = Path(task_info["output_dir"]) / "sys_info.csv"
        if sys_info_file.exists():
            df = pd.read_csv(sys_info_file)
            df["chunk_id"] = int(task_id)
            sys_info_dfs.append(df)

    if sys_info_dfs:
        sys_info_all = pd.concat(sys_info_dfs, ignore_index=True)
        sys_info_all.to_csv(results_dir / "sys_info_all.csv", index=False)
        print(f"Wrote {len(sys_info_all)} rows to {results_dir / 'sys_info_all.csv'}")

    # CLI mode: merge generated ligand SDFs across chunks
    if mode == "cli":
        merge_sdf_files(manifest, results_dir)

    # Summary
    summary = {
        "n_chunks_total": manifest["meta"]["n_chunks"],
        "n_chunks_with_metrics": len(metrics_dfs),
        "n_rows_metrics": len(metrics_all),
        "mode": mode,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate pipeline results")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json")
    args = parser.parse_args()
    aggregate(args.manifest)


if __name__ == "__main__":
    main()

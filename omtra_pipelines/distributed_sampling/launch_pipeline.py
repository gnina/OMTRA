#!/usr/bin/env python3
"""Main entry point for the distributed sampling pipeline.

Usage:
    # Full pipeline with dry-run (inspect generated files without submitting)
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml --dry-run

    # Submit the full pipeline
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml

    # Resume failed tasks
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml --resume

    # Check pipeline status
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml --status

    # Run only one stage
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml --stage sampling
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

import yaml

from omtra_pipelines.distributed_sampling.manifest import (
    build_manifest,
    build_manifest_cli,
    get_incomplete_tasks,
    load_manifest,
    write_manifest,
)
from omtra_pipelines.distributed_sampling.commands import (
    build_cli_metrics_command,
    build_cli_sampling_command,
    build_metrics_command,
    build_sampling_command,
    write_commands_file,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "slurm_templates"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch the OMTRA distributed sampling pipeline"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to pipeline YAML config"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate all files but do not submit SLURM jobs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing manifest, resubmitting only incomplete tasks",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "sampling", "metrics", "aggregate"],
        default="all",
        help="Which stage(s) to run (default: all)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print pipeline status and exit",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def render_slurm_script(template_path: Path, variables: dict) -> str:
    """Render a SLURM template with string.Template substitution.

    Uses $$ for literal $ in templates (bash variables like $$SLURM_ARRAY_TASK_ID).
    """
    raw = template_path.read_text()
    return Template(raw).safe_substitute(variables)


def make_array_spec(task_ids: list) -> str:
    """Build a SLURM --array spec from a list of task IDs.

    Produces compact range notation where possible:
      [0,1,2,3,5,7,8,9] -> "0-3,5,7-9"
    """
    if not task_ids:
        return ""

    task_ids = sorted(set(task_ids))
    ranges = []
    start = task_ids[0]
    end = start

    for tid in task_ids[1:]:
        if tid == end + 1:
            end = tid
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = tid
    ranges.append(f"{start}-{end}" if start != end else str(start))

    return ",".join(ranges)


def submit_slurm_job(
    script_path: Path, array_spec: str = None, dependency: str = None
) -> str:
    """Submit a SLURM job and return the job ID."""
    cmd = ["sbatch"]
    if array_spec:
        cmd.extend(["--array", array_spec])
    if dependency:
        cmd.extend(["--dependency", dependency])
    cmd.append(str(script_path))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"sbatch failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    return job_id


def print_status(manifest: dict) -> None:
    """Print a status table for the pipeline."""
    output_dir = Path(manifest["meta"]["output_dir"])
    status_dir = output_dir / "status"
    mode = manifest["meta"].get("mode", "dataset")

    print(f"\nPipeline: {output_dir}")
    if mode == "cli":
        print(f"Mode: CLI | Task: {manifest['meta']['task']}")
        print(f"Replicates: {manifest['meta']['n_replicates_total']} | Chunks: {manifest['meta']['n_chunks']}")
    else:
        print(f"Mode: Dataset | Task: {manifest['meta']['task']} | Dataset: {manifest['meta']['dataset']}")
        print(f"Systems: {manifest['meta']['n_systems']} | Chunks: {manifest['meta']['n_chunks']}")
    print()

    for stage, tasks in [("sampling", manifest["sampling_tasks"]), ("metrics", manifest["metrics_tasks"])]:
        n_total = len(tasks)
        n_done = sum(
            1 for tid in tasks if (status_dir / f"{stage}_{tid}.done").exists()
        )
        print(f"  {stage:12s}: {n_done}/{n_total} complete")

        # Show incomplete task IDs if any
        incomplete = [tid for tid in tasks if not (status_dir / f"{stage}_{tid}.done").exists()]
        if incomplete and len(incomplete) <= 20:
            print(f"  {'':12s}  incomplete: {', '.join(incomplete)}")
        elif incomplete:
            print(f"  {'':12s}  incomplete: {len(incomplete)} tasks")

    agg_done = (status_dir / "aggregate.done").exists()
    print(f"  {'aggregate':12s}: {'done' if agg_done else 'pending'}")
    print()


def get_mode(config: dict) -> str:
    """Return 'dataset' or 'cli'. Infer from config if 'mode' not explicit."""
    if "mode" in config:
        mode = config["mode"]
        if mode not in ("dataset", "cli"):
            raise ValueError(f"Invalid mode: {mode!r} (expected 'dataset' or 'cli')")
        return mode
    if "input_files" in config:
        return "cli"
    return "dataset"


def validate_cli_config(config: dict) -> None:
    """Validate CLI mode config has required fields."""
    required = ["task", "n_replicates_total", "replicates_per_chunk", "input_files"]
    missing = [k for k in required if k not in config]
    if missing:
        print(f"Error: CLI mode config missing required fields: {missing}", file=sys.stderr)
        sys.exit(1)

    input_files = config["input_files"]
    if not isinstance(input_files, dict) or not input_files:
        print("Error: 'input_files' must be a non-empty dict", file=sys.stderr)
        sys.exit(1)

    # At least one pocket definition method is required for protein tasks
    pocket_keys = {"pocket_ligand", "pocket_center", "pocket_residues"}
    has_pocket = any(input_files.get(k) for k in pocket_keys)
    has_protein = input_files.get("protein_file") is not None
    if has_protein and not has_pocket:
        print(
            "Warning: protein_file provided without a pocket definition "
            "(pocket_ligand, pocket_center, or pocket_residues). "
            "The omtra CLI will fail unless the task doesn't need a pocket."
        )


def convert_pharmacophore_to_xyz(pharm_file: Path, xyz_path: Path) -> None:
    """Convert a Pharmit JSON pharmacophore file to XYZ format for metrics.

    If the file is already in XYZ format (or SDF), it is copied as-is.
    """
    from omtra.constants import ph_idx_to_elem, ph_idx_to_type, ph_type_to_idx

    suffix = pharm_file.suffix.lower()
    if suffix == ".xyz":
        shutil.copy2(pharm_file, xyz_path)
        return

    if suffix == ".json":
        with open(pharm_file, "r") as f:
            data = json.load(f)

        points = data.get("points", [])
        enabled = [p for p in points if p.get("enabled", True)]
        if not enabled:
            raise ValueError(f"No enabled pharmacophore points in {pharm_file}")

        lines = [str(len(enabled)), "pharmacophore"]
        for p in enabled:
            name = p["name"]
            idx = ph_type_to_idx.get(name)
            if idx is None:
                raise ValueError(f"Unknown pharmacophore type: {name!r}")
            elem = ph_idx_to_elem[idx]
            lines.append(f"{elem}  {p['x']:.6f}  {p['y']:.6f}  {p['z']:.6f}")

        xyz_path.parent.mkdir(parents=True, exist_ok=True)
        xyz_path.write_text("\n".join(lines) + "\n")
        return

    # SDF or other — copy as-is (metrics code may handle it)
    shutil.copy2(pharm_file, xyz_path)


def prepare_ground_truth_files(manifest: dict, config: dict) -> None:
    """Copy input files into each chunk's output dir for metrics compatibility.

    Creates ``samples/chunk_K/sys_0_gt/`` with:
    - ``protein_0.pdb``  from protein_file
    - ``ligand.sdf``     from ligand_file (if provided)
    - ``pharmacophore.xyz`` from pharmacophore_file (converted if JSON)
    """
    input_files = config["input_files"]

    for task_info in manifest["sampling_tasks"].values():
        gt_dir = Path(task_info["output_dir"]) / "sys_0_gt"
        gt_dir.mkdir(parents=True, exist_ok=True)

        # Protein
        prot_src = input_files.get("protein_file")
        if prot_src:
            shutil.copy2(prot_src, gt_dir / "protein_0.pdb")

        # Ligand (ground truth for RMSD etc.)
        lig_src = input_files.get("ligand_file")
        if lig_src:
            shutil.copy2(lig_src, gt_dir / "ligand.sdf")

        # Pharmacophore
        pharm_src = input_files.get("pharmacophore_file")
        if pharm_src:
            convert_pharmacophore_to_xyz(Path(pharm_src), gt_dir / "pharmacophore.xyz")


def main():
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(config["output_dir"]).resolve()

    # --- Status mode ---
    if args.status:
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"No manifest found at {manifest_path}")
            sys.exit(1)
        manifest = load_manifest(manifest_path)
        print_status(manifest)
        return

    # --- Build or load manifest ---
    manifest_path = output_dir / "manifest.json"

    mode = get_mode(config)

    if args.resume:
        if not manifest_path.exists():
            print(f"Cannot resume: no manifest at {manifest_path}")
            sys.exit(1)
        manifest = load_manifest(manifest_path)
        mode = manifest["meta"].get("mode", "dataset")
        print(f"Resumed manifest from {manifest_path}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

        if mode == "cli":
            validate_cli_config(config)
            manifest = build_manifest_cli(config, output_dir)
            write_manifest(manifest, manifest_path)
            prepare_ground_truth_files(manifest, config)

            shutil.copy2(args.config, output_dir / "pipeline_config.yaml")
            print(f"Built CLI manifest: {manifest['meta']['n_chunks']} chunks, "
                  f"{manifest['meta']['n_replicates_total']} total replicates")
        else:
            manifest = build_manifest(config, output_dir)
            write_manifest(manifest, manifest_path)

            shutil.copy2(args.config, output_dir / "pipeline_config.yaml")
            print(f"Built manifest: {manifest['meta']['n_chunks']} chunks, "
                  f"{manifest['meta']['n_systems']} systems")

    # Create output directories
    for d in ["status", "logs", "samples", "metrics", "results",
              "work/scripts", "work/commands"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    # --- Determine which tasks to run ---
    if args.resume:
        sampling_ids = get_incomplete_tasks(manifest, "sampling")
        metrics_ids = get_incomplete_tasks(manifest, "metrics")
    else:
        sampling_ids = [int(tid) for tid in manifest["sampling_tasks"]]
        metrics_ids = [int(tid) for tid in manifest["metrics_tasks"]]

    # --- Generate command files ---
    # Sampling commands: one per task, lines indexed from 1 (for sed -n "${ID}p")
    # We write ALL tasks (not just incomplete) so line numbers stay consistent.
    # The array spec controls which tasks actually run.
    all_sampling_cmds = []
    all_metrics_cmds = []

    if mode == "cli":
        for i in range(manifest["meta"]["n_chunks"]):
            task_info = manifest["sampling_tasks"][str(i)]
            all_sampling_cmds.append(build_cli_sampling_command(task_info, config))
        for i in range(manifest["meta"]["n_chunks"]):
            task_info = manifest["metrics_tasks"][str(i)]
            all_metrics_cmds.append(build_cli_metrics_command(task_info, config))
    else:
        for i in range(manifest["meta"]["n_chunks"]):
            task_info = manifest["sampling_tasks"][str(i)]
            all_sampling_cmds.append(build_sampling_command(task_info, config))
        for i in range(manifest["meta"]["n_chunks"]):
            task_info = manifest["metrics_tasks"][str(i)]
            all_metrics_cmds.append(build_metrics_command(task_info, config))

    sampling_cmds_path = output_dir / "work" / "commands" / "sampling_commands.txt"
    write_commands_file(all_sampling_cmds, sampling_cmds_path)

    metrics_cmds_path = output_dir / "work" / "commands" / "metrics_commands.txt"
    write_commands_file(all_metrics_cmds, metrics_cmds_path)

    # --- Render SLURM scripts ---
    slurm_vars = {
        "partition_gpu": config.get("slurm", {}).get("partition_gpu", "dept_gpu"),
        "partition_cpu": config.get("slurm", {}).get("partition_cpu", "dept_cpu"),
        "cpus_per_task": str(config.get("slurm", {}).get("cpus_per_task", 4)),
        "mem": config.get("slurm", {}).get("mem", "32G"),
        "time_sampling": config.get("slurm", {}).get("time_sampling", "4:00:00"),
        "time_metrics": config.get("slurm", {}).get("time_metrics", "8:00:00"),
        "time_aggregate": config.get("slurm", {}).get("time_aggregate", "0:30:00"),
        "conda_env": config.get("slurm", {}).get("conda_env", "omtra"),
        "extra_sbatch_args": config.get("slurm", {}).get("extra_sbatch_args", ""),
        "log_dir": str(output_dir / "logs"),
        "status_dir": str(output_dir / "status"),
        "sampling_commands_file": str(sampling_cmds_path),
        "metrics_commands_file": str(metrics_cmds_path),
        "manifest_path": str(manifest_path),
    }

    scripts_dir = output_dir / "work" / "scripts"

    sampling_script = scripts_dir / "sampling.slurm"
    sampling_script.write_text(
        render_slurm_script(TEMPLATES_DIR / "sampling.slurm", slurm_vars)
    )

    metrics_script = scripts_dir / "metrics.slurm"
    metrics_script.write_text(
        render_slurm_script(TEMPLATES_DIR / "metrics.slurm", slurm_vars)
    )

    aggregate_script = scripts_dir / "aggregate.slurm"
    aggregate_script.write_text(
        render_slurm_script(TEMPLATES_DIR / "aggregate.slurm", slurm_vars)
    )

    # --- Summary ---
    run_sampling = args.stage in ("all", "sampling")
    run_metrics = args.stage in ("all", "metrics")
    run_aggregate = args.stage in ("all", "aggregate")

    sampling_array = make_array_spec(sampling_ids) if run_sampling else ""
    metrics_array = make_array_spec(metrics_ids) if run_metrics else ""

    print(f"\nOutput directory: {output_dir}")
    print(f"Commands files:  {sampling_cmds_path}")
    print(f"                 {metrics_cmds_path}")
    print(f"SLURM scripts:   {scripts_dir}")

    if run_sampling and sampling_array:
        print(f"\nSampling: {len(sampling_ids)} tasks, array={sampling_array}")
    if run_metrics and metrics_array:
        print(f"Metrics:  {len(metrics_ids)} tasks, array={metrics_array}")
    if run_aggregate:
        print("Aggregate: 1 job")

    if args.dry_run:
        print("\n[DRY RUN] No jobs submitted. Inspect files above.")
        return

    # --- Submit jobs ---
    sampling_job_id = None
    metrics_job_id = None

    if run_sampling and sampling_array:
        sampling_job_id = submit_slurm_job(sampling_script, array_spec=sampling_array)
        print(f"\nSubmitted sampling job: {sampling_job_id}")

    if run_metrics and metrics_array:
        dep = f"afterok:{sampling_job_id}" if sampling_job_id else None
        metrics_job_id = submit_slurm_job(metrics_script, array_spec=metrics_array, dependency=dep)
        print(f"Submitted metrics job:  {metrics_job_id}")

    if run_aggregate:
        dep = f"afterok:{metrics_job_id}" if metrics_job_id else None
        agg_job_id = submit_slurm_job(aggregate_script, dependency=dep)
        print(f"Submitted aggregate job: {agg_job_id}")

    print("\nPipeline submitted. Monitor with: --config ... --status")


if __name__ == "__main__":
    main()

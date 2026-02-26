#!/usr/bin/env python3
"""Main entry point for the distributed sampling pipeline.

Usage:
    # Full pipeline with dry-run (inspect generated files without submitting)
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml --dry-run

    # Submit the full pipeline
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml

    # With site-specific SLURM/paths config
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml --site cluster

    # With inline overrides (dotlist syntax, highest priority)
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml \
        sampling.n_timesteps=500 metrics.disable_gnina=true

    # Resume failed tasks
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml --resume

    # Check pipeline status
    python -m omtra_pipelines.distributed_sampling.launch_pipeline --config pipeline.yaml --status

Config loading order (later overrides earlier):
    1. defaults/default.yaml               (always — Tier 2+3 defaults)
    2. defaults/site/{name}.yaml           (if --site specified)
    3. User --config YAML file             (Tier 1 required + any overrides)
    4. CLI positional overrides            (key=value — highest priority)
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

from omegaconf import OmegaConf, DictConfig

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
CONFIGS_DIR = Path(__file__).resolve().parent / "defaults"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_layered_config(
    config_path: Path = None,
    site: str = None,
    overrides: list = None,
) -> DictConfig:
    """Load pipeline config with layered merging.

    Loading order (later overrides earlier):
        1. defaults/default.yaml             (always)
        2. defaults/site/{site}.yaml         (if site specified)
        3. User config file at config_path (if provided)
        4. CLI overrides from dotlist (if provided)

    Returns:
        Merged OmegaConf DictConfig.
    """
    layers = []

    # Layer 1: base defaults
    base_path = CONFIGS_DIR / "default.yaml"
    if base_path.exists():
        layers.append(OmegaConf.load(base_path))
    else:
        print(f"Warning: base config not found at {base_path}", file=sys.stderr)

    # Layer 2: site config
    if site:
        site_path = CONFIGS_DIR / "site" / f"{site}.yaml"
        if site_path.exists():
            layers.append(OmegaConf.load(site_path))
        else:
            print(f"Error: site config not found: {site_path}", file=sys.stderr)
            sys.exit(1)

    # Layer 3: user config file
    if config_path:
        if not config_path.exists():
            print(f"Error: config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        layers.append(OmegaConf.load(config_path))

    # Layer 4: CLI overrides
    if overrides:
        layers.append(OmegaConf.from_dotlist(overrides))

    if not layers:
        print("Error: no config provided. Use --config and/or --site.", file=sys.stderr)
        sys.exit(1)

    return OmegaConf.merge(*layers)


def _resolve_compat_keys(cfg: DictConfig) -> DictConfig:
    """Support legacy config key names by mapping them to new names.

    Old names are accepted and mapped to canonical names:
        sampling_chunk_size  -> systems_per_job
        n_replicates         -> replicates_per_system   (dataset mode only)
        replicates_per_chunk -> replicates_per_job

    When a legacy key is present, it always wins (the user explicitly set it).
    """
    container = OmegaConf.to_container(cfg, resolve=False)

    # sampling_chunk_size -> systems_per_job
    if "sampling_chunk_size" in container:
        container["systems_per_job"] = container.pop("sampling_chunk_size")

    # n_replicates -> replicates_per_system (only for dataset mode)
    mode = _get_mode_from_container(container)
    if mode == "dataset" and "n_replicates" in container:
        container["replicates_per_system"] = container.pop("n_replicates")

    # replicates_per_chunk -> replicates_per_job
    if "replicates_per_chunk" in container:
        container["replicates_per_job"] = container.pop("replicates_per_chunk")

    return OmegaConf.create(container)


def _get_mode_from_container(container: dict) -> str:
    """Infer mode from a raw config dict."""
    if container.get("mode") in ("dataset", "cli"):
        return container["mode"]
    if container.get("input_files"):
        return "cli"
    return "dataset"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch the OMTRA distributed sampling pipeline",
        epilog=(
            "Positional arguments after flags are treated as config overrides "
            "using dotlist syntax (e.g. sampling.n_timesteps=500)."
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to user pipeline YAML config (Tier 1 settings + any overrides)",
    )
    parser.add_argument(
        "--site", type=str, default=None,
        help="Site config name (loads defaults/site/{name}.yaml for Tier 3 settings)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate all files but do not submit SLURM jobs",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing manifest, resubmitting only incomplete tasks",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "sampling", "metrics", "aggregate"],
        default="all",
        help="Which stage(s) to run (default: all)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print pipeline status and exit",
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Config overrides in dotlist syntax (e.g. slurm.mem=64G)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mode detection and validation
# ---------------------------------------------------------------------------

def get_mode(cfg: DictConfig) -> str:
    """Return 'dataset' or 'cli'. Infer from config if 'mode' not explicit."""
    mode = OmegaConf.select(cfg, "mode", default=None)
    if mode is not None:
        if mode not in ("dataset", "cli"):
            raise ValueError(f"Invalid mode: {mode!r} (expected 'dataset' or 'cli')")
        return mode
    if OmegaConf.select(cfg, "input_files", default=None) is not None:
        return "cli"
    return "dataset"


def validate_cli_config(cfg: DictConfig) -> None:
    """Validate CLI mode config has required fields."""
    from omtra.tasks.register import task_name_to_class

    missing = []
    if not OmegaConf.select(cfg, "task", default=None):
        missing.append("task")
    if not OmegaConf.select(cfg, "n_replicates_total", default=None):
        missing.append("n_replicates_total")
    if not OmegaConf.select(cfg, "output_dir", default=None):
        missing.append("output_dir")

    if missing:
        print(f"Error: CLI mode config missing required fields: {missing}", file=sys.stderr)
        sys.exit(1)

    # Only require input_files when the task has fixed groups (i.e. conditioning inputs)
    task = task_name_to_class(cfg.task)
    if task.groups_fixed:
        if not OmegaConf.select(cfg, "input_files", default=None):
            print("Error: CLI mode config missing required field: input_files "
                  f"(task '{cfg.task}' has fixed groups)", file=sys.stderr)
            sys.exit(1)

        input_files = cfg.input_files
        if not input_files:
            print("Error: 'input_files' must be a non-empty dict", file=sys.stderr)
            sys.exit(1)

        # At least one pocket definition method is required for protein tasks
        pocket_keys = {"pocket_ligand", "pocket_center", "pocket_residues"}
        has_pocket = any(OmegaConf.select(input_files, k, default=None) for k in pocket_keys)
        has_protein = OmegaConf.select(input_files, "protein_file", default=None) is not None
        if has_protein and not has_pocket:
            print(
                "Warning: protein_file provided without a pocket definition "
                "(pocket_ligand, pocket_center, or pocket_residues). "
                "The omtra CLI will fail unless the task doesn't need a pocket."
            )


def validate_dataset_config(cfg: DictConfig) -> None:
    """Validate dataset mode config has required fields."""
    missing = []
    if not OmegaConf.select(cfg, "task", default=None):
        missing.append("task")
    if not OmegaConf.select(cfg, "checkpoint", default=None):
        missing.append("checkpoint")
    if not OmegaConf.select(cfg, "dataset", default=None):
        missing.append("dataset")
    if not OmegaConf.select(cfg, "output_dir", default=None):
        missing.append("output_dir")

    has_sys_idx = OmegaConf.select(cfg, "sys_idx_file", default=None) is not None
    has_n_systems = OmegaConf.select(cfg, "n_systems", default=None) is not None
    if not has_sys_idx and not has_n_systems:
        missing.append("sys_idx_file or n_systems")

    if missing:
        print(f"Error: Dataset mode config missing required fields: {missing}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Config to dict conversion for manifest/commands compatibility
# ---------------------------------------------------------------------------

def cfg_to_legacy_dict(cfg: DictConfig, mode: str) -> dict:
    """Convert the layered OmegaConf config to a plain dict with legacy key names.

    The manifest and command builders expect the old key names. This function
    maps the new canonical names back to the legacy format they understand.
    """
    d = OmegaConf.to_container(cfg, resolve=True)

    # Map new names -> legacy names for manifest/commands compatibility
    if mode == "dataset":
        d["sampling_chunk_size"] = d.get("systems_per_job", d.get("sampling_chunk_size", 10))
        d["n_replicates"] = d.get("replicates_per_system", d.get("n_replicates", 1))
    else:
        d["replicates_per_chunk"] = d.get("replicates_per_job", d.get("replicates_per_chunk", 100))

    return d


# ---------------------------------------------------------------------------
# Ground truth file preparation (CLI mode)
# ---------------------------------------------------------------------------

def convert_pharmacophore_to_xyz(pharm_file: Path, xyz_path: Path) -> None:
    """Convert a Pharmit JSON pharmacophore file to XYZ format for metrics.

    If the file is already in XYZ format (or SDF), it is copied as-is.
    """
    from omtra.constants import ph_idx_to_elem, ph_type_to_idx

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
    input_files = config.get("input_files", {})

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Load layered config ---
    cfg = load_layered_config(
        config_path=args.config,
        site=args.site,
        overrides=args.overrides,
    )
    cfg = _resolve_compat_keys(cfg)

    output_dir = Path(cfg.output_dir).resolve()
    mode = get_mode(cfg)

    # Convert to legacy dict for manifest/commands compatibility
    config = cfg_to_legacy_dict(cfg, mode)

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

    if args.resume:
        if not manifest_path.exists():
            print(f"Cannot resume: no manifest at {manifest_path}")
            sys.exit(1)
        manifest = load_manifest(manifest_path)
        mode = manifest["meta"].get("mode", "dataset")
        print(f"Resumed manifest from {manifest_path}")
    else:
        # Validate config
        if mode == "cli":
            validate_cli_config(cfg)
        else:
            validate_dataset_config(cfg)

        output_dir.mkdir(parents=True, exist_ok=True)

        if mode == "cli":
            manifest = build_manifest_cli(config, output_dir)
            write_manifest(manifest, manifest_path)
            prepare_ground_truth_files(manifest, config)

            # Save resolved config for reproducibility
            OmegaConf.save(cfg, output_dir / "pipeline_config.yaml")
            print(f"Built CLI manifest: {manifest['meta']['n_chunks']} chunks, "
                  f"{manifest['meta']['n_replicates_total']} total replicates")
        else:
            manifest = build_manifest(config, output_dir)
            write_manifest(manifest, manifest_path)

            OmegaConf.save(cfg, output_dir / "pipeline_config.yaml")
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
    slurm = config.get("slurm", {})
    slurm_vars = {
        "partition_gpu": slurm.get("partition_gpu", "dept_gpu"),
        "partition_cpu": slurm.get("partition_cpu", "dept_cpu"),
        "cpus_per_task": str(slurm.get("cpus_per_task", 4)),
        "mem": slurm.get("mem", "32G"),
        "time_sampling": slurm.get("time_sampling", "4:00:00"),
        "time_metrics": slurm.get("time_metrics", "8:00:00"),
        "time_aggregate": slurm.get("time_aggregate", "0:30:00"),
        "conda_env": slurm.get("conda_env", "omtra"),
        "extra_sbatch_args": slurm.get("extra_sbatch_args", ""),
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

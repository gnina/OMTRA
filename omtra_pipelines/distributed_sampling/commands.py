"""Command generation for sampling and metrics stages."""

from pathlib import Path
from typing import List


DOCKING_EVAL_SCRIPT = (
    Path(__file__).resolve().parent.parent / "docking_eval" / "docking_eval.py"
)


def _add_if_not_none(parts: list, flag: str, value) -> None:
    if value is not None:
        parts.extend([flag, str(value)])


def _add_flag(parts: list, flag: str, value: bool) -> None:
    if value:
        parts.append(flag)


def build_sampling_command(task_info: dict, config: dict) -> str:
    """Build a docking_eval.py --sample_only command for one sampling task.

    Args:
        task_info: One entry from manifest["sampling_tasks"].
        config: The full pipeline config dict.
    """
    sampling_opts = config.get("sampling", {})
    paths = config.get("paths", {})

    parts = [
        "python",
        str(DOCKING_EVAL_SCRIPT),
        "--ckpt_path", config["checkpoint"],
        "--task", config["task"],
        "--dataset", config["dataset"],
        "--split", config.get("split", "test"),
        "--sys_idx_file", task_info["chunk_file"],
        "--n_samples", str(task_info["n_systems"]),
        "--n_replicates", str(config.get("n_replicates", 1)),
        "--output_dir", task_info["output_dir"],
        "--sample_only",
    ]

    _add_if_not_none(parts, "--n_timesteps", sampling_opts.get("n_timesteps"))
    _add_if_not_none(parts, "--bs_per_gbmem", sampling_opts.get("bs_per_gbmem"))
    _add_if_not_none(parts, "--max_batch_size", sampling_opts.get("max_batch_size"))
    _add_flag(parts, "--stochastic_sampling", sampling_opts.get("stochastic_sampling", False))

    _add_if_not_none(parts, "--plinder_path", paths.get("plinder"))
    _add_if_not_none(parts, "--crossdocked_path", paths.get("crossdocked"))

    return " ".join(parts)


def build_metrics_command(task_info: dict, config: dict) -> str:
    """Build a docking_eval.py --samples_dir command for one metrics task.

    Args:
        task_info: One entry from manifest["metrics_tasks"].
        config: The full pipeline config dict.
    """
    metrics_opts = config.get("metrics", {})
    paths = config.get("paths", {})

    # Determine n_samples for metrics: dataset mode uses chunk n_systems,
    # CLI mode always has 1 system per chunk.
    n_samples = task_info.get("n_systems")
    if n_samples is None:
        n_samples = 1  # CLI mode: single system per chunk

    n_replicates = task_info.get("n_replicates", config.get("n_replicates", 1))

    parts = [
        "python",
        str(DOCKING_EVAL_SCRIPT),
        "--samples_dir", task_info["samples_dir"],
        "--task", config["task"],
        "--dataset", config.get("dataset") or "plinder",
        "--split", config.get("split") or "test",
        "--n_samples", str(n_samples),
        "--n_replicates", str(n_replicates),
        "--output_dir", task_info["output_dir"],
    ]

    _add_if_not_none(parts, "--timeout", metrics_opts.get("timeout"))
    _add_flag(parts, "--disable_gnina", metrics_opts.get("disable_gnina", False))
    _add_flag(parts, "--disable_pb_valid", metrics_opts.get("disable_pb_valid", False))
    _add_flag(parts, "--disable_posecheck", metrics_opts.get("disable_posecheck", False))
    _add_flag(parts, "--disable_rmsd", metrics_opts.get("disable_rmsd", False))
    _add_flag(parts, "--disable_strain", metrics_opts.get("disable_strain", False))
    _add_flag(parts, "--disable_interaction_recovery", metrics_opts.get("disable_interaction_recovery", False))
    _add_flag(parts, "--disable_pharm_match", metrics_opts.get("disable_pharm_match", False))
    _add_flag(parts, "--disable_ground_truth_metrics", metrics_opts.get("disable_ground_truth_metrics", False))

    _add_if_not_none(parts, "--plinder_path", paths.get("plinder"))
    _add_if_not_none(parts, "--crossdocked_path", paths.get("crossdocked"))

    return " ".join(parts)


def build_cli_sampling_command(task_info: dict, config: dict) -> str:
    """Build an ``omtra`` CLI command for sampling from files.

    Args:
        task_info: One entry from manifest["sampling_tasks"].
        config: The full pipeline config dict.
    """
    sampling_opts = config.get("sampling", {})
    input_files = config.get("input_files", {})

    parts = [
        "omtra",
        "--task", config["task"],
        "--n_samples", str(task_info["n_replicates"]),
        "--output_dir", task_info["output_dir"],
    ]

    # Checkpoint (optional — omtra CLI auto-resolves if omitted)
    if config.get("checkpoint"):
        parts.extend(["--checkpoint", config["checkpoint"]])

    # Input files
    _add_if_not_none(parts, "--protein_file", input_files.get("protein_file"))
    _add_if_not_none(parts, "--ligand_file", input_files.get("ligand_file"))
    _add_if_not_none(parts, "--pharmacophore_file", input_files.get("pharmacophore_file"))
    _add_if_not_none(parts, "--pocket_ligand", input_files.get("pocket_ligand"))
    _add_if_not_none(parts, "--pocket_center", input_files.get("pocket_center"))
    _add_if_not_none(parts, "--pocket_residues", input_files.get("pocket_residues"))
    _add_if_not_none(parts, "--bbox_length", input_files.get("bbox_length"))

    # Sampling options
    _add_if_not_none(parts, "--n_timesteps", sampling_opts.get("n_timesteps"))
    _add_flag(parts, "--stochastic_sampling", sampling_opts.get("stochastic_sampling", False))
    _add_if_not_none(parts, "--noise_scaler", sampling_opts.get("noise_scaler"))
    _add_if_not_none(parts, "--eps", sampling_opts.get("eps"))

    return " ".join(parts)


def build_cli_metrics_command(task_info: dict, config: dict) -> str:
    """Build a docking_eval.py --samples_dir command for CLI mode metrics.

    Args:
        task_info: One entry from manifest["metrics_tasks"].
        config: The full pipeline config dict.
    """
    metrics_opts = config.get("metrics", {})

    parts = [
        "python",
        str(DOCKING_EVAL_SCRIPT),
        "--samples_dir", task_info["samples_dir"],
        "--task", config["task"],
        "--dataset", config.get("dataset") or "plinder",
        "--split", config.get("split") or "test",
        "--n_samples", "1",
        "--n_replicates", str(task_info["n_replicates"]),
        "--output_dir", task_info["output_dir"],
    ]

    _add_if_not_none(parts, "--timeout", metrics_opts.get("timeout"))
    _add_flag(parts, "--disable_gnina", metrics_opts.get("disable_gnina", False))
    _add_flag(parts, "--disable_pb_valid", metrics_opts.get("disable_pb_valid", False))
    _add_flag(parts, "--disable_posecheck", metrics_opts.get("disable_posecheck", False))
    _add_flag(parts, "--disable_rmsd", metrics_opts.get("disable_rmsd", False))
    _add_flag(parts, "--disable_strain", metrics_opts.get("disable_strain", False))
    _add_flag(parts, "--disable_interaction_recovery", metrics_opts.get("disable_interaction_recovery", False))
    _add_flag(parts, "--disable_pharm_match", metrics_opts.get("disable_pharm_match", False))
    _add_flag(parts, "--disable_ground_truth_metrics", metrics_opts.get("disable_ground_truth_metrics", False))

    return " ".join(parts)


def write_commands_file(commands: List[str], path: Path) -> None:
    """Write one command per line, indexed from line 1 (for sed -n)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for cmd in commands:
            f.write(cmd + "\n")

"""Orchestration for computing docking evaluation metrics.

Provides the main ``compute_metrics`` loop, ``system_pairs_from_path`` for
reading sample directories, and helpers for determining which metrics apply
to a given task.

Functions moved here from ``omtra_pipelines/docking_eval/docking_eval.py``
so that ``routines/sample.py`` and the CLI can reuse them without importing
from the pipelines package.
"""

from typing import Dict, List, Optional
from pathlib import Path
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from rdkit import Chem

from omtra.tasks.base_task import Task
from omtra.constants import ph_idx_to_elem
from omtra.eval.metrics.rmsd import ligand_rmsd
from omtra.eval.metrics.pharmacophore import pharmacophore_match_from_dict
from omtra.eval.metrics.posebusters import pb_validate
from omtra.eval.metrics.gnina import gnina_score_and_minimize
from omtra.eval.metrics.posecheck import posecheck_all


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

VALID_EVAL_METRICS = {"posebusters", "gnina", "posecheck", "rmsd", "pharmacophore"}

CLI_TO_INTERNAL = {
    "posebusters": "pb_valid",
    "gnina": "gnina",
    "posecheck": "posecheck",
    "rmsd": "rmsd",
    "pharmacophore": "pharm_match",
}


# ---------------------------------------------------------------------------
# Task-aware helpers
# ---------------------------------------------------------------------------

def determine_pb_mode(task: Task) -> str:
    """Return PoseBusters config mode for *task*.

    ``"redock"`` when ligand identity is fixed (docking/conformer tasks),
    ``"dock"`` for de novo design.
    """
    if (
        "ligand_identity_condensed" in task.groups_generated
        or "ligand_identity" in task.groups_generated
    ):
        return "dock"
    return "redock"


def determine_applicable_metrics(
    task: Task,
    requested: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Build a ``metrics_to_run`` dict based on task properties and user request.

    Parameters
    ----------
    task : Task
        The sampling task.
    requested : list of str or None
        CLI metric names (from ``VALID_EVAL_METRICS``).
        ``None`` or empty list means "all applicable".

    Returns
    -------
    dict
        Keys are the internal metric names used by ``compute_metrics``:
        ``pb_valid``, ``gnina``, ``posecheck``, ``rmsd``, ``pharm_match``,
        ``ground_truth``, ``interaction_recovery``.
    """
    has_protein = "protein_identity" in task.groups_present
    lig_identity_generated = (
        "ligand_identity_condensed" in task.groups_generated
        or "ligand_identity" in task.groups_generated
    )
    has_pharmacophore = "pharmacophore" in task.groups_present

    # Determine which metrics are applicable for this task
    applicable = {
        "pb_valid": has_protein,
        "gnina": has_protein,
        "posecheck": has_protein,
        "rmsd": has_protein and not lig_identity_generated,
        "pharm_match": has_pharmacophore,
        "ground_truth": has_protein,
        "interaction_recovery": False,  # opt-in only
    }

    # If specific metrics requested, filter to only those
    if requested:
        internal_names = {CLI_TO_INTERNAL[m] for m in requested}
        for key in list(applicable):
            if key in ("ground_truth", "interaction_recovery"):
                continue
            if key not in internal_names:
                applicable[key] = False

    return applicable


# ---------------------------------------------------------------------------
# Timeout / sanitization utilities
# ---------------------------------------------------------------------------

def run_with_timeout(func, *args, timeout, **kwargs):
    """Run *func* in a subprocess with a wall-clock *timeout* (seconds).

    Returns ``None`` on timeout or exception.
    """
    def target(q, *a, **k):
        try:
            res = func(*a, **k)
            q.put(res)
        except Exception as e:
            q.put(e)

    q = mp.Queue()
    p = mp.Process(target=target, args=(q, *args), kwargs=kwargs)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"\n[TIMEOUT] {func.__name__} killed after {timeout}s \n", flush=True)
        return None

    result = q.get() if not q.empty() else None
    if isinstance(result, Exception):
        print(f"\n[ERROR] {func.__name__} failed: {result} \n", flush=True)
        return None
    return result


def repair_and_sanitize(lig, i):
    """Sanitize *lig* and reject radicals. Returns ``None`` on failure."""
    try:
        Chem.SanitizeMol(lig)
    except Exception as e:
        print(f"An error occurred during sanitization of ligand {i}: {e}")
        return None
    if any(atom.GetNumRadicalElectrons() > 0 for atom in lig.GetAtoms()):
        print(f"Ligand {i} has a radical")
        return None
    return lig


# ---------------------------------------------------------------------------
# Main compute loop
# ---------------------------------------------------------------------------

def compute_metrics(
    system_pairs: dict,
    pb_mode: str,
    metrics_to_run: Dict[str, bool],
    timeout: int,
    disable_strain: bool = False,
) -> pd.DataFrame:
    """Compute evaluation metrics for all system pairs.

    Parameters
    ----------
    system_pairs : dict
        Nested dict ``{sys_id: {pair_id: data_dict}}``, as produced by
        ``system_pairs_from_path`` or ``write_system_pairs``.
    pb_mode : str
        PoseBusters config mode (``"dock"`` or ``"redock"``).
    metrics_to_run : dict
        Boolean flags keyed by metric name.
    timeout : int
        Per-metric timeout in seconds.
    disable_strain : bool
        Skip strain energy in PoseCheck.

    Returns
    -------
    pd.DataFrame
        Multi-indexed by (sys_id, protein_id, gen_ligand_id).
    """
    # Build scaffold dataframe
    rows = []
    for sys_id, pairs in system_pairs.items():
        for _, data in pairs.items():
            for lig_id in data["gen_ligs_ids"]:
                rows.append(
                    {
                        "sys_id": sys_id,
                        "protein_id": data.get("gen_prot_id", "none"),
                        "gen_ligand_id": lig_id,
                    }
                )

    metrics = pd.DataFrame(rows)
    metrics.set_index(["sys_id", "protein_id", "gen_ligand_id"], inplace=True)

    for sys_id, pairs in system_pairs.items():
        for pair_id, data in pairs.items():

            print(f"{'–'*32}", flush=True)
            print(f"{sys_id}, {data['gen_ligs_ids']}, {data.get('gen_prot_id', 'none')}", flush=True)
            print(f"{'–'*32}", flush=True)

            # Sanitize generated ligands
            valid_gen_ligs = []
            valid_gen_lig_ids = []

            for i, lig in enumerate(data["gen_ligs"]):
                lig = repair_and_sanitize(lig, i)
                if lig is not None:
                    valid_gen_ligs.append(lig)
                    valid_gen_lig_ids.append(data["gen_ligs_ids"][i])

            # Sanitize ground truth ligand
            true_lig = data["true_lig"]
            try:
                Chem.SanitizeMol(true_lig)
            except Exception as e:
                metrics_to_run["ground_truth"] = False
                print(
                    f"An error encountered during sanitization of true ligand for system {sys_id}: {e}"
                )
                print("Automatically disabling ground truth metric computation. \n")

            all_indices = pd.MultiIndex.from_product(
                [[sys_id], [data["gen_prot_id"]], data["gen_ligs_ids"]],
                names=["sys_id", "protein_id", "gen_ligand_id"],
            )
            valid_lig_indices = pd.MultiIndex.from_product(
                [[sys_id], [data["gen_prot_id"]], valid_gen_lig_ids],
                names=["sys_id", "protein_id", "gen_ligand_id"],
            )

            metrics.loc[all_indices, "RDKit_valid"] = False
            metrics.loc[valid_lig_indices, "RDKit_valid"] = True

            # Resolve PB args
            pb_true_lig = true_lig if pb_mode == "redock" else None

            # ---- PoseBusters ----
            if metrics_to_run["pb_valid"]:
                pb_results = run_with_timeout(
                    pb_validate,
                    timeout=timeout,
                    gen_ligs=valid_gen_ligs,
                    mode=pb_mode,
                    true_lig=pb_true_lig,
                    prot_file=data["gen_prot_file"],
                )

                if pb_results is not None:
                    pb_results.index = valid_lig_indices
                    metrics.loc[valid_lig_indices, pb_results.columns] = pb_results
                else:
                    for i, gen_lig in enumerate(valid_gen_ligs):
                        pb_result_single = run_with_timeout(
                            pb_validate,
                            timeout=120,
                            gen_ligs=gen_lig,
                            mode=pb_mode,
                            true_lig=pb_true_lig,
                            prot_file=data["gen_prot_file"],
                        )
                        if pb_result_single is not None:
                            metrics.loc[
                                (sys_id, data["gen_prot_id"], valid_gen_lig_ids[i]),
                                pb_result_single.columns,
                            ] = pb_result_single.iloc[0].values
                        else:
                            print(
                                f"Could not resolve PoseBusters eval on single generated ligand {valid_gen_lig_ids[i]}\n"
                            )

                if metrics_to_run["ground_truth"]:
                    pb_true_results = run_with_timeout(
                        pb_validate,
                        timeout=120,
                        gen_ligs=true_lig,
                        mode="dock",
                        true_lig=None,
                        prot_file=data["true_prot_file"],
                    )
                    if pb_true_results is not None:
                        pb_true_results = pd.DataFrame(
                            [pb_true_results.iloc[0].values] * len(all_indices),
                            columns=pb_true_results.columns,
                            index=all_indices,
                        )
                        pb_true_results.columns = [
                            f"{col}_true" for col in pb_true_results.columns
                        ]
                        metrics.loc[all_indices, pb_true_results.columns] = pb_true_results

            # ---- PoseCheck ----
            if metrics_to_run["posecheck"]:
                posechk_results = run_with_timeout(
                    posecheck_all,
                    timeout=timeout,
                    ligs=valid_gen_ligs,
                    prot_file=data["gen_prot_file"],
                    true_lig=true_lig,
                    true_prot_file=data["true_prot_file"],
                    include_strain=not disable_strain,
                    include_interaction_recovery=metrics_to_run.get(
                        "interaction_recovery", False
                    ),
                )

                if posechk_results is not None:
                    posechk_results = pd.DataFrame(
                        posechk_results, index=valid_lig_indices
                    )
                    metrics.loc[valid_lig_indices, posechk_results.columns] = (
                        posechk_results
                    )
                else:
                    for i, gen_lig in enumerate(valid_gen_ligs):
                        posechk_result_single = run_with_timeout(
                            posecheck_all,
                            timeout=120,
                            ligs=[gen_lig],
                            prot_file=data["gen_prot_file"],
                            true_lig=true_lig,
                            true_prot_file=data["true_prot_file"],
                            include_strain=not disable_strain,
                            include_interaction_recovery=metrics_to_run.get(
                                "interaction_recovery", False
                            ),
                        )
                        if posechk_result_single is not None:
                            metrics.loc[
                                (sys_id, data["gen_prot_id"], valid_gen_lig_ids[i]),
                                list(posechk_result_single.keys()),
                            ] = pd.Series(
                                {k: v[0] for k, v in posechk_result_single.items()}
                            )
                        else:
                            print(
                                f"Could not resolve PoseCheck eval on single generated ligand {valid_gen_lig_ids[i]}\n"
                            )

                # ground truth ligand
                if metrics_to_run["ground_truth"]:
                    posechk_true_results = run_with_timeout(
                        posecheck_all,
                        timeout=120,
                        ligs=[true_lig],
                        prot_file=data["true_prot_file"],
                        include_strain=not disable_strain,
                    )
                    if posechk_true_results is not None:
                        flat_row = {k: v[0] for k, v in posechk_true_results.items()}
                        posechk_true_results = pd.DataFrame(
                            [flat_row] * len(all_indices), index=all_indices
                        )
                        posechk_true_results.columns = [
                            f"{col}_true" for col in posechk_true_results.keys()
                        ]
                        metrics.loc[all_indices, posechk_true_results.columns] = (
                            posechk_true_results
                        )

            # ---- GNINA ----
            if metrics_to_run["gnina"]:
                gnina_results = run_with_timeout(
                    gnina_score_and_minimize,
                    timeout=timeout,
                    lig_file=data["gen_ligs_file"],
                    prot_file=data["gen_prot_file"],
                )

                if gnina_results is not None:
                    gnina_results = pd.DataFrame(gnina_results)
                    gnina_results.index = pd.MultiIndex.from_product(
                        [[sys_id], [data["gen_prot_id"]], gnina_results.index],
                        names=["sys_id", "protein_id", "gen_ligand_id"],
                    )
                    metrics.loc[gnina_results.index, gnina_results.columns] = (
                        gnina_results
                    )

                # ground truth ligand
                if metrics_to_run["ground_truth"]:
                    gnina_true_results = run_with_timeout(
                        gnina_score_and_minimize,
                        timeout=timeout,
                        lig_file=data["true_lig_file"],
                        prot_file=data["true_prot_file"],
                    )
                    if gnina_true_results is not None:
                        flat_row = {
                            k: v[true_lig.GetProp("_Name")]
                            for k, v in gnina_true_results.items()
                        }
                        gnina_true_results = pd.DataFrame(
                            [flat_row] * len(all_indices), index=all_indices
                        )
                        gnina_true_results.columns = [
                            f"{col}_true" for col in gnina_true_results.columns
                        ]
                        metrics.loc[all_indices, gnina_true_results.columns] = (
                            gnina_true_results
                        )

            # ---- RMSD ----
            if metrics_to_run["rmsd"]:
                for i, gen_lig in enumerate(data["gen_ligs"]):
                    rmsd_result = run_with_timeout(
                        ligand_rmsd,
                        timeout=120,
                        gen_lig=gen_lig,
                        true_lig=true_lig,
                    )
                    if rmsd_result is not None:
                        metrics.loc[
                            (sys_id, data["gen_prot_id"], data["gen_ligs_ids"][i]),
                            "rmsd",
                        ] = rmsd_result

            # ---- Pharmacophore matching ----
            if metrics_to_run["pharm_match"]:
                pharm_results = run_with_timeout(
                    pharmacophore_match_from_dict,
                    timeout=timeout,
                    gen_ligs=valid_gen_ligs,
                    true_pharm=data["true_pharm"],
                )
                if pharm_results is not None:
                    pharm_results = pd.DataFrame(
                        pharm_results, index=valid_lig_indices
                    )
                    metrics.loc[valid_lig_indices, pharm_results.columns] = (
                        pharm_results
                    )

    return metrics


# ---------------------------------------------------------------------------
# Loading system pairs from disk
# ---------------------------------------------------------------------------

def system_pairs_from_path(
    samples_dir: Path,
    task: Task,
    n_samples: int,
    sample_start_idx: int,
    n_replicates: int,
) -> dict:
    """Read a sample directory into the ``system_pairs`` dict expected by
    ``compute_metrics``.

    Parameters
    ----------
    samples_dir : Path
        Root directory containing ``sys_*_gt/`` subdirectories.
    task : Task
        The sampling task (used to determine directory structure).
    n_samples : int
        Number of systems.
    sample_start_idx : int
        First system index.
    n_replicates : int
        Number of replicates per system.

    Returns
    -------
    dict
        ``{sys_name: {pair_id: data_dict}}``
    """
    system_pairs = {}

    if sample_start_idx is None:
        sample_start_idx = 0

    has_protein = "protein_identity" in task.groups_present

    for sys_idx in range(sample_start_idx, sample_start_idx + n_samples):

        sys_name = f"sys_{sys_idx}_gt"
        sys_dir = samples_dir / sys_name

        if not os.path.isdir(sys_dir):
            print(
                f"WARNING: Missing directory for system {sys_idx}. Skipping this system."
            )
            continue

        sys_pair = {}

        true_lig_file = sys_dir / "ligand.sdf"
        if not os.path.exists(true_lig_file):
            print(
                f"WARNING: Missing ground truth ligand file for system {sys_idx}. "
                "Depending on downstream metrics this may cause pipeline failures."
            )

        true_lig = Chem.SDMolSupplier(str(true_lig_file), sanitize=False, removeHs=False)[0]

        if has_protein:
            true_prot_file = sys_dir / "protein_0.pdb"
            true_prot_id = "protein_0"

        pharm_missing = False
        if "pharmacophore" in task.groups_present:
            pharm = {}
            true_pharm_file = sys_dir / "pharmacophore.xyz"

            if not os.path.exists(true_pharm_file):
                print(
                    f"WARNING: Missing pharmacophore file for system {sys_idx}. "
                    "Depending on downstream metrics this may cause pipeline failures."
                )
                pharm_missing = True
            else:
                pharm_data = np.loadtxt(true_pharm_file, skiprows=1, dtype=str)
                if pharm_data.ndim == 1:
                    pharm_data = pharm_data.reshape(1, -1)
                pharm["types_idx"] = [
                    ph_idx_to_elem.index(p) for p in pharm_data[:, 0].tolist()
                ]
                pharm["coords"] = pharm_data[:, 1:].astype(float)

        if "protein_structure" in task.groups_generated:
            # Flexible protein tasks: pair each replicate separately
            for rep_idx in range(n_replicates):
                pair = {}
                gen_lig_file = sys_dir / f"gen_ligands_{rep_idx}.sdf"

                if os.path.exists(gen_lig_file):
                    gen_ligs = [
                        mol
                        for mol in Chem.SDMolSupplier(
                            str(gen_lig_file), sanitize=False, removeHs=False
                        )
                        if mol is not None
                    ]
                    pair["gen_ligs"] = gen_ligs
                    pair["gen_ligs_file"] = gen_lig_file
                    pair["gen_ligs_ids"] = [mol.GetProp("_Name") for mol in gen_ligs]
                    pair["true_lig"] = true_lig
                    pair["true_lig_file"] = true_lig_file
                    if has_protein:
                        pair["gen_prot_file"] = sys_dir / f"gen_prot_{rep_idx}.pdb"
                        pair["gen_prot_id"] = f"gen_prot_{rep_idx}"
                        pair["true_prot_file"] = true_prot_file
                        pair["true_prot_id"] = true_prot_id

                        if not os.path.exists(true_prot_file):
                            print(
                                f"WARNING: Missing true protein file for system {sys_idx}. "
                                "Depending on downstream metrics this may cause pipeline failures."
                            )
                    else:
                        pair["gen_prot_id"] = "none"
                        pair["true_prot_id"] = "none"

                    if "pharmacophore" in task.groups_present and not pharm_missing:
                        pair["true_pharm"] = pharm

                    sys_pair[f"pair_{rep_idx}"] = pair
                else:
                    print(
                        f"WARNING: Missing file for generated ligand {rep_idx} for system {sys_idx}."
                    )

        else:
            # Rigid protein tasks: all replicates share one protein
            pair = {}
            gen_lig_file = sys_dir / "gen_ligands.sdf"

            if not os.path.exists(gen_lig_file):
                print(
                    f"WARNING: Missing generated ligands file for system {sys_idx}. Skipping this system."
                )
                continue

            gen_ligs = [
                mol
                for mol in Chem.SDMolSupplier(
                    str(gen_lig_file), sanitize=False, removeHs=False
                )
                if mol is not None
            ]
            pair["gen_ligs"] = gen_ligs
            pair["gen_ligs_file"] = gen_lig_file
            pair["gen_ligs_ids"] = [mol.GetProp("_Name") for mol in gen_ligs]
            pair["true_lig"] = true_lig
            pair["true_lig_file"] = true_lig_file

            if has_protein:
                if not os.path.exists(true_prot_file):
                    print(
                        f"WARNING: Missing true protein file for system {sys_idx}. Skipping this system."
                    )
                    continue

                pair["gen_prot_file"] = true_prot_file
                pair["gen_prot_id"] = true_prot_id
                pair["true_prot_file"] = true_prot_file
                pair["true_prot_id"] = true_prot_id
            else:
                pair["gen_prot_id"] = "none"
                pair["true_prot_id"] = "none"

            if "pharmacophore" in task.groups_present and not pharm_missing:
                pair["true_pharm"] = pharm

            sys_pair["pair_0"] = pair

        system_pairs[sys_name] = sys_pair

    return system_pairs

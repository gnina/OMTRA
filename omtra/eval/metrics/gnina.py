import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional
import torch

from rdkit import Chem

from omtra.utils import omtra_root


def _resolve_gnina_binary(gnina_binary: Optional[str] = None) -> Path:
    """Resolve the path to the gnina binary."""
    if gnina_binary is not None:
        path = Path(gnina_binary)
    else:
        path = Path(omtra_root()) / "gnina.1.3.2"

    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find GNINA binary at {path}. "
            "Download from https://github.com/gnina/gnina/releases/tag/v1.3.2"
        )
    return path


def _run_gnina(
    lig_file: str,
    prot_file: str,
    gnina_binary: Path,
    minimization: bool,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Run gnina scoring or minimization and parse results from the output SDF."""
    with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
        output_sdf = Path(tmp.name)

    try:
        if not minimization:
            scores = {
                "minimizedAffinity": {},
                "CNNscore": {},
                "CNNaffinity": {},
                "CNNaffinity_variance": {},
            }
            cmd = [
                str(gnina_binary),
                "-r", str(prot_file),
                "-l", str(lig_file),
                "--score_only",
                "-o", str(output_sdf),
                "--seed", "42",
            ]
        else:
            scores = {"minimizedAffinity": {}}
            cmd = [
                str(gnina_binary),
                "-r", str(prot_file),
                "-l", str(lig_file),
                "-o", str(output_sdf),
                "--minimize",
                "--seed", "42",
            ]
        
        torch_lib_path = str(Path(torch.__file__).parent / "lib")
        cudnn_lib_path = str(
            Path(torch.__file__).parent.parent
            / "nvidia"
            / "cudnn"
            / "lib"
        )
        env = os.environ.copy()
        existing_ld_path = env.get("LD_LIBRARY_PATH", "")
        paths = [torch_lib_path, cudnn_lib_path]
        if existing_ld_path:
            paths.append(existing_ld_path)

        env["LD_LIBRARY_PATH"] = ":".join(paths)

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            print("Error running GNINA:", flush=True)
            print(result.stderr)
            return None

        supplier = Chem.SDMolSupplier(str(output_sdf), sanitize=False, removeHs=False)
        for lig in supplier:
            if lig is None:
                continue
            lig_id = lig.GetProp("_Name")
            for name, vals in scores.items():
                vals[lig_id] = float(lig.GetProp(name))

    finally:
        output_sdf.unlink(missing_ok=True)

    return scores


def gnina_score(
    lig_file: str,
    prot_file: str,
    gnina_binary: Optional[str] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    """Score ligand poses with gnina (no minimization).

    Args:
        lig_file: Path to ligand SDF file.
        prot_file: Path to protein PDB file.
        gnina_binary: Path to gnina binary. Auto-resolved from omtra_root() if None.

    Returns:
        Dict mapping score names to {lig_name: value} dicts, or None on failure.
        Score names: minimizedAffinity, CNNscore, CNNaffinity, CNNaffinity_variance.
    """
    binary = _resolve_gnina_binary(gnina_binary)
    return _run_gnina(lig_file, prot_file, binary, minimization=False)


def gnina_minimize(
    lig_file: str,
    prot_file: str,
    gnina_binary: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Minimize ligand poses with gnina and return Vina affinities.

    Args:
        lig_file: Path to ligand SDF file.
        prot_file: Path to protein PDB file.
        gnina_binary: Path to gnina binary. Auto-resolved from omtra_root() if None.

    Returns:
        Dict mapping lig_name to minimized Vina affinity, or None on failure.
    """
    binary = _resolve_gnina_binary(gnina_binary)
    result = _run_gnina(lig_file, prot_file, binary, minimization=True)
    if result is None:
        return None
    return result["minimizedAffinity"]


def gnina_score_and_minimize(
    lig_file: str,
    prot_file: str,
    gnina_binary: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Run both gnina scoring and minimization, returning combined results.

    This is the equivalent of the original gnina() function in docking_eval.py.

    Returns:
        Dict with keys: minimizedAffinity, CNNscore, CNNaffinity,
        CNNaffinity_variance, vina_min. Empty dict on total failure.
    """
    binary = _resolve_gnina_binary(gnina_binary)
    results = {}

    score_results = _run_gnina(lig_file, prot_file, binary, minimization=False)
    if score_results is not None:
        results.update(score_results)

    min_results = _run_gnina(lig_file, prot_file, binary, minimization=True)
    if min_results is not None:
        results["vina_min"] = min_results["minimizedAffinity"]

    return results

from typing import List, Literal, Optional, Union

import pandas as pd
from rdkit import Chem
import posebusters as pb
from omtra.data.physical_checks import required_checks, evaluate_checks


def pb_validate(
    gen_ligs: Union[List[Chem.Mol], Chem.Mol],
    mode: Literal["dock", "redock", "mol", "regen"] = "dock",
    true_lig: Optional[Chem.Mol] = None,
    prot_file: Optional[str] = None,
    max_workers: int = 0,
) -> Optional[pd.DataFrame]:
    """Run PoseBusters validation on generated ligands.

    Args:
        gen_ligs: Single molecule or list of molecules to validate.
        mode: PoseBusters config mode. Use "redock" when ligand identity is
            fixed (e.g. docking/conformer tasks), "dock" for de novo design.
        true_lig: Ground truth ligand (required for "redock" mode).
        prot_file: Path to protein PDB file.
        max_workers: Number of parallel workers for PoseBusters.

    Returns:
        DataFrame with pb_* columns and a pb_valid boolean column,
        or None if gen_ligs is empty.
    """
    if isinstance(gen_ligs, list) and not gen_ligs:
        return None

    identity = mode in ('redock','regen')
    effective_mode = mode if prot_file is not None else ('regen' if identity else 'mol')
    buster = pb.PoseBusters(config=effective_mode, max_workers=max_workers)
    df_pb = buster.bust(gen_ligs, true_lig, prot_file)
    df_pb.columns = [f"pb_{col}" for col in df_pb.columns]
    required = required_checks(identity=identity, protein=prot_file is not None)
    physical, missing = evaluate_checks(df_pb, required)
    df_pb['pb_physical_valid'] = physical
    df_pb['metric_pb_missing_required'] = missing
    # Historical pb_valid includes redocking RMSD, while physical_valid does not.
    full = required + (['pb_rmsd_≤_2å'] if identity else [])
    df_pb['pb_valid'], _ = evaluate_checks(df_pb, full)

    return df_pb

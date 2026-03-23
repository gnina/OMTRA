from typing import List, Literal, Optional, Union

import pandas as pd
from rdkit import Chem
import posebusters as pb


def pb_validate(
    gen_ligs: Union[List[Chem.Mol], Chem.Mol],
    mode: Literal["dock", "redock", "mol"] = "dock",
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

    buster = pb.PoseBusters(config=mode, max_workers=max_workers)
    df_pb = buster.bust(gen_ligs, true_lig, prot_file)
    df_pb.columns = [f"pb_{col}" for col in df_pb.columns]
    df_pb["pb_valid"] = (
        df_pb[df_pb["pb_sanitization"] == True].values.astype(bool).all(axis=1)
    )

    return df_pb

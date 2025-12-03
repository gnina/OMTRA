"""
This module mirrors the logic from ``omtra.data.pharmacophores`` but lives inside 
the webapp so the API can extract pharmacophore centers without needing the full OMTRA 
python package mounted inside the container
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from rdkit import Chem


ph_idx_to_type: List[str] = [
    "Aromatic",
    "HydrogenDonor",
    "HydrogenAcceptor",
    "PositiveIon",
    "NegativeIon",
    "Hydrophobic",
    "Halogen",
]

ph_idx_to_elem: List[str] = ["P", "S", "F", "N", "O", "C", "Cl"]
ph_type_to_idx: Dict[str, int] = {ptype: idx for idx, ptype in enumerate(ph_idx_to_type)}


smarts_patterns: Dict[str, List[str]] = {
    "Aromatic": ["a1aaaaa1", "a1aaaa1"],
    "PositiveIon": ["[+,+2,+3,+4]", "[$(C(N)(N)=N)]", "[$(n1cc[nH]c1)]"],
    "NegativeIon": ["[-,-2,-3,-4]", "C(=O)[O-,OH,OX1]"],
    "HydrogenAcceptor": [
        "[#7&!$([nX3])&!$([NX3]-*=[!#6])&!$([NX3]-[a])&!$([NX4])&!$(N=C([C,N])N)]",
        "[$([O])&!$([OX2](C)C=O)&!$(*(~a)~a)]",
    ],
    "HydrogenDonor": [
        "[#7!H0&!$(N-[SX4](=O)(=O)[CX4](F)(F)F)]",
        "[#8!H0&!$([OH][C,S,P]=O)]",
        "[#16!H0]",
    ],
    "Hydrophobic": [
        "a1aaaaa1",
        "a1aaaa1",
        "[$([CH3X4,CH2X3,CH1X2])&!$(**[CH3X4,CH2X3,CH1X2])]",
        "[$(*([CH3X4,CH2X3,CH1X2])[CH3X4,CH2X3,CH1X2])&!$(*([CH3X4,CH2X3,CH1X2])([CH3X4,CH2X3,CH1X2])[CH3X4,CH2X3,CH1X2])]([CH3X4,CH2X3,CH1X2])[CH3X4,CH2X3,CH1X2]",
        "[CH2X4,CH1X3,CH0X2]~[CH3X4,CH2X3,CH1X2]",
        "[$([CH2X4,CH1X3,CH0X2]~[$([!#1]);!$([CH2X4,CH1X3,CH0X2])])]~[CH2X4,CH1X3,CH0X2]~[CH2X4,CH1X3,CH0X2]",
        "[$([S]~[#6])&!$(S~[!#6])]",
    ],
    "Halogen": [
        "[F;$(F-[#6]);!$(FC[F,Cl,Br,I])]",
        "[Cl;$(Cl-[#6]);!$(FC[F,Cl,Br,I])]",
        "[Br;$(Br-[#6]);!$(FC[F,Cl,Br,I])]",
        "[I;$(I-[#6]);!$(FC[F,Cl,Br,I])]",
    ],
}


def get_pharmacophores(mol: Chem.Mol, rec: Chem.Mol | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool]]:
    if rec is not None:
        raise NotImplementedError("Receptor-aware pharmacophore extraction is not supported")

    pharmacophore_store: Dict[str, List[np.ndarray]] = {feature: [] for feature in smarts_patterns}

    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)

    for feature, patterns in smarts_patterns.items():
        for pattern in patterns:
            smarts_mol = Chem.MolFromSmarts(pattern)
            matches = mol.GetSubstructMatches(smarts_mol, uniquify=True)
            if not matches:
                continue
            conformer = mol.GetConformer()
            positions = conformer.GetPositions()
            for match in matches:
                atoms = positions[list(match)]
                feature_location = np.mean(atoms, axis=0)
                pharmacophore_store[feature].append(feature_location)

    mol = Chem.RemoveHs(mol)

    positions: List[np.ndarray] = []
    type_indices: List[int] = []

    for feature, data in pharmacophore_store.items():
        pos_list = data
        if not pos_list:
            continue
        positions.extend(pos_list)
        type_idx = ph_type_to_idx[feature]
        type_indices.extend([type_idx] * len(pos_list))

    if not positions:
        return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 4, 3)), []

    return (
        np.array(positions, dtype=np.float32),
        np.array(type_indices, dtype=np.int64),
        np.zeros((len(positions), 4, 3), dtype=np.float32),
        [],
    )



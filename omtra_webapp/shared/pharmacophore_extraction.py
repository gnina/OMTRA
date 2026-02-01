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
ph_elem_to_idx: Dict[str, int] = {elem: idx for idx, elem in enumerate(ph_idx_to_elem)}


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


def load_pharmacophore_xyz(xyz_content: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse XYZ content for pharmacophores.
    Expected format:
    N_ATOMS
    Comment
    Type X Y Z
    """
    lines = xyz_content.strip().splitlines()
    if not lines:
        return np.zeros((0, 3)), np.zeros(0)
    
    # Basic XYZ check: skip header lines if they look like N_ATOMS / Comment
    start_idx = 0
    if lines[0].strip().isdigit():
        start_idx = 2
    
    positions = []
    type_indices = []
    
    unk_code = ph_type_to_idx["Hydrophobic"] # Fallback
    
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        
        kind = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
            
        # Map kind (can be type name OR element symbol)
        if kind in ph_type_to_idx:
            t_idx = ph_type_to_idx[kind]
        elif kind in ph_elem_to_idx:
            t_idx = ph_elem_to_idx[kind]
        else:
            t_idx = unk_code
            
        positions.append([x, y, z])
        type_indices.append(t_idx)
    
    if not positions:
        return np.zeros((0, 3)), np.zeros(0)
        
    return np.array(positions, dtype=np.float32), np.array(type_indices, dtype=np.int64)


def load_pharmacophore_json(json_content: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse JSON content for pharmacophores.
    Mirrors CLI logic from omtra/utils/file_to_graph.py::load_pharmacophore_json
    """
    import json
    try:
        data = json.loads(json_content)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"load_pharmacophore_json: Failed to parse JSON: {e}")
        return np.zeros((0, 3)), np.zeros(0)
        
    # Handle both {"points": [...]} and direct array
    points = data.get('points', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    
    # Filter for enabled points
    enabled_points = [p for p in points if isinstance(p, dict) and p.get('enabled', True)]
    
    if not enabled_points:
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"load_pharmacophore_json: No enabled points found in {len(points)} total points")
        return np.zeros((0, 3)), np.zeros(0)
    
    coords = []
    kinds = []
    
    for p in enabled_points:
        try:
            # Extract coordinates - CLI expects x, y, z keys
            coords.append([p['x'], p['y'], p['z']])
            # Extract name - CLI expects 'name' key
            kinds.append(p['name'])
        except KeyError as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"load_pharmacophore_json: Skipping point missing key: {e}")
            continue
    
    if not coords:
        return np.zeros((0, 3)), np.zeros(0)
    
    coords = np.asarray(coords, dtype=np.float32)
    kinds = np.asarray(kinds)
    
    # CLI logic: unique kinds, then map to indices
    unique_kinds, inverse = np.unique(kinds, return_inverse=True)
    unk_code = ph_type_to_idx.get('UNK', 0)  # Default to 0 if UNK not in mapping
    
    unique_codes = np.array([
        ph_type_to_idx[kind] if kind in ph_type_to_idx else unk_code
        for kind in unique_kinds
    ], dtype=np.int64)
    kind_idx = unique_codes[inverse]
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"load_pharmacophore_json: Successfully loaded {len(coords)} pharmacophores")
    logger.debug(f"  Types: {dict(zip(*np.unique(kinds, return_counts=True)))}")
    
    return coords, kind_idx




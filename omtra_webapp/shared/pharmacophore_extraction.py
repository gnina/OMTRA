"""
This module mirrors the logic from ``omtra.data.pharmacophores`` but lives inside 
the webapp so the API can extract pharmacophore centers without needing the full OMTRA 
python package mounted inside the container
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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


def _pharmacophore_type_index(kind: str, *, line_desc: str) -> int:
    if kind in ph_type_to_idx:
        return ph_type_to_idx[kind]
    if kind in ph_elem_to_idx:
        return ph_elem_to_idx[kind]
    allowed = ", ".join(ph_idx_to_type + ph_idx_to_elem)
    raise ValueError(f"Unknown pharmacophore type '{kind}' at {line_desc}. Allowed values: {allowed}")


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
        raise ValueError("Pharmacophore XYZ is empty")
    
    # Basic XYZ check: skip header lines if they look like N_ATOMS / Comment
    start_idx = 0
    if lines[0].strip().isdigit():
        start_idx = 2
    
    positions = []
    type_indices = []

    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid pharmacophore XYZ line {i + 1}: expected 'TYPE X Y Z'")
        
        kind = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            raise ValueError(f"Invalid coordinates on pharmacophore XYZ line {i + 1}: expected numeric X Y Z")
        if not np.isfinite([x, y, z]).all():
            raise ValueError(f"Invalid coordinates on pharmacophore XYZ line {i + 1}: coordinates must be finite")

        t_idx = _pharmacophore_type_index(kind, line_desc=f"XYZ line {i + 1}")
            
        positions.append([x, y, z])
        type_indices.append(t_idx)
    
    if not positions:
        raise ValueError("Pharmacophore XYZ contains no feature rows")
        
    return np.array(positions, dtype=np.float32), np.array(type_indices, dtype=np.int64)


def load_pharmacophore_json(json_content: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse JSON content for pharmacophores.
    Mirrors CLI logic from omtra/utils/file_to_graph.py::load_pharmacophore_json
    """
    import json
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid pharmacophore JSON: {e}") from e
        
    # Handle both {"points": [...]} and direct array
    if isinstance(data, dict):
        points = data.get('points')
    elif isinstance(data, list):
        points = data
    else:
        raise ValueError("Pharmacophore JSON must be an array or an object with a 'points' array")

    if not isinstance(points, list):
        raise ValueError("Pharmacophore JSON must contain a 'points' array")
    
    # Filter for enabled points
    enabled_points = []
    for i, point in enumerate(points):
        if not isinstance(point, dict):
            raise ValueError(f"Invalid pharmacophore JSON point {i}: expected an object")
        if point.get('enabled', True):
            enabled_points.append((i, point))
    
    if not enabled_points:
        raise ValueError("Pharmacophore JSON contains no enabled points")
    
    coords = []
    type_indices = []
    
    for original_idx, point in enabled_points:
        missing = [key for key in ("x", "y", "z", "name") if key not in point]
        if missing:
            raise ValueError(f"Invalid pharmacophore JSON point {original_idx}: missing required key(s): {', '.join(missing)}")

        try:
            x, y, z = float(point["x"]), float(point["y"]), float(point["z"])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid pharmacophore JSON point {original_idx}: coordinates must be numeric") from e
        if not np.isfinite([x, y, z]).all():
            raise ValueError(f"Invalid pharmacophore JSON point {original_idx}: coordinates must be finite")

        kind = str(point["name"])
        coords.append([x, y, z])
        type_indices.append(_pharmacophore_type_index(kind, line_desc=f"JSON point {original_idx}"))
    
    coords = np.asarray(coords, dtype=np.float32)
    kind_idx = np.asarray(type_indices, dtype=np.int64)
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"load_pharmacophore_json: Successfully loaded {len(coords)} pharmacophores")
    logger.debug(f"  Type indices: {dict(zip(*np.unique(kind_idx, return_counts=True)))}")
    
    return coords, kind_idx




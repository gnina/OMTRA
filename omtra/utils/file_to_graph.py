import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rdkit import Chem

from omtra.tasks.tasks import Task
from omtra.constants import (
    lig_atom_type_map,
    npnde_atom_type_map,
    charge_map,
    ph_idx_to_type,
    ph_idx_to_elem,
    residue_map,
    protein_element_map,
    protein_atom_map,
)
from omtra.data.graph import build_complex_graph
from omtra.data.xace_ligand import MoleculeTensorizer, add_k_hop_edges
from omtra.data.plinder import StructureData, LigandData, BackboneData
from omtra.data.condensed_atom_typing import CondensedAtomTyper
from omtra.utils.embedding import residue_sinusoidal_encoding


# loaders
def load_protein_biotite(
    protein_file: Path,
    return_npnde_mols: bool = False,
):
    try:
        from biotite.structure.io import pdb
        from biotite.structure.io.pdbx import CIFFile, get_structure
    except ImportError as e:
        raise ImportError("biotite is required: pip install biotite") from e

    suffix = protein_file.suffix.lower()
    if suffix == ".pdb":
        st = pdb.PDBFile.read(str(protein_file)).get_structure(model=1)
    elif suffix == ".cif":
        cif_file = CIFFile.read(str(protein_file))
        st = get_structure(cif_file, model=1, include_bonds=False)
    else:
        raise ValueError(f"Unsupported protein format: {suffix}")

    import biotite.structure as struc
    # removing waters and hydrogens
    st = st[st.res_name != "HOH"]
    st = st[st.element != "H"]
    st = st[st.element != "D"]

    if st.array_length() == 0:
        raise ValueError("Protein structure has no atoms")

    npnde_mols: Dict[str, Chem.Mol] = {}
    if return_npnde_mols:
        st.bonds = struc.connect_via_residue_names(st)
        st, npnde_mols = _split_protein_and_npndes(st)
        if st.array_length() == 0:
            raise ValueError("Protein structure has no amino-acid atoms after npnde extraction")

    coords = st.coord
    backbone_mask = struc.filter_peptide_backbone(st)
    backbone = BackboneData(
        coords=coords[backbone_mask],
        res_ids=st.res_id[backbone_mask],
        res_names=st.res_name[backbone_mask],
        chain_ids=st.chain_id[backbone_mask],
    )

    structure_data = StructureData(
        coords=coords,
        atom_names=st.atom_name,
        elements=st.element,
        res_ids=st.res_id,
        res_names=st.res_name,
        chain_ids=st.chain_id,
        backbone_mask=backbone_mask,
        backbone=backbone,
        cif=None,
    )

    if return_npnde_mols:
        return structure_data, npnde_mols
    return structure_data


def _is_protein(mol_array) -> bool:
    """Return True if every atom in ``mol_array`` is an amino-acid atom."""
    import biotite.structure as struc

    mask = struc.filter_amino_acids(mol_array)
    return int(sum(mask)) == len(mol_array)


def _split_protein_and_npndes(atom_array) -> Tuple["object", Dict[str, Chem.Mol]]:
    """Split a biotite AtomArray into amino-acid atoms and non-polymer molecules.
    """
    from biotite.structure import molecule_iter

    if atom_array.array_length() == 0:
        return atom_array, {}

    try:
        from biotite.interface.rdkit import to_mol
    except ImportError:
        aa_mask = _aa_mask(atom_array)
        return atom_array[aa_mask], {}

    protein_mask = np.zeros(len(atom_array), dtype=bool)
    npnde_mols: Dict[str, Chem.Mol] = {}
    npnde_count = 0
    current_idx = 0

    for molecule in molecule_iter(atom_array):
        molecule_len = len(molecule)
        if _is_protein(molecule):
            protein_mask[current_idx:current_idx + molecule_len] = True
        else:
            npnde_count += 1
            npnde_mol = to_mol(molecule)
            if npnde_mol is not None:
                npnde_mols[f"npnde_{npnde_count}"] = npnde_mol
        current_idx += molecule_len

    cleaned = atom_array[protein_mask]
    return cleaned, npnde_mols


def _aa_mask(atom_array):
    import biotite.structure as struc

    return struc.filter_amino_acids(atom_array)


def _build_edge_mask(fixed_atom_mask: np.ndarray, xace_edge_idxs: torch.Tensor) -> np.ndarray:
    """Derive edge mask from atom mask: an edge is fixed iff both endpoints are fixed."""
    ei = xace_edge_idxs.detach().cpu().numpy()
    if ei.ndim != 2:
        raise ValueError("Unexpected edge_idxs shape for fixed-edge mask")
    if ei.shape[0] == 2:
        src, dst = ei[0], ei[1]
    else:
        src, dst = ei[:, 0], ei[:, 1]
    return (fixed_atom_mask[src] & fixed_atom_mask[dst]).astype(np.int64)


def _fixed_masks_from_brics(
    mol: Chem.Mol,
    xace_edge_idxs: torch.Tensor,
    fixed_brics_fragment_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build atom/edge fixed masks from BRICS fragment ids.

    WARNING: This calls fragment_molecule(mol) which uses BRICS bond detection.
    If mol has been kekulized (clearAromaticFlags=True), the fragmentation will
    differ from the original. Prefer _fixed_masks_from_brics_precomputed when
    fragments were computed before kekulization.
    """
    from omtra.data.extra_ligand_features import fragment_molecule

    frags = fragment_molecule(mol).squeeze(-1)
    return _fixed_masks_from_brics_precomputed(frags, xace_edge_idxs, fixed_brics_fragment_ids)


def _fixed_masks_from_brics_precomputed(
    frags: np.ndarray,
    xace_edge_idxs: torch.Tensor,
    fixed_brics_fragment_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build atom/edge fixed masks from pre-computed BRICS fragment assignments."""
    if fixed_brics_fragment_ids:
        frag_ids_arr = np.asarray(list(fixed_brics_fragment_ids), dtype=np.int64)
        present = np.unique(frags[frags >= 0])
        missing = np.setdiff1d(frag_ids_arr, present)
        if missing.size > 0:
            raise ValueError(
                f"BRICS fragment id(s) {missing.tolist()} not present in ligand "
                f"(available fragment indices: {sorted(present.tolist())})"
            )
        fixed_atom_mask = np.isin(frags, frag_ids_arr).astype(np.int64)
    else:
        fixed_atom_mask = np.zeros(len(frags), dtype=np.int64)

    fixed_edge_mask = _build_edge_mask(fixed_atom_mask, xace_edge_idxs)
    return fixed_atom_mask, fixed_edge_mask


def _fixed_masks_from_atom_indices(
    n_atoms: int,
    xace_edge_idxs: torch.Tensor,
    fixed_atom_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build atom/edge fixed masks from atom indices."""
    fixed_atom_mask = np.zeros(n_atoms, dtype=np.int64)
    if fixed_atom_indices:
        idx_arr = np.asarray(list(fixed_atom_indices), dtype=np.int64)
        if np.any(idx_arr < 0) or np.any(idx_arr >= n_atoms):
            bad = idx_arr[(idx_arr < 0) | (idx_arr >= n_atoms)]
            raise ValueError(
                f"fixed atom index(ices) {bad.tolist()} out of range for ligand with "
                f"{n_atoms} atoms (valid: 0..{n_atoms - 1})"
            )
        fixed_atom_mask[idx_arr] = 1
    fixed_edge_mask = _build_edge_mask(fixed_atom_mask, xace_edge_idxs)
    return fixed_atom_mask, fixed_edge_mask


def load_ligand_rdkit(
    ligand_file: Path,
    compute_condensed: bool = False,
    fixed_brics_fragments: Optional[Sequence[int]] = None,
    fixed_atom_indices: Optional[Sequence[int]] = None,
) -> LigandData:
    supplier = Chem.SDMolSupplier(str(ligand_file))
    mol = next(supplier)
    if mol is None:
        raise ValueError(f"Failed to read ligand from {ligand_file}")
    if mol.GetNumAtoms() == 0:
        raise ValueError("Ligand has zero atoms")
    if not mol.GetNumConformers():
        raise ValueError("Ligand has no 3D conformer")

    # Compute BRICS fragmentation before featurization bc
    # featurize_molecules changes BRICS bond detection
    pre_kekulize_brics_frags = None
    if fixed_brics_fragments is not None:
        from omtra.data.extra_ligand_features import fragment_molecule
        pre_kekulize_brics_frags = fragment_molecule(mol).squeeze(-1)

    tensorizer = MoleculeTensorizer(lig_atom_type_map, n_cpus=1)
    valid_mols, failed, failures, _ = tensorizer.featurize_molecules([mol])
    if failed:
        raise ValueError(f"Ligand featurization failed: {failures}")
    xace = valid_mols[0]
    xace.to_torch()

    atom_cond_a = None
    if compute_condensed:
        cond_typer = CondensedAtomTyper(fake_atoms=False)
        

        from omtra.data.extra_ligand_features import ligand_properties, fragment_molecule
        extra_feats = ligand_properties(mol)
        fragment_feats = fragment_molecule(mol)
        
        extra_feats = np.concatenate([extra_feats, fragment_feats], axis=1) # (n_atoms, 6)
        
        extra_feats = extra_feats[:, :-1]  # (n_atoms, 5)
        
        atom_cond_a = cond_typer.feats_to_cond_a(
            a=xace.a,
            c=xace.c, 
            extra_feats=extra_feats
        )

    fixed_atom_mask = None
    fixed_edge_mask = None
    if fixed_atom_indices is not None or fixed_brics_fragments is not None:
        n_atoms = xace.x.shape[0]
        combined_atom_mask = np.zeros(n_atoms, dtype=np.int64)
        if fixed_atom_indices is not None:
            atom_mask, _ = _fixed_masks_from_atom_indices(
                n_atoms, xace.edge_idxs, fixed_atom_indices
            )
            combined_atom_mask |= atom_mask
        if fixed_brics_fragments is not None:
            brics_atom_mask, _ = _fixed_masks_from_brics_precomputed(
                pre_kekulize_brics_frags, xace.edge_idxs, fixed_brics_fragments
            )
            combined_atom_mask |= brics_atom_mask
        fixed_atom_mask = combined_atom_mask
        fixed_edge_mask = _build_edge_mask(fixed_atom_mask, xace.edge_idxs)

    return LigandData(
        coords=xace.x,
        bond_types=xace.e,
        bond_indices=np.asarray(xace.edge_idxs.cpu().numpy()),
        is_covalent=False,
        ccd="LIG",
        sdf=str(ligand_file),
        atom_types=xace.a,
        atom_charges=xace.c,
        atom_impl_H=getattr(xace, 'impl_H', None),
        atom_aro=getattr(xace, 'aro', None),
        atom_hyb=getattr(xace, 'hyb', None),
        atom_ring=getattr(xace, 'ring', None),
        atom_chiral=getattr(xace, 'chiral', None),
        atom_cond_a=atom_cond_a,
        fragments=None,
        fixed_atom_mask=fixed_atom_mask,
        fixed_edge_mask=fixed_edge_mask,
    )


def load_pharmacophore_xyz(pharm_file: Path):
    with open(pharm_file, 'r') as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError("Pharmacophore file too short")
    data_lines = lines[2:]
    coords, kinds = [], []
    for ln in data_lines:
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 4:
            continue
        kinds.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not coords:
        raise ValueError("No pharmacophore points found")
    coords = np.asarray(coords, dtype=np.float32)
    kinds = np.asarray(kinds)
    
    # Convert type names to element symbols if needed
    # ph_idx_to_type maps to ph_idx_to_elem at same index
    type_to_elem = {ptype: elem for ptype, elem in zip(ph_idx_to_type, ph_idx_to_elem)}
    
    unique_kinds, inverse = np.unique(kinds, return_inverse=True)
    unk_code = ph_idx_to_type.index('UNK') if 'UNK' in ph_idx_to_type else 0
    unique_codes = np.array([
        ph_idx_to_elem.index(kind) if kind in ph_idx_to_elem else (
            ph_idx_to_elem.index(type_to_elem[kind]) if kind in type_to_elem else unk_code
        )
        for kind in unique_kinds
    ], dtype=np.int64)
    kind_idx = unique_codes[inverse]
    
    return coords, kind_idx

def load_pharmacophore_json(pharm_file: Path):
    import json
    with open(pharm_file, 'r') as f:
        data = json.load(f)
    
    points = data.get('points', [])
    enabled_points = [p for p in points if p.get('enabled', True)]
    
    if not enabled_points:
        raise ValueError("No enabled pharmacophore points found")
    
    coords = []
    kinds = []
    
    for p in enabled_points:
        coords.append([p['x'], p['y'], p['z']])
        kinds.append(p['name'])
    
    coords = np.asarray(coords, dtype=np.float32)
    kinds = np.asarray(kinds)
    
    unique_kinds, inverse = np.unique(kinds, return_inverse=True)
    unk_code = ph_idx_to_type.index('UNK') if 'UNK' in ph_idx_to_type else 0
    unique_codes = np.array([
        ph_idx_to_type.index(kind) if kind in ph_idx_to_type else unk_code
        for kind in unique_kinds
    ], dtype=np.int64)
    kind_idx = unique_codes[inverse]
    
    return coords, kind_idx

def extract_backbone_data(backbone_atoms) -> BackboneData:
    """Extract backbone data from backbone atoms (N, CA, C per residue)."""
    unique_compound_keys = sorted(
        set(zip(backbone_atoms.chain_id.tolist(), backbone_atoms.res_id.tolist())),
        key=lambda item: (str(item[0]), int(item[1])),
    )
    num_residues = len(unique_compound_keys)

    coords = np.zeros((num_residues, 3, 3))
    res_ids = np.zeros(num_residues, dtype=int)
    res_names_list = []
    chain_ids_list = []

    for i, (chain_id, res_id) in enumerate(unique_compound_keys):
        res_id = int(res_id)

        res_mask = (
            (backbone_atoms.chain_id == chain_id)
            & (backbone_atoms.res_id == res_id)
        )
        res_atoms = backbone_atoms[res_mask]

        res_ids[i] = res_id
        res_names_list.append(res_atoms.res_name[0])
        chain_ids_list.append(chain_id)

        for j, atom_name in enumerate(["N", "CA", "C"]):
            atom_mask = res_atoms.atom_name == atom_name
            if np.any(atom_mask):
                coords[i, j] = res_atoms.coord[atom_mask][0]
            else:
                # if atom is missing
                coords[i, j] = np.zeros(3)

    res_names = np.array(res_names_list)
    chain_ids = np.array(chain_ids_list)

    return BackboneData(
        coords=coords,
        res_ids=res_ids,
        res_names=res_names,
        chain_ids=chain_ids,
    )

def _atoms_to_residue_mask(atom_array, atom_mask):
    """include all atoms from residues with any atom selected"""
    close_res_ids = atom_array.res_id[atom_mask]
    close_chain_ids = atom_array.chain_id[atom_mask]
    unique_res_pairs = set(zip(close_res_ids, close_chain_ids))
    mask = np.zeros(len(atom_array), dtype=bool)
    for res_id, chain_id in unique_res_pairs:
        mask |= (
            (atom_array.res_id == res_id)
            & (atom_array.chain_id == chain_id)
        )
    return mask

def extract_pocket(
    receptor: StructureData,
    reference_coords: np.ndarray,
    pocket_cutoff: float = 8.0,
) -> Optional[StructureData]:
    try:
        import biotite.structure as struc
    except ImportError as e:
        raise ImportError("biotite is required: pip install biotite") from e
    
    atom_array = receptor.to_atom_array()
    
    atom_array = atom_array[atom_array.res_name != "HOH"]
    atom_array = atom_array[atom_array.element != "H"]
    atom_array = atom_array[atom_array.element != "D"]
    
    if len(atom_array) == 0:
        return None
    
    receptor_cell_list = struc.CellList(atom_array, cell_size=pocket_cutoff)
    
    close_atom_indices = []
    for ref_coord in reference_coords:
        indices = receptor_cell_list.get_atoms(ref_coord, radius=pocket_cutoff)
        close_atom_indices.extend(indices)
    
    if len(close_atom_indices) == 0:
        return None
    
    atom_mask = np.zeros(len(atom_array), dtype=bool)
    atom_mask[close_atom_indices] = True
    residue_mask = _atoms_to_residue_mask(atom_array, atom_mask)
    pocket_indices = np.where(residue_mask)[0]
    
    if len(pocket_indices) == 0:
        return None
    
    pocket_atoms = atom_array[pocket_indices]
    
    backbone_atoms = pocket_atoms[struc.filter_peptide_backbone(pocket_atoms)]
    if len(backbone_atoms) == 0:
        return None
    
    backbone_data = extract_backbone_data(backbone_atoms)
    if backbone_data is None:
        return None
    
    bb_mask = struc.filter_peptide_backbone(pocket_atoms)
    
    return StructureData(
        coords=pocket_atoms.coord,
        atom_names=pocket_atoms.atom_name,
        elements=pocket_atoms.element,
        res_ids=pocket_atoms.res_id,
        res_names=pocket_atoms.res_name,
        chain_ids=pocket_atoms.chain_id,
        backbone_mask=bb_mask,
        backbone=backbone_data,
        cif=None,
    )


def _create_pocket_from_indices(
    receptor: StructureData,
    selector,
    error_context: str,
) -> StructureData:

    import biotite.structure as struc
    
    atom_array = receptor.to_atom_array()
    mask = selector(atom_array)
    pocket_indices = np.where(mask)[0]
    
    if len(pocket_indices) == 0:
        raise ValueError(f"No receptor atoms found for {error_context}")
    
    pocket_atoms = atom_array[pocket_indices]
    backbone_atoms = pocket_atoms[struc.filter_peptide_backbone(pocket_atoms)]
    
    if len(backbone_atoms) == 0:
        raise ValueError(f"No backbone atoms found for {error_context}")
    
    backbone_data = extract_backbone_data(backbone_atoms)
    if backbone_data is None:
        raise ValueError(f"Failed to extract backbone data for {error_context}")
    
    bb_mask = struc.filter_peptide_backbone(pocket_atoms)
    
    return StructureData(
        coords=pocket_atoms.coord,
        atom_names=pocket_atoms.atom_name,
        elements=pocket_atoms.element,
        res_ids=pocket_atoms.res_id,
        res_names=pocket_atoms.res_name,
        chain_ids=pocket_atoms.chain_id,
        backbone_mask=bb_mask,
        backbone=backbone_data,
        cif=None,
    )

def featurize_npnde_mols(
    npnde_mols: Dict[str, Chem.Mol],
) -> Dict[str, LigandData]:
    if not npnde_mols:
        return {}

    keys = list(npnde_mols.keys())
    mols = list(npnde_mols.values())

    tensorizer = MoleculeTensorizer(atom_map=npnde_atom_type_map, n_cpus=1)
    xace_mols, failed_idxs, _failure_counts, _tcv_counts = tensorizer.featurize_molecules(mols)

    kept_keys = [key for i, key in enumerate(keys) if i not in failed_idxs]

    npnde_data: Dict[str, LigandData] = {}
    for i, key in enumerate(kept_keys):
        xace = xace_mols[i]
        npnde_data[key] = LigandData(
            sdf=None,
            ccd=None,
            coords=np.asarray(xace.x, dtype=np.float32),
            atom_types=np.asarray(xace.a, dtype=np.int64),
            atom_charges=np.asarray(xace.c, dtype=np.int64),
            bond_types=np.asarray(xace.e, dtype=np.int64),
            bond_indices=np.asarray(xace.edge_idxs, dtype=np.int64),
            is_covalent=False,
            linkages=None,
        )
    return npnde_data


def _encode_charges(charges: torch.Tensor, charge_map_tensor: torch.Tensor) -> torch.Tensor:
    return torch.searchsorted(charge_map_tensor, charges)


def build_npnde_graph_data(
    npndes: Dict[str, LigandData],
    charge_map_tensor: torch.Tensor,
    graph_config,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:

    node_data: Dict[str, Dict[str, torch.Tensor]] = {}
    edge_data: Dict[str, Dict[str, torch.Tensor]] = {}
    edge_idxs: Dict[str, torch.Tensor] = {}

    node_data["npnde"] = {
        "x_1_true": torch.zeros((0, 3), dtype=torch.float32),
        "a_1_true": torch.zeros((0,), dtype=torch.long),
        "c_1_true": torch.zeros((0,), dtype=torch.long),
    }
    edge_data["npnde_to_npnde"] = {"e_1_true": torch.zeros((0,), dtype=torch.long)}
    edge_idxs["npnde_to_npnde"] = torch.zeros((2, 0), dtype=torch.long)

    if not npndes:
        return node_data, edge_idxs, edge_data

    all_coords: List[torch.Tensor] = []
    all_atom_types: List[torch.Tensor] = []
    all_atom_charges: List[torch.Tensor] = []
    all_bond_types: List[torch.Tensor] = []
    all_bond_indices: List[torch.Tensor] = []

    node_offset = 0
    for _, ligand_data in npndes.items():
        coords = torch.from_numpy(np.asarray(ligand_data.coords, dtype=np.float32)).float()
        atom_types = torch.from_numpy(np.asarray(ligand_data.atom_types, dtype=np.int64)).long()
        atom_charges = torch.from_numpy(np.asarray(ligand_data.atom_charges, dtype=np.int64)).long()

        all_coords.append(coords)
        all_atom_types.append(atom_types)
        all_atom_charges.append(atom_charges)

        has_bonds = (
            ligand_data.bond_types is not None
            and ligand_data.bond_indices is not None
            and np.asarray(ligand_data.bond_types).shape[0] > 0
        )
        if has_bonds:
            bond_types = torch.from_numpy(np.asarray(ligand_data.bond_types, dtype=np.int64)).long()
            bond_indices = torch.from_numpy(np.asarray(ligand_data.bond_indices, dtype=np.int64)).long()
            adjusted = bond_indices.clone()
            adjusted[:, 0] += node_offset
            adjusted[:, 1] += node_offset
            all_bond_types.append(bond_types)
            all_bond_indices.append(adjusted)

        node_offset += coords.shape[0]

    combined_coords = torch.cat(all_coords, dim=0) if all_coords else torch.zeros((0, 3), dtype=torch.float32)
    combined_atom_types = torch.cat(all_atom_types, dim=0) if all_atom_types else torch.zeros((0,), dtype=torch.long)
    combined_atom_charges = torch.cat(all_atom_charges, dim=0) if all_atom_charges else torch.zeros((0,), dtype=torch.long)

    if all_bond_types and all_bond_indices:
        combined_bond_types = torch.cat(all_bond_types, dim=0)
        combined_bond_indices = torch.cat(all_bond_indices, dim=0)
        k = graph_config["edges"]["npnde_to_npnde"]["params"]["k"]
        npnde_x, npnde_a, npnde_c, npnde_e, npnde_edge_idxs = add_k_hop_edges(
            combined_coords,
            combined_atom_types,
            combined_atom_charges,
            combined_bond_types,
            combined_bond_indices,
            k=k,
        )
        npnde_c = _encode_charges(npnde_c, charge_map_tensor)
        node_data["npnde"] = {
            "x_1_true": npnde_x,
            "a_1_true": npnde_a,
            "c_1_true": npnde_c,
        }
        edge_data["npnde_to_npnde"] = {"e_1_true": npnde_e}
        edge_idxs["npnde_to_npnde"] = npnde_edge_idxs
    else:
        combined_atom_charges = _encode_charges(combined_atom_charges, charge_map_tensor)
        node_data["npnde"] = {
            "x_1_true": combined_coords,
            "a_1_true": combined_atom_types,
            "c_1_true": combined_atom_charges,
        }

    return node_data, edge_idxs, edge_data


def _filter_npndes_by_pocket(
    npnde_mols: Dict[str, Chem.Mol],
    reference_coords: np.ndarray,
    cutoff: float,
) -> Dict[str, Chem.Mol]:
    """Keep only cofactor mols with any atom within ``cutoff`` of ``reference_coords``."""
    if not npnde_mols or reference_coords is None or len(reference_coords) == 0:
        return {}

    ref = np.asarray(reference_coords, dtype=np.float32)
    if ref.ndim == 1:
        ref = ref[None, :]

    kept: Dict[str, Chem.Mol] = {}
    cutoff_sq = float(cutoff) ** 2
    for key, mol in npnde_mols.items():
        try:
            conf = mol.GetConformer()
        except ValueError:
            continue
        positions = np.asarray(conf.GetPositions(), dtype=np.float32)
        if positions.size == 0:
            continue
        diffs = positions[:, None, :] - ref[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
        if np.any(d2 <= cutoff_sq):
            kept[key] = mol
    return kept


def _residue_spec_mask(atom_array, residue_specs):
    masks = []
    for spec in residue_specs:
        if len(spec) == 2:
            chain_id, res_id = spec
            masks.append((atom_array.chain_id == chain_id) & (atom_array.res_id == res_id))
        else:
            raise ValueError(f"Invalid residue spec: {spec}")
    if not masks:
        return np.zeros(len(atom_array), dtype=bool)
    return np.any(masks, axis=0)

# graph construction
def create_conditional_graphs_from_files(
    task: Task,
    n_samples: int,
    device: torch.device,
    protein_file: Optional[Path] = None,
    ligand_file: Optional[Path] = None,
    pocket_definition: Optional[dict] = None,
    pharmacophore_file: Optional[Path] = None,
    pocket_cutoff: Optional[float] = 8.0,
    use_pocket: bool = True,
    fixed_brics_fragments: Optional[Sequence[int]] = None,
    fixed_atom_indices: Optional[Sequence[int]] = None,
):
    receptor = None
    npnde_mols: Dict[str, Chem.Mol] = {}
    if protein_file is not None:
        receptor, npnde_mols = load_protein_biotite(protein_file, return_npnde_mols=True)

    needs_condensed = 'ligand_identity_condensed' in task.groups_present
    ligand = (
        load_ligand_rdkit(
            ligand_file,
            compute_condensed=needs_condensed,
            fixed_brics_fragments=fixed_brics_fragments,
            fixed_atom_indices=fixed_atom_indices,
        )
        if ligand_file is not None
        else None
    )
    
    # Load pharmacophore from file (JSON, XYZ, or SDF)
    if pharmacophore_file is not None:
        suffix = pharmacophore_file.suffix.lower()
        if suffix == '.json':
            pharm_coords, pharm_types = load_pharmacophore_json(pharmacophore_file)
        elif suffix in ['.sdf', '.mol', '.mol2']:
            from omtra.data.pharmacophores import get_pharmacophores
            supplier = Chem.SDMolSupplier(str(pharmacophore_file))
            mol = next(supplier)
            if mol is None or mol.GetNumAtoms() == 0 or not mol.GetNumConformers():
                raise ValueError(f"Invalid ligand from {pharmacophore_file} for pharmacophore extraction")
            P, X, _, _ = get_pharmacophores(mol, rec=None)
            if len(P) == 0:
                raise ValueError(f"No pharmacophore features extracted from {pharmacophore_file}")
            pharm_coords, pharm_types = P, X
        else:
            pharm_coords, pharm_types = load_pharmacophore_xyz(pharmacophore_file)
    else:
        pharm_coords, pharm_types = (None, None)
    
    npnde_reference_coords: Optional[np.ndarray] = None

    if use_pocket and receptor is not None and pocket_cutoff is not None:
        if pocket_definition is not None:
            pocket_type = pocket_definition.get('type')
            
            if pocket_type == 'file':
                # Use ligand atoms to extract pocket
                pocket_ligand_file = pocket_definition['value']
                pocket_ligand = load_ligand_rdkit(pocket_ligand_file, compute_condensed=False)
                if pocket_ligand is not None:
                    reference_coords = pocket_ligand.coords 
                    pocket = extract_pocket(receptor, reference_coords, pocket_cutoff=pocket_cutoff)
                    if pocket is not None:
                        receptor = pocket
                    npnde_reference_coords = np.asarray(reference_coords, dtype=np.float32)

            elif pocket_type == 'coords':
                # for Pocketeer alpha sphere centers.
                coords_value = pocket_definition['value']
                if isinstance(coords_value, (str, Path)):
                    reference_coords = np.load(coords_value)
                else:
                    reference_coords = np.asarray(coords_value, dtype=np.float32)
                pocket = extract_pocket(receptor, reference_coords, pocket_cutoff=pocket_cutoff)
                if pocket is not None:
                    receptor = pocket
                npnde_reference_coords = np.asarray(reference_coords, dtype=np.float32)
                        
            elif pocket_type == 'center':
                # Create pocket from bounding box around center point
                center_point = np.array(pocket_definition['value'], dtype=np.float32)
                bbox_length = pocket_definition.get('bbox_length', 23.0)
                half_bbox = bbox_length / 2
                
                def residue_selector(atom_array):
                    atom_mask = np.all(np.abs(atom_array.coord - center_point) <= half_bbox, axis=1)
                    return _atoms_to_residue_mask(atom_array, atom_mask)
                
                receptor = _create_pocket_from_indices(
                    receptor,
                    selector=residue_selector,
                    error_context=f"bounding box (size {bbox_length}Å) around center {center_point}",
                )
                npnde_reference_coords = center_point[None, :]
                
            elif pocket_type == 'residues':
                # Create pocket from specified residues
                residue_specs = pocket_definition['value']
                receptor = _create_pocket_from_indices(
                    receptor,
                    selector=lambda aa: _residue_spec_mask(aa, residue_specs),
                    error_context=f"specified residues: {residue_specs}",
                )
                npnde_reference_coords = receptor.coords if receptor is not None else None
        else:
            # fall back to protein center of mass
            reference_coords = np.mean(receptor.coords, axis=0, keepdims=True)
            pocket = extract_pocket(receptor, reference_coords, pocket_cutoff=pocket_cutoff)
            if pocket is not None:
                receptor = pocket
            npnde_reference_coords = np.asarray(reference_coords, dtype=np.float32)


    if npnde_mols:
        if npnde_reference_coords is not None and pocket_cutoff is not None:
            npnde_mols = _filter_npndes_by_pocket(
                npnde_mols,
                npnde_reference_coords,
                cutoff=float(pocket_cutoff),
            )
    npnde_data = featurize_npnde_mols(npnde_mols) if npnde_mols else {}

    charge_map_tensor = torch.tensor(charge_map)
    from omegaconf import OmegaConf
    from omtra.utils import omtra_root
    graph_config_path = Path(omtra_root()) / 'configs' / 'graph' / 'default.yaml'
    graph_config = OmegaConf.load(graph_config_path)
    
    # cache repeated .index() calls
    unk_atom_code = protein_atom_map.index('UNK')
    unk_elem_code = protein_element_map.index('X')
    unk_res_code = residue_map.index('UNK')

    graphs = []
    for _ in range(n_samples):
        node_data = {}
        edge_idxs = {}
        edge_data = {}

        if ligand is not None:
            lig_xace = ligand.to_xace_mol(dense=True)
            
            node_data['lig'] = {
                'x_1_true': lig_xace.x,
            }
            
            if 'ligand_identity_condensed' in task.groups_present and hasattr(lig_xace, 'cond_a'):
                node_data['lig']['cond_a_1_true'] = lig_xace.cond_a
            else:
                # use standard a/c tokenization if present
                node_data['lig']['a_1_true'] = lig_xace.a
                # map charges to charge_map indices
                lig_c = torch.searchsorted(charge_map_tensor, lig_xace.c)
                node_data['lig']['c_1_true'] = lig_c


            edge_idxs['lig_to_lig'] = lig_xace.edge_idxs
            edge_data['lig_to_lig'] = {
                'e_1_true': lig_xace.e,
            }

            if len(task.partial_modalities_fixed) > 0:
                if lig_xace.fixed_atom_mask is None or lig_xace.fixed_edge_mask is None:
                    raise ValueError(
                        "This task uses partial ligand conditioning; provide fixed atoms via "
                        "--fixed-atoms and/or fixed BRICS fragments via --fixed-brics-fragments along with --ligand_file."
                    )
                node_data['lig']['atom_mask_1_true'] = lig_xace.fixed_atom_mask.long()
                edge_data['lig_to_lig']['edge_mask_1_true'] = lig_xace.fixed_edge_mask.long()

        # no ligand file provided, declare the 'lig' node type with zero nodes
        if ligand is None and ('ligand_structure' in task.groups_generated or 'ligand_identity_condensed' in task.groups_generated):
            node_data['lig'] = {
                'x_1_true': torch.zeros((0, 3), dtype=torch.float32)
            }
            edge_idxs.setdefault('lig_to_lig', torch.empty(2, 0, dtype=torch.long))
            edge_data.setdefault('lig_to_lig', {})

        # protein nodes
        if receptor is not None:
            prot_x = torch.from_numpy(receptor.coords).float()
            
            unique_names, inverse = np.unique(receptor.atom_names.astype(str), return_inverse=True)
            unique_codes = np.array([
                protein_atom_map.index(name) if name in protein_atom_map else unk_atom_code
                for name in unique_names
            ], dtype=np.int64)
            a_idx = unique_codes[inverse]
            
            unique_elems, inverse = np.unique(receptor.elements.astype(str), return_inverse=True)
            unique_codes = np.array([
                protein_element_map.index(elem) if elem in protein_element_map else unk_elem_code
                for elem in unique_elems
            ], dtype=np.int64)
            e_idx = unique_codes[inverse]
            
            unique_names, inverse = np.unique(receptor.res_names.astype(str), return_inverse=True)
            unique_codes = np.array([
                residue_map.index(name) if name in residue_map else unk_res_code
                for name in unique_names
            ], dtype=np.int64)
            r_idx = unique_codes[inverse]
            
            unique_chains = sorted(set(receptor.chain_ids.astype(str)))
            chain_to_idx = {chain: idx for idx, chain in enumerate(unique_chains)}
            chain_idx = np.array([chain_to_idx[chain_id] for chain_id in receptor.chain_ids.astype(str)], dtype=np.int64)
            
            node_data['prot_atom'] = {
                'x_1_true': prot_x,
                'a_1_true': torch.from_numpy(a_idx).long(),
                'e_1_true': torch.from_numpy(e_idx).long(),
                'res_id': torch.from_numpy(receptor.res_ids.astype(np.int64)).long(),
                'res_names': torch.from_numpy(r_idx).long(),
                'res_names_1_true': torch.from_numpy(r_idx).long(),
                'chain_id': torch.from_numpy(chain_idx).long(),
                'backbone_mask': torch.from_numpy(receptor.backbone_mask.astype(bool)).bool(),
            }

            prot_res_ids = node_data['prot_atom']['res_id'].numpy()
            prot_chain_ids = node_data['prot_atom']['chain_id'].numpy()
            contiguous_residue_idxs = np.zeros_like(prot_res_ids)
            for chain in np.unique(prot_chain_ids):
                mask = prot_chain_ids == chain
                unique_res = np.unique(prot_res_ids[mask])
                res_to_idx = {res: i for i, res in enumerate(unique_res)}
                contiguous_residue_idxs[mask] = np.vectorize(res_to_idx.get)(prot_res_ids[mask])
            residue_idx_tensor = torch.from_numpy(contiguous_residue_idxs).long()
            pos_enc = residue_sinusoidal_encoding(residue_idx_tensor, d_model=64)
            node_data['prot_atom']['pos_enc_1_true'] = pos_enc

            bb_coords = torch.from_numpy(receptor.backbone.coords).float()
            bb_res_ids = torch.from_numpy(receptor.backbone.res_ids.astype(np.int64)).long()
            
            unique_names, inverse = np.unique(receptor.backbone.res_names.astype(str), return_inverse=True)
            unique_codes = np.array([
                residue_map.index(name) if name in residue_map else unk_res_code
                for name in unique_names
            ], dtype=np.int64)
            bb_res_names = torch.from_numpy(unique_codes[inverse]).long()
            
            bb_chain_ids = torch.tensor([chain_to_idx[c] for c in receptor.backbone.chain_ids], dtype=torch.long)
            node_data['prot_res'] = {
                'x_1_true': bb_coords,
                'res_id': bb_res_ids,
                'a_1_true': bb_res_names,
                'chain_id': bb_chain_ids,
            }

        # pharmacophore nodes
        if pharm_coords is not None:
            node_data['pharm'] = {
                'x_1_true': torch.from_numpy(pharm_coords).float(),
                'a_1_true': torch.from_numpy(pharm_types).long(),
            }

        # npnde nodes
        if npnde_data:
            npnde_node_data, npnde_edge_idxs, npnde_edge_data = build_npnde_graph_data(
                npnde_data, charge_map_tensor, graph_config
            )
            node_data.update(npnde_node_data)
            edge_idxs.update(npnde_edge_idxs)
            for etype, feats in npnde_edge_data.items():
                edge_data[etype] = feats

        g = build_complex_graph(
            node_data=node_data,
            edge_idxs=edge_idxs,
            edge_data=edge_data,
            task=task,
            graph_config=graph_config,
        )
        graphs.append(g.to(device))

    return graphs

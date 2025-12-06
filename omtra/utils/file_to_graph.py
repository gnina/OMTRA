import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

from rdkit import Chem

from omtra.tasks.tasks import Task
from omtra.constants import (
    lig_atom_type_map,
    charge_map,
    ph_idx_to_type,
    ph_idx_to_elem,
    residue_map,
    protein_element_map,
    protein_atom_map,
)
from omtra.data.graph import build_complex_graph
from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.data.plinder import StructureData, LigandData, BackboneData
from omtra.data.condensed_atom_typing import CondensedAtomTyper
from omtra.utils.embedding import residue_sinusoidal_encoding


# loaders
def load_protein_biotite(protein_file: Path) -> StructureData:
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

    coords = st.coord
    if coords.size == 0:
        raise ValueError("Protein structure has no atoms")
    backbone_mask = struc.filter_peptide_backbone(st)
    backbone = BackboneData(
        coords=coords[backbone_mask],
        res_ids=st.res_id[backbone_mask],
        res_names=st.res_name[backbone_mask],
        chain_ids=st.chain_id[backbone_mask],
    )

    return StructureData(
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


def load_ligand_rdkit(ligand_file: Path, compute_condensed: bool = False) -> LigandData:
    supplier = Chem.SDMolSupplier(str(ligand_file))
    mol = next(supplier)
    if mol is None:
        raise ValueError(f"Failed to read ligand from {ligand_file}")
    if mol.GetNumAtoms() == 0:
        raise ValueError("Ligand has zero atoms")
    if not mol.GetNumConformers():
        raise ValueError("Ligand has no 3D conformer")

    tensorizer = MoleculeTensorizer(lig_atom_type_map, n_cpus=1)
    valid_mols, failed, failures, _ = tensorizer.featurize_molecules([mol])
    if failed:
        raise ValueError(f"Ligand featurization failed: {failures}")
    xace = valid_mols[0].sparse_to_dense()
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

    return LigandData(
        coords=xace.x,
        bond_types=xace.e,
        bond_indices=xace.edge_idxs.T,
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


def extract_backbone_data(backbone_atoms) -> BackboneData:
    """Extract backbone data from backbone atoms (N, CA, C per residue)."""
    import biotite.structure as struc
    
    compound_keys = np.array(
        [f"{chain}_{res}" for chain, res in zip(backbone_atoms.chain_id, backbone_atoms.res_id)]
    )

    unique_compound_keys = np.unique(compound_keys)
    num_residues = len(unique_compound_keys)

    coords = np.zeros((num_residues, 3, 3))
    res_ids = np.zeros(num_residues, dtype=int)
    res_names_list = []
    chain_ids_list = []

    for i, compound_key in enumerate(unique_compound_keys):
        chain_id, res_id = compound_key.split("_")
        res_id = int(res_id)

        res_mask = (backbone_atoms.chain_id == chain_id) & (backbone_atoms.res_id == res_id)
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
    
    close_res_ids = atom_array.res_id[close_atom_indices]
    close_chain_ids = atom_array.chain_id[close_atom_indices]
    unique_res_pairs = set(zip(close_res_ids, close_chain_ids))
    
    pocket_indices = []
    for res_id, chain_id in unique_res_pairs:
        res_mask = (atom_array.res_id == res_id) & (atom_array.chain_id == chain_id)
        res_indices = np.where(res_mask)[0]
        pocket_indices.extend(res_indices)
    
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


# graph construction
def create_conditional_graphs_from_files(
    task: Task,
    n_samples: int,
    device: torch.device,
    protein_file: Optional[Path] = None,
    ligand_file: Optional[Path] = None,
    pharmacophore_file: Optional[Path] = None,
    pocket_cutoff: Optional[float] = 8.0,
    use_pocket: bool = True,
):
    receptor = load_protein_biotite(protein_file) if protein_file is not None else None
    
    needs_condensed = 'ligand_identity_condensed' in task.groups_present
    ligand = load_ligand_rdkit(ligand_file, compute_condensed=needs_condensed) if ligand_file is not None else None
    pharm_coords, pharm_types = (load_pharmacophore_xyz(pharmacophore_file) if pharmacophore_file is not None else (None, None))
    
    if use_pocket and receptor is not None and pocket_cutoff is not None:
        reference_coords = None
        
        if ligand is not None:
            lig_coords = ligand.coords
            if isinstance(lig_coords, torch.Tensor):
                reference_coords = lig_coords.cpu().numpy()
            else:
                reference_coords = np.asarray(lig_coords)
        elif pharm_coords is not None:
            reference_coords = pharm_coords
        else:
            reference_coords = np.mean(receptor.coords, axis=0, keepdims=True)
        
        pocket = extract_pocket(receptor, reference_coords, pocket_cutoff=pocket_cutoff)
        if pocket is not None:
            receptor = pocket

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
                charge_map_tensor = torch.tensor(charge_map)
                lig_c = torch.searchsorted(charge_map_tensor, lig_xace.c)
                node_data['lig']['c_1_true'] = lig_c


            edge_idxs['lig_to_lig'] = lig_xace.edge_idxs
            edge_data['lig_to_lig'] = {
                'e_1_true': lig_xace.e,
            }

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
            unk_code = protein_atom_map.index('UNK')
            unique_codes = np.array([
                protein_atom_map.index(name) if name in protein_atom_map else unk_code
                for name in unique_names
            ], dtype=np.int64)
            a_idx = unique_codes[inverse]
            
            unique_elems, inverse = np.unique(receptor.elements.astype(str), return_inverse=True)
            unknown_elem_code = protein_element_map.index('X')
            unique_codes = np.array([
                protein_element_map.index(elem) if elem in protein_element_map else unknown_elem_code
                for elem in unique_elems
            ], dtype=np.int64)
            e_idx = unique_codes[inverse]
            
            unique_names, inverse = np.unique(receptor.res_names.astype(str), return_inverse=True)
            unk_code = residue_map.index('UNK')
            unique_codes = np.array([
                residue_map.index(name) if name in residue_map else unk_code
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
            unk_code = residue_map.index('UNK')
            unique_codes = np.array([
                residue_map.index(name) if name in residue_map else unk_code
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
            n_points = len(pharm_coords)
            dummy_vectors = torch.zeros((n_points, 4, 3), dtype=torch.float32)
            dummy_interactions = torch.zeros((n_points,), dtype=torch.bool)
            
            node_data['pharm'] = {
                'x_1_true': torch.from_numpy(pharm_coords).float(),
                'a_1_true': torch.from_numpy(pharm_types).long(),
                'v_1_true': dummy_vectors,
                'i_1_true': dummy_interactions,
            }

        from omegaconf import OmegaConf
        from omtra.utils import omtra_root
        graph_config_path = Path(omtra_root()) / 'configs' / 'graph' / 'default.yaml'
        graph_config = OmegaConf.load(graph_config_path)
        
        g = build_complex_graph(
            node_data=node_data,
            edge_idxs=edge_idxs,
            edge_data=edge_data,
            task=task,
            graph_config=graph_config,
        )
        graphs.append(g.to(device))

    return graphs

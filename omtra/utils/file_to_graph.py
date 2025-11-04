import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

from omtra.utils.embedding import residue_sinusoidal_encoding

from rdkit import Chem

from omtra.tasks.tasks import Task
from omtra.constants import (
    lig_atom_type_map,
    charge_map,
    ph_idx_to_type,
    residue_map,
    protein_element_map,
    protein_atom_map,
)
from omtra.data.graph import build_complex_graph
from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.data.plinder import StructureData, LigandData, BackboneData
from omtra.data.condensed_atom_typing import CondensedAtomTyper



# loaders
def load_protein_biotite(protein_file: Path) -> StructureData:
    try:
        from biotite.structure.io import pdb
    except ImportError as e:
        raise ImportError("biotite is required: pip install biotite") from e

    suffix = protein_file.suffix.lower()
    if suffix == ".pdb":
        st = pdb.PDBFile.read(str(protein_file)).get_structure()
    # elif suffix == ".cif":
    #     st = mmcif.CifFile.read(str(protein_file)).get_structure()
    else:
        raise ValueError(f"Unsupported protein format: {suffix}")

    if st.coord.ndim == 3:
        st = st[0]

    coords = st.coord
    if coords.size == 0:
        raise ValueError("Protein structure has no atoms")

    import biotite.structure as struc
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

    atom_cond_a = None
    if compute_condensed:
        cond_typer = CondensedAtomTyper(fake_atoms=False)
        

        from omtra.data.extra_ligand_features import ligand_properties, fragment_molecule
        extra_feats = ligand_properties(mol)
        fragment_feats = fragment_molecule(mol)
        
        extra_feats = np.concatenate([extra_feats, fragment_feats], axis=1) # (n_atoms, 6)
        
        extra_feats = extra_feats[:, :-1]  # (n_atoms, 5)
        
        atom_cond_a = cond_typer.feats_to_cond_a(
            a=xace.a.numpy(),
            c=xace.c.numpy(), 
            extra_feats=extra_feats
        )

    return LigandData(
        coords=xace.x.numpy(),
        bond_types=xace.e.numpy(),
        bond_indices=xace.edge_idxs.numpy().T if hasattr(xace.edge_idxs, 'numpy') else np.array(xace.edge_idxs).T,
        is_covalent=False,
        ccd="LIG",
        sdf=str(ligand_file),
        atom_types=xace.a.numpy(),
        atom_charges=xace.c.numpy(),
        atom_impl_H=getattr(xace, 'impl_H', None).numpy() if getattr(xace, 'impl_H', None) is not None else None,
        atom_aro=getattr(xace, 'aro', None).numpy() if getattr(xace, 'aro', None) is not None else None,
        atom_hyb=getattr(xace, 'hyb', None).numpy() if getattr(xace, 'hyb', None) is not None else None,
        atom_ring=getattr(xace, 'ring', None).numpy() if getattr(xace, 'ring', None) is not None else None,
        atom_chiral=getattr(xace, 'chiral', None).numpy() if getattr(xace, 'chiral', None) is not None else None,
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
    
    unique_kinds, inverse = np.unique(kinds, return_inverse=True)
    unk_code = ph_idx_to_type.index('UNK') if 'UNK' in ph_idx_to_type else 0
    unique_codes = np.array([
        ph_idx_to_type.index(kind) if kind in ph_idx_to_type else unk_code
        for kind in unique_kinds
    ], dtype=np.int64)
    kind_idx = unique_codes[inverse]
    
    return coords, kind_idx


# graph construction
def create_conditional_graphs_from_files(
    task: Task,
    n_samples: int,
    device: torch.device,
    protein_file: Optional[Path] = None,
    ligand_file: Optional[Path] = None,
    pharmacophore_file: Optional[Path] = None,
    input_files_dir: Optional[Path] = None,
):
    # resolve files from directory if provided
    if input_files_dir is not None:
        if protein_file is None:
            pf = input_files_dir / "protein.pdb"
            protein_file = pf if pf.exists() else (input_files_dir / "protein.cif" if (input_files_dir / "protein.cif").exists() else None)
        if ligand_file is None and (input_files_dir / "ligand.sdf").exists():
            ligand_file = input_files_dir / "ligand.sdf"
        if pharmacophore_file is None and (input_files_dir / "pharmacophore.xyz").exists():
            pharmacophore_file = input_files_dir / "pharmacophore.xyz"

    required = set(task.groups_fixed)
    if 'protein_identity' in required and protein_file is None:
        raise ValueError("Task requires protein_file")
    if 'ligand_identity' in required and ligand_file is None and 'ligand_identity_condensed' in required and ligand_file is None:
        raise ValueError("Task requires ligand_file")
    if 'pharmacophore' in required and pharmacophore_file is None:
        raise ValueError("Task requires pharmacophore_file")

    receptor = load_protein_biotite(protein_file) if protein_file is not None else None
    
    needs_condensed = 'ligand_identity_condensed' in task.groups_present
    ligand = load_ligand_rdkit(ligand_file, compute_condensed=needs_condensed) if ligand_file is not None else None
    pharm_coords, pharm_types = (load_pharmacophore_xyz(pharmacophore_file) if pharmacophore_file is not None else (None, None))

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
            unique_codes = np.array([
                protein_element_map.index(elem) for elem in unique_elems
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

            # Standardize residue IDs and create positional encodings
            unique_chains_sorted = sorted(set(receptor.chain_ids.astype(str)))
            chain_to_idx_map = {chain: idx for idx, chain in enumerate(unique_chains_sorted)}
            prot_chain_ids_mapped = torch.tensor([chain_to_idx_map[c] for c in receptor.chain_ids.astype(str)], dtype=torch.long)

            # Standardize residue IDs (normalize within each chain)
            standardized_res_ids = torch.zeros_like(torch.from_numpy(receptor.res_ids).long())
            unique_chains_tensor = torch.unique(prot_chain_ids_mapped, sorted=True)

            for chain in unique_chains_tensor:
                chain_mask = prot_chain_ids_mapped == chain
                chain_res_ids = torch.from_numpy(receptor.res_ids).long()[chain_mask]
                min_res_id = torch.min(chain_res_ids)
                standardized_res_ids[chain_mask] = chain_res_ids - min_res_id

            # Create positional encodings
            res_id_embed_dim = 64  # default from vector_field
            protein_position_encodings = residue_sinusoidal_encoding(standardized_res_ids, res_id_embed_dim)

            # Add to node_data
            node_data['prot_atom']['pos_enc_1_true'] = protein_position_encodings

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
            dummy_vectors = torch.zeros((n_points, 3), dtype=torch.float32)
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

    print(f"[DEBUG] Finished creating all graphs")
    return graphs
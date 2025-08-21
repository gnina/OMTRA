import dgl
import torch
from typing import Tuple, List, Union, Dict
from pathlib import Path
import omtra.constants as constants
from omtra.constants import (
    lig_atom_type_map,
    npnde_atom_type_map,
    bond_type_map,
    charge_map,
    protein_element_map,
    protein_atom_map,
    residue_map,
)
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D
import numpy as np
import biotite.structure as struc
from biotite.interface import rdkit as bt_rdkit
import os
from copy import deepcopy
from omtra.tasks.modalities import name_to_modality
from collections import defaultdict
from omtra.tasks.register import task_name_to_class
from omtra.data.condensed_atom_typing import CondensedAtomTyper
from omtra.tasks.tasks import Task

from omtra.data.graph.utils import (
    copy_graph,
    get_upper_edge_mask,
)

from biotite.structure.io.pdbx import CIFFile
from biotite.structure.io.pdb import PDBFile



class SampledSystem:
    """
    Convert a DGLGraph into objects ready for evaluation.
    """

    def __init__(
        self,
        g: dgl.DGLHeteroGraph,
        task: Task,
        traj: Dict[str, torch.Tensor] = None,
        fake_atoms: bool = False,  # whether the molecule contains fake atoms,
        ctmc_mol: bool = True,
        exclude_charges: bool = False,  # TODO: remove  this option and all its effects
        ligand_atom_type_map: List[str] = lig_atom_type_map,
        npnde_atom_type_map: List[str] = npnde_atom_type_map,
        protein_atom_type_map: List[str] = protein_atom_map,
        residue_map: List[str] = residue_map,
        bond_type_map: List[str] = bond_type_map,
        charge_map: List[int] = charge_map,
        protein_element_map: List[str] = protein_element_map,
        cond_a_typer: CondensedAtomTyper = None
    ):
        self.task = task
        self.traj = traj
        self.fake_atoms = fake_atoms
        self.ctmc_mol = ctmc_mol
        self.exclude_charges = exclude_charges
        self.ligand_atom_type_map = ligand_atom_type_map
        self.npnde_atom_type_map = npnde_atom_type_map
        self.protein_atom_type_map = protein_atom_type_map
        self.residue_map = residue_map
        self.bond_type_map = bond_type_map
        self.charge_map = charge_map
        self.protein_element_map = protein_element_map
        self._cached_protein_array = None
        
        self.rdkit_ligand = None
        self.rdkit_ref_ligand = None
        self.rdkit_protein = None
        self.rdkit_ref_protein = None
        
        if self.fake_atoms:
            self.ligand_atom_type_map = deepcopy(self.ligand_atom_type_map)
            self.ligand_atom_type_map.append("Sn")  # fake atoms appear as Sn

        if self.ctmc_mol:
            self.ligand_atom_type_map = deepcopy(self.ligand_atom_type_map)
            self.ligand_atom_type_map.append("Se")  # masked atoms appear as Se
        
        self.has_condensed_typing = ('ligand_identity_condensed' in task.groups_present) or ('ligand_identity_condensed' in task.groups_generated)

        if self.has_condensed_typing:
            # decode condensed atom type representation to explicit form 
            self.cond_a_typer = cond_a_typer
            self.g = self.decode_conda(g)
        else:
            self.g = g

    def decode_conda(self, g):
        # decodes condensed atom tupes to explicit atom features
        lig_g = dgl.node_type_subgraph(g, ntypes=["lig"])
        lig_ndata_feats = list(lig_g.nodes["lig"].data.keys())
    
        cond_a_feats = [(feat, suffix) for feat in lig_ndata_feats if "cond_a" in feat for _, suffix in [feat.split("cond_a")]]

        for cond_a_feat, suffix in cond_a_feats:
            lig_feats_dict = self.cond_a_typer.cond_a_to_feats(lig_g.nodes["lig"].data[cond_a_feat])

            for feat, val in lig_feats_dict.items():
                g.nodes["lig"].data[f"{feat}{suffix}"] = torch.tensor(val, device=lig_g.device)
            
            del g.nodes["lig"].data[cond_a_feat]
        return g

    def to(self, device: str):
        self.g = self.g.to(device)

        if self.traj:
            for k in self.traj:
                self.traj[k] = self.traj[k].to(device)

        return self

    def get_n_lig_atoms(self) -> int:
        n_lig_atoms = self.g.num_nodes(ntype="lig")
        if self.fake_atoms:
            fake_atom_token_idx  = self.ligand_atom_type_map.index('Sn')
            atom_type_idxs = self.g.nodes['lig'].data['a_1']
            fake_atom_mask = atom_type_idxs == fake_atom_token_idx
            n_fake_atoms = fake_atom_mask.sum().item()
            n_lig_atoms -= n_fake_atoms
        return n_lig_atoms

    def get_atom_arr(self, reference: bool = False):
        """
        Get the system data represented as Biotite AtomArray
        :return: Biotite AtomArray
        """
        # TODO: need to handle masked/fake elements
        if reference:
            feat_suffix = "1_true"
        else:
            feat_suffix = "1"
        ntypes = ["prot_atom", "lig", "npnde"]

        atom_arrays = []

        for ntype in ntypes:
            if self.g.num_nodes(ntype=ntype) == 0:
                continue
            atom_array = struc.AtomArray(self.g.num_nodes(ntype=ntype))
            coords = self.g.nodes[ntype].data[f"x_{feat_suffix}"].numpy()

            if ntype == "prot_atom":
                atypes = self.g.nodes[ntype].data[f"a_1_true"].numpy()
                atom_type_map_array = np.array(self.protein_atom_type_map, dtype=object)
                atom_names = atom_type_map_array[atypes]

                eltypes = self.g.nodes[ntype].data[f"e_1_true"].numpy()
                element_type_map_array = np.array(
                    self.protein_element_map, dtype=object
                )
                elements = element_type_map_array[eltypes]

                res_ids = self.g.nodes[ntype].data["res_id"].numpy()
                res_types = self.g.nodes[ntype].data["res_names"].numpy()
                res_type_map_array = np.array(self.residue_map, dtype=object)
                res_names = res_type_map_array[res_types]

                chain_ids = self.g.nodes[ntype].data["chain_id"].numpy()
                hetero = np.full_like(atom_names, False, dtype=bool)
                atom_array.coord = coords

                atom_array.set_annotation("atom_name", atom_names)
                atom_array.set_annotation("element", elements)
                atom_array.set_annotation("res_id", res_ids)
                atom_array.set_annotation("res_name", res_names)
                atom_array.set_annotation("chain_id", chain_ids)
                atom_array.set_annotation("hetero", hetero)
                atom_array.bonds = struc.connect_via_distances(atom_array)

            if ntype == "lig":
                atypes = self.g.nodes[ntype].data[f"a_{feat_suffix}"].numpy()
                atom_type_map_array = np.array(self.ligand_atom_type_map, dtype=object)
                elements = atom_type_map_array[atypes]
                atom_names = struc.create_atom_names(elements)

                res_id = 0
                res_ids = np.full_like(atypes, res_id, dtype=int)
                res_names = np.full_like(atypes, "LIG", dtype=object)
                chain_ids = np.full_like(atypes, "LIG", dtype=object)
                hetero = np.full_like(atom_names, True, dtype=bool)

                charge_types = self.g.nodes[ntype].data[f"c_{feat_suffix}"].numpy()
                charge_map_array = np.array(self.charge_map, dtype=object)
                charges = charge_map_array[charge_types]

                bond_types = self.g.edges["lig_to_lig"].data[f"e_{feat_suffix}"].numpy()
                bond_types = bond_types.astype(int)
                bond_src_idxs, bond_dst_idxs = self.g.edges(etype="lig_to_lig")
                bond_src_idxs, bond_dst_idxs = (
                    bond_src_idxs.numpy(),
                    bond_dst_idxs.numpy(),
                )

                upper_edge_mask = get_upper_edge_mask(
                    self.g, etype="lig_to_lig"
                ).numpy()
                bond_types[bond_types == 5] = 0
                bond_types[bond_types == 4] = (
                    9  # NOTE: generic aromatic bond is 9 in biotite
                )

                bond_mask = (bond_types != 0) & upper_edge_mask
                bond_types = bond_types[bond_mask]
                bond_src_idxs = bond_src_idxs[bond_mask]
                bond_dst_idxs = bond_dst_idxs[bond_mask]

                bond_array = np.stack(
                    [bond_src_idxs, bond_dst_idxs, bond_types], axis=1
                ).astype(int)

                atom_array.coord = coords
                atom_array.set_annotation("charge", charges)
                atom_array.set_annotation("atom_name", atom_names)
                atom_array.set_annotation("element", elements)
                atom_array.set_annotation("res_id", res_ids)
                atom_array.set_annotation("res_name", res_names)
                atom_array.set_annotation("chain_id", chain_ids)
                atom_array.set_annotation("hetero", hetero)
                atom_array.bonds = struc.BondList(len(atom_array), bond_array)

            if ntype == "npnde":
                atypes = self.g.nodes[ntype].data[f"a_{feat_suffix}"].numpy()
                atom_type_map_array = np.array(self.npnde_atom_type_map, dtype=object)
                elements = atom_type_map_array[atypes]
                atom_names = struc.create_atom_names(elements)

                res_id = 0
                res_ids = np.full_like(atypes, res_id, dtype=int)
                res_names = np.full_like(atypes, "NPND", dtype=object)
                # TODO: might need to modify dataset to track individual npnde chains
                chain_ids = np.full_like(atypes, "NPND", dtype=object)
                hetero = np.full_like(atom_names, True, dtype=bool)
                charge_types = self.g.nodes[ntype].data[f"c_{feat_suffix}"].numpy()
                charge_map_array = np.array(self.charge_map, dtype=object)
                charges = charge_map_array[charge_types]

                bond_types = (
                    self.g.edges["npnde_to_npnde"].data[f"e_{feat_suffix}"].numpy()
                )
                bond_types = bond_types.astype(int)
                bond_src_idxs, bond_dst_idxs = self.g.edges(etype="npnde_to_npnde")
                bond_src_idxs, bond_dst_idxs = (
                    bond_src_idxs.numpy(),
                    bond_dst_idxs.numpy(),
                )

                upper_edge_mask = get_upper_edge_mask(
                    self.g, etype="npnde_to_npnde"
                ).numpy()
                bond_types[bond_types == 5] = 0
                bond_types[bond_types == 4] = (
                    9  # NOTE: generic aromatic bond is 9 in biotite
                )

                bond_mask = (bond_types != 0) & upper_edge_mask
                bond_types = bond_types[bond_mask]
                bond_src_idxs = bond_src_idxs[bond_mask]
                bond_dst_idxs = bond_dst_idxs[bond_mask]

                bond_array = np.stack(
                    [bond_src_idxs, bond_dst_idxs, bond_types], axis=1
                ).astype(int)

                atom_array.coord = coords
                atom_array.set_annotation("atom_name", atom_names)
                atom_array.set_annotation("element", elements)
                atom_array.set_annotation("res_id", res_ids)
                atom_array.set_annotation("res_name", res_names)
                atom_array.set_annotation("chain_id", chain_ids)
                atom_array.set_annotation("charge", charges)
                atom_array.set_annotation("hetero", hetero)
                atom_array.bonds = struc.BondList(len(atom_array), bond_array)

            atom_arrays.append(atom_array)
            system_arr = struc.concatenate(atom_arrays)

        return system_arr
    
    def construct_system_array(self, g=None):
        """
        Refactoring get_atom_arr to be more modular/work better for trajectories (leaving above as is for now for eval stuff)
        Get the system data represented as Biotite AtomArray
        :return: Biotite AtomArray
        """
        arrs = []
        prot = self.get_protein_array(g=g)
        arrs.append(prot)
        
        ligdata = self.extract_ligdata_from_graph(g=g, ctmc_mol=self.ctmc_mol, show_fake_atoms=True)
        ligdata = self.convert_ligdata_to_biotite(*ligdata)
        lig = self.build_atom_array(*ligdata)
        arrs.append(lig)
        
        npndedata = self.extract_ligdata_from_graph(g=g, ctmc_mol=self.ctmc_mol, show_fake_atoms=True, npnde=True)
        if npndedata:
            npndedata = self.convert_ligdata_to_biotite(*npndedata, npnde=True)
            npnde = self.build_atom_array(*npndedata)
            arrs.append(npnde)
            
        system_arr = struc.concatenate(arrs)
        return system_arr

    def build_bond_list(self, bond_src_idxs, bond_dst_idxs, bond_types, n_atoms):
        bond_types[bond_types == 4] = 9  # NOTE: generic aromatic bond is 9 in biotite
        bond_array = np.stack(
            [bond_src_idxs, bond_dst_idxs, bond_types], axis=1
        ).astype(int)
        bond_list = struc.BondList(n_atoms, bond_array)
        return bond_list

    def build_atom_array(
        self,
        coords,
        atom_name,
        element,
        res_id,
        res_name,
        chain_id,
        hetero,
        charge=None,
        include_bonds: bool = True,
        bond_src_idxs=None,
        bond_dst_idxs=None,
        bond_types=None,
    ):
        n_nodes = len(atom_name)
        atom_array = struc.AtomArray(n_nodes)
        
        if isinstance(coords, torch.Tensor):
            coords = coords.numpy()
        atom_array.coord = coords
        atom_array.set_annotation("atom_name", atom_name)
        atom_array.set_annotation("element", element)
        atom_array.set_annotation("res_id", res_id)
        atom_array.set_annotation("res_name", res_name)
        atom_array.set_annotation("chain_id", chain_id)
        atom_array.set_annotation("hetero", hetero)
        if charge is not None:
            # TODO: why is this a generator object ?
            atom_array.set_annotation("charge", charge)
        
        if not include_bonds:
            return atom_array
        
        if (
            bond_src_idxs is not None
            and bond_dst_idxs is not None
            and bond_types is not None
        ):
            bond_list = self.build_bond_list(bond_src_idxs, bond_dst_idxs, bond_types, atom_array.array_length())
            atom_array.bonds = bond_list
        else:
            atom_array.bonds = struc.connect_via_distances(self.get_protein_array(reference=True, include_bonds=False))
            
        return atom_array

    def get_protein_array(self, g=None, reference: bool = False, include_bonds: bool = True):
        if g is None and reference and include_bonds:
            if self._cached_protein_array is not None:
                return self._cached_protein_array
        coords, atom_names, elements, res_ids, res_names, chain_ids, hetero = (
            self.extract_protdata_from_graph(g=g, reference=reference)
        )
        arr = self.build_atom_array(
            coords=coords,
            atom_name=atom_names,
            element=elements,
            res_id=res_ids,
            res_name=res_names,
            chain_id=chain_ids,
            hetero=hetero,
            include_bonds=include_bonds,
        )
        if g is None and reference and include_bonds:
            self._cached_protein_array = arr
        return arr

    def get_rdkit_ligand(self) -> Union[None, Chem.Mol]:
        if self.rdkit_ligand is not None:
            return self.rdkit_ligand
        ligdata = self.extract_ligdata_from_graph(ctmc_mol=self.ctmc_mol)
        rdkit_mol = self.build_molecule(*ligdata)
        self.rdkit_ligand = rdkit_mol
        return rdkit_mol
    
    def get_gt_ligand(self, g=None):
        if g is None:
            g_dummy = self.g.clone()
        else:
            g_dummy = g.clone()

        if self.has_condensed_typing:
            g_dummy = self.decode_conda(g_dummy)       

        g_dummy = move_feats_to_t1('denovo_ligand', g_dummy, t="1_true")     
        
        ligdata = self.extract_ligdata_from_graph(g=g_dummy, ctmc_mol=self.ctmc_mol)
        rdkit_mol = self.build_molecule(*ligdata)
        return rdkit_mol
    
    def get_rdkit_ref_ligand(self) -> Union[None, Chem.Mol]:
        if self.rdkit_ref_ligand is not None:
            return self.rdkit_ref_ligand
        ligdata = self.extract_ligdata_from_graph(ctmc_mol=self.ctmc_mol, ref=True)
        rdkit_mol = self.build_molecule(*ligdata)
        self.rdkit_ref_ligand = rdkit_mol
        return rdkit_mol
    
    def get_rdkit_protein(self):
        if self.rdkit_protein is not None:
            return self.rdkit_protein
        prot_arr = self.get_protein_array()
        prot_mol = bt_rdkit.to_mol(prot_arr)
        self.rdkit_protein = prot_mol
        return prot_mol

    def get_rdkit_ref_protein(self):
        if self.rdkit_ref_protein is not None:
             return self.rdkit_ref_protein
        prot_arr = self.get_protein_array(reference=True)
        prot_mol = bt_rdkit.to_mol(prot_arr)
        self.rdkit_ref_protein = prot_mol
        return prot_mol

    def convert_ligdata_to_biotite(
        self,
        positions,
        atom_types,
        atom_charges,
        bond_types,
        bond_src_idxs,
        bond_dst_idxs,
        npnde: bool = False,
    ):
        atom_names = struc.create_atom_names(atom_types)
        elements = atom_types
        res_id = 0
        res_ids = np.full_like(atom_names, res_id, dtype=int)
        if npnde:
            res_names = np.full_like(atom_names, "NPND", dtype=object)
        else:
            res_names = np.full_like(atom_names, "LIG", dtype=object)
        # TODO: might need to modify dataset to track individual npnde chains
        if npnde:
            chain_ids = np.full_like(atom_names, "NPND", dtype=object)
        else:
            chain_ids = np.full_like(atom_names, "LIG", dtype=object)

        hetero = np.full_like(atom_names, True, dtype=bool)
        return (
            positions,
            atom_names,
            elements,
            res_ids,
            res_names,
            chain_ids,
            hetero,
            atom_charges,
            bond_src_idxs,
            bond_dst_idxs,
            bond_types,
        )

    def extract_ligdata_from_graph(
        self,
        g=None,
        ctmc_mol: bool = False,
        show_fake_atoms: bool = False,
        npnde: bool = False,
        ref: bool = False
    ) -> Tuple[torch.Tensor, List[str], List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        if g is None:
            g = self.g

        if not npnde:
            atom_type_map = list(self.ligand_atom_type_map)
            lig_g = dgl.node_type_subgraph(g, ntypes=["lig"])
            lig_ndata_feats = list(lig_g.nodes["lig"].data.keys())
            lig_edata_feats = list(lig_g.edges["lig_to_lig"].data.keys())
            
        else:
            if g.num_nodes(ntype="npnde") == 0:
                return None
            atom_type_map = list(self.npnde_atom_type_map)
            lig_g = dgl.node_type_subgraph(g, ntypes=["npnde"])
            lig_ndata_feats = list(lig_g.nodes["npnde"].data.keys())
            lig_edata_feats = list(lig_g.edges["npnde_to_npnde"].data.keys())

        lig_g = dgl.to_homogeneous(lig_g, ndata=lig_ndata_feats, edata=lig_edata_feats)

        
        # if fake atoms are present, identify them
        if self.fake_atoms and not show_fake_atoms:
            # TODO: need to update atom map to include fake atoms
            fake_atom_token_idx = len(atom_type_map) - 2
            fake_atom_mask = lig_g.ndata["a_1"] == fake_atom_token_idx
            fake_atom_idxs = torch.where(fake_atom_mask)[0]
            lig_g.remove_nodes(fake_atom_idxs)

        # extract node-level features
        if ref:
            positions: torch.Tensor = lig_g.ndata["x_1_true"].clone()
        else:
            positions: torch.Tensor = lig_g.ndata["x_1"]
            
        atom_types = lig_g.ndata["a_1"]
        atom_types: List[str] = [atom_type_map[int(atom)] for atom in atom_types]

        if self.exclude_charges:
            atom_charges = None
        else:
            charge_data = lig_g.ndata["c_1"].clone()

            # set masked charges to 0
            if ctmc_mol:
                masked_charge = charge_data == len(self.charge_map)
                neutral_index = self.charge_map.index(0)
                charge_data[masked_charge] = neutral_index

            atom_charges: List[int] = [self.charge_map[int(charge)] for charge in charge_data]

        # get bond types and atom indicies for every edge, convert types from simplex to integer
        bond_types = lig_g.edata["e_1"].clone()
        masked_bonds = bond_types == len(self.bond_type_map)
        bond_types[masked_bonds] = 0  # set masked bonds to 0 (unbonded)
        bond_src_idxs, bond_dst_idxs = lig_g.edges()

        # get just the upper triangle of the adjacency matrix
        # TODO: need to use lig_g not self.g for upper edge mask
        upper_edge_mask = get_upper_edge_mask(lig_g, etype=None)
        bond_types = bond_types[upper_edge_mask]
        bond_src_idxs = bond_src_idxs[upper_edge_mask]
        bond_dst_idxs = bond_dst_idxs[upper_edge_mask]

        # get only non-zero bond types
        bond_mask = bond_types != 0
        bond_types = bond_types[bond_mask]
        bond_src_idxs = bond_src_idxs[bond_mask]
        bond_dst_idxs = bond_dst_idxs[bond_mask]

        return (
            positions,
            atom_types,
            atom_charges,
            bond_types,
            bond_src_idxs,
            bond_dst_idxs,
        )

    def extract_protdata_from_graph(self, g=None, reference: bool = False):
        if g is None:
            g = self.g

        if reference:
            feat_suffix = "1_true"
        else:
            feat_suffix = "1"

        coords = g.nodes["prot_atom"].data[f"x_{feat_suffix}"].numpy()
        atypes = g.nodes["prot_atom"].data[f"a_1_true"].numpy()
        atom_type_map_array = np.array(self.protein_atom_type_map, dtype="U3")
        atom_names = atom_type_map_array[atypes]

        eltypes = g.nodes["prot_atom"].data[f"e_1_true"].numpy()
        element_type_map_array = np.array(self.protein_element_map, dtype="U2")
        elements = element_type_map_array[eltypes]

        res_ids = g.nodes["prot_atom"].data["res_id"].numpy()
        res_types = g.nodes["prot_atom"].data["res_names"].numpy()
        res_type_map_array = np.array(self.residue_map, dtype="U3")
        res_names = res_type_map_array[res_types]

        chain_ids = g.nodes["prot_atom"].data["chain_id"].numpy()
        hetero = np.full_like(atom_names, False, dtype=bool)
        return coords, atom_names, elements, res_ids, res_names, chain_ids, hetero

    def build_molecule(
        self,
        positions,
        atom_types,
        atom_charges,
        bond_types,
        bond_src_idxs,
        bond_dst_idxs,
    ):
        """Builds a rdkit molecule from the given atom and bond information."""
        # create a rdkit molecule and add atoms to it
        mol = Chem.RWMol()
        for atom_type, charge in zip(atom_types, atom_charges):
            a = Chem.Atom(atom_type)
            if charge != 0:
                a.SetFormalCharge(int(charge))
            mol.AddAtom(a)

        # add bonds to rdkit molecule
        for bond_type, src_idx, dst_idx in zip(
            bond_types, bond_src_idxs, bond_dst_idxs
        ):
            src_idx = int(src_idx)
            dst_idx = int(dst_idx)
            mol.AddBond(src_idx, dst_idx, self.bond_type_map[bond_type])

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            return None

        # Set coordinates
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            x, y, z = positions[i]
            x, y, z = float(x), float(y), float(z)
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf)

        return mol

    def build_traj(self, ep_traj=False, lig=True, prot=False, pharm=False):
        if self.traj is None:
            raise ValueError("No trajectory data available.")

        if not any([lig, prot, pharm]):
            raise ValueError("at least one of lig, prot, or pharm must be True.")

        g_dummy = copy_graph(self.g, n_copies=1)[0]

        traj_keys = list(self.traj.keys())
        if ep_traj:
            for k in traj_keys:
                if "pred" in k:
                    test_key = k
                    break
        else:
            test_key = traj_keys[0]

        n_frames = self.traj[test_key].shape[0]
        if lig:
            # TODO: this was so that we can align trajectory frames to the last frame, need to do this
            lig_x_final = self.traj["lig_x"][-1]

        traj_mols = defaultdict(list)
        for frame_idx in range(n_frames):
            # place the current traj frame on the dummy graph as the t=1 values
            for m_name in self.traj.keys():
                if "pred" in m_name:
                    continue

                m = name_to_modality(m_name)

                if m.is_node:
                    data_src = g_dummy.nodes[m.entity_name].data
                else:
                    data_src = g_dummy.edges[m.entity_name].data

                if ep_traj:
                    traj_key = f"{m.name}_pred"
                else:
                    traj_key = m.name

                data_src[f"{m.data_key}_1"] = self.traj[traj_key][frame_idx]

            if lig:
                ligdata = self.extract_ligdata_from_graph(
                    g=g_dummy, ctmc_mol=self.ctmc_mol, show_fake_atoms=True
                )
                rdkit_mol = self.build_molecule(*ligdata)
                traj_mols["lig"].append(rdkit_mol)
                
            if prot:
                bt_arr = self.get_protein_array(g=g_dummy)
                traj_mols["prot"].append(bt_arr)

            if pharm:
                traj_mols['pharm'].append(self.get_pharmacophore_from_graph(g=g_dummy, kind="predicted"))
                
        return traj_mols
    
    def get_pharmacophore_from_graph(self, g=None, kind="predicted", xyz=False):
        if g is None:
            g = self.g

        if kind == "predicted":
            suffix = "1"
        elif kind == 'gt':
            suffix = "1_true"
        else:
            raise ValueError("kind must be either 'predicted' or 'gt'.")

        coords = g.nodes['pharm'].data[f'x_{suffix}'].numpy()
        pharm_types_idx = g.nodes['pharm'].data[f'a_{suffix}'].numpy().tolist()
        pharm_types = [ constants.ph_idx_to_type[idx] for idx in pharm_types_idx ]
        pharm_types_elems = [ constants.ph_idx_to_elem[idx] for idx in pharm_types_idx ] 

        if xyz:
            return pharm_to_xyz(coords, pharm_types_elems)
        else:
            return {
                'coords': coords,
                'types': pharm_types,
                'types_idx': pharm_types_idx,
                'types_elems': pharm_types_elems,
            }
    
    def write_ligand(self, output_file: str, 
                     trajectory: bool = False, 
                     endpoint: bool = False, 
                     ground_truth: bool = True,
                     g = None):
        """Write a ligand or a ligand trajectory to an sdf file."""
        output_file = Path(output_file)
        if not output_file.suffix == ".sdf":
            raise ValueError("Output file must have .sdf extension.")
        if trajectory:
            mols = self.build_traj(ep_traj=endpoint, lig=True)['lig']
        elif ground_truth:
            mols = [self.get_gt_ligand(g=g)]
        else:
            mols = [self.get_rdkit_ligand()]
        write_mols_to_sdf(mols, str(output_file))

    def write_protein(self, 
            output_file: str, 
            trajectory: bool = False, 
            endpoint: bool = False,
            ground_truth: bool = False,
        ):
        """Write a protein or a protein trajectory to a cif file."""
        output_file = Path(output_file)
        if not output_file.suffix == ".cif":
            raise ValueError("Output file must have .cif extension.")
        if trajectory:
            arrs = self.build_traj(ep_traj=endpoint, prot=True)['prot']
        else:
            arrs = [self.get_protein_array(reference=ground_truth)]
        
        write_arrays_to_cif(arrs, str(output_file))
        
    
    def write_protein_pdb(self, 
            output_dir: str,
            filename: str,
            ground_truth: bool = False,
            
        ):
        """Write a protein a PDB file."""
        output_dir = Path(output_dir)
        arrs = [self.get_protein_array(reference=ground_truth)]
        write_arrays_to_pdb(arrs, output_dir, filename)


    def write_pharmacophore(self, 
        output_file, 
        trajectory: bool = False, 
        endpoint: bool = False,
        ground_truth: bool = False,
        g=None):

        output_file = Path(output_file)
        if not output_file.suffix == ".xyz":
            raise ValueError("Output file must have .xyz extension.")
        
        if trajectory:
            pharms = self.build_traj(ep_traj=endpoint, pharm=True)['pharm']
            pharms = [ pharm_to_xyz(pharm['coords'], pharm['types_elems']) for pharm in pharms ]
        else:
            kind = 'gt' if ground_truth else 'predicted'
            pharms = [self.get_pharmacophore_from_graph(g=g, kind=kind, xyz=True)]

        xyz_content =''.join(pharms)
        with open(output_file, 'w') as f:
            f.write(xyz_content)

    def compute_valencies(self):
        """Compute the valencies of every atom in the molecule. Returns a tensor of shape (num_atoms,)."""
        n_atoms = self.get_n_lig_atoms()
        _, _, _, bond_types, bond_src_idxs, bond_dst_idxs = (
            self.extract_ligdata_from_graph()
        )
        adj = torch.zeros((n_atoms, n_atoms))
        adjusted_bond_types = bond_types.clone().float()
        adjusted_bond_types[adjusted_bond_types == 4] = 1.5
        adjusted_bond_types = adjusted_bond_types.float()
        adj[bond_src_idxs, bond_dst_idxs] = adjusted_bond_types
        adj[bond_dst_idxs, bond_src_idxs] = adjusted_bond_types
        valencies = torch.sum(adj, dim=-1).long()
        return valencies

def write_arrays_to_cif(arrays, filename):
    cif_file = CIFFile()
    arr_stack = struc.stack(arrays)
    struc.io.pdbx.set_structure(cif_file, arr_stack, include_bonds=True)
    cif_file.write(filename)

def write_arrays_to_pdb(arrays, output_dir, filename):
    for i, arr in enumerate(arrays):
        out_path = output_dir / f"{filename}_{i}.pdb"
        pdb_file = PDBFile()
        pdb_file.set_structure(arr)
        pdb_file.write(str(out_path))

def pharm_to_xyz(pos: torch.Tensor, pharm_elements: List[str]):
    out = f'{len(pos)}\n'
    for i in range(len(pos)):
        elem = pharm_elements[i]
        out += f"{elem} {pos[i, 0]:.3f} {pos[i, 1]:.3f} {pos[i, 2]:.3f}\n"
    return out


def write_mols_to_sdf(mols: List[Chem.Mol], filename: Union[str, Path]):
    """Write a list of rdkit molecules to an sdf file."""
    filename = Path(filename)
    if not filename.suffix == ".sdf":
        raise ValueError("Output file must have .sdf extension.")
    sdwriter = Chem.SDWriter(str(filename))
    sdwriter.SetKekulize(False)
    for mol in mols:
        if mol is not None:
            sdwriter.write(mol)
    sdwriter.close()

def move_feats_to_t1(task_name: str, g: dgl.DGLHeteroGraph, t: str = '0'):
    task = task_name_to_class(task_name)
    for m in task.modalities_present:

        num_entries = g.num_nodes(m.entity_name) if m.is_node else g.num_edges(m.entity_name)
        if num_entries == 0:
            continue

        data_src = g.nodes if m.is_node else g.edges
        dk = m.data_key
        en = m.entity_name

        if t == '0' and m in task.modalities_fixed:
            data_to_copy = data_src[en].data[f'{dk}_1_true']
        else:
            data_to_copy = data_src[en].data[f'{dk}_{t}']

        data_src[en].data[f'{dk}_1'] = data_to_copy

    return g
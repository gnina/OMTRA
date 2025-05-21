import dgl
import torch
from typing import Tuple, List, Union, Dict
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
from copy import deepcopy
from omtra.tasks.modalities import name_to_modality
from collections import defaultdict

'''
def get_upper_edge_mask(g: dgl.DGLHeteroGraph, etype: str):
    """Returns a boolean mask for the edges that lie in the upper triangle of the adjacency matrix for each molecule in the batch."""
    # this algorithm assumes that the edges are ordered such that the upper triangle edges come first, followed by the lower triangle edges for each graph in the batch
    # and then those graph-wise edges are concatenated together
    # you can see that this is indeed how the edges are constructed by inspecting data_processing.dataset.MoleculeDataset.__getitem__
    edges_per_mol = g.batch_num_edges(etype=etype)
    ul_pattern = torch.tensor([1, 0]).repeat(g.batch_size).to(g.device)
    n_edges_pattern = (edges_per_mol / 2).int().repeat_interleave(2)
    upper_edge_mask = ul_pattern.repeat_interleave(n_edges_pattern).bool()
    return upper_edge_mask
'''


def get_upper_edge_mask(g: dgl.DGLHeteroGraph, etype: str):
    src, dst = g.edges(etype=etype)
    return src < dst


def get_node_batch_idxs_ntype(g: dgl.DGLHeteroGraph, ntype: str):
    """Returns a tensor of integers indicating which graph each node belongs to for a given node type."""
    node_batch_idx = torch.arange(g.batch_size, device=g.device)
    node_batch_idx = node_batch_idx.repeat_interleave(g.batch_num_nodes(ntype=ntype))
    return node_batch_idx


def get_edge_batch_idxs_etype(g: dgl.DGLHeteroGraph, etype: str):
    """Returns a tensor of integers indicating which batch each edge belongs to."""
    edge_batch_idx = torch.arange(g.batch_size, device=g.device)
    edge_batch_idx = edge_batch_idx.repeat_interleave(g.batch_num_edges(etype=etype))
    return edge_batch_idx


def get_node_batch_idxs(g: dgl.DGLHeteroGraph):
    node_batch_idxs = {}
    for ntype in g.ntypes:
        node_batch_idxs[ntype] = get_node_batch_idxs_ntype(g, ntype)
    return node_batch_idxs


def get_edge_batch_idxs(g: dgl.DGLHeteroGraph):
    edge_batch_idxs = {}
    for etype in g.etypes:
        edge_batch_idxs[etype] = get_edge_batch_idxs_etype(g, etype)
    return edge_batch_idxs


def get_batch_idxs(g: dgl.DGLHeteroGraph) -> Tuple[dict, dict]:
    """Returns two tensors of integers indicating which molecule each node and edge belongs to."""
    node_batch_idx = get_node_batch_idxs(g)
    edge_batch_idx = get_edge_batch_idxs(g)
    return node_batch_idx, edge_batch_idx


def get_batch_info(g: dgl.DGLHeteroGraph) -> Tuple[dict, dict]:
    batch_num_nodes = {}
    for ntype in g.ntypes:
        batch_num_nodes[ntype] = g.batch_num_nodes(ntype)

    batch_num_edges = {}
    for etype in g.canonical_etypes:
        batch_num_edges[etype] = g.batch_num_edges(etype)

    return batch_num_nodes, batch_num_edges


def get_edges_per_batch(
    edge_node_idxs: torch.Tensor, batch_size: int, node_batch_idxs: torch.Tensor
):
    device = edge_node_idxs.device
    batch_idxs = torch.arange(batch_size, device=device)
    batches_with_edges, edges_per_batch = torch.unique_consecutive(
        node_batch_idxs[edge_node_idxs], return_counts=True
    )
    edges_per_batch_full = torch.zeros_like(batch_idxs)
    edges_per_batch_full[batches_with_edges] = edges_per_batch
    return edges_per_batch_full


def copy_graph(g: dgl.DGLHeteroGraph, n_copies: int) -> List[dgl.DGLHeteroGraph]:
    """Create n_copies copies of an unbatched DGL heterogeneous graph."""

    # get edge indicies
    e_idxs_dict = {}
    for etype in g.canonical_etypes:
        e_idxs_dict[etype] = g.edges(form="uv", etype=etype)

    # get number of nodes
    num_nodes_dict = {}
    for ntype in g.ntypes:
        num_nodes_dict[ntype] = g.num_nodes(ntype=ntype)

    # make copies of graph
    g_copies = [
        dgl.heterograph(e_idxs_dict, num_nodes_dict=num_nodes_dict, device=g.device)
        for _ in range(n_copies)
    ]

    # transfer over node features
    for ntype in g.ntypes:
        for feat_name in g.nodes[ntype].data.keys():
            src_feat = (
                g.nodes[ntype].data[feat_name].detach()
            )  # get the feature on the source graph

            # add a clone to each copy
            for copy_idx in range(n_copies):
                g_copies[copy_idx].nodes[ntype].data[feat_name] = src_feat.clone()

    # transfer over edge features
    for etype in g.canonical_etypes:
        for feat_name in g.edges[etype].data.keys():
            src_feat = g.edges[etype].data[feat_name].detach()
            for copy_idx in range(n_copies):
                g_copies[copy_idx].edges[etype].data[feat_name] = src_feat.clone()

    return g_copies


def build_lig_edge_idxs(n_atoms: int) -> torch.Tensor:
    """Generate edge indicies for lig_to_lig; a fully-connected graph but with upper and lower triangle edges separated."""
    upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)

    # get lower triangle edges by swapping source and destination of upper_edge_idxs
    lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

    edges = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
    return edges


class SampledSystem:
    """
    Convert a DGLGraph into objects ready for evaluation.
    """

    def __init__(
        self,
        g: dgl.DGLHeteroGraph,
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
    ):
        self.g = g
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

        if self.fake_atoms:
            self.ligand_atom_type_map = deepcopy(self.ligand_atom_type_map)
            self.ligand_atom_type_map.append("Sn")  # fake atoms appear as Sn

        if self.ctmc_mol:
            self.ligand_atom_type_map = deepcopy(self.ligand_atom_type_map)
            self.ligand_atom_type_map.append("Se")  # masked atoms appear as Se

    def to(self, device: str):
        self.g = self.g.to(device)

        if self.traj:
            for k in self.traj:
                self.traj[k] = self.traj[k].to(device)

        return self

    def get_n_lig_atoms(self) -> int:
        n_lig_atoms = self.g.num_nodes(ntype="lig")
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
        if (
            bond_src_idxs is not None
            and bond_dst_idxs is not None
            and bond_types is not None
        ):
            bond_list = self.build_bond_list(bond_src_idxs, bond_dst_idxs, bond_types, atom_array.array_length())
            atom_array.bonds = bond_list
        else:
            atom_array.bonds = struc.connect_via_distances(atom_array)
            
        return atom_array

    def get_protein_array(self, g=None, reference: bool = False):
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
        )
        return arr

    def get_rdkit_ligand(self) -> Union[None, Chem.Mol]:
        ligdata = self.extract_ligdata_from_graph(ctmc_mol=self.ctmc_mol)
        rdkit_mol = self.build_molecule(*ligdata)
        return rdkit_mol

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
    ):
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
        positions = lig_g.ndata["x_1"]

        # extract node-level features
        positions = lig_g.ndata["x_1"]
        atom_types = lig_g.ndata["a_1"]
        atom_types = [atom_type_map[int(atom)] for atom in atom_types]

        if self.exclude_charges:
            atom_charges = None
        else:
            charge_data = lig_g.ndata["c_1"].clone()

            # set masked charges to 0
            if ctmc_mol:
                masked_charge = charge_data == len(self.charge_map)
                neutral_index = self.charge_map.index(0)
                charge_data[masked_charge] = neutral_index

            atom_charges = [self.charge_map[int(charge)] for charge in charge_data]

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

        coords = self.g.nodes["prot_atom"].data[f"x_{feat_suffix}"].numpy()
        atypes = self.g.nodes["prot_atom"].data[f"a_1_true"].numpy()
        atom_type_map_array = np.array(self.protein_atom_type_map, dtype="U3")
        atom_names = atom_type_map_array[atypes]

        eltypes = self.g.nodes["prot_atom"].data[f"e_1_true"].numpy()
        element_type_map_array = np.array(self.protein_element_map, dtype="U2")
        elements = element_type_map_array[eltypes]

        res_ids = self.g.nodes["prot_atom"].data["res_id"].numpy()
        res_types = self.g.nodes["prot_atom"].data["res_names"].numpy()
        res_type_map_array = np.array(self.residue_map, dtype="U3")
        res_names = res_type_map_array[res_types]

        chain_ids = self.g.nodes["prot_atom"].data["chain_id"].numpy()
        hetero = np.full_like(atom_names, False, dtype=bool)
        return coords, atom_names, elements, res_ids, res_names, chain_ids, hetero

    def extract_pharm_from_graph(self, g=None):
        if g is None:
            g = self.g

        coords = g.nodes["pharm"].data["x_1_true"].numpy()
        pharm_types = g.nodes["pharm"].data["a_1_true"].numpy()
        pharm_vecs = g.nodes["pharm"].data["v_1_true"].numpy()

        return coords, pharm_types, pharm_vecs

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

        if pharm:
            raise NotImplementedError(
                "Pharmacophore trajectory building not implemented yet."
            )
        if prot:
            print("warning: protein trajectory building being tested")

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
                
        return traj_mols

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

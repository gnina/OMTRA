import dgl
import torch
from typing import Tuple, List, Union
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
        fake_atoms: bool = False,  # whether the molecule contains fake atoms,
        exclude_charges: bool = False,
        ligand_atom_type_map: List[str] = lig_atom_type_map,
        npnde_atom_type_map: List[str] = npnde_atom_type_map,
        protein_atom_type_map: List[str] = protein_atom_map,
        residue_map: List[str] = residue_map,
        bond_type_map: List[str] = bond_type_map,
        charge_map: List[int] = charge_map,
        protein_element_map: List[str] = protein_element_map,
    ):
        self.g = g
        self.fake_atoms = fake_atoms
        self.exclude_charges = exclude_charges
        self.ligand_atom_type_map = ligand_atom_type_map
        self.npnde_atom_type_map = npnde_atom_type_map
        self.protein_atom_type_map = protein_atom_type_map
        self.residue_map = residue_map
        self.bond_type_map = bond_type_map
        self.charge_map = charge_map
        self.protein_element_map = protein_element_map
    
    def get_n_lig_atoms(self) -> int:
        n_lig_atoms = self.g.num_nodes(ntype="lig")
        return n_lig_atoms 

    def get_atom_arr(self, reference: bool = False):
        """
        Get the system data represented as Biotite AtomArray
        :return: Biotite AtomArray
        """
        if reference:
            feat_suffix = "1_true"
        else:
            feat_suffix = "1"
        ntypes = ["prot_atom", "lig", "npnde"]
        n_atoms = 0
        for ntype in ntypes:
            n_atoms += self.g.num_nodes(ntype=ntype)

        atom_array = struc.AtomArray(n_atoms)

        coords = []
        atom_names = []
        elements = []
        res_ids = []
        res_names = []
        chain_ids = []

        for ntype in ntypes:
            if self.g.num_nodes(ntype=ntype) == 0:
                continue
            coords.append(self.g.nodes[ntype].data[f"x_{feat_suffix}"].numpy())

            if ntype == "prot_atom":
                atypes = self.g.nodes[ntype].data[f"a_1_true"].numpy()
                atom_type_map_array = np.array(self.protein_atom_type_map, dtype=object)
                anames = atom_type_map_array[atypes]
                atom_names.append(anames)

                eltypes = self.g.edges[ntype].data[f"e_1_true"].numpy()
                element_type_map_array = np.array(
                    self.protein_element_map, dtype=object
                )
                elnames = element_type_map_array[eltypes]
                elements.append(elnames)

                res_ids.append(self.g.nodes[ntype].data["res_id"].numpy())
                res_types = self.g.nodes[ntype].data["res_name"].numpy()
                res_type_map_array = np.array(self.residue_map, dtype=object)
                resnames = res_type_map_array[res_types]
                res_names.append(resnames)

                chain_ids.append(self.g.nodes[ntype].data["chain_id"].numpy())

            if ntype == "lig":
                atypes = self.g.nodes[ntype].data[f"a_{feat_suffix}"].numpy()
                atom_type_map_array = np.array(self.ligand_atom_type_map, dtype=object)
                anames = atom_type_map_array[atypes]
                atom_names.append(anames)
                elements.append(anames)
                res_id = np.max(res_ids) + 1
                res_ids.append(np.full_like(atypes, res_id, dtype=int))
                res_names.append(np.full_like(atypes, "LIG", dtype=object))
                chain_ids.append(np.full_like(atypes, "LIG", dtype=object))

            if ntype == "npnde":
                atypes = self.g.nodes[ntype].data[f"a_{feat_suffix}"].numpy()
                atom_type_map_array = np.array(self.npnde_atom_type_map, dtype=object)
                anames = atom_type_map_array[atypes]
                atom_names.append(anames)
                elements.append(anames)

                res_id = np.max(res_ids) + 1
                res_ids.append(np.full_like(atypes, res_id, dtype=int))
                res_names.append(np.full_like(atypes, "NPNDE", dtype=object))
                # TODO: might need to modify dataset to track individual npnde chains
                chain_ids.append(np.full_like(atypes, "NPNDE", dtype=object))

        atom_array.coord = coords

        atom_array.set_annotation("atom_name", atom_names)
        atom_array.set_annotation("element", elements)
        atom_array.set_annotation("res_id", res_ids)
        atom_array.set_annotation("res_name", res_names)
        atom_array.set_annotation("chain_id", chain_ids)

        return atom_array

    def get_rdkit_ligand(self) -> Union[None, Chem.Mol]:
        (
            positions,
            atom_types,
            atom_charges,
            bond_types,
            bond_src_idxs,
            bond_dst_idxs,
        ) = self.extract_ligdata_from_graph()
        rdkit_mol = self.build_molecule(
            positions,
            atom_types,
            atom_charges,
            bond_src_idxs,
            bond_dst_idxs,
            bond_types,
        )
        return rdkit_mol

    def extract_ligdata_from_graph(
        self,
        ctmc_mol: bool = False,
        show_fake_atoms: bool = False,
    ):
        atom_type_map = list(self.ligand_atom_type_map)
        lig_g = dgl.node_type_subgraph(self.g, ntypes=["lig"])

        lig_ndata_feats = list(lig_g.nodes["lig"].data.keys())
        lig_edata_feats = list(lig_g.edges["lig_to_lig"].data.keys())

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
            atom_charges = (
                self.charge_map[int(charge)] for charge in lig_g.ndata["c_1"]
            )

        # get bond types and atom indicies for every edge, convert types from simplex to integer
        bond_types = lig_g.edata["e_1"]
        bond_types[bond_types == 5] = 0  # set masked bonds to 0
        bond_src_idxs, bond_dst_idxs = lig_g.edges()

        # get just the upper triangle of the adjacency matrix
        upper_edge_mask = get_upper_edge_mask(self.g, etype="lig_to_lig")
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

    def build_molecule(
        self,
        positions,
        atom_types,
        atom_charges,
        bond_src_idxs,
        bond_dst_idxs,
        bond_types,
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
    
    def compute_valencies(self):
        """Compute the valencies of every atom in the molecule. Returns a tensor of shape (num_atoms,)."""
        n_atoms = self.get_n_lig_atoms()
        _, _, _, bond_types, bond_src_idxs, bond_dst_idxs = self.extract_ligdata_from_graph()
        adj = torch.zeros((n_atoms, n_atoms))
        adjusted_bond_types = bond_types.clone()
        adjusted_bond_types[adjusted_bond_types == 4] = 1.5
        adjusted_bond_types = adjusted_bond_types.float()
        adj[bond_src_idxs, bond_dst_idxs] = adjusted_bond_types
        adj[bond_dst_idxs, bond_src_idxs] = adjusted_bond_types
        valencies = torch.sum(adj, dim=-1).long()
        return valencies

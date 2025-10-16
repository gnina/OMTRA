from typing import Dict, List, Optional
import numpy as np
from rdkit import Chem
import dgl
import torch
import traceback
from multiprocessing import Pool
from dataclasses import dataclass
from typing import Tuple, Union
from collections import defaultdict
import math
from omtra.constants import lig_atom_type_map, extra_feats_map
from omtra.data.condensed_atom_typing import CondensedAtomTyper

from omtra.utils.misc import combine_tcv_counts, bad_mol_reporter

from rdkit import RDLogger

# Disable RDKit warnings and errors
RDLogger.DisableLog('rdApp.*')

@dataclass
class MolXACE:
    x: Optional[Union[np.ndarray, torch.Tensor]] = None
    a: Optional[Union[np.ndarray, torch.Tensor]] = None
    c: Optional[Union[np.ndarray, torch.Tensor]] = None
    e: Optional[Union[np.ndarray, torch.Tensor]] = None  # corresponds to edge attributes (bond orders)

    impl_H: Optional[Union[np.ndarray, torch.Tensor]] = None
    aro: Optional[Union[np.ndarray, torch.Tensor]] = None
    hyb: Optional[Union[np.ndarray, torch.Tensor]] = None
    ring: Optional[Union[np.ndarray, torch.Tensor]] = None
    chiral: Optional[Union[np.ndarray, torch.Tensor]] = None
    rdkit_mol: Optional[Chem.Mol] = None  # RDKit molecule

    cond_a: Optional[Union[np.ndarray, torch.Tensor]] = None    # condensed atom typing

    edge_idxs: Optional[Union[np.ndarray, torch.Tensor]] = None   # corresponds to edge index (upper triangular edges)
    tcv_counts: Optional[dict] = None
    failure_mode: Optional[str] = None


    def sparse_to_dense(self):
        """Converts the sparse representation of the molecule to a dense representation."""
        if self.edge_idxs is None or self.e is None:
            raise ValueError("bond_idxs and bond_types must be set to convert to dense representation.")
        
        # Create a dense adjacency matrix
        if self.cond_a is not None:
            n_atoms = self.cond_a.shape[0]
        else:
            n_atoms = self.a.shape[0]
            
        e_tensor = self.e
        edge_idxs_tensor = self.edge_idxs
        
        adj = torch.zeros((n_atoms, n_atoms), dtype=e_tensor.dtype)
        adj[edge_idxs_tensor[:, 0], edge_idxs_tensor[:, 1]] = e_tensor
        
        # Get the upper-triangular indices
        upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
        
        # Get the edge labels for the upper-triangular edges
        upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]

        lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))
        
        # Create the final edge representation
        edge_idxs = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
        bond_types = torch.cat((upper_edge_labels, upper_edge_labels))

        if self.cond_a is not None:     # condensed atom typing
            dense_xace = MolXACE(
                x=self.x,
                cond_a=self.cond_a,
                e=bond_types,
                edge_idxs=edge_idxs
            )

        elif self.impl_H is not None:   # Extra features
            dense_xace = MolXACE(
                x=self.x,
                a=self.a,
                c=self.c,
                impl_H=self.impl_H,
                aro=self.aro,
                hyb=self.hyb,
                ring=self.ring,
                chiral=self.chiral,
                e=bond_types,
                edge_idxs=edge_idxs
            )

        else:                       # regular 
            dense_xace = MolXACE(
                x=self.x,
                a=self.a,
                c=self.c,
                e=bond_types,
                edge_idxs=edge_idxs
            )
            
        return dense_xace


class MoleculeTensorizer():
    """Class for converting RDKit molecules to tensors for use in graph neural networks."""

    def __init__(self, atom_map: List[str], n_cpus=1):
        self.n_cpus = n_cpus
        self.atom_map = atom_map
        self.atom_map_dict = {atom: i for i, atom in enumerate(atom_map)}

        if self.n_cpus == 1:
            self.pool = None
        else:
            self.pool = Pool(self.n_cpus)

        self.explicit_hydrogens = False

    def featurize_molecules(self, molecules) -> Tuple[MolXACE, List[int], np.ndarray]:
        """Featurizes a list of RDKit molecules into MolXACE data classes."""
        valid_molecules = []
        failed_idxs = []
        failure_counts = defaultdict(int)

        if self.n_cpus == 1:
            for idx, molecule in enumerate(molecules):
                molxace = rdmol_to_xace(molecule, self.atom_map_dict, self.explicit_hydrogens)
                if molxace.failure_mode is not None:
                    failed_idxs.append(idx)
                    failure_counts[molxace.failure_mode] += 1
                else:
                    valid_molecules.append(molxace)
        else:
            args = [(molecule, self.atom_map_dict, self.explicit_hydrogens) for molecule in molecules]
            results = self.pool.starmap(rdmol_to_xace, args)
            for idx, molxace in enumerate(results):
                if molxace is None or molxace.x is None:
                    failed_idxs.append(idx)
                    failure_counts[molxace.failure_mode] += 1
                else:
                    valid_molecules.append(molxace)

        num_failed = len(failed_idxs)

        # combine unique valencies from all valid molecules
        tcv_counts = combine_tcv_counts([molxace.tcv_counts for molxace in valid_molecules])

        return valid_molecules, failed_idxs, failure_counts, tcv_counts


def rdmol_to_xace(molecule: Chem.rdchem.Mol, atom_map_dict: Dict[str, int], explicit_hydrogens=False) -> MolXACE:
    """Converts an RDKit molecule to a MolXACE data class containing the positions, atom types, atom charges, bond types, bond indexes, and unique valencies."""
    try:
        Chem.SanitizeMol(molecule)
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except Exception as e:
        traceback.print_exc()
        return MolXACE(failure_mode="sanitization/kekulization")

    # remove hydrogens if explicit hydrogens are not desired
    if not explicit_hydrogens:
        molecule = Chem.RemoveHs(molecule)
        try:
            Chem.Kekulize(molecule, clearAromaticFlags=True)
        except Exception as e:
            traceback.print_exc()
            return MolXACE(failure_mode="kekulization")

    num_fragments = len(Chem.GetMolFrags(molecule, sanitizeFrags=False))
    if num_fragments > 1:
        return MolXACE(failure_mode="multiple fragments")

    try:
        positions = molecule.GetConformer().GetPositions()
    except Exception as e:
        traceback.print_exc()
        return MolXACE(failure_mode="positions")

    num_atoms = molecule.GetNumAtoms()
    atom_types = np.zeros(num_atoms, dtype=int)
    atom_charges = np.zeros(num_atoms, dtype=int)
    for i, atom in enumerate(molecule.GetAtoms()):
        try:
            atom_types[i] = atom_map_dict[atom.GetSymbol()]
        except KeyError:
            # print(f"Atom {atom.GetSymbol()} not in atom map", flush=True)
            return MolXACE(failure_mode="atom map")

        atom_charges[i] = atom.GetFormalCharge()

    # get one-hot encoded bonds (only existing bonds) using the upper-triangular portion of the adjacency matrix
    adj = Chem.rdmolops.GetAdjacencyMatrix(molecule, useBO=True)
    
    arom_idxs = np.where(adj == 1.5)
    assert arom_idxs[0].size == 0, "Aromatic bonds should not be present in the adjacency matrix."

    edge_index = np.nonzero(np.triu(adj))  # tuple of arrays for upper triangle indices
    if len(edge_index[0]) == 0:
        # if no bonds exist, create empty arrays with proper shape
        bond_types = np.array([], dtype=int)
        bond_idxs = np.empty((0, 2), dtype=int)
    else:
        bond_idxs = np.stack(edge_index, axis=-1)  # shape (n_edges, 2)
        bond_types = adj[bond_idxs[:, 0], bond_idxs[:, 1]].astype(np.int32)

    # compute valencies and unique valencies information
    valencies = np.sum(adj, axis=1)

    # check for trivalent oxygen
    if np.any((atom_types == atom_map_dict["O"]) & (valencies == 3)):
        return MolXACE(failure_mode="trivalent oxygen")

    tcv = np.stack([atom_types, atom_charges, valencies], axis=1).astype(np.int8)
    unique_valencies, counts = np.unique(tcv, axis=0, return_counts=True)
    tcv_counts = {}
    for row, count in zip(unique_valencies, counts):
        tcv_counts[tuple(row)] = count

    return MolXACE(
        x=torch.from_numpy(positions).float(),
        a=torch.from_numpy(atom_types).long(),
        c=torch.from_numpy(atom_charges).long(),
        e=torch.from_numpy(bond_types).long(),
        edge_idxs=torch.from_numpy(bond_idxs).long(),
        tcv_counts=tcv_counts,
        rdkit_mol=molecule
    )

def add_fake_atoms(mol: MolXACE, fake_atom_p: float, cond_a_typer: CondensedAtomTyper=None):
    n_real_atoms, _ = mol.x.shape
    max_num_fake_atoms = math.ceil(n_real_atoms*fake_atom_p)
    num_fake_atoms = torch.randint(low=0, high=max_num_fake_atoms, size=(1,))

    anchor_atom_idxs = torch.randint(low=0, high=n_real_atoms, size=(num_fake_atoms,))
    fake_atom_positions = mol.x[anchor_atom_idxs]
    # TODO: think about how to decide fake atom positions
    # currently: gaussians around anchor atom 
    # possibilities: collapse on nearest atom, random placement in molecule interior,
    # fixed distance from acnhor atom
    fake_atom_positions = fake_atom_positions + torch.randn_like(fake_atom_positions)
    mol.x = torch.cat((mol.x, fake_atom_positions), dim=0)

    # TODO: dataset class initialize CondensedAtomTyper and pass to add_fake_atoms. Right now done by Pharmit dataclass

    if mol.cond_a is not None:  # condensed atom typing
        fake_atom_cond_a =  torch.full_like(
            mol.cond_a[anchor_atom_idxs], 
            fill_value=cond_a_typer.fake_atom_idx)
        mol.cond_a = torch.cat((mol.cond_a, fake_atom_cond_a), dim=0)

    else: 
        fake_atom_charges = torch.zeros_like(mol.c[anchor_atom_idxs])
        fake_atom_types = torch.full_like(mol.a[anchor_atom_idxs], fill_value=len(lig_atom_type_map))

        # combine fake atoms with real atoms
        mol.a = torch.cat((mol.a, fake_atom_types), dim=0)
        mol.c = torch.cat((mol.c, fake_atom_charges), dim=0)
        
        if mol.impl_H is not None:  # extra features
            for extra_feat in extra_feats_map.keys():
                old_feat = getattr(mol, extra_feat)     # Old atom extra features
                fake_atom_feats = torch.zeros_like(old_feat[anchor_atom_idxs])
                new_feat = torch.cat((old_feat, fake_atom_feats), dim=0)    
                setattr(mol, extra_feat, new_feat)      # Add fake atom extra features
    
    return mol




def add_k_hop_edges(x, a, c, e, edge_idxs, k=2):
    n_atoms = x.shape[0]
    
    edge_to_bond = {}
    for i in range(edge_idxs.shape[0]):
        src, dst = edge_idxs[i, 0].item(), edge_idxs[i, 1].item()
        edge_to_bond[(src, dst)] = e[i].item()
    
    src_list, dst_list = [], []
    bond_types = []
    
    for (src, dst), bond_type in edge_to_bond.items():
        if (dst, src) not in edge_to_bond:
            src_list.append(dst)
            dst_list.append(src)
            bond_types.append(bond_type)
    
    if src_list:
        reverse_edges = torch.tensor(list(zip(src_list, dst_list)), dtype=edge_idxs.dtype)
        bond_tensor = torch.tensor(bond_types, dtype=e.dtype)
        
        edge_idxs = torch.cat([edge_idxs, reverse_edges], dim=0)
        e = torch.cat([e, bond_tensor])
        
        for i in range(len(src_list)):
            edge_to_bond[(src_list[i], dst_list[i])] = bond_types[i]
    
    g = dgl.graph((edge_idxs[:, 0], edge_idxs[:, 1]), num_nodes=n_atoms)
    
    k_hop_g = dgl.khop_graph(g, k)
    k_hop_edges = k_hop_g.edges()
    
    new_edges = []
    
    for i, j in zip(k_hop_edges[0].tolist(), k_hop_edges[1].tolist()):
        edge = (i, j)
        if edge not in edge_to_bond:
            new_edges.append([i, j])
            
            reverse_edge = (j, i)
            if reverse_edge not in edge_to_bond and [j, i] not in new_edges:
                new_edges.append([j, i])
    
    if not new_edges:
        print("Warning: No new edges found from k-hop graph")
        return x, a, c, e, edge_idxs.t()
    
    new_edge_tensor = torch.tensor(new_edges, dtype=edge_idxs.dtype)
    
    k_hop_edge_type = torch.zeros(len(new_edges), dtype=e.dtype)
    
    updated_edge_idxs = torch.cat([edge_idxs, new_edge_tensor], dim=0)
    updated_e = torch.cat([e, k_hop_edge_type])
    
    return x, a, c, updated_e, updated_edge_idxs.t()
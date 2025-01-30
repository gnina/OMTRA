from typing import Dict, List
import numpy as np
from rdkit import Chem
import torch

from multiprocessing import Pool

class MoleculeTensorizer():

    """Class for converting RDKit molecules to tensors for use in graph neural networks."""

    def __init__(self, atom_map: str, n_cpus=1):
        self.n_cpus = n_cpus
        self.atom_map = atom_map
        self.atom_map_dict = {atom: i for i, atom in enumerate(atom_map)}

        if self.n_cpus == 1:
            self.pool = None
        else:
            self.pool = Pool(self.n_cpus)

        if 'H' in atom_map:
            self.explicit_hydrogens = True
        else:    
            self.explicit_hydrogens = False

    def featurize_molecules(self, molecules):


        all_positions, all_atom_types, all_atom_charges, all_bond_types, all_bond_idxs = [], [], [], [], []

        if self.n_cpus == 1:
            for molecule in molecules:
                positions, atom_types, atom_charges, bond_types, bond_idxs = rdmol_to_xace(molecule, self.atom_map_dict)
                all_positions.append(positions)
                all_atom_types.append(atom_types)
                all_atom_charges.append(atom_charges)
                all_bond_types.append(bond_types)
                all_bond_idxs.append(bond_idxs)

        else:
            args = [(molecule, self.atom_map_dict) for molecule in molecules]
            results = self.pool.starmap(rdmol_to_xace, args)
            for positions, atom_types, atom_charges, bond_types, bond_idxs in results:
                all_positions.append(positions)
                all_atom_types.append(atom_types)
                all_atom_charges.append(atom_charges)
                all_bond_types.append(bond_types)
                all_bond_idxs.append(bond_idxs)

        # find molecules that failed to featurize and count them
        num_failed = 0
        failed_idxs = []
        for i in range(len(all_positions)):
            if all_positions[i] is None:
                num_failed += 1
                failed_idxs.append(i)

        # remove failed molecules
        all_positions = [pos for i, pos in enumerate(all_positions) if i not in failed_idxs]
        all_atom_types = [atom for i, atom in enumerate(all_atom_types) if i not in failed_idxs]
        all_atom_charges = [charge for i, charge in enumerate(all_atom_charges) if i not in failed_idxs]
        all_bond_types = [bond for i, bond in enumerate(all_bond_types) if i not in failed_idxs]
        all_bond_idxs = [idx for i, idx in enumerate(all_bond_idxs) if i not in failed_idxs]

        return all_positions, all_atom_types, all_atom_charges, all_bond_types, all_bond_idxs, num_failed, failed_idxs



def rdmol_to_xace(molecule: Chem.rdchem.Mol, atom_map_dict: Dict[str, int], explicit_hydrogens=False):
    """Converts an rdkit molecule to a tuple of numpy arrays containing the positions, atom types, atom charges, bond types, and edge index."""

    # kekulize the molecule
    try:
        Chem.Kekulize(molecule)
    except Chem.KekulizeException as e:
        print(f"Kekulization failed for molecule {molecule.GetProp('_Name')}", flush=True)
        return None, None, None, None, None

    # if explicit_hydrogens is False, remove all hydrogens from the molecule
    if not explicit_hydrogens:
        molecule = Chem.RemoveHs(molecule)

    # get positions
    positions = molecule.GetConformer().GetPositions()

    # get atom elements as a string
    # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atom_types = np.zeros(molecule.GetNumAtoms(), dtype=int)
    atom_charges = np.zeros_like(atom_types)
    for i, atom in enumerate(molecule.GetAtoms()):
        try:
            atom_types[i] = atom_map_dict[atom.GetSymbol()]
        except KeyError:
            print(f"Atom {atom.GetSymbol()} not in atom map", flush=True)
            return None, None, None, None, None
        
        atom_charges[i] = atom.GetFormalCharge()

    # get one-hot encoded of existing bonds only (no non-existing bonds)
    adj = Chem.rdmolops.GetAdjacencyMatrix(molecule, useBO=True)
    edge_index = np.triu(adj).nonzero()  # upper triangular portion of adjacency matrix

    # note that because we take the upper-triangular portion of the adjacency matrix, there is only one edge per bond
    # at training time for every edge (i,j) in edge_index, we will also add edges (j,i)
    # we also only retain existing bonds, but then at training time we will add in edges for non-existing bonds
    bond_types = adj[edge_index[0], edge_index[1]]

    # compute the number of upper-edge pairs
    atom_types = atom_types

    # TODO: we should centralize the data typing for all our datasets
    # and also make more informed decisions
    atom_charges = atom_charges.astype(np.int32)
    edge_attr = bond_types.astype(np.int32)

    return positions, atom_types, atom_charges, edge_attr, edge_index, 


def sparse_to_dense(x, a, c, e, edge_idxs):
    """Converts a sparse xace ligand to a dense xace ligand.
    
    We are overloading the word "sparse" here. The graph representation in both the input
    and output is "sparse" in the classic sense of sparse tensors or sparse adjacency matrices.
    But this is not the sparse to dense conversion that is being referred to in the function name.

    When we store xace ligands to disk, we only record bond orders for existing bonds.
    If there is no bond, we don't record the edge or bond order on that edge. Moreover,
    we only record upper-edge pairs (where dst node index > src node index).
    Of course for training our model must predict the absence of bonds between atoms. 
    And we need edge labels for both directions (upper and lower triangle edges).
    So, here we need to modify the list of edges and the edge labels to include:
     - non-bonded edges 
     - upper and lower triangle edges
    This is what is meant by "sparse to dense". We should adopt better terminology; a future problem :p
    """
    # reconstruct ligand adjacency matrix
    n_atoms = x.shape[0]
    adj = torch.zeros((n_atoms, n_atoms), dtype=edge_idxs.dtype)

    # fill in the values of the adjacency matrix specified by e (bond orders)
    # this fills in bond orders for the upper triangle of the adjacency matrix
    adj[edge_idxs[:, 0], edge_idxs[:, 1]] = e

    # get upper triangle of adjacency matrix
    upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1) # has shape (2, n_upper_edges)
    upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]

    # get lower triangle edges by swapping source and destination of upper_edge_idxs
    lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

    # construct final edge representation: this includes upper-lower edges and unbonded edges
    edge_idxs = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
    e = torch.cat((upper_edge_labels, upper_edge_labels))
    return x, a, c, e, edge_idxs
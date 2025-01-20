from typing import Dict, List
import numpy as np
from rdkit import Chem

from multiprocessing import Pool

class MoleculeFeaturizer():

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
                positions, atom_types, atom_charges, bond_types, bond_idxs = rdmol_to_xae(molecule, self.atom_map_dict)
                all_positions.append(positions)
                all_atom_types.append(atom_types)
                all_atom_charges.append(atom_charges)
                all_bond_types.append(bond_types)
                all_bond_idxs.append(bond_idxs)

        else:
            args = [(molecule, self.atom_map_dict) for molecule in molecules]
            results = self.pool.starmap(rdmol_to_xae, args)
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

        return all_positions, all_atom_types, all_atom_charges, all_bond_types, all_bond_idxs, num_failed



def rdmol_to_xae(molecule: Chem.rdchem.Mol, atom_map_dict: Dict[str, int], explicit_hydrogens=True):
    """Converts an rdkit molecule to a tuple of numpy arrays containing the positions, atom types, bond types, and edge index."""

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
    positions = np.from_numpy(positions)

    # get atom elements as a string
    # atom_types_str = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atom_types = np.zeros(molecule.GetNumAtoms()).long()
    atom_charges = np.zeros_like(atom_types)
    for i, atom in enumerate(molecule.GetAtoms()):
        try:
            atom_types[i] = atom_map_dict[atom.GetSymbol()]
        except KeyError:
            print(f"Atom {atom.GetSymbol()} not in atom map", flush=True)
            return None, None, None, None, None
        
        atom_charges[i] = atom.GetFormalCharge()

    # get one-hot encoded of existing bonds only (no non-existing bonds)
    adj = np.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(molecule, useBO=True))
    edge_index = adj.triu().nonzero().contiguous() # upper triangular portion of adjacency matrix

    # note that because we take the upper-triangular portion of the adjacency matrix, there is only one edge per bond
    # at training time for every edge (i,j) in edge_index, we will also add edges (j,i)
    # we also only retain existing bonds, but then at training time we will add in edges for non-existing bonds

    bond_types = adj[edge_index[:, 0], edge_index[:, 1]]

    # count the number of pairs of atoms which are bonded
    n_bonded_pairs = edge_index.shape[0]
    edge_attr = bond_types

    # compute the number of upper-edge pairs
    atom_types = atom_types

    # TODO: we should centralize the data typing for all our datasets
    # and also make more informed decisions
    atom_charges = atom_charges.type(np.int32)
    edge_attr = bond_types.type(np.int32)

    return positions, atom_types, atom_charges, edge_attr, edge_index, #  bond_order_counts
from typing import Dict, List, Optional
import numpy as np
from rdkit import Chem
import torch
import traceback
from multiprocessing import Pool
from dataclasses import dataclass
from typing import Tuple
from collections import defaultdict

@dataclass
class MolXACE:
    positions: Optional[np.ndarray] = None
    atom_types: Optional[np.ndarray] = None
    atom_charges: Optional[np.ndarray] = None
    bond_types: Optional[np.ndarray] = None  # corresponds to edge attributes (bond orders)
    bond_idxs: Optional[np.ndarray] = None   # corresponds to edge index (upper triangular edges)
    unique_valencies: Optional[np.ndarray] = None
    failure_mode: Optional[str] = None


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
                if molxace is None or molxace.positions is None:
                    failed_idxs.append(idx)
                    failure_counts[molxace.failure_mode] += 1
                else:
                    valid_molecules.append(molxace)

        num_failed = len(failed_idxs)

        # combine unique valencies from all valid molecules
        unique_valencies = np.concatenate([molxace.unique_valencies for molxace in valid_molecules], axis=0)
        unique_valencies = np.unique(unique_valencies, axis=0)


        return valid_molecules, failed_idxs, failure_counts, unique_valencies


def rdmol_to_xace(molecule: Chem.rdchem.Mol, atom_map_dict: Dict[str, int], explicit_hydrogens=False) -> Optional[MolXACE]:
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
            print(f"Atom {atom.GetSymbol()} not in atom map", flush=True)
            return MolXACE(failure_mode="atom map")
        atom_charges[i] = atom.GetFormalCharge()

    # get one-hot encoded bonds (only existing bonds) using the upper-triangular portion of the adjacency matrix
    adj = Chem.rdmolops.GetAdjacencyMatrix(molecule, useBO=True)
    edge_index = np.nonzero(np.triu(adj))  # tuple of arrays for upper triangle indices
    if len(edge_index[0]) == 0:
        # if no bonds exist, create empty arrays with proper shape
        bond_types = np.array([], dtype=int)
        bond_idxs = np.empty((0, 2), dtype=int)
    else:
        bond_idxs = np.stack(edge_index, axis=-1)  # shape (n_edges, 2)
        bond_types = adj[bond_idxs[:, 0], bond_idxs[:, 1]].astype(np.int32)

    # compute valencies and unique valencies information
    # Note: torch simulation of np.sum; here we use numpy first then convert to torch
    valencies = np.sum(adj, axis=1)
    tcv = np.stack([atom_types, atom_charges, valencies], axis=1)
    unique_valencies = np.unique(tcv, axis=0)

    return MolXACE(
        positions=positions,
        atom_types=atom_types,
        atom_charges=atom_charges,
        bond_types=bond_types,
        bond_idxs=bond_idxs,
        unique_valencies=unique_valencies
    )


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
    n_atoms = x.shape[0]
    adj = torch.zeros((n_atoms, n_atoms), dtype=edge_idxs.dtype)
    adj[edge_idxs[:, 0], edge_idxs[:, 1]] = e
    upper_edge_idxs = torch.triu_indices(n_atoms, n_atoms, offset=1)
    upper_edge_labels = adj[upper_edge_idxs[0], upper_edge_idxs[1]]
    lower_edge_idxs = torch.stack((upper_edge_idxs[1], upper_edge_idxs[0]))

    # construct final edge representation: this includes upper-lower edges and unbonded edges
    edge_idxs = torch.cat((upper_edge_idxs, lower_edge_idxs), dim=1)
    e = torch.cat((upper_edge_labels, upper_edge_labels))
    return x, a, c, e, edge_idxs
import numpy as np
import dgl
import zarr

from rdkit import Chem
from rdkit.Chem import BRICS

from omtra.tasks.register import task_name_to_class
from omtra.eval.system import SampledSystem
from omtra.tasks.tasks import Task


def ligand_properties(mol: Chem.Mol) -> np.ndarray:
    """
    Parameters:
        mol (Chem.Mol): RDKit ligand

    Returns: 
        np.ndarray: Additional ligand features (n_atoms, 6)
    """

    implicit_Hs = []    # Number of implicit hydrogens (int)
    aromaticity = []    # Whether the atom is in an aromatic ring (binary flag)
    hybridization = []  # Hydridization (int)
    in_ring = []        # Whether the atom is in a ring (binary flag)
    chiral_center = []         # Whether the atom is a chiral center (binary flag)

    # Collect indices of chiral atoms
    try:
        chiral_centers = set(idx for idx, _ in Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    except:
        chiral_centers = set()

    for atom in mol.GetAtoms():
        implicit_Hs.append(atom.GetNumImplicitHs())
        aromaticity.append(int(atom.GetIsAromatic()))
        hybridization.append(int(atom.GetHybridization()))
        in_ring.append(int(atom.IsInRing()))
        chiral_center.append(int(atom.GetIdx() in chiral_centers))
    
    new_feats = np.array([
        implicit_Hs,
        aromaticity,
        hybridization,
        in_ring,
        chiral_center
    ], dtype=np.int8).T

    return new_feats


def fragment_molecule(mol: Chem.Mol) -> np.ndarray:
    """ 
    Parameters:
        mol (Chem.Mol): RDKit ligand

    Returns:
        np.ndarray: Index of the BRICS fragment for each atom (n_atoms, 1) 
    """

    broken = BRICS.BreakBRICSBonds(mol) # cut molecule at BRICS bonds and replace with dummy atoms labeled [*]

    # find connected components
    comps = Chem.GetMolFrags(broken, asMols=False)     # returns tuple of tuples. each tuple is a connected component

    # build mapping from each original atom to fragment
    N = mol.GetNumAtoms()
    atom_to_fragment = [-1] * N

    for frag_idx, comp in enumerate(comps):
        for ai in comp:
            atom = broken.GetAtomWithIdx(ai)
            if atom.GetSymbol() != "*" and ai < N: # not part of a BRICS bond
                atom_to_fragment[ai] = frag_idx

    atom_to_fragment = np.array(atom_to_fragment, dtype=np.int8)

    return atom_to_fragment[:, np.newaxis]


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


def dgl_to_rdkit(g):
    """ Converts one DGL molecule to RDKit ligand """

    task_name: str = 'denovo_ligand'
    task: Task = task_name_to_class(task_name)

    g = move_feats_to_t1(task_name, g, '1_true')
    rdkit_ligand = SampledSystem(g, task=task).get_rdkit_ligand()
    return rdkit_ligand


        
class BlockWriter:
    def __init__(self, store_path: str, array_name: str):

        # Open Pharmit Zarr store
        self.root = zarr.open(store_path, mode='r+')
        self.lig_node_group = self.root['ligand']

        # Check that Zarr array was correctly made
        if array_name not in self.lig_node_group:
            raise KeyError(f"Zarr array '{array_name}' not found in 'ligand' group.")

        self.new_feats_array = self.lig_node_group[array_name]


    def save_chunk(self, contig_idxs: np.ndarray, new_feats: np.ndarray):
        for i, atom_props in enumerate(new_feats):
            start_idx = contig_idxs[i][0]
            end_idx = contig_idxs[i][1]

            # write features to zarr store
            self.new_feats_array[start_idx:end_idx] = atom_props

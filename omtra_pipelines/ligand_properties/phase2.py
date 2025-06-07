import argparse
from typing import List, Dict
from pathlib import Path
import numpy as np
import dgl
import zarr

from rdkit import Chem
from rdkit.Chem import BRICS

from omtra.tasks.register import task_name_to_class
from omtra.eval.system import SampledSystem


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

    # TODO: Check for errors in fragment generation

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

    g = move_feats_to_t1('denovo_ligand', g, '1_true')
    rdkit_ligand = SampledSystem(g).get_rdkit_ligand()
    return rdkit_ligand


def process_pharmit_block(block_start_idx: int, block_size: int):
    """ 
    Parameters:
        block_start_idx (int): Index of the first ligand in the block
        block_size (int): Number of ligands in the block

    Returns:
        new_feats (List[np.ndarray]): Feature arrays per contiguous atom block.
        contig_idxs (List[Tuple[int, int]]): Start/end atom indices for each contiguous block.
        failed_idxs (List[int]): Indices of ligands that failed processing.
    """

    global pharmit_dataset

    # Load Pharmit dataset object
    n_mols = len(pharmit_dataset)
    block_end_idx = min(block_start_idx + block_size, n_mols)

    contig_idxs = []
    new_feats = []
    failed_idxs = []

    cur_contig_feats = []
    contig_start_idx = None
    contig_end_idx = None

    for idx in range(block_start_idx, block_end_idx):
        
        start_idx, end_idx = pharmit_dataset.retrieve_atom_idxs(idx)

        try:
            g = pharmit_dataset[('denovo_ligand', idx)]
            mol = dgl_to_rdkit(g)
            Chem.SanitizeMol(mol)

            atom_props = ligand_properties(mol)                         # (n_atoms, 5)
            fragments = fragment_molecule(mol)                          # (n_atoms, 1)
            atom_props = np.concatenate((atom_props, fragments), axis=1)# (n_atoms, 6)

            assert atom_props.shape[0] == (end_idx - start_idx), f"Mismatch in atom counts: computed properties for {atom_props.shape[0]} atoms but expected {(end_idx - start_idx)}"

            if contig_start_idx is None:
                contig_start_idx = start_idx

            cur_contig_feats.append(atom_props)
            contig_end_idx = end_idx  # always update with latest good molecule

        except Exception as e:
            print(f"Failed to compute features for molecule {idx}: {e}. Creating new contig from {contig_start_idx}-{contig_end_idx}")
            failed_idxs.append(idx)

            # Close current contiguous chunk (if any)
            if cur_contig_feats:
                feat_array = np.vstack(cur_contig_feats)
                contig_idxs.append((contig_start_idx, contig_end_idx))
                new_feats.append(feat_array)

                # Reset
                cur_contig_feats = []
                contig_start_idx = None
                contig_end_idx = None

    # After final molecule, flush last chunk if present
    if cur_contig_feats:
        atom_props = np.vstack(cur_contig_feats)
        contig_idxs.append((contig_start_idx, contig_end_idx))
        new_feats.append(atom_props)

    return new_feats, contig_idxs, failed_idxs

        
class BlockWriter:
    def __init__(self, store_path: str):
        # Open Pharmit Zarr store
        self.root = zarr.open(store_path, mode='r+')
        self.lig_node_group = self.root['lig/node']

    def save_chunk(self, array_name: str, contig_idxs: np.ndarray, new_feats: np.ndarray):

        # Check that Zarr array was correctly made
        if array_name not in self.lig_node_group:
            raise KeyError(f"Zarr array '{array_name}' not found in 'lig/node' group.")

        for i, atom_props in enumerate(new_feats):
            start_idx = contig_idxs[i][0]
            end_idx = contig_idxs[i][1]

            print(f"Writing from {start_idx} to {end_idx}")

            # write features to zarr store
            self.lig_node_group[array_name][start_idx:end_idx] = atom_props

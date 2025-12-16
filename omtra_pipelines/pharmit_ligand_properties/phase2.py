import argparse
from typing import List, Dict
from pathlib import Path
import numpy as np
import dgl
import zarr

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem

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

def get_chirality(mol, mol_idx, conf_id: int = -1):
    """
    Return an (n_atoms,) int array with:
      0 = not chiral, 1 = R, 2 = S, 3 = E, 4 = Z
    Uses 3D coordinates (AssignStereochemistryFrom3D) to set stereo,
    then assigns CIP labels.
    """

    n_atoms = mol.GetNumAtoms()
    mol_h = Chem.AddHs(mol, addCoords=True)

    if mol_h.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformers.")
    if conf_id >= mol_h.GetNumConformers():
        raise ValueError(f"conf_id {conf_id} out of range.")

    out = np.zeros(mol_h.GetNumAtoms(), dtype=int)

    try:
        # Perceive stereo from 3D, then assign CIP labels
        Chem.AssignStereochemistryFrom3D(mol_h, confId=conf_id, replaceExistingTags=True)
        Chem.AssignCIPLabels(mol_h)
    
    except Exception:
        print(f"Failed to compute chirality for molecule {mol_idx}. Returning all 0's", flush=True)
        return out[:n_atoms][:, None]

    # Atom R/S
    for atom in mol_h.GetAtoms():
        if atom.HasProp('_CIPCode'):
            code = atom.GetProp('_CIPCode')
            if code == 'R':
                out[atom.GetIdx()] = 1
            elif code == 'S':
                out[atom.GetIdx()] = 2

    # Bond E/Z → mark both atoms on each double bond
    for bond in mol_h.GetBonds():
        st = bond.GetStereo()
        if st == Chem.BondStereo.STEREOE:
            out[bond.GetBeginAtomIdx()] = max(out[bond.GetBeginAtomIdx()], 3)
            out[bond.GetEndAtomIdx()]   = max(out[bond.GetEndAtomIdx()],   3)
        elif st == Chem.BondStereo.STEREOZ:
            out[bond.GetBeginAtomIdx()] = max(out[bond.GetBeginAtomIdx()], 4)
            out[bond.GetEndAtomIdx()]   = max(out[bond.GetEndAtomIdx()],   4)

    return out[:n_atoms][:, None]

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

    g = move_feats_to_t1('denovo_ligand', g, '1_true')
    task = task_name_to_class('denovo_ligand')
    rdkit_ligand = SampledSystem(g, task=task).get_rdkit_ligand()
    return rdkit_ligand


        
class BlockWriter:
    def __init__(self, store_path: str, atom_array_name: str, edge_array_name: str):

        # Open Pharmit Zarr store
        self.root = zarr.open(store_path, mode='r+')
        
        self.lig_node_group = self.root['lig/node']
        self.lig_edge_group = self.root['lig/edge']

        # Check that Zarr array was correctly made
        if atom_array_name not in self.lig_node_group:
            raise KeyError(f"Zarr array '{atom_array_name}' not found in 'lig/node' group.")
        if edge_array_name not in self.lig_edge_group:
            raise KeyError(f"Zarr array '{edge_array_name}' not found in 'lig/edge' group.")

        self.new_atom_feats_array = self.lig_node_group[atom_array_name]
        self.new_edge_feats_array = self.lig_edge_group[edge_array_name]


    def save_chunk(self, atom_contig_idxs: np.ndarray, new_atom_feats: np.ndarray, edge_contig_idxs: np.ndarray, new_edge_feats: np.ndarray):
        for i, atom_feats in enumerate(new_atom_feats):
            atom_start_idx = atom_contig_idxs[i][0]
            atom_end_idx = atom_contig_idxs[i][1]

            # write features to zarr store
            self.new_atom_feats_array[atom_start_idx:atom_end_idx] = atom_feats
        
        for i, edge_feats in enumerate(new_edge_feats):
            edge_start_idx = edge_contig_idxs[i][0]
            edge_end_idx = edge_contig_idxs[i][1]

            # write features to zarr store
            self.new_edge_feats_array[edge_start_idx:edge_end_idx] = edge_feats
            

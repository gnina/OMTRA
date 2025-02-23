import argparse
from pathlib import Path
import gzip
from rdkit import Chem
import pymysql
import itertools
import re

from rdkit.Chem import AllChem as Chem
import numpy as np
import os
from multiprocessing import Pool
import pickle
from functools import partial


from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.utils.graph import build_lookup_table
from omtra.data.pharmit_pharmacophores import get_lig_only_pharmacophore
from tempfile import TemporaryDirectory
import time


def batch_generator(iterable, batch_size, n_chunks):
    """  
    Gets batches of conformer files

    Args: 
        iterable: Generator that crawls the conformer files
        batch_size: Size of the batches
    
    Returns:
        batch: List of conformer file paths of length batch_size (or remaining files)
    """
    batch = []
    batches_served = 0
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batches_served += 1
            batch = []  # Reset the batch
            if n_chunks is not None and batches_served >= n_chunks:
                break

    if batch:  # Remaining items that didn't fill a complete batch
        yield batch

class PharmitDBConnector():

    def __init__(self, spoof_db=False):
        if spoof_db:
            self.conn = None
        else:
            self.conn = pymysql.connect(
                host="localhost",
                user="pharmit", 
                db="conformers",)

class NameFinder(PharmitDBConnector):

    def query_name(self, smiles: str):

        if self.conn is None:
            return ['PubChem', 'ZINC', 'MolPort']

        with self.conn.cursor() as cursor:
            cursor.execute("SELECT name FROM names WHERE smile = %s", (smiles,))
            names = cursor.fetchall()
        names = list(itertools.chain.from_iterable(names))
        return self.extract_prefixes(names)
    

    def query_name_from_smiles(self, smiles_list: list[str]):
        if self.conn is None:
            return [['PubChem', 'ZINC', 'MolPort'] for smiles in smiles_list], []

        # Ensure the input is unique to avoid unnecessary duplicates in the result -> Need to preserve list order to match to other lists. TODO: find smarter way to handle duplicates
        #smiles_list = list(set(smiles_list))

        with self.conn.cursor() as cursor:
            # Use the IN clause to query multiple SMILES strings
            query = "SELECT smile, name FROM names WHERE smile IN %s"
            cursor.execute(query, (tuple(smiles_list),))
            results = cursor.fetchall()

        # Organize results into a dictionary: {smile: [names]}
        smiles_to_names = {smile: None for smile in smiles_list}
        
        for smile, name in results:
            if smiles_to_names[smile] is None:
                smiles_to_names[smile] = []
            smiles_to_names[smile].append(name)
        
        failed_idxs = [smiles_list.index(smile) for smile in smiles_to_names if smiles_to_names[smile] is None]  # Get the indices of failed smile in smiles_list
        names = [names for smile, names in smiles_to_names.items() if names is not None] # Remove None entries 

        return names, failed_idxs
    

    def extract_prefixes(self, names):
        """
        Extracts prefixes from a list of names where the prefix consists of 
        all characters at the start of the string that are not numbers or special characters.
        
        Args:
            names (list of str): A list of strings representing molecule names.
            
        Returns:
            list of str: A list of prefixes extracted from the names.
        """
        prefixes = set()
        for name in names:
            # Use a regex to match all letters at the start of the string
            match = re.match(r'^[A-Za-z]+', name)
            if match:
                prefixes.add(match.group(0))
            else:
                continue
        return list(prefixes)
    


class DBCrawler(PharmitDBConnector):

    def __init__(self, *args, max_num_queries=None, query_size: int = 1000, **kwargs):
        self.query_size = query_size
        self.max_num_queries = max_num_queries
        super().__init__(*args, **kwargs)

    def __iter__(self):
        try:
            with self.conn.cursor() as cursor:
                last_id = 0  # Start from the first row
                n_queries = 0
                
                while True:
                    if n_queries >= self.max_num_queries:
                        return

                    rows = self._fetch_next_batch(cursor, last_id)
                    if not rows:  # No more rows to fetch
                        break

                    id_vals, smiles, sdf_files = zip(*rows)
                    
                    # Yield current batch
                    yield (smiles, sdf_files)

                    # Update last_id to the ID of the last row in this batch
                    last_id = id_vals[-1]
                    n_queries += 1
        finally:
            pass

    def _fetch_next_batch(self, cursor, last_id):
        query = """
            SELECT id, smile, sdfloc FROM structures
            WHERE id > %s
            ORDER BY id
            LIMIT %s
        """
        cursor.execute(query, (last_id, self.query_size))
        return cursor.fetchall()

def get_pharmacophore_data(conformer_files, ph_type_idx, tmp_path: Path = None):

    # create a temporary directory if one is not provided
    delete_tmp_dir = False
    if tmp_path is None:
        delete_tmp_dir = True
        tmp_dir = TemporaryDirectory()
        tmp_path = Path(tmp_dir.name)


    # collect all pharmacophore data
    all_x_pharm = []
    all_a_pharm = []
    failed_pharm_idxs = []

    for idx, conf_file in enumerate(conformer_files):
        x_pharm, a_pharm = get_lig_only_pharmacophore(conf_file, tmp_path, ph_type_idx)
        if x_pharm is None:
            failed_pharm_idxs.append(idx)
            continue
        all_x_pharm.append(x_pharm)
        all_a_pharm.append(a_pharm)

    # delete temporary directory if it was created
    if delete_tmp_dir:
        tmp_dir.cleanup()

    return all_x_pharm, all_a_pharm, failed_pharm_idxs


def save_chunk_to_disk(tensors, chunk_data_file, chunk_info_file):
    
    positions = tensors['positions']
    atom_types = tensors['atom_types']
    atom_charges = tensors['atom_charges']
    bond_types = tensors['bond_types']
    bond_idxs = tensors['bond_idxs']
    x_pharm = tensors['x_pharm']
    a_pharm = tensors['a_pharm']
    databases = tensors['databases']

    # Record the number of nodes and edges in each molecule and convert to numpy arrays
    batch_num_nodes = np.array([x.shape[0] for x in positions])
    batch_num_edges = np.array([eidxs.shape[0] for eidxs in bond_idxs])
    batch_num_pharm_nodes = np.array([x.shape[0] for x in x_pharm])

    # concatenate all the data together
    x = np.concatenate(positions, axis=0)
    a = np.concatenate(atom_types, axis=0)
    c = np.concatenate(atom_charges, axis=0)
    e = np.concatenate(bond_types, axis=0)
    edge_index = np.concatenate(bond_idxs, axis=0)
    x_pharm = np.concatenate(x_pharm, axis=0)
    a_pharm = np.concatenate(a_pharm, axis=0)
    db = np.concatenate(databases, axis=0)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's node features
    node_lookup = build_lookup_table(batch_num_nodes)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's edge features
    edge_lookup = build_lookup_table(batch_num_edges)

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's pharmacophore node features
    pharm_node_lookup = build_lookup_table(batch_num_pharm_nodes)

    # Tensor dictionary
    chunk_data_dict = { 
        'lig_x': x,
        'lig_a': a,
        'lig_c': c,
        'node_lookup': node_lookup,
        'lig_e': e,
        'lig_edge_idx': edge_index,
        'edge_lookup': edge_lookup,
        'pharm_x': x_pharm,
        'pharm_a': a_pharm,
        'pharm_lookup': pharm_node_lookup,
        'database': databases
    }


    # Save tensor dictionary to npz file
    with open(chunk_data_file, 'wb') as f:
        np.savez_compressed(f, **chunk_data_dict)
    

    
    # Chunk data file info dictionary
    chunk_info_dict = {
        'File': chunk_data_file,
        'Mols': len(node_lookup),
        'Atoms': len(x),
        'Edges': len(e),
        'Pharm': len(x_pharm)
    }

    
    # Dump info dictionary in pickle files
    with open(chunk_info_file, "wb") as f:
        pickle.dump(chunk_info_dict, f)

    print('Wrote chunk:', chunk_info_dict['File'])


def generate_library_tensor(names, database_list):
    """
    Generates a binary tensor indicating whether each molecule belongs to any of the specified libraries.

    Args:
        names (list of list of str): A list of lists containing the database names for each molecule.

    Returns:
        np.ndarray: A binary tensor of shape (num_mols, num_libraries) where each element is 1 if the molecule belongs to the library, otherwise 0.
    """
    num_mols = len(names)
    num_libraries = len(database_list)
    
    # Initialize the binary tensor with zeros
    library_tensor = np.zeros((num_mols, num_libraries), dtype=int)
    
    for i, molecule_names in enumerate(names):
        for j, db in enumerate(database_list):
            if db in molecule_names:
                library_tensor[i, j] = 1
    
    return library_tensor
    

def compute_rmsd(mol1, mol2):
    return Chem.CalcRMS(mol1, mol2)


def minimize_molecule(molecule: Chem.rdchem.Mol):

    # create a copy of the original ligand
    lig = Chem.Mol(molecule)

    # Add hydrogens
    lig_H = Chem.AddHs(lig, addCoords=True)
    Chem.SanitizeMol(lig_H)

    try:
        ff = Chem.UFFGetMoleculeForceField(lig_H,ignoreInterfragInteractions=False)
    except Exception as e:
        print("Failed to get force field:", e)
        return None

    # documentation for this function call, incase we want to play with number of minimization steps or record whether it was successful: https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html#rdkit.ForceField.rdForceField.ForceField.Minimize
    try:
        ff.Minimize(maxIts=400)
    except Exception as e:
        print("Failed to minimize molecule")
        return None

    # Get the minimized positions for molecule with H's
    cpos = lig_H.GetConformer().GetPositions()

    # Original ligand with no H's
    conf = lig.GetConformer()

    for (i,xyz) in enumerate(cpos[-lig.GetNumAtoms():]):
        conf.SetAtomPosition(i,xyz)
    
    return lig


def remove_counterions_batch(mols: list[Chem.Mol], counterions: list[str]):
    for idx in range(len(mols)):
        mol = mols[idx]
        for i, atom in enumerate(mol.GetAtoms()):
            if str(atom.GetSymbol()) in counterions:
                print(f"Atom {atom.GetSymbol()} is a known counterion. Removing and minimizing structure.")
                mol_cpy = Chem.EditableMol(mol)
                mol_cpy.RemoveAtom(i)
                mol_cpy = mol_cpy.GetMol()
                mol = minimize_molecule(mol_cpy)
                mols[idx] = mol
    return mols
    
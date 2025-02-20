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

# Global variable to hold the NameFinder object
name_finder = None

def worker_initializer(spoof_db):
    global name_finder
    name_finder = NameFinder(spoof_db=spoof_db)
 
def parse_args():
    p = argparse.ArgumentParser(description='Process pharmit data')

    # temporary default path for development
    # don't hard-code a path here. just make a symbolic link to my pharmit_small directory in the same place in your repo,
    # or run the code with --db_dir=/path/to/pharmit_small
    p.add_argument('--db_dir', type=Path, default='./pharmit_small/')
    p.add_argument('--spoof_db', action='store_true', help='Spoof the database connection, for offline development')

    p.add_argument('--chunk_info_dir', type=Path, 
                   help='Output directory for information on data chunk files.', 
                   default='outputs/phase1_chunk_info')
    p.add_argument('--chunk_data_dir', type=Path, 
                   help='Output directory for tensor chunks.', 
                   default='outputs/phase1_chunk_data')

    p.add_argument('--atom_type_map', type=list, default=["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"])
    p.add_argument('--batch_size', type=int, default=50, help='Number of conformer files to batch togther.')
    p.add_argument('--pharm_types', type=list, default=['Aromatic','HydrogenDonor','HydrogenAcceptor','Hydrophobic','NegativeIon','PositiveIon'], help='Pharmacophore center types.')
    p.add_argument('--counterions', type=list, default=['Na', 'Ca', 'K', 'Mg', 'Al', 'Zn'])
    p.add_argument('--databases', type=list, default=["CHEMBL", "ChemDiv", "CSC", "Z", "CSF", "MCULE","MolPort", "NSC", "PubChem", "MCULE-ULTIMATE","LN", "LNL", "ZINC"])

    p.add_argument('--n_cpus', type=int, default=2, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--n_chunks', type=int, default=None, help='Number of to process. If None, process all. This is only for testing purposes.')

    args = p.parse_args()
    return args


"""
def extract_pharmacophore_data(mol):
    
    Parses pharmacophore data from an RDKit molecule object into a dictionary.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object containing pharmacophore data.

    Returns:
        dict: Parsed pharmacophore data with types as keys and lists of tuples as values.
    
    pharmacophore_data = mol.GetProp("pharmacophore") if mol.HasProp("pharmacophore") else None
    if pharmacophore_data is None:
        return None

    parsed_data = []
    lines = pharmacophore_data.splitlines()
    for line in lines:
        parts = line.split()
        if len(parts) >= 4:
            ph_type = parts[0]  # Pharmacophore type
            try:
                coordinates = tuple(map(float, parts[1:4]))  # Extract the 3 float values
                parsed_data.append((ph_type, coordinates))
            except ValueError:
                print(f"Skipping line due to parsing error: {line}")

    return parsed_data
"""

class NameFinder():

    def __init__(self, spoof_db=False):

        if spoof_db:
            self.conn = None
        else:
            self.conn = pymysql.connect(
                host="localhost",
                user="pharmit", 
                db="conformers",)
                # password="",
                # unix_socket="/var/run/mysqld/mysqld.sock")
            self.cursor = self.conn.cursor()

    def query_name(self, smiles: str):

        if self.conn is None:
            return ['PubChem', 'ZINC', 'MolPort']

        with self.conn.cursor() as cursor:
            cursor.execute("SELECT name FROM names WHERE smile = %s", (smiles,))
            names = cursor.fetchall()
        names = list(itertools.chain.from_iterable(names))
        return self.extract_prefixes(names)
    

    def query_name_batch(self, smiles_list: list[str]):
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
            if smile not in smiles_to_names:
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
    
    def query_smiles_from_file(self, conformer_file: Path):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT smile FROM structures WHERE sdfloc = %s", (str(conformer_file),))
            smiles = cursor.fetchall()
        return smiles


    def query_smiles_from_file_batch(self, conformer_files: list[Path]):

        if self.conn is None:
            return ['CC' for file in conformer_files], []

        file_to_smile = {Path(file): None for file in conformer_files}  # Dictionary to map conformer file to smile

        # failure will be different than a mysqlerror; there just wont be an entry if the file is not in the database
        with self.conn.cursor() as cursor:
            query = "SELECT sdfloc, smile FROM structures WHERE sdfloc IN %s"
            cursor.execute(query, (tuple(str(file) for file in conformer_files),))
            results = cursor.fetchall()

        for sdfloc, smile in results:
            file_to_smile[Path(sdfloc)] = smile  # Update with successfull queries

        failed_idxs = []
        for i, file in enumerate(conformer_files):
            if file_to_smile[Path(file)] is None:
                failed_idxs.append(i)
        
        smiles = [smile for smile in file_to_smile.values() if smile is not None] # Remove None entries 

        return smiles, failed_idxs
    

def read_mol_from_conf_file(conf_file):    # Returns Mol representaton of first conformer
    with gzip.open(conf_file, 'rb') as gzipped_sdf:
        suppl = Chem.ForwardSDMolSupplier(gzipped_sdf)
        try:
            for mol in suppl:
                if mol is not None:
                    return mol # Changed from break
            if mol is None:
                #print(f"Failed to parse a molecule from {conf_file}")
                return None
        except Exception as e:
            #print("Error parsing file", conf_file)
            return None


def crawl_conformer_files(db_dir: Path):
    for data_dir in db_dir.iterdir():
        conformers_dir = data_dir / 'conformers'
        for conformer_subdir in conformers_dir.iterdir():
            for conformer_file in conformer_subdir.iterdir():
                yield conformer_file


def batch_generator(iterable, batch_size):
    """  
    Gets batches of conformer files

    Args: 
        iterable: Generator that crawls the conformer files
        batch_size: Size of the batches
    
    Returns:
        batch: List of conformer file paths of length batch_size (or remaining files)
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []  # Reset the batch

    if batch:  # Remaining items that didn't fill a complete batch
        yield batch


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
        np.savez(f, **chunk_data_dict)
    

    
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


def process_batch(conformer_files, atom_type_map, ph_type_idx, database_list):
    global name_finder
    mol_tensorizer = MoleculeTensorizer(atom_map=atom_type_map)

    # Get RDKit Mol objects
    mols = [read_mol_from_conf_file(file) for file in conformer_files]
    # Find molecules that failed to featurize and count them
    failed_mol_idxs = []
    for i in range(len(mols)):
        if mols[i] is None:
            failed_mol_idxs.append(i)

    if len(failed_mol_idxs) > 0:
        #print("Mol objects for", len(failed_mol_idxs), "could not be found, removing")
        mols = [mol for i, mol in enumerate(mols) if i not in failed_mol_idxs]
        conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_mol_idxs]

    # (BATCHED) SMILES representations
    smiles, failed_smiles_idxs = name_finder.query_smiles_from_file_batch(conformer_files)
    # Remove molecules that couldn't get SMILES data
    if len(failed_smiles_idxs) > 0:
       #print("SMILEs for", len(failed_smiles_idxs), "conformer files could not be found, removing")
        mols = [mol for i, mol in enumerate(mols) if i not in failed_smiles_idxs]
        conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_smiles_idxs]


    # (BATCHED) Database source
    names, failed_names_idxs = name_finder.query_name_batch(smiles)
    # Remove molecules that couldn't get database data
    if len(failed_names_idxs) > 0:
        #print("Database sources for", len(failed_names_idxs), "could not be found, removing")
        mols = [mol for i, mol in enumerate(mols) if i not in failed_names_idxs]
        conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_names_idxs]


    # Get pharmacophore data
    x_pharm, a_pharm, failed_pharm_idxs = get_pharmacophore_data(conformer_files, ph_type_idx)
    # Remove ligands where pharmacophore generation failed
    if len(failed_pharm_idxs) > 0 :
        #print("Failed to generate pharmacophores for,", len(failed_pharm_idxs), "molecules, removing")
        mols = [mol for i, mol in enumerate(mols) if i not in failed_pharm_idxs]
        names = [name for i, name in enumerate(names) if i not in failed_pharm_idxs]
        
    
    # Get XACE data
    positions, atom_types, atom_charges, bond_types, bond_idxs, num_xace_failed, failed_xace_idxs = mol_tensorizer.featurize_molecules(mols) # (BATCHED) Tensor representation of molecules
    # Remove molecules that failed to get xace data
    if len(failed_xace_idxs) > 0:
        #print("XACE data for,", num_xace_failed, "molecules could not be found, removing")
        mols = [mol for i, mol in enumerate(mols) if i not in failed_xace_idxs]
        names = [name for i, name in enumerate(names) if i not in failed_xace_idxs]
        x_pharm = [x for i, x in enumerate(x_pharm) if i not in failed_xace_idxs]
        a_pharm = [a for i, a in enumerate(a_pharm) if i not in failed_xace_idxs]
    
    
    # Tensorize database sources
    databases  = generate_library_tensor(names, database_list)

    # Save tensors in dictionary
    tensors = {'positions': positions, 'atom_types': atom_types, 'atom_charges': atom_charges, 'bond_types': bond_types, 'bond_idxs': bond_idxs, 'x_pharm': x_pharm, 'a_pharm': a_pharm, 'databases': databases}
    return tensors

def run_parallel(args, batch_iter):
    with Pool(processes=args.n_cpus, initializer=worker_initializer, initargs=(spoof_db,)) as pool:
        for chunk_idx, conformer_files in enumerate(batch_iter):

            if args.n_chunks is not None and chunk_idx > args.n_chunks:
                break

            chunk_data_file = f"{args.chunk_data_dir}/data_chunk_{chunk_idx}.npz"
            chunk_info_file = f"{args.chunk_info_dir}/data_chunk_{chunk_idx}.pkl"

            pool.apply_async(
                process_batch, 
                args=(conformer_files, atom_type_map, ph_type_idx, database_list), 
                callback=partial(save_chunk_to_disk, 
                        chunk_data_file=chunk_data_file, 
                        chunk_info_file=chunk_info_file))

        
        pool.close()
        pool.join()

def run_single(args, batch_iter):
    worker_initializer(args.spoof_db)
    for chunk_idx, conformer_files in enumerate(batch_iter):

        if args.n_chunks is not None and chunk_idx > args.n_chunks:
            break

        chunk_data_file = f"{args.chunk_data_dir}/data_chunk_{chunk_idx}.npz"
        chunk_info_file = f"{args.chunk_info_dir}/data_chunk_{chunk_idx}.pkl"

        tensors = process_batch(conformer_files, atom_type_map, ph_type_idx, database_list)
        save_chunk_to_disk(tensors, chunk_data_file, chunk_info_file)


if __name__ == '__main__':

    args = parse_args()
    database_list = args.databases
    atom_type_map = args.atom_type_map
    spoof_db = args.spoof_db
    ph_type_idx = {type:idx for idx, type in enumerate(args.pharm_types)}

    # Make output directories
    os.makedirs(args.chunk_data_dir, exist_ok=True)
    os.makedirs(args.chunk_info_dir, exist_ok=True)

    path_iter = crawl_conformer_files(args.db_dir)
    batch_iter = batch_generator(path_iter, args.batch_size)

    start_time = time.time()

    if args.n_cpus == 1:
        run_single(args, batch_iter)
    else:
        run_parallel(args, batch_iter)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")

    # TODO: can you combine the conformer_file -> smiles -> names into one query rather than two? one query that is batched?


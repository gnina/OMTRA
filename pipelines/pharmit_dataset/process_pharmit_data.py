import argparse
from pathlib import Path
import gzip
from rdkit import Chem
import pymysql
import itertools
import re

import pickle
import numpy as np
import gc
import csv
import os

from omtra.data.xae_ligand import MoleculeTensorizer

# TODO: this script should actually take as input just a hydra config 
# - but Ramith is setting up our hydra stuff yet, and we don't 
# yet know what the config for this dataset processing component will look like
# so for now just argparse, and once its written it'll be easy/concrete to 
# port into a hydra config
def parse_args():
    p = argparse.ArgumentParser(description='Process pharmit data')

    # temporary default path for development
    p.add_argument('--db_dir', type=Path, default='/net/galaxy/home/koes/icd3/moldiff/OMTRA/pipelines/pharmit_dataset/pharmit_small') # OLD: './pharmit_small/'
    p.add_argument('--spoof_db', action='store_true', help='Spoof the database connection, for offline development')

    p.add_argument('--atom_type_map', type=list, default=["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"])

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

        try:
            with self.conn.cursor() as cursor:
                query = "SELECT sdfloc, smile FROM structures WHERE sdfloc IN %s"
                cursor.execute(query, (tuple(str(file) for file in conformer_files),))
                results = cursor.fetchall()

            for sdfloc, smile in results:
                file_smile_map[Path(sdfloc)] = smile  # Update with successfull queries

        except pymysql.MySQLError as e: # Indicate error in finding SMILE using conformer file
            print(f"Database query failed: {e}")
        
        failed_idxs = [conformer_files.index(file) for file in file_to_smile if file_to_smile[file] is None]  # Get the indices of failed file in conformer_files
        smiles = [smile for file, smile in file_to_smile.items() if smile is not None] # Remove None entries 

        return smiles, failed_idxs
    


def read_mol_from_conf_file(conf_file):    # Returns Mol representaton of first conformer
    with gzip.open(conf_file, 'rb') as gzipped_sdf:
        suppl = Chem.ForwardSDMolSupplier(gzipped_sdf)
        try:
            for mol in suppl:
                if mol is not None:
                    return mol # Changed from break
            if mol is None:
                print(f"Failed to parse a molecule from {conf_file}")
                return None
        except Exception as e:
            print("Error parsing file", conf_file)
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


def save_tensors_to_disk(tensors, output_path, chunk_number, chunk_size):
    os.makedirs(output_path, exist_ok=True)

    with open(f"{output_path}/chunk_{chunk_number}.pkl", 'wb') as f:
        pickle.dump(tensors, f)

    new_row = [[f"chunk_{chunk_number}.pkl", chunk_size]]
    with open(f"{output_path}/chunk_data.csv", 'a', newline='') as file:   # Save chunk size to csv file
        writer = csv.writer(file)
        writer.writerows(new_row)


if __name__ == '__main__':
    args = parse_args()

    mol_tensorizer = MoleculeTensorizer(atom_map=args.atom_type_map)
    name_finder = NameFinder(spoof_db=True) # args.spoof_db

    output_path = './tensor_chunks' # Outout tensor filepath

    # Clean output_path directory
    for filename in os.listdir(output_path):
        file_path = os.path.join(output_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    batch_size = 10    # Batch size for queries and processing to disk (memory clearing)
    chunks = 0


    for conformer_files in batch_generator(crawl_conformer_files(args.db_dir), batch_size):
        chunks += 1

        # RDKit Mol objects
        mols = [read_mol_from_conf_file(file) for file in conformer_files]
        # find molecules that failed to featurize and count them
        failed_mol_idxs = []
        for i in range(len(mols)):
            if mols[i] is None:
                failed_mol_idxs.append(i)

        if len(failed_mol_idxs) > 0:
            print("Mol objects for", len(failed_mol_idxs), "could not be found, removing")
            mols = [mol for i, mol in enumerate(mols) if i not in failed_mol_idxs]
            conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_mol_idxs]
    

        smiles, failed_smiles_idxs = name_finder.query_smiles_from_file_batch(conformer_files) # (BATCHED) SMILES representations
        # Remove molecules that couldn't get SMILES data
        if len(failed_smiles_idxs) > 0:
            print("SMILEs for", len(failed_smiles_idxs), "conformer files could not be found, removing")
            mols = [mol for i, mol in enumerate(mols) if i not in failed_smiles_idxs]
            conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_smiles_idxs]


        names, failed_names_idxs = name_finder.query_name_batch(smiles)  # (BATCHED) Database source
        # Remove molecules that couldn't get database data
        if len(failed_names_idxs) > 0:
            print("Database sources for", len(failed_names_idxs), "could not be found, removing")
            mols = [mol for i, mol in enumerate(mols) if i not in failed_names_idxs]
            conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_names_idxs]
            smiles = [smile for i, smile in enumerate(smiles) if i not in failed_names_idxs]

        
        positions, atom_types, atom_charges, bond_types, bond_idxs, num_xace_failed, failed_xace_idxs = mol_tensorizer.featurize_molecules(mols) # (BATCHED) Tensor representation of molecules
        # Remove molecules that failed to get xace data
        if len(failed_xace_idxs) > 0:
            print("XACE date for,", num_xace_failed, "molecules could not be found, removing")
            mols = [mol for i, mol in enumerate(mols) if i not in failed_xace_idxs]
            conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_xace_idxs]
            smiles = [smile for i, smile in enumerate(smiles) if i not in failed_xace_idxs]
            names = [name for i, name in enumerate(names) if i not in failed_xace_idxs]
    

        # TODO: Tensorize database name (Somayeh)
        # TODO: Generate pharmacore data using pharmit & convert to tensors (Nate)
        # TODO: Merge tensors
        
        # Save tensors to disk 
        #save_tensors_to_disk(tensors, output_path, chunks, len(mols))   # TODO: Chunk size should probably be determined from the size of the tensor

        del mols, smiles, names, positions, atom_types, atom_charges, bond_types, bond_idxs, tensors   # Clear memory after saving
        gc.collect()  # Clear unused memory

        print(f"Processed batch {chunks}, memory cleared")

    


    """
    for conformer_file in crawl_conformer_files(args.db_dir):
        mol = read_mol_from_conf_file(conformer_file)   # RDKit Mol object
        smiles = name_finder.query_smiles_from_file(conformer_file) # (BATCH) SMILES representation.
        names = name_finder.query_name(smiles)  # Molecule name
        pharmacophore_data = extract_pharmacophore_data(mol)
        xae_mol = mol_tensorizer.featurize_molecules([mol])
        # TODO: if pharmacophore data is not found, generatate it using pharmit
        if pharmacophore_data is None:
            print(f"Failed to parse pharmacophore data for {smiles}")
    """

    # TODO: convert pharmacophore and names into tensors
    # TODO: can you combine the conformer_file -> smiles -> names into one query rather than two? one query that is batched?
    # TODO: process molecules in batches; 
    #     this includes using the NameFinder.query_batch method instead of NameFinder.query
    #     you can also batch with NameFinder.query_smiles_from_file_batch in stead of NameFinder.query_smiles_from_file
    #     MoleculeTensorizer can handle batches of molecules
    # TODO: parallelize processing: hand chunks of conformer files to subprocesses
    # TODO: write molecules to disk in chunks
        

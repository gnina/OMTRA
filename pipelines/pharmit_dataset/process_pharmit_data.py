import argparse
from pathlib import Path
import gzip
from rdkit import Chem
import pymysql
import itertools

# TODO: this script should actually take as input just a hydra config 
# - but Ramith is setting up our hydra stuff yet, and we don't 
# yet know what the config for this dataset processing component will look like
# so for now just argparse, and once its written it'll be easy/concrete to 
# port into a hydra config
def parse_args():
    p = argparse.ArgumentParser(description='Process pharmit data')

    # temporary default path for development
    p.add_argument('--conf_file', type=Path, default='/home/icd3/OMTRA/pipelines/pharmit_dataset/tmp_conformer_inspection/100.sdf.gz')
    p.add_argument('--skip_query', action='store_true', help='Skip querying the database for names')

    args = p.parse_args()
    return args

def extract_pharmacophore_data(mol):
    """
    Parses pharmacophore data from an RDKit molecule object into a dictionary.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object containing pharmacophore data.

    Returns:
        dict: Parsed pharmacophore data with types as keys and lists of tuples as values.
    """
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

import re




class NameFinder():

    def __init__(self):
        self.conn = pymysql.connect(
            host="localhost",
            user="pharmit", 
            db="conformers",)
            # password="",
            # unix_socket="/var/run/mysqld/mysqld.sock")
        self.cursor = self.conn.cursor()

    def query(self, smiles: str):
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT name FROM names WHERE smile = %s", (smiles,))
            names = cursor.fetchall()
        names = list(itertools.chain.from_iterable(names))
        return self.extract_prefixes(names)
    
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

if __name__ == '__main__':
    args = parse_args()


    # get the first conformer from the file
    conformer_file_path = args.conf_file
    with gzip.open(conformer_file_path, 'rb') as gzipped_sdf:
        suppl = Chem.ForwardSDMolSupplier(gzipped_sdf)
        for mol in suppl:
            if mol is not None:
                break
        if mol is None:
            raise ValueError(f"Failed to parse a molecule from {conformer_file_path}")
        
    # extract pharmacophore data from the molecule
    pharmacophore_data = extract_pharmacophore_data(mol)

    # get smiles string
    smiles = Chem.MolToSmiles(mol)

    print(f"Pharmacophore data: {pharmacophore_data}")
    print(f"SMILES: {smiles}")
    
    if not args.skip_query:
        name_finder = NameFinder()
        names = name_finder.query(smiles)
        print(f"Names: {names}")
        

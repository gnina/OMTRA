import argparse
from pathlib import Path
import gzip
from rdkit import Chem
import pymysql
import itertools
import re

from omtra.data.xae_ligand import MoleculeTensorizer

# TODO: this script should actually take as input just a hydra config 
# - but Ramith is setting up our hydra stuff yet, and we don't 
# yet know what the config for this dataset processing component will look like
# so for now just argparse, and once its written it'll be easy/concrete to 
# port into a hydra config
def parse_args():
    p = argparse.ArgumentParser(description='Process pharmit data')

    # temporary default path for development
    p.add_argument('--db_dir', type=Path, default='./pharmit_small/')
    p.add_argument('--spoof_db', action='store_true', help='Spoof the database connection, for offline development')

    p.add_argument('--atom_type_map', type=list, default=["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"])

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

    def query(self, smiles: str):

        if self.conn is None:
            return ['PubChem', 'ZINC', 'MolPort']

        with self.conn.cursor() as cursor:
            cursor.execute("SELECT name FROM names WHERE smile = %s", (smiles,))
            names = cursor.fetchall()
        names = list(itertools.chain.from_iterable(names))
        return self.extract_prefixes(names)
    
    def query_batch(self, smiles_list: list[str]):
        if self.conn is None:
            return {smiles: ['PubChem', 'ZINC', 'MolPort'] for smiles in smiles_list}

        # Ensure the input is unique to avoid unnecessary duplicates in the result
        smiles_list = list(set(smiles_list))

        with self.conn.cursor() as cursor:
            # Use the IN clause to query multiple SMILES strings
            query = "SELECT smile, name FROM names WHERE smile IN %s"
            cursor.execute(query, (tuple(smiles_list),))
            results = cursor.fetchall()

        # Organize results into a dictionary: {smile: [names]}
        smiles_to_names = {}
        for smile, name in results:
            if smile not in smiles_to_names:
                smiles_to_names[smile] = []
            smiles_to_names[smile].append(name)

        # Add SMILES strings that had no matches to ensure a complete dictionary
        for smile in smiles_list:
            if smile not in smiles_to_names:
                smiles_to_names[smile] = []

        return {smile: self.extract_prefixes(names) for smile, names in smiles_to_names.items()}
    
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
    
def read_mol_from_conf_file(conf_file):
    with gzip.open(conf_file, 'rb') as gzipped_sdf:
        suppl = Chem.ForwardSDMolSupplier(gzipped_sdf)
        for mol in suppl:
            if mol is not None:
                break
        if mol is None:
            raise ValueError(f"Failed to parse a molecule from {conf_file}")
    return mol

def crawl_conformer_files(db_dir: Path):
    for data_dir in db_dir.iterdir():
        conformers_dir = data_dir / 'conformers'
        for conformer_subdir in conformers_dir.iterdir():
            for conformer_file in conformer_subdir.iterdir():
                yield conformer_file


if __name__ == '__main__':
    args = parse_args()

    mol_tensorizer = MoleculeTensorizer(atom_type_map=args.atom_type_map)
    name_finder = NameFinder(spoof_db=args.spoof_db)

    for conformer_file in crawl_conformer_files(args.db_dir):
        mol = read_mol_from_conf_file(conformer_file)
        smiles = Chem.MolToSmiles(mol)
        names = name_finder.query(smiles)
        pharmacophore_data = extract_pharmacophore_data(mol)
        xae_mol = mol_tensorizer.featurize_molecules([mol])

    # TODO: convert pharmacophore and names into tensors
    # TODO: process molecules in batches; 
    #     this includes using the NameFinder.query_batch method instead of NameFinder.query
    #     MoleculeTensorizer can handle batches of molecules
    #     but for optimial efficiency we may want to paralellize
    # TODO: write molecules to disk in chunks
        

import argparse
from pathlib import Path
import gzip
from rdkit import Chem
import pymysql
import itertools
import re
from typing import Tuple
import subprocess
import functools

from rdkit.Chem import AllChem as Chem
import numpy as np
import os
from multiprocessing import Pool
import pickle
from functools import partial
import random
import math
from collections import defaultdict


from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.utils.graph import build_lookup_table
from omtra.data.pharmit_pharmacophores import get_lig_only_pharmacophore
from omtra.data.pharmacophores import get_pharmacophores
from tempfile import TemporaryDirectory

def read_mol_from_conf_file(conf_file):    # Returns Mol representaton of first conformer

    if conf_file is None:
        return None

    try:
        if not Path(conf_file).exists():
            return None
    except PermissionError:
        return None

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
        is_data_dir = data_dir.is_dir() and re.match(r'data\d{2}', data_dir.name)
        if not is_data_dir:
            continue
        conformers_dir = data_dir / 'conformers'
        for conformer_subdir in conformers_dir.iterdir():
            for conformer_file in conformer_subdir.iterdir():
                yield conformer_file

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
        smiles_to_names = defaultdict(list)
        
        for smile, name in results:
            smiles_to_names[smile].append(name)
        
        # important to note here we are encoding a very important bit of logic here
        # namely, we heavily filter naems for molecules. any name that does not
        # have a "proper" prefix is removed
        # we define a prefix as some sequence of num-numeric characters at the start of the string
        names = []
        failed_idxs = []
        for i, smile in enumerate(smiles_list):
            if smiles_to_names[smile] is None:
                failed_idxs.append(i)
                names.append(None)
            else:
                names.append(self.extract_prefixes(smiles_to_names[smile]))

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
        return prefixes
    


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

    @functools.cached_property
    def row_count(self):
        """
        Efficiently retrieves the number of rows in the 'structures' table.
        
        Returns:
            int: The number of rows in the table.
        """
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM structures")
            count = cursor.fetchone()[0]
        return count
    
    def __len__(self):
        if self.max_num_queries == float('inf'):
            n_rows = self.row_count()
        else:
            n_rows = self.max_num_queries*self.query_size

        n_whole_queries, remainder = divmod(n_rows, self.query_size)
        return n_whole_queries + (remainder > 0)

    def get_random_names(self, total_names, n_queries):
        """
        Randomly sample total_names values from the names table by selecting random rows 
        from the "structures" table (using its indexed "id" column) and joining with the 
        "names" table on the "smiles" column.

        Parameters:
            connection: a pymysql Connection object.
            total_names: the total number of name values to retrieve.
            n_queries: the total number of queries to execute (each query fetches roughly total_names/n_queries rows).

        Returns:
            A list of strings representing the 'name' values from the joined tables.
        """
        # Get the minimum and maximum id values from the structures table.
        query = "SELECT MIN(id) AS min_id, MAX(id) AS max_id FROM structures"
        with self.conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(query)
            result = cursor.fetchone()
            min_id = result['min_id']
            max_id = result['max_id']
        
        if min_id is None or max_id is None:
            return []  # Table is empty.

        # Calculate the approximate number of rows to fetch per query.
        rows_per_query = math.ceil(total_names / n_queries)
        results = []

        for _ in range(n_queries):
            random_id = random.randint(min_id, max_id)
            # Query: select the name values by joining structures and names, starting from a random id.
            query = (
                "SELECT names.name "
                "FROM structures "
                "JOIN names ON structures.smile = names.smile "
                "WHERE structures.id >= %s "
                "LIMIT " + str(rows_per_query)
            )
            with self.conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(query, (random_id,))
                rows = cursor.fetchall()
                for row in rows:
                    results.append(row['name'])
                    if len(results) >= total_names:
                        return results

        return results

def get_pharmacophore_data(mols):

    # collect all pharmacophore data
    all_x_pharm = []
    all_a_pharm = []
    all_v_pharm = []
    
    failed_pharm_idxs = []
    for idx, mol in enumerate(mols):
        x_pharm, a_pharm, v_pharm, _ = get_pharmacophores(mol)
        if x_pharm is None:
            failed_pharm_idxs.append(idx)
            continue
        
        all_x_pharm.append(x_pharm)
        all_a_pharm.append(a_pharm)
        all_v_pharm.append(v_pharm)

    return all_x_pharm, all_a_pharm, all_v_pharm, failed_pharm_idxs

def get_pharmacophore_data_pharmit(conformer_files, ph_type_idx, tmp_path: Path = None):

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


class ChunkSaver():

    def __init__(self, output_dir: Path, 
        register_write_interval: int = 10, # how frequently we record the chunks that have been processed to disk
        chunk_offload_threshold_mb: int = 1000 # how many MB of data to store locally before offloading to masuda
        ):

        self.output_dir = output_dir
        self.register_write_interval = register_write_interval
        self.chunk_offload_threshold_mb = chunk_offload_threshold_mb

        # Make output directories
        chunk_data_dir = self.output_dir / 'chunk_data'
        chunk_info_dir = self.output_dir / 'chunk_info'
        chunk_data_dir.mkdir(parents=True, exist_ok=True)
        chunk_info_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_data_dir = chunk_data_dir
        self.chunk_info_dir = chunk_info_dir

        # get register file, load register if it exists
        self.register_file = output_dir / 'register.pkl'
        if self.register_file.exists():
            with open(self.register_file, 'rb') as f:
                self.register: set = pickle.load(f)
        else:
            self.register = set()

        bytes_stored_local = sum(f.stat().st_size for f in self.chunk_data_dir.glob('*.npz'))
        self.mb_stored_local = bytes_stored_local / (1024 ** 2)
        self.n_chunks_since_register_write = 0

    def chunk_processed(self, chunk_idx: int) -> bool:
        chunk_data_file, _ = self.idx_to_chunk_files(chunk_idx)
        return str(chunk_data_file) in self.register

    def add_chunk_to_register(self, chunk_data_file: Path):
        self.register.add(str(chunk_data_file))
        self.mb_stored_local += chunk_data_file.stat().st_size / (1024 ** 2)
        self.n_chunks_since_register_write += 1

        if self.n_chunks_since_register_write >= self.register_write_interval:
            self.write_register()

        if self.mb_stored_local >= self.chunk_offload_threshold_mb:
            self.offload_chunks()

    def offload_chunks(self):

        if self.mb_stored_local == 0:
            return

        print("Offloading chunks to masuda")

        data_files = self.chunk_data_dir / '*.npz'
        info_files = self.chunk_info_dir / '*.pkl'
        data_dst_path = '/home/ian/projects/mol_diffusion/OMTRA/omtra_pipelines/pharmit_dataset/outputs/phase1/chunk_data'
        info_dst_path = '/home/ian/projects/mol_diffusion/OMTRA/omtra_pipelines/pharmit_dataset/outputs/phase1/chunk_info'
        scp_transfer(str(data_files), "masuda-tunnel", data_dst_path)
        scp_transfer(str(info_files), "masuda-tunnel", info_dst_path)

        # remove files from local storage
        for f in self.chunk_data_dir.glob('*.npz'):
            f.unlink()
        for f in self.chunk_info_dir.glob('*.pkl'):
            f.unlink()

        self.mb_stored_local = 0

    def write_register(self):
        with open(self.register_file, 'wb') as f:
            pickle.dump(self.register, f)
        self.n_chunks_since_register_write = 0

    def idx_to_chunk_files(self, idx: int) -> Tuple[Path, Path]:
        chunk_data_file = self.chunk_data_dir / f"chunk_data_{idx}.npz"
        chunk_info_file = self.chunk_info_dir / f"chunk_info_{idx}.pkl"
        return chunk_data_file, chunk_info_file


    def save_chunk_to_disk(self, tensors, chunk_idx):

        chunk_data_file, chunk_info_file = self.idx_to_chunk_files(chunk_idx)
        
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

        # Convert data types
        chunk_data_dict['lig_x'] = chunk_data_dict['lig_x'].astype(np.float32)
        chunk_data_dict['pharm_x'] = chunk_data_dict['pharm_x'].astype(np.float32)
        chunk_data_dict['lig_a'] = chunk_data_dict['lig_a'].astype(np.int8)
        chunk_data_dict['pharm_a'] = chunk_data_dict['pharm_a'].astype(np.int8)
        chunk_data_dict['lig_c'] = chunk_data_dict['lig_c'].astype(np.int8)


        # Save tensor dictionary to npz file
        with open(chunk_data_file, 'wb') as f:
            np.savez_compressed(f, **chunk_data_dict)
        

        # Chunk data file info dictionary
        chunk_info_dict = {
            'file': chunk_data_file,
            'n_mols': len(node_lookup),
            'n_atoms': len(x),
            'n_edges': len(e),
            'n_pharm': len(x_pharm)
        }
        
        # Dump info dictionary in pickle files
        with open(chunk_info_file, "wb") as f:
            pickle.dump(chunk_info_dict, f)

        self.add_chunk_to_register(chunk_data_file)

def scp_transfer(local_path, remote_host, remote_path):
    command = f"scp {local_path} {remote_host}:{remote_path}"
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
        # print(result.stdout)  # Print the stdout from the scp command
        # print(result.stderr)  # Print the stderr from the scp command
    except subprocess.CalledProcessError as e:
        print(f"SCP failed with error:\n{e.stderr}")
        raise RuntimeError(f"SCP command failed: {e}") from e


def generate_library_tensor(names, 
                            database_list, 
                            filter_unknown=True,
                            other_category=False):
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
    library_tensor = np.zeros((num_mols, num_libraries), dtype=bool)
    
    for i, molecule_names in enumerate(names):
        for j, db in enumerate(database_list):
            if db in molecule_names:
                library_tensor[i, j] = 1

    if filter_unknown and other_category:
        raise ValueError("Cannot filter unknown and include an 'other' category at the same time.")

    # Find rows where all columns have a value of 0
    if filter_unknown:
        db_found = np.any(library_tensor, axis=1)
        failed_idxs = np.where(~db_found)[0].tolist()
        failed_idxs = set(failed_idxs)
        library_tensor = library_tensor[db_found]
        return library_tensor, failed_idxs

    if other_category:
        db_found = np.any(library_tensor, axis=1)
        library_tensor = np.concatenate([library_tensor, ~db_found[:, None]], axis=1)
    
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

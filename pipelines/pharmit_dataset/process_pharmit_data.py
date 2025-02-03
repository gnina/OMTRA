import argparse
from pathlib import Path
import gzip
from rdkit import Chem
import pymysql
import itertools
import re

from rdkit.Chem import AllChem as Chem
import numpy as np
import gc
import csv
import os
from collections import defaultdict
import zarr
import time
import subprocess
import json


from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.utils.graph import build_lookup_table

# TODO: this script should actually take as input just a hydra config 
# - but Ramith is setting up our hydra stuff yet, and we don't 
# yet know what the config for this dataset processing component will look like
# so for now just argparse, and once its written it'll be easy/concrete to 
# port into a hydra config
def parse_args():
    p = argparse.ArgumentParser(description='Process pharmit data')

    # temporary default path for development
    # don't hard-code a path here. just make a symbolic link to my pharmit_small directory in the same place in your repo,
    # or run the code with --db_dir=/path/to/pharmit_small
    p.add_argument('--db_dir', type=Path, default='./pharmit_small/') # OLD: './pharmit_small/'
    p.add_argument('--spoof_db', action='store_true', help='Spoof the database connection, for offline development')

    p.add_argument('--atom_type_map', type=list, default=["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"])

    p.add_argument('--batch_size', type=int, default=50, help='Number of conformer files to batch togther.')

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
    


def get_pharmacophore_data(conformer_files):
    tmp = 'pharmacophores'
    phfile = os.path.join(tmp,"ph.json") # File to save pharmacophore data temporarily
    x_pharm = []
    a_pharm = []
    failed_pharm_idxs = []
    ph_type_to_idx = {'Aromatic': 0,
    'HydrogenDonor': 1,
    'HydrogenAcceptor':2,
    'Hydrophobic':3,
    'NegativeIon':4,
    'PositiveIon':5}

    for i in range(len(conformer_files)):
        file = conformer_files[i]
        
        # Get pharmacophore data
        cmd = f'./pharmit pharma -in {file} -out {phfile}'   # command for pharmit to get pharmacophore data
        subprocess.check_call(cmd,shell=True)

        #some files have another json object in them - only take first
        #in actuality, it is a bug with how pharmit/openbabel is dealing
        #with gzipped sdf files that causes only one molecule to be read
        decoder = json.JSONDecoder()
        ph = decoder.raw_decode(open(phfile).read())[0]
        
        # Read generated data into numpy arrays
        if ph['points']:
            x_pharm.append(np.array([(p['x'],p['y'],p['z']) for p in ph['points'] if p['enabled']]))
            a_pharm.append(np.array([ph_type_to_idx[p['name']] for p in ph['points'] if p['enabled']]))
        else:
            # Failed to get data --> store index
            failed_pharm_idxs.append(i)
        
    return x_pharm, a_pharm, failed_pharm_idxs


def save_tensors_to_zarr(outdir, positions, atom_types, atom_charges, bond_types, bond_idxs, x_pharm, a_pharm, databases):

    # Record the number of nodes and edges in each molecule and convert to numpy arrays
    batch_num_nodes = np.array([x.shape[0] for x in positions])
    batch_num_edges = np.array([eidxs.shape[0] for eidxs in bond_idxs])
    batch_num_pharm_nodes = np.array([x.shape[0] for x in x_pharm])
    batch_num_db_nodes = np.array([x.shape[0] for x in databases]) 

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

    # create an array of indicies to keep track of the start_idx and end_idx of each molecule's database locations
    db_node_lookup = build_lookup_table(batch_num_db_nodes)

    print("Shape of x:", x.shape)
    print("Shape of a:", a.shape)
    print("Shape of c:", c.shape)
    print("Shape of e:", e.shape)
    print("Shape of edge_index:", edge_index.shape)
    print("Shape of x_pharm:", x_pharm.shape)
    print("Shape of a_pharm:", a_pharm.shape)
    print("Shape of db:", db.shape)
    print("Shape of node_lookup:", node_lookup.shape)
    print("Shape of edge_lookup:", edge_lookup.shape)
    print("Shape of pharm_node_lookup:", pharm_node_lookup.shape)
    print("Shape of db_node_lookup:", db_node_lookup.shape)


    graphs_per_chunk = 50 # very important parameter
    id = str(int(time.time() * 1000))[-8:]
    filename = f"test_ligand_dataset_{id}.zarr"
    store = zarr.storage.LocalStore(f"{outdir}/{filename}")

    # Create a root group
    root = zarr.group(store=store)

    ntypes = ['lig', 'db', 'pharm']

    ntype_groups = {}
    for ntype in ntypes:
        ntype_groups[ntype] = root.create_group(ntype)


    lig_node = ntype_groups['lig'].create_group('node')
    lig_edge_data = ntype_groups['lig'].create_group('edge')

    pharm_node_data = ntype_groups['pharm'].create_group('node')
    db_node_data = ntype_groups['db'].create_group('node')

    # Store tensors under different keys with specified chunk sizes

    # some simple heuristics to decide chunk sizes for node and edge data
    mean_lig_nodes_per_graph = int(np.mean(batch_num_nodes))
    mean_ll_edges_per_graph = int(np.mean(batch_num_edges))
    mean_pharm_nodes_per_graph = int(np.mean([x.shape[0] for x in x_pharm]))
    mean_db_nodes_per_graph = int(np.mean(batch_num_db_nodes))

    nodes_per_chunk = graphs_per_chunk * mean_lig_nodes_per_graph
    ll_edges_per_chunk = graphs_per_chunk * mean_ll_edges_per_graph 
    pharm_nodes_per_chunk = graphs_per_chunk * mean_pharm_nodes_per_graph
    db_nodes_per_chunk = graphs_per_chunk * mean_db_nodes_per_graph

    # create arrays for node data
    lig_node.create_array('x', shape=x.shape, chunks=(nodes_per_chunk, 3), dtype=x.dtype)
    lig_node.create_array('a', shape=a.shape, chunks=(nodes_per_chunk,), dtype=a.dtype)
    lig_node.create_array('c', shape=c.shape, chunks=(nodes_per_chunk,), dtype=c.dtype)
    
    # create arrays for edge data
    lig_edge_data.create_array('e', shape=e.shape, chunks=(ll_edges_per_chunk,), dtype=e.dtype)
    lig_edge_data.create_array('edge_index', shape=edge_index.shape, chunks=(ll_edges_per_chunk, 2), dtype=edge_index.dtype)

    # create arrays for pharmacophore node data
    pharm_node_data.create_array('x', shape=x_pharm.shape, chunks=(pharm_nodes_per_chunk, 3), dtype=x_pharm.dtype)
    pharm_node_data.create_array('a', shape=a_pharm.shape, chunks=(pharm_nodes_per_chunk,), dtype=a_pharm.dtype)
    pharm_node_data.create_array('graph_lookup', shape=pharm_node_lookup.shape, chunks=pharm_node_lookup.shape, dtype=pharm_node_lookup.dtype)

    # create arrays for database data
    db_node_data.create_array('db', shape=db.shape, chunks=(db_nodes_per_chunk,), dtype=db.dtype)  # TODO: edit to include actual dimension of array
    db_node_data.create_array('graph_lookup', shape=db_node_lookup.shape, chunks=db_node_lookup.shape, dtype=db_node_lookup.dtype)

    # because node_lookup and edge_lookup are relatively small, we may get away with not chunking them
    lig_node.create_array('graph_lookup', shape=node_lookup.shape, chunks=node_lookup.shape, dtype=node_lookup.dtype)
    lig_edge_data.create_array('graph_lookup', shape=edge_lookup.shape, chunks=edge_lookup.shape, dtype=edge_lookup.dtype)

    # write data to the arrays
    lig_node['x'][:] = x
    lig_node['a'][:] = a
    lig_node['c'][:] = c
    lig_node['graph_lookup'][:] = node_lookup

    lig_edge_data['e'][:] = e
    lig_edge_data['edge_index'][:] = edge_index
    lig_edge_data['graph_lookup'][:] = edge_lookup

    pharm_node_data['x'][:] = x_pharm
    pharm_node_data['a'][:] = a_pharm
    pharm_node_data['graph_lookup'][:] = pharm_node_lookup
    
    db_node_data['db'][:] = db
    db_node_data['graph_lookup'][:] = db_node_lookup
    

    print(root.tree())
    return filename



if __name__ == '__main__':
    args = parse_args()
    mol_tensorizer = MoleculeTensorizer(atom_map=args.atom_type_map)
    name_finder = NameFinder(spoof_db=args.spoof_db) 

    # Known counterions: https://www.sciencedirect.com/topics/chemistry/counterion#:~:text=About%2070%25%20of%20the%20counter,most%20common%20cation%20is%20Na%2B.
    counterions = ['Na', 'Ca', 'K', 'Mg', 'Al', 'Zn']

    outdir = 'pharmit_data'
    id = str(int(time.time() * 1000))[-8:]
    chunk_data = f"{outdir}/data_{id}.txt"

    batch_size = 1000    # Batch size for queries and processing to disk (memory clearing)
    chunks = 0
    
    for conformer_files in batch_generator(crawl_conformer_files(args.db_dir), batch_size):
        chunks += 1

        # Get RDKit Mol objects
        mols = [read_mol_from_conf_file(file) for file in conformer_files]
        # Find molecules that failed to featurize and count them
        failed_mol_idxs = []
        for i in range(len(mols)):
            if mols[i] is None:
                failed_mol_idxs.append(i)

        if len(failed_mol_idxs) > 0:
            print("Mol objects for", len(failed_mol_idxs), "could not be found, removing")
            mols = [mol for i, mol in enumerate(mols) if i not in failed_mol_idxs]
            conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_mol_idxs]

        # Check for counterions and remove
        mols = remove_counterions_batch(mols, counterions)


        # (BATCHED) SMILES representations
        smiles, failed_smiles_idxs = name_finder.query_smiles_from_file_batch(conformer_files)
        # Remove molecules that couldn't get SMILES data
        if len(failed_smiles_idxs) > 0:
            print("SMILEs for", len(failed_smiles_idxs), "conformer files could not be found, removing")
            mols = [mol for i, mol in enumerate(mols) if i not in failed_smiles_idxs]
            conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_smiles_idxs]


        # (BATCHED) Database source
        names, failed_names_idxs = name_finder.query_name_batch(smiles)
        # Remove molecules that couldn't get database data
        if len(failed_names_idxs) > 0:
            print("Database sources for", len(failed_names_idxs), "could not be found, removing")
            mols = [mol for i, mol in enumerate(mols) if i not in failed_names_idxs]
            conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_names_idxs]


        # Get pharmacophore data
        x_pharm, a_pharm, failed_pharm_idxs = get_pharmacophore_data(conformer_files)
        # Remove ligands where pharmacophore generation failed
        if len(failed_pharm_idxs) > 0 :
            print("Failed to generate pharmacophores for,", len(failed_pharm_idxs), "molecules, removing")
            mols = [mol for i, mol in enumerate(mols) if i not in failed_pharm_idxs]
            names = [name for i, name in enumerate(names) if i not in failed_pharm_idxs]
            
        
        # Get XACE data
        positions, atom_types, atom_charges, bond_types, bond_idxs, num_xace_failed, failed_xace_idxs = mol_tensorizer.featurize_molecules(mols) # (BATCHED) Tensor representation of molecules
        # Remove molecules that failed to get xace data
        if len(failed_xace_idxs) > 0:
            print("XACE date for,", num_xace_failed, "molecules could not be found, removing")
            mols = [mol for i, mol in enumerate(mols) if i not in failed_xace_idxs]
            names = [name for i, name in enumerate(names) if i not in failed_xace_idxs]
            x_pharm = [x for i, x in enumerate(x_pharm) if i not in failed_xace_idxs]
            a_pharm = [a for i, a in enumerate(a_pharm) if i not in failed_xace_idxs]
        


        # TODO: Tensorize database name (Somayeh)
        # INPUT: List of
        # OUTPUT: List of numpy arrays of encodings 


        # Change bond ID representation
        new_bond_idxs = []
        for ligand in bond_idxs:
            bonds = []
            for i in range(len(ligand[0])):
                bonds.append([ligand[0][i], ligand[1][i]])
            new_bond_idxs.append(np.array(bonds))

            # this could be done in numpy one-liner, but why are we even doing it?
        
        # Format and save tensors to disk
        zarr_store = save_tensors_to_zarr(outdir, positions, atom_types, atom_charges, bond_types, new_bond_idxs, x_pharm, a_pharm, [np.array([])])
        print(f"Processed batch {chunks}")
        print("––––––––––––––––––––––––––––––––––––––––––––––––")

        # Record number of molecules in zarr store to txt file
        with open(chunk_data, "a") as file:
            line = f"{zarr_store} \t {len(mols)} \n"
            file.write(line)




    # TODO: convert pharmacophore and names into tensors
    # TODO: can you combine the conformer_file -> smiles -> names into one query rather than two? one query that is batched?
    # TODO: process molecules in batches; 
    #     this includes using the NameFinder.query_batch method instead of NameFinder.query
    #     you can also batch with NameFinder.query_smiles_from_file_batch in stead of NameFinder.query_smiles_from_file
    #     MoleculeTensorizer can handle batches of molecules
    # TODO: parallelize processing: hand chunks of conformer files to subprocesses
    # TODO: write molecules to disk in chunks
        

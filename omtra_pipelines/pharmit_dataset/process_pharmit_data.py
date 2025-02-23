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

from omtra_pipelines.pharmit_dataset.phase1 import *

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
    p.add_argument('--batches_per_query', type=int, default=4, help='Number of batches per query.')

    args = p.parse_args()
    return args

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
        is_data_dir = data_dir.is_dir() and re.match(r'data\d{2}', data_dir.name)
        if not is_data_dir:
            continue
        conformers_dir = data_dir / 'conformers'
        for conformer_subdir in conformers_dir.iterdir():
            for conformer_file in conformer_subdir.iterdir():
                yield conformer_file



    

def process_batch(chunk_data, atom_type_map, ph_type_idx, database_list):
    global name_finder
    mol_tensorizer = MoleculeTensorizer(atom_map=atom_type_map)

    # chunk data is a list of tuples, each tuple contains (conformer_file, smile)

    smiles, conformer_files = chunk_data

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


    # (BATCHED) Database source
    names, failed_names_idxs = name_finder.query_name_from_smiles(smiles)
    # TODO: name parsing is broken 

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
    with Pool(processes=args.n_cpus, initializer=worker_initializer, initargs=(args.spoof_db,)) as pool:
        for chunk_idx, chunk_data in enumerate(batch_iter):

            chunk_data_file = f"{args.chunk_data_dir}/data_chunk_{chunk_idx}.npz"
            chunk_info_file = f"{args.chunk_info_dir}/data_chunk_{chunk_idx}.pkl"

            pool.apply_async(
                process_batch, 
                args=(chunk_data, atom_type_map, ph_type_idx, database_list), 
                callback=partial(save_chunk_to_disk, 
                        chunk_data_file=chunk_data_file, 
                        chunk_info_file=chunk_info_file))

        
        pool.close()
        pool.join()

def run_single(args, batch_iter):
    worker_initializer(args.spoof_db)
    for chunk_idx, chunk_data in enumerate(batch_iter):

        chunk_data_file = f"{args.chunk_data_dir}/data_chunk_{chunk_idx}.npz"
        chunk_info_file = f"{args.chunk_info_dir}/data_chunk_{chunk_idx}.pkl"

        tensors = process_batch(chunk_data, atom_type_map, ph_type_idx, database_list)
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

    batches_per_query = args.batches_per_query

    if args.n_chunks is None:
        args.n_chunks = float('inf')

    db_crawler = DBCrawler(query_size=args.batch_size, 
                           max_num_queries=args.n_chunks,
                           spoof_db=spoof_db)

    start_time = time.time()

    if args.n_cpus == 1:
        run_single(args, db_crawler)
    else:
        run_parallel(args, db_crawler)

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")

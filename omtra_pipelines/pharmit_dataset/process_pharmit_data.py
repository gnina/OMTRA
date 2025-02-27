import argparse
from pathlib import Path
import traceback
import shutil
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial

from omtra.data.xace_ligand import MoleculeTensorizer
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

    

    p.add_argument('--output_dir', type=Path,
                   help='Output directory for processed data.', 
                   default=Path('./outputs/phase1'))
    
    p.add_argument('--overwrite', action='store_true', help='Remove anything in existing output directory.')

    p.add_argument('--atom_type_map', type=list, default=["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"])
    p.add_argument('--batch_size', type=int, default=50, help='Number of conformer files to batch togther.')
    p.add_argument('--pharm_types', type=list, default=['Aromatic','HydrogenDonor','HydrogenAcceptor','Hydrophobic','NegativeIon','PositiveIon'], help='Pharmacophore center types.')
    p.add_argument('--counterions', type=list, default=['Na', 'Ca', 'K', 'Mg', 'Al', 'Zn'])
    p.add_argument('--databases', type=list, default=["CHEMBL", "ChemDiv", "CSC", "Z", "CSF", "MCULE","MolPort", "NSC", "PubChem", "MCULE-ULTIMATE","LN", "LNL", "ZINC"])
    p.add_argument('--max_num_atoms', type=int, default=120, help='Maximum number of atoms in a molecule.')
    p.add_argument('--chunk_offload_threshold', type=int, default=1000, help='Threshold for offloading chunks to disk, in MB.')
    p.add_argument('--register_write_interval', type=int, default=10, help='Interval for recording processed chunks.')

    p.add_argument('--n_cpus', type=int, default=2, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--n_chunks', type=int, default=None, help='Number of to process. If None, process all. This is only for testing purposes.')

    args = p.parse_args()
    return args

def process_batch(chunk_data, atom_type_map, ph_type_idx, database_list, max_num_atoms):
    global name_finder
    mol_tensorizer = MoleculeTensorizer(atom_map=atom_type_map)

    # chunk data is a list of tuples, each tuple contains (conformer_file, smile)

    smiles, conformer_files = chunk_data

    # (BATCHED) Database source
    names, failed_names_idxs = name_finder.query_name_from_smiles(smiles)

    # Remove molecules that couldn't get database data
    if len(failed_names_idxs) > 0:
        #print("Database sources for", len(failed_names_idxs), "could not be found, removing")
        conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_names_idxs]

    # Tensorize database sources
    databases = generate_library_tensor(
        names, 
        database_list, 
        filter_unknown=False,
        other_category=True
    )
    # databases, failed_idxs = generate_library_tensor(names, database_list, filter_unknown=True)
    # mols = [  mol for i, mol in enumerate(mols) if i not in failed_idxs]
    # conformer_files = [file for i, file in enumerate(conformer_files) if i not in failed_idxs]
    # smiles = [smile for i, smile in enumerate(smiles) if i not in failed_idxs]

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
        failed_mask = np.zeros(len(mols), dtype=bool)
        failed_mask[failed_mol_idxs] = True
        databases = databases[~failed_mask]

    # filter molecules with too many atoms
    too_big_idxs = []
    for i, mol in enumerate(mols):
        if mol.GetNumAtoms() > max_num_atoms:
            too_big_idxs.append(i)
    too_big_idxs = set(too_big_idxs)
    if len(too_big_idxs) > 0:
        mols = [mol for i, mol in enumerate(mols) if i not in too_big_idxs]
        failed_mask = np.zeros(len(mols), dtype=bool)
        failed_mask[list(too_big_idxs)] = True
        databases = databases[~failed_mask]

    # Get pharmacophore data
    x_pharm, a_pharm, failed_pharm_idxs = get_pharmacophore_data(conformer_files, ph_type_idx)
    # Remove ligands where pharmacophore generation failed
    if len(failed_pharm_idxs) > 0 :
        #print("Failed to generate pharmacophores for,", len(failed_pharm_idxs), "molecules, removing")
        x_pharm = [x for i, x in enumerate(x_pharm) if i not in failed_pharm_idxs]
        a_pharm = [a for i, a in enumerate(a_pharm) if i not in failed_pharm_idxs]
        mols = [mol for i, mol in enumerate(mols) if i not in failed_pharm_idxs]
        failed_mask = np.zeros(len(mols), dtype=bool)
        failed_mask[failed_pharm_idxs] = True
        databases = databases[~failed_mask]
        
    
    # Get XACE data
    positions, atom_types, atom_charges, bond_types, bond_idxs, num_xace_failed, failed_xace_idxs = mol_tensorizer.featurize_molecules(mols) # (BATCHED) Tensor representation of molecules
    # Remove molecules that failed to get xace data
    if len(failed_xace_idxs) > 0:
        #print("XACE data for,", num_xace_failed, "molecules could not be found, removing")
        x_pharm = [x for i, x in enumerate(x_pharm) if i not in failed_xace_idxs]
        a_pharm = [a for i, a in enumerate(a_pharm) if i not in failed_xace_idxs]
        failed_mask = np.zeros(len(x_pharm), dtype=bool)
        failed_mask[failed_xace_idxs] = True
        databases = databases[~failed_mask]
    

    # Save tensors in dictionary
    tensors = {
        'positions': positions, 
        'atom_types': atom_types, 
        'atom_charges': atom_charges, 
        'bond_types': bond_types, 
        'bond_idxs': bond_idxs, 
        'x_pharm': x_pharm, 
        'a_pharm': a_pharm, 
        'databases': databases}
    
    return tensors

def save_and_update(result, chunk_idx, pbar, chunk_saver):
    # Save the result using your existing callback logic.
    chunk_saver.save_chunk_to_disk(result, chunk_idx=chunk_idx)
    # Update the progress bar by one step.
    pbar.update(1)

def error_and_update(error, pbar, error_counter):
    """Handle errors, update error counter and the progress bar."""
    print(f"Error: {error}")
    traceback.print_exception(type(error), error, error.__traceback__)
    # Increment the error counter (using a mutable container)
    error_counter[0] += 1
    # Optionally, update the tqdm bar's postfix to show the current error count.
    pbar.set_postfix({'errors': error_counter[0]})
    # Advance the progress bar, since this job is considered done.
    pbar.update(1)


def run_parallel(n_cpus: int, spoof_db: bool, batch_iter: DBCrawler, 
                 chunk_saver: ChunkSaver, process_args: tuple, 
                 max_pending: int = None):
    # Set a default limit if not provided
    if max_pending is None:
        max_pending = n_cpus * 2  # adjust this factor as needed

    total_tasks = len(batch_iter)
    pbar = tqdm(total=total_tasks, desc="Processing", unit="chunks")
    # Use a mutable container to track errors.
    error_counter = [0]

    with Pool(processes=n_cpus, initializer=worker_initializer, initargs=(spoof_db,)) as pool:
        pending = []
        for chunk_idx, chunk_data in enumerate(batch_iter):

            if chunk_saver.chunk_processed(chunk_idx):
                pbar.update(1)
                continue

            # Wait until the number of pending jobs is below the threshold.
            # We remove finished tasks from the list, and if still too many remain,
            # we sleep briefly before re-checking.
            while len(pending) >= max_pending:
                # Filter out jobs that have finished
                pending = [r for r in pending if not r.ready()]
                if len(pending) >= max_pending:
                    time.sleep(0.1)  # brief pause before checking again

            # Wrap the original success callback to also update the progress bar.
            callback_fn = partial(save_and_update, chunk_idx=chunk_idx, pbar=pbar, chunk_saver=chunk_saver)
            # Wrap the error callback to update the progress bar and error counter.
            error_callback_fn = partial(error_and_update, pbar=pbar, error_counter=error_counter)

            # Submit the job and add its AsyncResult to the pending list
            result = pool.apply_async(
                process_batch, 
                args=(chunk_data, *process_args), 
                callback=callback_fn,
                error_callback=error_callback_fn
            )
            pending.append(result)

        # After submitting all jobs, wait for any remaining tasks to complete.
        for result in pending:
            result.wait()

        pool.close()
        pool.join()

def run_single(spoof_db, batch_iter, chunk_saver, process_args: tuple):
    worker_initializer(spoof_db)
    pbar = tqdm(total=len(batch_iter), desc="Processing", unit="chunks")
    for chunk_idx, chunk_data in enumerate(batch_iter):

        if chunk_saver.chunk_processed(chunk_idx):
            pbar.update(1)
            continue

        tensors = process_batch(chunk_data, *process_args)
        chunk_saver.save_chunk_to_disk(tensors, chunk_idx=chunk_idx)
        pbar.update(1)


if __name__ == '__main__':

    args = parse_args()
    database_list = args.databases
    atom_type_map = args.atom_type_map
    spoof_db = args.spoof_db
    ph_type_idx = {type:idx for idx, type in enumerate(args.pharm_types)}

    if args.n_chunks is None:
        args.n_chunks = float('inf')

    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    chunk_saver = ChunkSaver(
        output_dir=args.output_dir,
        register_write_interval=args.register_write_interval,
        chunk_offload_threshold_mb=args.chunk_offload_threshold
    )

    db_crawler = DBCrawler(query_size=args.batch_size, 
                           max_num_queries=args.n_chunks,
                           spoof_db=spoof_db)
    
    process_args = (atom_type_map, ph_type_idx, database_list, args.max_num_atoms)

    start_time = time.time()

    if args.n_cpus == 1:
        run_single(spoof_db, db_crawler, chunk_saver, process_args)
    else:
        run_parallel(args.n_cpus, spoof_db, db_crawler, chunk_saver, process_args)

    # off load last remaining chunks to masuda
    chunk_saver.offload_chunks()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")

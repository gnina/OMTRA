import argparse
from pathlib import Path
import traceback
import shutil
from tqdm import tqdm

from rdkit import RDLogger

# Disable RDKit warnings and errors
RDLogger.DisableLog('rdApp.*')

from multiprocessing import Pool
from functools import partial

from omtra.data.xace_ligand import MoleculeTensorizer
from omtra.utils.misc import combine_tcv_counts
import time


from omtra_pipelines.pharmit_dataset.guts.phase1 import *

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
    p.add_argument('--spoof_db', action='store_true', help='Spoof the database connection, for offline development')

    

    p.add_argument('--output_dir', type=Path,
                   help='Output directory for processed data.', 
                   default=Path('./outputs/phase1'))
    
    p.add_argument('--overwrite', action='store_true', help='Remove anything in existing output directory.')

    p.add_argument('--atom_type_map', type=list, default=["C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B"])
    p.add_argument('--batch_size', type=int, default=50, help='Number of conformer files to batch togther.')
    p.add_argument('--pharm_types', type=list, default=['Aromatic','HydrogenDonor','HydrogenAcceptor','Hydrophobic','NegativeIon','PositiveIon'], help='Pharmacophore center types.')
    p.add_argument('--counterions', type=list, default=['Na', 'Ca', 'K', 'Mg', 'Al', 'Zn'])
    p.add_argument('--databases', type=list, default=["CHEMBL", "ChemDiv", "CSC", "Z", "CSF", "MCULE","MolPort", "NSC", "PubChem", "MCULE-ULTIMATE","LN", "LNL", "ZINC"])
    p.add_argument('--max_num_atoms', type=int, default=120, help='Maximum number of atoms in a molecule.')
    p.add_argument('--min_num_atoms', type=int, default=4, help='Minimum number of atoms in a molecule.')
    p.add_argument('--chunk_offload_threshold', type=int, default=2000, help='Threshold for offloading chunks to disk, in MB.')
    p.add_argument('--register_write_interval', type=int, default=10, help='Interval for recording processed chunks.')

    p.add_argument('--n_cpus', type=int, default=2, help='Number of CPUs to use for parallel processing.')
    p.add_argument('--n_chunks', type=int, default=10, help='Number of to process. If None, process all. This is only for testing purposes.') # set to small 
    p.add_argument('--dev', action='store_true', help='Run in development mode.')   # use 

    args = p.parse_args()

    if args.dev:
        args.output_dir = args.output_dir.parent / f'dev_{args.output_dir.name}'

    return args

def process_batch(chunk_data, atom_type_map, ph_type_idx, database_list, max_num_atoms, min_num_atoms):
    """Filter tensors and original source records together at every processing stage."""
    global name_finder
    if len(chunk_data) != 3:
        raise ValueError('Pharmit batches must retain database IDs, SMILES and source SDF locators')
    source_ids, smiles, conformer_files = chunk_data
    if not len(source_ids) == len(smiles) == len(conformer_files):
        raise ValueError('Pharmit source columns differ in length')
    entries = list(zip(source_ids, smiles, conformer_files))
    names, failures = name_finder.query_name_from_smiles(smiles)
    if len(names) != len(entries):
        raise ValueError('Source-library lookup lost row correspondence')
    keep = [i for i in range(len(entries)) if i not in set(failures)]
    entries = [entries[i] for i in keep]
    names = [names[i] for i in keep]
    databases = generate_library_tensor(names, database_list, filter_unknown=False, other_category=True)

    read = [read_mol_from_conf_file(entry[2], return_record_index=True) for entry in entries]
    keep = [i for i, (mol, _) in enumerate(read) if mol is not None and min_num_atoms <= mol.GetNumAtoms() <= max_num_atoms]
    entries = [entries[i] for i in keep]
    mols = [read[i][0] for i in keep]
    record_indices = [read[i][1] for i in keep]
    databases = databases[keep]

    tensorizer = MoleculeTensorizer(atom_map=atom_type_map)
    xace, failures, failure_modes, _ = tensorizer.featurize_molecules(mols)
    if failure_modes:
        print(failure_modes, flush=True)
    keep = [i for i in range(len(mols)) if i not in set(failures)]
    entries = [entries[i] for i in keep]
    mols = [mols[i] for i in keep]
    record_indices = [record_indices[i] for i in keep]
    databases = databases[keep]
    if len(xace) != len(mols):
        raise ValueError('Tensorizer lost source correspondence')

    x_pharm, a_pharm, v_pharm, failures = get_pharmacophore_data(mols)
    keep = [i for i in range(len(mols)) if i not in set(failures)]
    entries = [entries[i] for i in keep]
    mols = [mols[i] for i in keep]
    record_indices = [record_indices[i] for i in keep]
    xace = [xace[i] for i in keep]
    databases = databases[keep]
    if not len(entries) == len(xace) == len(x_pharm) == len(a_pharm) == len(v_pharm):
        raise ValueError('Pharmacophore processing lost source correspondence')

    from omtra_pipelines.pharmit_dataset.guts.provenance import source_record
    records = [source_record(*entry, record_index, mol, tensor, atom_type_map)
               for entry, record_index, mol, tensor in zip(entries, record_indices, mols, xace)]
    return {
        'positions': [m.x for m in xace], 'atom_types': [m.a for m in xace],
        'atom_charges': [m.c for m in xace], 'bond_types': [m.e for m in xace],
        'bond_idxs': [m.edge_idxs for m in xace],
        'x_pharm': x_pharm, 'a_pharm': a_pharm, 'v_pharm': v_pharm,
        'databases': databases, 'source_records': records,
        'tcv_counts': combine_tcv_counts([m.tcv_counts for m in xace]),
    }

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

    # write the traceback to a file named 'error_log.txt'
    with open('error_log.txt', 'a') as f:
        traceback.print_exception(type(error), error, error.__traceback__, file=f)


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

    with Pool(processes=n_cpus, initializer=worker_initializer, initargs=(spoof_db,), maxtasksperchild=2) as pool:
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
        chunk_offload_threshold_mb=args.chunk_offload_threshold,
        dev_mode=args.dev
    )

    db_crawler = DBCrawler(query_size=args.batch_size, 
                           max_num_queries=args.n_chunks,
                           spoof_db=spoof_db)
    
    process_args = (atom_type_map, ph_type_idx, database_list, args.max_num_atoms, args.min_num_atoms)

    start_time = time.time()

    if args.n_cpus == 1:
        run_single(spoof_db, db_crawler, chunk_saver, process_args)
    else:
        run_parallel(args.n_cpus, spoof_db, db_crawler, chunk_saver, process_args)

    # offload the last remaining chunks (no-op unless OMTRA_OFFLOAD_HOST/DIR are set)
    chunk_saver.offload_chunks()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.1f} seconds")

import argparse
from pathlib import Path
import zarr
from copy import deepcopy
import numpy as np
import pickle
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('zarr_store', type=Path, help='Path to the zarr store where clusters will be added.')
    p.add_argument('clusters_file', type=Path, help='Path to the file containing clusters to be added.')
    p.add_argument('--write', action='store_true', help='Actually write clusters to zarr store, otherwise will do a dry run.')
    p.add_argument('--dry_run_output', type=Path, default=Path('./dry_run_sys_lookup.pkl'))
    args = p.parse_args()
    return args

def main(args):

    # read system lookup from zarr store
    store = zarr.storage.LocalStore(str(args.zarr_store))
    root = zarr.open(store=store, mode='a')
    sys_lookup = deepcopy(root.attrs['system_lookup'])

    # read clusters from file
    cdata = np.load(args.clusters_file)
    cluster_assignments = cdata['all_mol_cluster_assignments'].tolist()
    unique_mol_ids = cdata['unique_mapping'].tolist()

    # add cluster assignments to the system lookup
    for system_idx in range(len(sys_lookup)):
        sys_lookup[system_idx]['cluster_id'] = cluster_assignments[system_idx]
        sys_lookup[system_idx]['unique_mol_id'] = unique_mol_ids[system_idx]

    if args.write:
        # write updated system lookup back to zarr store
        root.attrs['system_lookup'] = sys_lookup
        print(f"Updated system lookup with cluster assignments and saved to {args.zarr_store}.")
    else:
        with open(args.dry_run_output, 'wb') as f:
            pickle.dump(sys_lookup, f)
        print(f"Updated system lookup with cluster assignments. Dry run output saved to {args.dry_run_output}.")
        
        # load dry run output and try to convert to pandas dataframe 
        with open(args.dry_run_output, 'rb') as f:
            dry_run_data = pickle.load(f)
        df = pd.DataFrame(dry_run_data)
        print("Dry run output converted to pandas DataFrame:")
        print(df.head(1))



if __name__ == "__main__":
    args = parse_args()
    main(args)
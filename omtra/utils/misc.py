import os
from collections import defaultdict
from rdkit import Chem
import traceback
import uuid
from pathlib import Path

class classproperty:
    def __init__(self, func):
        self.fget = func
    def __get__(self, instance, owner):
        return self.fget(owner)

def get_zarr_store_size(store_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(store_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def combine_tcv_counts(tcv_counts_list) -> defaultdict:
    combined_tcv_counts = defaultdict(int)
    for tcv_counts in tcv_counts_list:
        for tcv, count in tcv_counts.items():
            combined_tcv_counts[tcv] += count
    return combined_tcv_counts

def bad_mol_reporter(mol, note=None):
    uuid_str = str(uuid.uuid4())[:4]
    bad_mols_dir = Path("./bad_mols/")
    bad_mols_dir.mkdir(exist_ok=True)
    error_filepath = bad_mols_dir / f"error_{uuid_str}.txt"
    with open(error_filepath, 'w') as error_file:
        traceback.print_exc(file=error_file)
        if note:
            error_file.write(f"\n\n{note}")

    Chem.MolToMolFile(mol, str(bad_mols_dir / f"mol_{uuid_str}.sdf"), kekulize=False)
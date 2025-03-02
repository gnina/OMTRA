import os
from collections import defaultdict

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
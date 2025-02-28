import os

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
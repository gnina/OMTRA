import zarr

def list_zarr_arrays(root: zarr.Group) -> list:
    """
    Recursively crawls a Zarr store and returns paths to all arrays.

    Parameters:
        store_path (str): Path to the Zarr store.

    Returns:
        list: A list of paths to all arrays in the Zarr store.
    """
    def _find_arrays(group, prefix=""):
        arrays = []
        for name, item in group.members():
            current_path = f"{prefix}/{name}".lstrip("/")
            if isinstance(item, zarr.Group):
                # Recursively explore the group
                arrays.extend(_find_arrays(item, current_path))
            elif isinstance(item, zarr.Array):
                # Add the array path
                arrays.append(current_path)
        return arrays

    # Recursively find all arrays
    return _find_arrays(root)
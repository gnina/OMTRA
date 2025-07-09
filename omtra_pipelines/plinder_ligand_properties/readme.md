## Computes New Atom Features for Plinder Ligands

### 1. Creating an empty Zarr array for new atom features
The script `phase1.py` is used to create an empty Zarr array.

The script opens the Zarr store and creates an empty Zarr array of size (n_atoms, n_feats) with an attribute 'features' that identifies each feature in the array. It does not overwrite an existing array with the same name and path.

Example usage:

```python
python phase1.py \
    --plinder_path /net/galaxy/home/koes/icd3/moldiff/OMTRA/data/plinder \ # path to the Plinder dataset
    --store_name train  \ # train/val
    --n_feats 6 \ # number of extra features
    --array_name extra_feats \ # name of the new zarr array
    --feat_names ['impl_H', 'aro', 'hyb', 'ring', 'chiral', 'frag']  \ # name of the extra features
```

### 2. Creating an empty Zarr array for new atom features
The script `run_phase2.py` is used to create compute the new atom features and write these to the empty Zarr array created by `phase1.py`. It processes all Plinder versions at once.

The script gets the DGL graphs of Plinder ligands using the PlinderDataset class and computes new atom features:
    impl_H: Number of implicit hydrogens (categorical)
    aro: Whether the atom is in an aromatic ring (binary)
    hyb: Hybridization  (categorical)
    ring: Whether the atom is in a ring (binary)
    chiral: Whether the atom is a chiral center (binary)
    frag: Fragment identity (categorical)

Functions for ligand property computation, fragment computation, and writing data to the Zarr array are found in `phase2.py`. This process can be parallelized by setting `n_cpus` > 1. Indices of failed atoms are written to `failed_ligands.txt` and errors associated with writing to the Zarr store are logged in `error_log.txt`.

Example usage:

```python
python run_phase2.py \
    --plinder_path /net/galaxy/home/koes/icd3/moldiff/OMTRA/data/plinder \ # path to the Plinder dataset
    --store_name train  \ # train/val
    --array_name extra_feats \ # name of the new zarr array
    --n_cpus 1 \ # number of cpus. 1 cpu uses run_single (un-parallelized)
    --output_dir Path('./outputs/phase2')  \ # output directory for error files
```
We're going to cluster all the ligands in the plinder dataset. This is the approach:

1. convert ligand in zarr to rdkit, convert rdkit mol to fingerprint
2. compute pairwise distance matrix
3. apply agglomerative clustering
4. perform some data analysis; do all nucleotides go to one cluster? how big are the clusters?
5. finalize cluster assignments, write to zarr store along with cluster frequencies

ok so this is like, high-level, some of these steps require more detail here.

# converting zarr to fingerprints

- SCRIPT 1: this process is embarassingly parallel. we need a script that can turn zarr into fingerprints, and can take as input a block size and block index. we can also use the ZarrDataset class directly, or some super simple custom version of it, so we get cached reading of the data. 

- SCRIPT 2: we therefore need a script that looks at the zarr store, takes in a block size, and writes a file of commands that will run the previous script


# pipeline setup

1. `make_fp_cmds.py` this will write a file `make_fp_cmds.sh`
```console
python scripts/make_fp_cmds.py /home/ian/projects/mol_diffusion/OMTRA/data/plinder/exp --block-size=5000 --output_dir=./fps/exp/blocks
./make_fp_cmds.sh
```
```console
python scripts/make_fp_cmds.py /home/ian/projects/mol_diffusion/OMTRA/data/plinder/pred --block-size=5000 --output_dir=./fps/pred/blocks
./make_fp_cmds.sh
```
2. run `make_fp_cmds.sh` to generate fingerprints; the commands here are just calling get_fingerprints.py
3. `merge_fps.py` - this will collect all fingerprints, deduplicate them, and cluster them, and write all the results to a single numpy file

```console
python scripts/merge_fps.py fps/no_links/blocks fps/no_links/clusters.npz
python scripts/merge_fps.py fps/exp/blocks fps/exp/clusters.npz
python scripts/merge_fps.py fps/pred/blocks fps/pred/clusters.npz
```

3. we can run `scripts/add_clusters_to_zarr.py` to insert cluster assignments into the zarr stores


```console
python scripts/add_clusters_to_zarr.py /home/ian/projects/mol_diffusion/OMTRA/data/plinder/no_links/train.zarr fps/no_links/clusters.npz
```
4. 
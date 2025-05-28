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
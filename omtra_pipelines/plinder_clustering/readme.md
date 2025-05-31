# how to run da pipeline

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

note that because this script modifies the zarr store in place, this is kind of dangerous, the script by default only does a dry run. you have to add the `--write` flag to actually write the changes. 

```console
python scripts/add_clusters_to_zarr.py /home/ian/projects/mol_diffusion/OMTRA/data/plinder/no_links/train.zarr fps/no_links/clusters.npz
python scripts/add_clusters_to_zarr.py /home/ian/projects/mol_diffusion/OMTRA/data/plinder/exp/train.zarr fps/exp/clusters.npz
python scripts/add_clusters_to_zarr.py /home/ian/projects/mol_diffusion/OMTRA/data/plinder/pred/train.zarr fps/pred/clusters.npz
```


# TODO:
- [ ] once cluster assignments are in zarr store, plinder dataset class must compute p_skip and hand over to chunk tracker
- [ ] parameter for controlling how close to uniform sampling over clusters
- [ ] measure how many things are skipped by the weighted sampling
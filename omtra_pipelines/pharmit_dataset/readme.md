This pipeline processes data on jabba. We also periodically shuffle processed data from jabba back to masuda. 

To be able to shuttle things from jabba to masuda we have to setup a reverse tunnel. The command for doing so is:

```console
autossh -M 20000 -f -N -R 2222:localhost:22 jabba
```

## running phase 1:
```console
python run_phase1.py --batch_size=50000 --n_cpus=24 --overwrite 2>&1 | tee -a output.log
```

`2>&1 | tee -a output.log`

## running phase 2:

```console
python run_phase2.py --store_dir=/home/ian/projects/mol_diffusion/OMTRA/data/pharmit --store_name=train.zarr --n_chunks_zarr=4000 --n_cpus=10
```



# TODO:
- [ ] dataset splits? datamodule currently expects a train and val set
- [ ] fix zarr dataset graphs_per_chunk 
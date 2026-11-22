# Pharmit dataset pipeline

Phase 1 (`run_phase1.py`) processes the raw Pharmit database into chunks. Phase 2 (`run_phase2.py`)
assembles the chunks into a Zarr store.

Phase 1 can optionally copy finished chunks to another machine to save local disk. Set
`OMTRA_OFFLOAD_HOST` (an ssh host) and `OMTRA_OFFLOAD_DIR` (a directory on that host). If they are
not set, chunks stay in the local output directory.

## running phase 2:

```console
python run_phase2.py --store_dir=data/pharmit --store_name=train.zarr --n_chunks_zarr=4000 --n_cpus=10
```

# TODO:
- [ ] dataset splits? datamodule currently expects a train and val set
- [ ] fix zarr dataset graphs_per_chunk

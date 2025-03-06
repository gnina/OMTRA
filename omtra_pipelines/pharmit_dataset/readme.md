This pipeline processes data on jabba. We also periodically shuffle processed data from jabba back to masuda. 

To be able to shuttle things from jabba to masuda we have to setup a reverse tunnel. The command for doing so is:

```console
autossh -M 20000 -f -N -R 2222:localhost:22 jabba
```

# TODO:
- [x] phase2 needs to extract/merge n_node distirbutions
- [x] where do we store n_node distributions? needs to be outside zarr store
    - so in a processed_data_dir, there are zarr stores and histogram files
- [x] need to collect valid valencies
    - will need to coordinate changes to xace module with plinder pipeline
- [x] double-check xace module does kekulization / sanitization
- [ ] dataset splits? datamodule currently expects a train and val set
- [ ] fix zarr dataset graphs_per_chunk 
- [x] write out bad mols on pharmacophore error
- [ ] pipeline should use hydra configs instead of argparse
- [ ] write atom type map into zarr store?

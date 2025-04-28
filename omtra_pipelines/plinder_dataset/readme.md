# OMTRA Pipelines
## Plinder Dataset

### Storage with Linked Structures
The script `store_linked_structures.py` can be used to store Plinder systems with corresponding experimental/predicted structures. 

This script will write logs to a file set by an environment variable `LOG_FILE_PATH`. 

Example usage:

```python
python store_linked_structures.py \
    --data omtra_pipelines/plinder_dataset/plinder_filtered.parquet \ # path to parquet file with filtering information 
    --split train \ # train/val/test/unassigned
    --type apo \ # apo (experimental)/ pred (AF)
    --output plinder/apo/train.zarr \ # path for zarr store
    --pocket_cutoff 8.0 \ # angstrom cutoff for pocket extraction (defaults to 8)
    --num_systems 10 \ # optional, only stores num_systems systems for testing
    --num_cpus 32 \ # defaults to 1
    --max_pending 64 \ # Maximum number of pending jobs (default: 2*num_cpus)
    --embeddings \ # optional, Generate ESM3 embeddings for associated protein pockets
```

### Storage without Linked Structures
The script `store_unlinked_structures.py` can be used to store Plinder systems. 

This script will write logs to a file set by an environment variable `LOG_FILE_PATH`. 

Example usage:

```python
python store_unlinked_structures.py \
    --data omtra_pipelines/plinder_dataset/plinder_filtered.parquet \ # path to parquet file with filtering information 
    --split train \ # train/val/test/unassigned
    --output plinder/no_links/train.zarr \ # path for zarr store
    --pocket_cutoff 8.0 \ # angstrom cutoff for pocket extraction (defaults to 8)
    --num_systems 10 \ # optional, only stores num_systems systems for testing
    --num_cpus 32 \ # defaults to 1
    --max_pending 64 \ # Maximum number of pending jobs (default: 2*num_cpus)
    --embeddings \ # optional, Generate ESM3 embeddings for associated protein pockets
```

### Plinder Pipeline
In `plinder_pipeline.py`, a `SystemProcessor` class is defined to create `SystemData` objects for a Plinder system. The definition of a `SystemData` object is shown below. For each system in Plinder, a filtering protocol is defined to distinguish between focal ligands and non-protein, non-designable entities. For each focal ligand of a system, a `SystemData` object is created (any other focal ligands stored in npndes). The object also includes the corresponding pocket and pharmacophore, as well as the linked experimental or AF predicted structure (optionally). The linked structure is aligned to the receptor (holo) structure and they are cropped to contain the same atoms/residues. 

```python
class SystemData:
    system_id: str
    ligand_id: str
    receptor: StructureData
    ligand: LigandData
    pharmacophore: PharmacophoreData
    pocket: StructureData
    npndes: Optional[Dict[str, LigandData]] = None
    link_id: Optional[str] = None
    link_type: Optional[str] = None
    link: Optional[StructureData] = None
```
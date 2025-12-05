# Pharmit Dataset

Where is the pharmit dataset? How do we download it? Somebody tell us please!!!!

The pharmit dataset is available for independent use separate from OMTRA. The dataset class can be found under `pharmit_utils/pharmit.py`. An instance of the `PharmitDataset` class can be configured to return RDKit molecules or a dictionary of tensors.

### Usage
| Argument | Default | Description | 
|----------|-------------|-------------| 
| `data_dir` | Requires | Path to Pharmit dataset Zarr store. |
| `split` | Required | Data split. |
| `return_type` | Required | Options: `rdkit` or `dict`. If `rdkit`, ligands are returned as RDKit molecules. Extra features and pharmacophore data will not be returned. If `dict`, ligand data will be returned as a dictionary of tensors. Extra features and pharmacophore data will be stored as nested dictionaries under the keys `extra_feats` and `pharm`, respectively. |
| `include_pharmacophore` | `False` | Include pharmacophore features. |
| `include_extra_feats` | `False` | Include atom extra features: implicit hydrogens, aromatic flag, hybridization, ring flag, chiral flag. |
| `n_chunks_cache` | `4` |  |

#### Example Usage
```console
dataset = PharmitDataset(data_dir='/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit',
                         split='test',
                         return_type='rdkit')
mol = dataset[0]
```
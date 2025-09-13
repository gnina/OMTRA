# Processing the Crossdocked Dataset
The CrossDocked2020 dataset is a large-scale structure-based machine learning dataset containing 22.5 million poses of ligands cross-docked into 18,450 protein complexes from the Protein Data Bank. This approach provides a more realistic evaluation of model performance by testing on protein structures different from those used in training, better reflecting real-world drug discovery scenarios.

The dataset includes internal splits (created by our lab using `.types` files) and external splits (alternative partitioning by other researchers via `split_by_name.pt`) for training and validation.

The fully processed data can be located at: 

External Splits: /net/galaxy/home/koes/jmgupta/omtra_2/data/crossdocked/external_split
Internal Splits: 


## External Split Processing

To process the external splits of the Crossdocked dataset, you must use the `split_by_name.pt` file located at:

/net/galaxy/home/koes/jmgupta/omtra_2/omtra_pipelines/crossdocked_dataset/crossdocked_external_splits/split_by_name.pt

### File Structure

This file contains a PyTorch dictionary with the following structure:

- **Data Type**: Dictionary
- **Keys**: `['train', 'test']`
- **Training Set**: 100,000 samples
- **Test Set**: 100 samples

### Data Format

Each sample is a tuple containing:
1. **PDB file path** (protein structure): `*.pdb`
2. **SDF file path** (ligand structure): `*.sdf`

#### Example Data Points
```python
# Training set examples:
('DYR_STAAU_2_158_0/4xe6_X_rec_3fqc_55v_lig_tt_docked_4_pocket10.pdb', 
 'DYR_STAAU_2_158_0/4xe6_X_rec_3fqc_55v_lig_tt_docked_4.sdf')

('TRY1_BOVIN_66_246_0/1k1j_A_rec_1yp9_uiz_lig_tt_docked_1_pocket10.pdb', 
 'TRY1_BOVIN_66_246_0/1k1j_A_rec_1yp9_uiz_lig_tt_docked_1.sdf')

```
### Processing Script Overview

The main processing script is `run_crossdocked_processing_external_splits.py` which:
- Loads the external split file (`split_by_name.pt`)
- Processes protein-ligand pairs in parallel batches
- Outputs processed data to Zarr format for training and validation

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--cd_directory` | `/net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types` | Crossdocked types file directory |
| `--pocket_cutoff` | `8.0` | Pocket cutoff distance (Angstroms) |
| `--zarr_output_dir` | `test_external_output.zarr` | Output directory for processed Zarr files |
| `--root_dir` | `/net/galaxy/home/koes/paf46_shared/cd2020_v1.3` | Root directory for crossdocked data |
| `--max_batches` | `None` | Maximum number of batches to process (None = all) |
| `--batch_size` | `500` | Number of ligand-receptor pairs per batch |
| `--n_cpus` | `8` | Number of CPUs for parallel processing |
| `--max_pending` | `32` | Maximum pending jobs in the processing pool |

### Example Running with SLURM

Create a SLURM script:
```bash
#!/bin/bash
#SBATCH -J crossdocked
#SBATCH --partition=dept_cpu
#SBATCH -o slurm_output/crossdocked_processing/%A_%a.out
#SBATCH -e slurm_output/crossdocked_processing/%A_%a.out
#SBATCH --cpus-per-task 18

hostname

source ~/.bashrc
mamba activate omtra

python omtra_pipelines/crossdocked_dataset/run_crossdocked_processing_external_splits.py \
  --cd_directory /net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types \
  --pocket_cutoff 8.0 \
  --zarr_output_dir /net/galaxy/home/koes/jmgupta/omtra_2/data/crossdocked/external_split \
  --root_dir /net/galaxy/home/koes/paf46_shared/cd2020_v1.3 \
  --max_batches None \
  --batch_size 500 \
  --n_cpus 16 \
  --max_pending 32
```

## Internal Processing

The internal splits use predefined train/val pairs from the Crossdocked dataset's `.types` files. These files can be located at: 

/net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types

### File Structure





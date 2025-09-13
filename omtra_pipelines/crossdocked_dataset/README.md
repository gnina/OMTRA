# Processing the Crossdocked Dataset
The CrossDocked2020 dataset is a large-scale structure-based machine learning dataset containing 22.5 million poses of ligands cross-docked into 18,450 protein complexes from the Protein Data Bank. This approach provides a more realistic evaluation of model performance by testing on protein structures different from those used in training, better reflecting real-world drug discovery scenarios.

The dataset includes internal splits (created by our lab using `.types` files) and external splits (alternative partitioning by other researchers via `split_by_name.pt`) for training and validation.

The fully processed data can be located at: 

**External Splits:** /net/galaxy/home/koes/jmgupta/omtra_2/data/crossdocked/external_split

**Internal Splits:** 

## File Directory

### Processing Pipeline Overview

The pipeline uses different entry points based on the split type: `run_crossdocked_processing.py` for internal splits or `run_crossdocked_processing_external_splits.py` for external splits. Both scripts instantiate the `CrossdockedNoLinksZarrConverter` class from `crossdocked_unlink_zarr.py`, which in turn uses the `SystemProcessor` class from `pipeline_components.py` for molecular feature extraction.

### Core Processing Files

- **`crossdocked_unlink_zarr.py`**: Handles batch creation from receptor and ligand file paths, coordinates batch processing, and manages writing processed data to Zarr format
- **`pipeline_components.py`**: Core processing module that extracts and calculates molecular features from ligand and receptor structures for storage
- **`run_crossdocked_processing.py`**: Main script for processing internal splits of the CrossDocked dataset using `.types` files
- **`run_crossdocked_processing_external_splits.py`**: Main script for processing external splits using the `split_by_name.pt` file

### Utilities and Testing

- **`test_crossdocked.py`**: Testing script for inspecting and validating processed dataset outputs
- **`README.md`**: Documentation for the dataset processing pipeline

### External Splits

- **`crossdocked_external_splits/`**: Directory containing external split definitions and related files
  - **`split_by_name.pt`**: PyTorch file defining custom train/test splits created by external researchers

### Processing Workflow

1. **Script Entry**: Choose `run_crossdocked_processing.py` (internal) or `run_crossdocked_processing_external_splits.py` (external)
2. **Data Reading**: `CrossdockedNoLinksZarrConverter` reads file paths and creates processing batches
3. **Feature Extraction**: `SystemProcessor` processes molecular structures and calculates features
4. **Batch Processing & Storage**: `CrossdockedNoLinksZarrConverter` coordinates parallel batch processing and writes to Zarr format

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

The `.types` files contain tab/space-separated lines where each line represents a protein-ligand pair. The format is:

#### Example Entry

1 0.0000 1.1921 1A1D_CYBSA_1_341_0/1j0c_A_rec_0.gninatypes 1A1D_CYBSA_1_341_0/1j0c_A_rec_1j0d_5pa_lig_tt_min_0.gninatypes #-9.1514

- **Column 1**: Label (must be "1" for valid entries)
- **Column 2**: Numerical value 
- **Column 3**: Numerical value
- **Column 4**: Relative path to receptor `.gninatypes` file
- **Column 5**: Relative path to ligand `.gninatypes` file
- **Comment**: Binding affinity value (e.g., #-9.1514)

#### Processing Notes
- Only lines starting with "1" are processed (valid entries)
- File paths are relative to the dataset root directory
- Both receptor and ligand files must exist for the pair to be included
- Lines with fewer than 5 columns are skipped
- Files use `.gninatypes` format (GNINA's input format)

#### Types Files Used
The internal split processing uses three predefined types file pairs. Each pair represents a complete train/validation split, resulting in three separate internal splits (split0, split1, split2) for training.
```python
types_files_pairs = [
    ("it2_tt_v1.3_0_train0.types", "it2_tt_v1.3_0_test0.types"),
    ("it2_tt_v1.3_0_train1.types", "it2_tt_v1.3_0_test1.types"),
    ("it2_tt_v1.3_0_train2.types", "it2_tt_v1.3_0_test2.types")
]
```
### Internal Split Dataset Sizes

| Split | Training Set | Test Set | Total |
|-------|-------------|----------|-------|
| Split 0 | 14,429,208 | 8,137,241 | 22,566,449 |
| Split 1 | 15,103,135 | 7,463,314 | 22,566,449 |
| Split 2 | 15,600,555 | 6,965,894 | 22,566,449 |

*Note: These numbers represent the total lines in each `.types` file, including both valid and invalid entries. The actual number of processed receptor-ligand pairs may be lower after filtering for valid entries.*

### Processing Script Overview

The main processing script is `run_crossdocked_processing.py` which:
- Loads all 6 types file shown above
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

# write your command line commands here
python omtra_pipelines/crossdocked_dataset/run_crossdocked_processing_external_splits.py \
  --cd_directory /net/galaxy/home/koes/paf46_shared/cd2020_v1.3/types \
  --pocket_cutoff 8.0 \
  --zarr_output_dir /net/galaxy/home/koes/jmgupta/omtra_2/data/crossdocked/external_split \
  --root_dir /net/galaxy/home/koes/paf46_shared/cd2020_v1.3 \
  --max_batches None \
  --batch_size 500 \
  --n_cpus 16 \
  --max_pending 32


#eval $cmd
exit
```
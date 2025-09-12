# Evaluates Models on Protein-Conditioned Tasks
## Overview
This script evaluates models on a user-specified protein-conditioned (+pharm-conditioned) task across various metrics. While some automatic task-specific metric disabling has been pre built in, users can disable any metric using the appropriate `--disable_METRIC` flag. The script also automatically computes metrics for the ground truth ligand when applicable. This feature can also be disabled.

### Tasks supported
| Task Name | Description | 
|----------|-------------|
| `rigid_docking_condensed` | Rigid docking |
| `fixed_protein_ligand_denovo_condensed` | Rigid protein, de novo ligand generation |
| `flexible_docking_condensed` | Flexible docking |
| `protein_ligand_denovo_condensed` | Flexible docking |
| `rigid_docking_pharmacophore_condensed` | Pharmacophore-conditioned rigid docking |
| `fixed_protein_pharmacophore_ligand_denovo_condensed` | Pharmacophore-conditioned, rigid protein, de novo ligand generation |
| `flexible_docking_pharmacophore_condensed` | Pharmacophore-conditioned flexible docking |
| `protein_pharmacophore_ligand_denovo_condensed` | Pharmacophore-conditioned, flexible protein, de novo ligand generation |


## Usage
#### Inputs/Outputs
| Argument | Default | Description | 
|----------|-------------|-------------| 
| `--ckpt_path` | `None` | Path to model checkpoint. |
| `--samples_dir` | `None` | Path to samples. Use existing samples, do not sample a model. |
| `--output_dir` | `./outputs/TASK_NAME/` |  Output directory. |
| `--sys_info_file` | `output_dir/TASK_NAME_sys_info.csv` | Path to a system info file (optional). This is configured automatically if you sample using Plinder. Users can also use this argument to specify a file path to additional system info. It must have columns: `sys_id`, `protein_id`, `gen_ligand_id` to merge with the metrics dataframe. |

#### Sampling
| Argument | Default | Description | 
|----------|-------------|-------------| 
| `--task` | Required | Task to sample for (e.g. rigid_docking_condensed). |
| `--n_samples` | Required | Number of samples. |
| `--n_replicates` | Required | Number of replicates per sample. |
| `--n_timesteps` | `250` | Number of integration steps to take when sampling. |
| `--stochastic_sampling` | `False` | If set, perform stochastic sampling. |
| `--noise_scaler` | `1` | Noise scaling param for stochastic sampling. |
| `--eps` | `0.01` | g(t) param for stochastic sampling. |
| `--max_batch_size` | `500` | Maximum number of systems to sample per batch. |
| `--dataset` | `plinder` | Dataset. Options: Plinder or Crossdocked |
| `--split` | `val` | Data split (i.e., train, val). |
| `--dataset_start_idx` | `0` | Index in the dataset to start sampling from. |
| `--plinder_path` | 'None' | Path to the Plinder dataset (optional). |

#### Metrics
| Argument | Default | Description | 
|----------|-------------|-------------| 
| `--timeout` | `12000` | Maximum running time in seconds for any eval metric. |
| `--disable_pb_valid` | `False` | Disables PoseBusters validity check. |
| `--disable_gnina` | `False` | Disables GNINA docking score calculation. |
| `--disable_posecheck` | `False`| Disables strain, clashes, and pocket-ligand interaction computation. |
| `--disable_rmsd` | `False` | Disables RMSD computation between generated ligand and ground truth ligand. |
| `--disable_interaction_recovery` | `False` | Disables analysis of interaction recovery by generated ligands. |
| `--disable_pharm_match` | `False` | Disables computations of matching pharmacophores by generated ligands. |
| `--disable_ground_truth_metrics` | `False` | Disables all relevant metrics on the truth ligand. |


## Getting Samples
### 1. From Model Checkpoint
The script will sample the model when passed the `--ckpt_path` and write sample files to `output_dir/samples/`. The format of the outputted files depends on the task. See below.

#### Fixed protein (protein structure fixed)
```text
output_dir/samples
├── sys_1_gt
│   ├── gen_ligands.sdf
│   ├── ligand.sdf
│   ├── protein_0.pdb
│   └── pharmacophore.xyz   (if applicable)
│
├── sys_2_gt
│   ├── gen_ligands.sdf
│   ├── ligand.sdf
│   ├── protein_0.pdb
│   └── pharmacophore.xyz   (if applicable) 
... 
│
└── sys_N_SAMPLES_gt
    ├── gen_ligands.sdf
    ├── ligand.sdf
    ├── protein_0.pdb
    └── pharmacophore.xyz   (if applicable)
```

Note: In the case of a fixed protein, all generated ligands are in one sdf file.

#### Flexible protein (protein structure generated)
```text
output_dir/samples
├── sys_1_gt 
│   ├── gen_ligands_0.sdf
│   ├── gen_ligands_1.sdf
│   ... 
│   ├── gen_ligands_N_REPLICATES.sdf 
│   ├── gen_prot_0.sdf 
│   ├── gen_prot_1.sdf 
│   ... 
│   ├── gen_prot_N_REPLICATES.sdf 
│   ├── ligand.sdf 
│   ├── protein_0.pdb  
│   └── pharmacophore.xyz   (if applicable) 
│ 
├── sys_2_gt 
│   ├── gen_ligands_0.sdf 
│   ├── gen_ligands_1.sdf 
│   ... 
│   ├── gen_ligands_N_REPLICATES.sdf 
│   ├── gen_prot_0.sdf 
│   ├── gen_prot_1.sdf 
│   ... 
│   ├── gen_prot_N_REPLICATES.sdf 
│   ├── ligand.sdf 
│   ├── protein_0.pdb 
│   └── pharmacophore.xyz   (if applicable) 
...
│ 
└── sys_N_SAMPLES_gt 
    ├── gen_ligands_0.sdf
    ├── gen_ligands_1.sdf 
    ... 
    ├── gen_ligands_N_REPLICATES.sdf 
    ├── gen_prot_0.sdf 
    ├── gen_prot_1.sdf 
    ... 
    ├── gen_prot_N_REPLICATES.sdf 
    ├── ligand.sdf 
    ├── protein_0.pdb  
    └── pharmacophore.xyz   (if applicable) 
```

#### Example Usage:
``` console
python docking_eval.py \
    --ckpt_path PATH/last.ckpt \
    --task rigid_docking_condensed \
    --n_samples 10 \
    --n_replicates 10 \
    --max_batch_size 600 \
    --timeout 1200 
```

### 2. From Samples Directory
If  `--samples_dir` is passed, the script reads in files from the directory. Given the task, there is a strict required format for the files in said directory. This should match the above section.

Note: Every generated ligand should have a property `_Name` set to `gen_ligands_IDX` 

#### Example Usage:
```console
python docking_eval.py \
    --samples_dir OMTRA_ROOT/omtra_pipelines/docking_eval/outputs/rigid_docking_condensed/samples \
    --sys_info_file OMTRA_ROOT/omtra_pipelines/docking_eval/outputs/rigid_docking_condensed/rigid_docking_condensed_sys_info.csv \
    --task rigid_docking_condensed \
    --n_samples 10 \
    --n_replicates 10 \
    --max_batch_size 600 \
    --timeout 1200 
```

## Outputs
| Name | Type | Description | 
|----------|-------------|-------------| 
| `OUTPUT_DIR/TASK_NAME_metrics.csv` | `csv` | All of the computed metrics. Metrics for the ground truth ligand are followed by a `_true` tag. Note: some metrics cannot be computed if the ligand fails to sanitize. These metrics are set to `NaN` values. |
| `OUTPUT_DIR/samples/TASK_NAME_sys_info.csv` | `csv` | File with system information. This is automatically configured when sampling from the Plinder or Crossdocked dataset. No system info file is created if we  (not sampling from a checkpoint).  |


Useful tip: The notebook `notebooks/docking_evaluation.ipynb` contains code for analyzing the models across the various computed metrics.


# OMTRA
A Multi-Task Generative model for Structure-Based Drug Design

![OMTRA]()

-----------------------------------------------------------------------------------------------------
# Building the Environment

We recommend building the envornment using pip inside of a virtual environment. Our recommended procedure is:

## Create conda/mamba environment

```bash
# Example using conda
conda create -n omtra python=3.11
conda activate omtra
```

### Option 1: run the build script:

```bash
git clone https://github.com/gnina/OMTRA.git
cd OMTRA
chmod +x build_env.sh
./build_env.sh
```

This script installs the CUDA-enabled versions of PyTorch, DGL, and PyG, and then installs the OMTRA package and its dependencies.

### Option 2: manual installation

```bash
pip install uv

# 1. Install CUDA dependencies
uv pip install -r requirements-cuda.txt

# 2. Install OMTRA
uv pip install -e .
```
-----------------------------------------------------------------------------------------------------
# Sampling 
There are two main ways to sample a trained OMTRA model.

## Option 1: OMTRA Webapp
The OMTRA webapp and documentation can be found here [Webapp]().

## Option 2: rountines/sample.py
Models: 

### Usage
| Argument | Default | Description | 
|----------|-------------|-------------| 
| `checkpoint` | Required | Path to model checkpoint. |
| `--task` | Required | Task to sample for (e.g. denovo_ligand). |
| `--dataset` | `pharmit` | Dataset to sample from (e.g. pharmit). |
| `--split` | `val` | Which data split to use. |
| `--dataset_start_idx` | `0` | Index in the dataset to start sampling from. |
| `--sys_idx_file` | `None` | Path to a file with pre-selected system indices. |
| `--pharmit_path` | `None` | Path to the Pharmit dataset (optional). |
| `--plinder_path` | `None` | Path to the Plinder dataset (optional). |
| `--crossdocked_path` | `None` | Path to the Crossdocked dataset (optional). |
| `--n_samples` | `100` | Number of samples to draw. |
| `--n_replicates` | `1` | For conditional sampling: number of replicates per input sample. |
| `--n_timseteps` | `250` | Number of integration steps to take when sampling. |
| `--use_gt_n_lig_atoms` | `store_true` | If set, use the number of ground truth ligand atoms for de novo design. |
| `--n_lig_atom_margin` | `0.075` | Number of atoms in the ligand will be +/- this margin from number of atoms in the ground truth ligand, only if --use_gt_n_lig_atoms is set. |
| `--stochastic_sampling` | `store_true` | If set, perform stochastic sampling. |
| `--noise_scaler` | `1.0` | Scaling factor for noise if using stochastic sampling. |
| `--eps` | `0.01` | g(t) param for stochastic sampling. |
| `--visualize` | `store_true` | If set, visualize the sampling process. |
| `--metrics` | `store_true` | If set, compute metrics for the samples. |
| `--output_dir` | `ckpt_path.parent.parent` | Directory for outputs. |


#### Example Usage
```console
python routines/sample.py 
    <PATH>/<CHECKPOINT>.ckpt \ 
    --task=fixed_protein_ligand_denovo_condensed \
    --n_samples=10 \
    --n_replicates=10 \
    --use_gt_n_lig_atoms \
    --visualize \
```

-----------------------------------------------------------------------------------------------------
# Train

## Task Groups
### Tasks Supported
| Task Name | Description | 
|----------|-------------|
| `denovo_ligand_condensed` | Unconditional de novo ligand generation |
| `ligand_conformer_condensed` | Unconditional ligand conformer generation |
| `denovo_ligand_from_pharmacophore` | Pharmacophore-conditioned de novo ligand generation |
| `ligand_conformer_from_pharmacophore` | Pharmacophore-conditioned ligand conformer generation |
| `rigid_docking_condensed` | Rigid docking |
| `fixed_protein_ligand_denovo_condensed` | Rigid protein, de novo ligand generation |
| `rigid_docking_pharmacophore_condensed` | Pharmacophore-conditioned rigid docking |
| `fixed_protein_pharmacophore_ligand_denovo_condensed` | Pharmacophore-conditioned, rigid protein, de novo ligand generation |

### Datasets Supported
| Dataset Name | Description | Tasks |
|----------|-------------|-------------|
| `pharmit` |  | Tasks without a protein |
| `plinder` |  | Any |
| `crossdocked` |  | Any |


### Specifying a Task Group
`task_group` config files are used to specify which tasks the OMTRA model supports. An example can be seen in `configs/task_group/prot_protpharm_cond.yaml` which would create a version of OMTRA that is trained on all supported tasks in varying proportions. The genreal structure of the task group is as follows:

1. `task_phases` 
This is essentially a list. Each item of the list describes a "phase" of training OMTRA; the idea is that different phases can have different task mixtures. For example, maybe you want to focus heavily on unconditional ligand generation intitally and then start to incorporate pocket-conditioned ligand generation in a second phase. Each phase has a duration (measured in minibatches) and a list of tasks + the probability of training on each task. The probabilities need not sum to one; they will be normalized. In otherwords, this specicies for each phase of trainnig, what is p(task) for each training batch.

2. `datset_task_coupling`
This is a dictionary where each key is a task and the value is a list specifying the dataset we will use for that task, along with the probability of using the dataset for that task. In other words, the dataset task coupling is directly specifying the probability distribution p(dataset|task).

## Training Commands
Default parameters are specified in config files under `configs/<GROUP>/default.yaml`. Parameters can be overwritten using command line arguments of the form `<GROUP>.<PARAMETER>=value`. 

#### Example Usage
```console
python routines/train.py           
    name=chirality_ph5050 \             # Run name
    task_group=pharmit5050_cond_a \     # Task group
    edges_per_batch=230000 \            # Batch size
    trainer.val_check_interval=600 
    max_steps=600000 \                  # Total train time
    plinder_pskip_factor=1.0 \
    num_workers=6  \  
    trainer=distributed \               # Multi-GPU training
    trainer.devices=2 \                 # Multi-GPU training
    graph=more_prot \
    model.train_t_dist=beta \
    model.t_alpha=1.8 \
    model.cat_loss_weight=0.8 \
    model.time_scaled_loss=true
    model/aux_losses=pairs \
```

-----------------------------------------------------------------------------------------------------
# Pharmit Dataset
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


-----------------------------------------------------------------------------------------------------
# Evaluations
A pipeline for generating evaluation metrics for conditional tasks can be found in `omtra_pipelines/docking_eval/docking_eval.py`. The script can be used to sample an OMTRA model from a checkpoint and subsequently compute metrics, or compute metrics on pre-obtained samples. A detailed usage description can be found in `omtra_pipelines/docking_eval/readme.md`.


-----------------------------------------------------------------------------------------------------
# Results
The following section describes how to recreate the published results.

#### Plinder De Novo Design
```console

```

#### Crossdocked De Novo Design
```console

```

#### Plinder Docking
```console

```

#### Posebusters Docking 
```console

```

# OMTRA
A Multi-Task Generative model for Structure-Based Drug Design

![OMTRA](assets/omtra_fig.png)

-----------------------------------------------------------------------------------------------------
# Installation

We recommend building the envornment using pip inside of a virtual environment. Our recommended procedure is:

Create conda/mamba environment

```bash
# Create conda/mamba environment
mamba create -n omtra python=3.11
mamba activate omtra

# Clone, run build script
git clone https://github.com/gnina/OMTRA.git
cd OMTRA
chmod +x build_env.sh
./build_env.sh
```

The build script installs the CUDA-enabled versions of PyTorch, DGL, and PyG, and then installs the OMTRA package and its dependencies.

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
| `--visualize` | `store_true` | If set, output will contain sampling trajectories rather than the final sampled state. |
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
# Using Docker

## Requirements

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Model weights downloaded to `OMTRA/checkpoints/` directory

## Start CLI with pre-built image

1. **Setup:**
   ```bash
   source docker-cli-setup.sh
   ```

   To make this permanent, add to your `~/.bashrc` or `~/.zshrc`:
   ```bash
   source /path/to/OMTRA/docker-cli-setup.sh
   ```

2. **Use the CLI:**
   ```bash
   omtra --task <task> [options]
   ```
   The docker image will be automatically pulled from registry on first use.

#### Examples

```bash
omtra --task denovo_ligand_condensed \
  --n_samples 100 \
  --output_dir outputs/samples \
  --metrics

omtra --task fixed_protein_ligand_denovo_condensed \
  --protein_file protein.pdb \
  --ligand_file ref_ligand.sdf \
  --n_samples 50

```

#TODO: does CLI require building environment first or does it use the docker image?
#TODO: i think we need a set of instructions for use that encompass everything you want to do, either with or without the docker image. Q im not clear on: what is what docker image for? what is it not for?
#TODO: give a nice table of options for the CLI like we did for sample.py


## Web Application

### Quick Start

```bash
cd omtra_webapp
docker-compose up -d
```

The webapp will be available at http://localhost:5900 (or the port specified in your `.env` file).

See [`omtra_webapp/START.md`](omtra_webapp/START.md) for more details

-----------------------------------------------------------------------------------------------------
# Training
Refer to [docs/training.md](docs/training.md) for details on training OMTRA models.

-----------------------------------------------------------------------------------------------------
# Pharmit Dataset
Refer to [docs/pharmit_dataset.md](docs/pharmit_dataset.md) for details on the Pharmit dataset and how to use it.

# Evals and Reproducing Paper Results
Refer to [docs/reproducing_results.md](docs/reproducing_results.md) for instructions on reproducing the published results.
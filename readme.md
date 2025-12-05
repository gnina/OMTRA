# OMTRA
A Multi-Task Generative model for Structure-Based Drug Design

![OMTRA](assets/omtra_fig.png)

-----------------------------------------------------------------------------------------------------

## Table of Contents
- [Installation](#installation)
  - [Manual Installation (Recommended)](#manual-installation-recommended)
  - [Docker Installation](#docker-installation)
- [Model Weights](#model-weights)
- [Sampling](#sampling)
  - [CLI Reference](#cli-reference)
  - [Available Tasks](#available-tasks)
  - [CLI Examples](#cli-examples)
  - [Web Application](#omtra-web-application)
- [Training](#training)
- [Additional Documentation](#additional-documentation)

-----------------------------------------------------------------------------------------------------

# Installation

There are two ways to set up OMTRA:
1. **Manual Installation** — Build the environment manually in a conda/mamba environment (recommended for most users)
2. **Docker Installation** — Use a Docker container for isolated, reproducible environments

### System Requirements

- Linux System
- NVIDIA GPU with CUDA support (CUDA 12.1 recommended)
- Python 3.11

## Manual Installation (Recommended)

This approach gives you direct control over the environment and is recommended for development and most use cases.

```bash
# Create and activate conda/mamba environment
mamba create -n omtra python=3.11
mamba activate omtra

# Clone the repository
git clone https://github.com/gnina/OMTRA.git
cd OMTRA

# Run the build script
chmod +x build_env.sh
./build_env.sh
```

The build script installs:
- CUDA-enabled versions of PyTorch, DGL, and PyG
- OMTRA package and all dependencies

After installation, the `omtra` command will be available:
```bash
omtra --task <task> [options]
```

## Docker Installation

> **⚠️ Note:** Docker support is still a work in progress. The instructions below describe the intended workflow, but the pre-built image may not yet be available in the registry. You can build the image locally in the meantime.

Docker provides an isolated environment and is particularly useful for deployment or if you want to use the web application interface.

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- Model weights downloaded to `OMTRA/checkpoints/` directory (see [Model Weights](#model-weights))

### Option A: Build the Docker Image Locally

If the pre-built image is not available, you can build it yourself:

```bash
cd OMTRA
docker build -t omtra/cli:latest .
```

Then set up the CLI wrapper:

```bash
source docker-cli-setup.sh
```

### Option B: Use Pre-built Image (Coming Soon)

Once the image is published, it will be automatically pulled when you first use the CLI:

```bash
source docker-cli-setup.sh
omtra --task <task> [options]
```

### Making the CLI Available Permanently

Add the following to your shell configuration (`~/.bashrc` or `~/.zshrc`):

```bash
source /path/to/OMTRA/docker-cli-setup.sh
```

### Customizing the Docker Image

You can specify a custom image name by setting the `OMTRA_CLI_IMAGE` environment variable before sourcing the setup script:

```bash
export OMTRA_CLI_IMAGE="my-registry/omtra:v1.0"
source docker-cli-setup.sh
```

To disable GPU support (for testing on CPU-only machines):
```bash
export OMTRA_NO_GPU=1
```

-----------------------------------------------------------------------------------------------------

# Model Weights

> **⚠️ Note:** Model checkpoints not uploaded yet, the instructions below contain placeholders for the remote storage location. Check back in a few hours :)

Download the pre-trained model weights using `wget`:

```bash
cd OMTRA
wget -r -np -nH --cut-dirs=2 -P omtra/trained_models https://example.com/path/to/trained_models/
```

Or using `curl`:

```bash
cd OMTRA
curl -L https://example.com/path/to/trained_models.tar.gz | tar -xz -C omtra/
```

This will create the `omtra/trained_models/` directory with the checkpoint files. The CLI automatically selects the appropriate checkpoint based on the task. You can also specify a checkpoint explicitly with the `--checkpoint` flag.

-----------------------------------------------------------------------------------------------------

# Sampling

There are two ways to sample from a trained OMTRA model:
1. **Command-Line Interface (CLI)** — For scripting and batch processing
2. **Web Application** — For interactive exploration

## CLI Reference

### Basic Usage

```bash
omtra --task <task> [options]
```

The `omtra` command is available after either installation method. With manual installation, it's installed via `pip install -e .`. With Docker, the `docker-cli-setup.sh` script creates a shell function that wraps the Docker container.

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--task` | string | *required* | The sampling task to perform (see [Available Tasks](#available-tasks)) |
| `--checkpoint` | path | auto | Path to model checkpoint (auto-detected from task if not provided) |
| `--n_samples` | int | 100 | Number of samples to generate |
| `--n_timesteps` | int | 250 | Number of integration steps during sampling |
| `--output_dir` | path | None | Directory to save output files |
| `--metrics` | flag | False | Compute evaluation metrics on generated samples |

### Input File Arguments

For conditional generation tasks, you can provide input structures directly:

| Argument | Type | Description |
|----------|------|-------------|
| `--protein_file` | path | Protein structure file (PDB or CIF format) |
| `--ligand_file` | path | Ligand structure file (SDF format) |
| `--pharmacophore_file` | path | Pharmacophore constraints file (XYZ format) |

When input files are provided, `--n_samples` specifies how many samples to generate for that single input system.

### Advanced Sampling Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--stochastic_sampling` | flag | False | Enable stochastic (vs deterministic) sampling |
| `--noise_scaler` | float | 1.0 | Scaling factor for noise in stochastic sampling |
| `--eps` | float | 0.01 | Small epsilon value for numerical stability |
| `--visualize` | flag | False | Generate visualization of sampling trajectory |

### Ligand Size Control

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_gt_n_lig_atoms` | flag | False | Match ground truth ligand atom count |
| `--n_lig_atom_margin` | float | 0.15 | Margin (±%) around ground truth atom count |
| `--n_lig_atoms_mean` | float | None | Mean for normal distribution of atom counts |
| `--n_lig_atoms_std` | float | None | Std dev for normal distribution of atom counts |

## Available Tasks

OMTRA supports multiple drug design tasks. Use the `--task` argument to select one:

### Unconditional Generation
| Task | Description |
|------|-------------|
| `denovo_ligand_condensed` | Generate novel drug-like molecules from scratch |

### Protein-Conditioned Generation
| Task | Description |
|------|-------------|
| `fixed_protein_ligand_denovo_condensed` | Design ligands for a fixed protein binding site |
| `protein_ligand_denovo_condensed` | Joint generation of ligand with flexible protein |

### Docking Tasks
| Task | Description |
|------|-------------|
| `rigid_docking_condensed` | Dock a known ligand into a fixed protein structure |
| `flexible_docking_condensed` | Dock with protein flexibility |
| `expapo_conditioned_ligand_docking_condensed` | Docking starting from experimental apo structure |
| `predapo_conditioned_ligand_docking_condensed` | Docking starting from predicted apo structure |

### Conformer Generation
| Task | Description |
|------|-------------|
| `ligand_conformer_condensed` | Generate 3D conformations for a given ligand |

### Pharmacophore-Conditioned Tasks
| Task | Description |
|------|-------------|
| `denovo_ligand_pharmacophore_condensed` | Generate ligand and pharmacophore jointly |
| `denovo_ligand_from_pharmacophore_condensed` | Design ligand matching a given pharmacophore |
| `ligand_conformer_from_pharmacophore_condensed` | Generate conformer satisfying pharmacophore |
| `fixed_protein_pharmacophore_ligand_denovo_condensed` | Design ligand for protein with pharmacophore constraints |
| `rigid_docking_pharmacophore_condensed` | Dock ligand with pharmacophore constraints |

## CLI Examples

### Generate Novel Molecules (Unconditional)
```bash
omtra --task denovo_ligand_condensed \
  --n_samples 100 \
  --output_dir outputs/denovo_samples \
  --metrics
```

### Structure-Based Drug Design (Protein-Conditioned)
```bash
omtra --task fixed_protein_ligand_denovo_condensed \
  --protein_file my_protein.pdb \
  --ligand_file reference_ligand.sdf \
  --n_samples 50 \
  --output_dir outputs/sbdd_samples
```
The reference ligand is used to define the binding site center. If omitted, the protein center of mass is used.

### Molecular Docking
```bash
omtra --task rigid_docking_condensed \
  --protein_file protein.pdb \
  --ligand_file ligand.sdf \
  --n_samples 10 \
  --output_dir outputs/docking
```

### Conformer Generation
```bash
omtra --task ligand_conformer_condensed \
  --ligand_file molecule.sdf \
  --n_samples 20 \
  --output_dir outputs/conformers
```

### Pharmacophore-Guided Design
```bash
omtra --task denovo_ligand_from_pharmacophore_condensed \
  --pharmacophore_file constraints.xyz \
  --n_samples 100 \
  --output_dir outputs/pharm_guided
```

### Debug Mode
Set the `OMTRA_DEBUG` environment variable for full stack traces:
```bash
OMTRA_DEBUG=1 omtra --task denovo_ligand_condensed --n_samples 10
```

-----------------------------------------------------------------------------------------------------

## OMTRA Web Application

The web application provides an interactive interface for exploring OMTRA's capabilities.

### Prerequisites
- Docker and Docker Compose installed
- Pre-built webapp images (or build from source)

### Starting the Web Application

```bash
cd omtra_webapp
docker-compose up -d
```

The webapp will be available at http://localhost:5900 (or the port specified in your `.env` file).

### Stopping the Web Application

```bash
cd omtra_webapp
docker-compose down
```

See [`omtra_webapp/START.md`](omtra_webapp/START.md) for detailed configuration options and building from source.

-----------------------------------------------------------------------------------------------------

# Training

Refer to [docs/training.md](docs/training.md) for details on training OMTRA models.

-----------------------------------------------------------------------------------------------------

# Additional Documentation

- [Pharmit Dataset](docs/pharmit_dataset.md) — Details on the Pharmit dataset and how to use it
- [Reproducing Results](docs/reproducing_results.md) — Instructions for reproducing published results

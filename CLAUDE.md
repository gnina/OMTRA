# CLAUDE.md - OMTRA Codebase Guide

OMTRA is a multi-task flow matching generative model for structure-based drug design. It generates small molecules conditioned on protein pockets and/or pharmacophore constraints. Paper: https://arxiv.org/abs/2512.05080

## Quick Reference

```bash
# CLI sampling
omtra --task <task_name> --n_samples 100 --output_dir outputs/

# Training
python routines/train.py name=my_run task_group=protein max_steps=100000

# Tests
pytest tests/
```

## Architecture Overview

### Core Abstraction: Tasks as Modality Partitions

A **Task** defines what to generate vs. condition on by partitioning **modalities** into three sets:
- `groups_generated`: sampled by the flow matching model
- `groups_fixed`: provided as conditioning input
- `groups_absent`: not present in the graph

**Modality groups** (defined in `omtra/tasks/modalities.py`):
- `ligand_structure`: ligand atom positions (`lig_x`)
- `ligand_identity_condensed`: atom types + bonds (`lig_cond_a`, `lig_e_condensed`)
- `protein_structure`: protein atom positions (`prot_atom_x`)
- `protein_identity`: residue names, atom names, elements
- `pharmacophore`: pharmacophore positions + types (`pharm_x`, `pharm_a`)

**Example task definition** (`omtra/tasks/tasks_condensed.py`):
```python
@register_task("rigid_docking_condensed")
class RigidDockingCondensed(Task):
    groups_fixed = ["ligand_identity_condensed", "protein_identity", "protein_structure"]
    groups_generated = ["ligand_structure"]
```

### Key Tasks

| Task | Generates | Conditions On |
|------|-----------|---------------|
| `denovo_ligand_condensed` | ligand atoms + positions | nothing |
| `fixed_protein_ligand_denovo_condensed` | ligand atoms + positions | protein pocket |
| `rigid_docking_condensed` | ligand positions | ligand identity + protein |
| `ligand_conformer_condensed` | ligand positions | ligand identity |
| `denovo_ligand_from_pharmacophore_condensed` | ligand | pharmacophore |
| `fixed_protein_pharmacophore_ligand_denovo_condensed` | ligand | protein + pharmacophore |

### Model Architecture

`omtra/models/omtra.py` - Main PyTorch Lightning module
- Extends FlowMol3 to heterogeneous graphs with type-specific convolutions
- SE(3)-equivariant via GVP (Geometric Vector Perceptron) layers
- Nodes carry: positions (x), scalar features (s), vector features (v)
- 4 ConvolutionBlocks, each with 2 graph convolutions + position/edge updates

`omtra/models/vector_field.py` - The neural network predicting the velocity field
- Edge-type-specific message functions
- Node-type-specific update functions
- Outputs: positions directly, categorical logits via MLP heads

### Data Pipeline

**Datasets** (`omtra/dataset/`):
- `pharmit.py`: 500M 3D conformers (ligand-only tasks)
- `plinder.py`: Protein-ligand complexes with rigorous splits
- `crossdocked.py`: Protein-ligand complexes (legacy splits for comparison)

**Graph construction** (`omtra/data/graph/`):
- Heterogeneous graph with node types: ligand, protein atom, pharmacophore, other (cofactors/ions)
- Edge types: ligand-ligand, ligand-protein, protein-protein, etc.
- Pocket cropped to 8Å around ligand

**Multi-task training** (`omtra/dataset/multitask.py`, `omtra/dataset/data_module.py`):
- `task_phases`: defines task mixture over training phases
- `dataset_task_coupling`: maps tasks to datasets with probabilities

## Directory Structure

```
omtra/                      # Main package
├── models/
│   ├── omtra.py           # Main model (PyTorch Lightning)
│   ├── vector_field.py    # GNN architecture
│   ├── gvp.py             # SE(3)-equivariant layers
│   └── conditional_paths/ # Flow matching interpolants
├── tasks/
│   ├── modalities.py      # Modality definitions
│   ├── tasks_condensed.py # Task definitions (use these)
│   ├── tasks.py           # Legacy task definitions
│   └── base_task.py       # Task base class
├── dataset/
│   ├── plinder.py         # Plinder dataset loader
│   ├── pharmit.py         # Pharmit dataset loader
│   ├── crossdocked.py     # CrossDocked dataset loader
│   └── data_module.py     # Lightning DataModule
├── data/
│   ├── graph/             # Graph construction
│   ├── pharmacophores.py  # Pharmacophore extraction
│   └── condensed_atom_typing.py
├── eval/
│   ├── system.py          # SampledSystem class for evaluation
│   └── evals.py           # Evaluation metrics
├── priors/                # Prior distributions for flow matching
└── utils/

configs/                   # Hydra configs
├── config.yaml           # Main config
├── model/                # Model configs
├── task_group/           # Task mixture configs
├── trainer/              # Training configs
└── eval/                 # Evaluation configs

routines/
├── train.py              # Training entrypoint
└── sample.py             # Sampling entrypoint

omtra_webapp/             # Web interface
├── api/                  # FastAPI backend
├── worker/               # Celery worker for sampling
└── frontend-react/       # React frontend

omtra_pipelines/          # Data processing pipelines
├── plinder_dataset/      # Plinder preprocessing
├── pharmit_dataset/      # Pharmit preprocessing
└── docking_eval/         # Docking evaluation scripts

cli.py                    # CLI entrypoint (`omtra` command)
```

## Configuration System (Hydra)

Config files in `configs/` are composed hierarchically:

```yaml
# configs/config.yaml
defaults:
  - model: default
  - task_group: protein      # Which tasks to train on
  - trainer: default
  - graph: default
  - prior: [conformer_indep, denovo_permute]
```

Key config groups:
- `task_group`: defines task phases and dataset coupling
- `model`: architecture hyperparameters
- `trainer`: Lightning trainer settings
- `graph`: graph construction parameters

Override at command line:
```bash
python routines/train.py model.n_layers=6 trainer.devices=2
```

## Common Development Tasks

### Adding a new task

1. Define the task in `omtra/tasks/tasks_condensed.py`:
```python
@register_task("my_new_task")
class MyNewTask(Task):
    groups_fixed = [...]
    groups_generated = [...]
    priors = {...}
    conditional_paths = {...}
```

2. Add to a task group config in `configs/task_group/`

### Modifying the model

- Architecture changes: `omtra/models/vector_field.py`
- Training logic: `omtra/models/omtra.py` (training_step, validation_step)
- Loss functions: in `omtra.py` and `omtra/aux_losses/`

### Adding evaluation metrics

- Add to `omtra/eval/evals.py` or create new file in `omtra/eval/`
- Register in `omtra/eval/register.py`

## Key Concepts

### Flow Matching
- Interpolates between prior distribution (noise) and target (data)
- Continuous variables: ODE with learned velocity field
- Discrete variables (atom types, bonds): Continuous-time Markov chains
- Multi-modal: simultaneous transport on all modalities

### Condensed Atom Typing
Tasks ending in `_condensed` use a simplified atom typing that encodes element + implicit hydrogens + charge + hybridization into a single categorical. This is the preferred approach.

### Priors
Defined per-modality in `omtra/tasks/prior_collections.py`:
- `gaussian`: standard Gaussian for positions
- `uniform`: uniform over categories
- `target_dependent_gaussian`: Gaussian centered on target (for docking)
- `apo_exp`/`apo_pred`: use apo structure as prior

### Pharmacophores
7 types: Aromatic, HydrogenDonor, HydrogenAcceptor, PositiveIon, NegativeIon, Hydrophobic, Halogen

Extracted from ligands or provided via JSON/XYZ files.

## Testing

```bash
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests (slower)
```

## Weights & Checkpoints

Pre-trained weights: `omtra/trained_models/`
- Task-to-checkpoint mapping in `omtra/utils/checkpoints.py`
- CLI auto-selects checkpoint based on task

## Environment

- Python 3.11, PyTorch, DGL, PyTorch Geometric
- CUDA 12.1 recommended
- Install via `./build_env.sh` or Docker

### Machine-Specific Info

**masuda** (dev machine):
- Path: `/home/ian/projects/mol_diffusion/OMTRA`
- Conda env: `omtra`
- Pharmit data: `data/pharmit` (may not exist locally)
- Use for: development, testing, code changes

**cluster** (experiment machine):
- Path: `/net/galaxy/home/koes/icd3/moldiff/OMTRA`
- Conda env: `omtra`
- Pharmit data: `data/pharmit`
- Use for: training runs, computing marginals, experiments
- SLURM: `sbatch --array=1-N train_multigpu_2.slurm train_cmds.txt`

### Current Session Handoff

**Branch**: `noisy-paths-stage1`

**Date**: 2026-01-24

**Training runs in progress** (SLURM job 51893727):
- `noisy_stage1_uniform` (array task 2): Stage 1 with α=0.15 (~3.75% max corruption)
- `noisy_stage2_marginal` (array task 3): Stage 2 with α=0.15 (~3.75% max corruption)
- `noisy_stage1_alpha32` (array task 4): Stage 1 with α=0.32 (8% max corruption)

**Baseline already ran**: `noisy_baseline` completed ~99k steps (job 51893591_3)

**Next steps (for another session)**:
1. Implement Stage 4: Corruption classification head
   - Add auxiliary head to predict which tokens are corrupted vs clean
   - Train jointly with main denoising objective
   - Use predictions to guide remasking at inference
2. Implement Stage 5: Modified sampling with corruption-aware remasking
   - At inference, use corruption head predictions to identify likely errors
   - Remask high-confidence corrupted tokens and resample

---

## Active Research: Noisy Paths Experiment

**Goal**: Address train-test mismatch in discrete flow matching where the denoiser never sees incorrect-but-unmasked tokens during training, but encounters them at inference due to its own errors.

**Research Plan** (from `noisy_paths.zip`):

| Stage | Description | Status | Branch |
|-------|-------------|--------|--------|
| 1 | **Uniform corruption**: Replace some unmasked tokens with uniform samples during training | Implemented | `noisy-paths-stage1` |
| 2 | **Data-marginal corruption**: Use empirical token distribution instead of uniform | Implemented | `noisy-paths-stage1` |
| 3 | **Model-induced corruption**: Sample from current denoiser to create corruptions | Planned | - |
| 4 | **Corruption classification head**: Add head to classify clean vs corrupted tokens | **Next** | - |
| 5 | **Modified sampling**: Follow three-way marginal path at inference | Planned | - |

**Three-way conditional path**:
```
p_t(x | x_1) = (t - β_t/2)·δ_{x_1} + β_t·p_corrupt + (1 - t - β_t/2)·δ_mask
```
where `β_t = α·t·(1-t)` and `p_corrupt` varies by stage:
- Stage 1: uniform distribution
- Stage 2: empirical data marginal
- Stage 3: denoiser's own predictions

**Key files**:
- `omtra/models/conditional_paths/paths.py`: Core three-way path implementation
- `configs/model/conditional_paths/noisy.yaml`: Stage 1 config (α=0.15, ~3.75% max)
- `configs/model/conditional_paths/noisy_marginal.yaml`: Stage 2 config (α=0.15)
- `configs/model/conditional_paths/noisy_alpha32.yaml`: Higher corruption config (α=0.32, 8% max)
- `docs/noisy_paths.md`: Detailed documentation

**Corruption level math**:
- `β_t = noise_alpha * t * (1-t)` peaks at t=0.5
- `β_max = noise_alpha * 0.25`
- α=0.15 → 3.75% max corruption
- α=0.32 → 8% max corruption

**Training with noisy paths**:
```bash
# Stage 1 (uniform corruption, default α=0.15)
python routines/train.py model/conditional_paths=noisy name=noisy_experiment ...

# Stage 2 (data-marginal corruption)
python routines/train.py model/conditional_paths=noisy_marginal name=noisy_experiment ...

# Stage 1 with higher corruption (α=0.32, 8% max)
python routines/train.py model/conditional_paths=noisy_alpha32 name=noisy_experiment ...
```

**Command file**: `noisy_paths_cmds.txt` - 4 training commands (no comments, array-friendly)
- Line 1: baseline (no noisy paths)
- Line 2: stage1 uniform (α=0.15)
- Line 3: stage2 marginal (α=0.15)
- Line 4: stage1 alpha32 (α=0.32)

**Pharmit zarr structure** (for computing marginals):
```
{split}.zarr/
├── lig/node/
│   ├── graph_lookup    # (n_graphs, 2) -> start/end indices
│   ├── x               # (n_atoms, 3) -> positions
│   ├── a               # (n_atoms,) -> atom types
│   ├── c               # (n_atoms,) -> charges
│   └── extra_feats     # (n_atoms, 5) -> impl_H, aro, hyb, ring, chiral
└── lig/edge/
    ├── graph_lookup    # (n_graphs, 2) -> start/end indices
    ├── e               # (n_edges,) -> bond types (SPARSE: only non-zero!)
    └── edge_index      # (n_edges, 2) -> src/dst node indices
```

**Note**: Edge data is sparse—only stores non-zero bond orders. For marginal computation, must estimate fraction of "no bond" (type 0) edges from graph structure.

### Stage 4 Implementation Notes (for next session)

**Corruption classification head design**:
1. Add binary classification head parallel to categorical prediction heads in `vector_field.py`
2. For each discrete token position, predict P(corrupted | x_t, t)
3. Ground truth: track which tokens were corrupted during `sample_xt()` in paths.py
4. Loss: binary cross-entropy, weighted to handle class imbalance (most tokens are clean)

**Key integration points**:
- `omtra/models/conditional_paths/paths.py`: Already tracks corruption via `is_corrupted` mask in `sample_xt()`
- `omtra/models/vector_field.py`: Add new MLP head for corruption prediction
- `omtra/models/omtra.py`: Add corruption classification loss to training_step

**Inference changes** (Stage 5):
- During sampling, use corruption head to identify likely errors
- Remask tokens where P(corrupted) > threshold
- Continue denoising from remasked state

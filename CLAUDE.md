# CLAUDE.md - OMTRA Codebase Guide

> Also read `claude.local/CLAUDE.md` for machine-specific configuration.

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

`omtra/models/omtra.py` — Main PyTorch Lightning module
- Wraps VectorField, handles training/validation loops and loss computation
- Multi-task training with task-specific modality partitioning

`omtra/models/vector_field.py` — The neural network predicting the velocity field
- **Hybrid GVP + Transformer architecture**:
  1. **Pre-convs**: Initial GVP convolutions for geometric message passing
  2. **Edge embedding**: Updates edge features after pre-convs
  3. **Transformer**: Main processing via `TransformerWrapper` (cross-type attention)
  4. **Position updates**: MLP-based position prediction
  5. **Output heads**: Categorical logits for atom types, bond orders
- Global conditioning via AdaLN (task + time embeddings projected together)
- Nodes carry: positions (x), scalar features (s), vector features (v)

`omtra/models/transformer.py` — Transformer module
- `TransformerWrapper`: Packs heterogeneous graph nodes into unified sequence
  - Projects (scalars + coords + flattened vectors) → d_model
  - Cross-type self-attention across all node types (lig, prot_atom, pharm)
  - Supports standard `TransformerEncoderLayer` or `DiTLayer` (QK-norm + AdaLN)
- `LigandPairBiasEmbedder`: Pair-biased attention for ligand nodes
  - RBF-encoded pairwise distances as attention bias
  - Uses `AttentionPairBias` layer (inspired by AlphaFold)
- `PairTransformerLayer`: Transformer layer with pair bias injection

`omtra/models/gvp.py` — GVP layers (used in pre-convs)
- `HeteroGVPConv`: Heterogeneous graph convolution with GVP message functions
- SE(3)-equivariant via Geometric Vector Perceptrons

`omtra/models/dit.py` — DiT layer with AdaLN conditioning
`omtra/models/adaln.py` — Adaptive Layer Normalization
`omtra/models/layers.py` — Shared layer primitives


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

### Eval Metrics Module

`omtra/eval/metrics/` — Standalone metric functions for evaluating sampled molecules. Accepts standard inputs (RDKit Mol, file paths, numpy arrays), returns per-ligand results. No timeout/multiprocessing wrapping — that's the caller's concern.

**Public API:**
- `ligand_rmsd(mol, ref_mol)` — RMSD between generated and reference ligand
- `pharmacophore_match(mol, ref_mol)` / `pharmacophore_match_from_dict(mol, pharm_dict)` — pharmacophore feature recovery
- `gnina_score(mol, protein_path)` / `gnina_minimize(...)` / `gnina_score_and_minimize(...)` — CNN scoring and minimization via gnina
- `pb_validate(mol, protein_path)` — PoseBusters validity checks
- `posecheck_clashes(...)` / `posecheck_strain(...)` / `posecheck_interactions(...)` / `posecheck_interaction_recovery(...)` / `posecheck_all(...)` — PoseCheck suite
- `compute_metrics(...)` — orchestrates multiple metrics over a set of sampled systems

Separate from the train-time eval system in `omtra/eval/evals.py` (which operates on `SampledSystem` lists).

## Directory Structure

```
omtra/                      # Main package
├── models/
│   ├── omtra.py           # Main model (PyTorch Lightning)
│   ├── vector_field.py    # Hybrid GVP + Transformer architecture
│   ├── transformer.py     # TransformerWrapper, PairTransformerLayer
│   ├── dit.py             # DiT layer (AdaLN conditioning)
│   ├── adaln.py           # Adaptive Layer Normalization
│   ├── layers.py          # Shared layer primitives
│   ├── gvp.py             # SE(3)-equivariant GVP layers
│   ├── self_conditioning.py
│   ├── interpolant_scheduler.py
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
│   ├── evals.py           # Train-time eval (operates on SampledSystem lists)
│   └── metrics/           # Standalone metric functions
│       ├── rmsd.py        # ligand_rmsd
│       ├── pharmacophore.py # pharmacophore_match
│       ├── gnina.py       # gnina_score, gnina_minimize
│       ├── posebusters.py # pb_validate
│       ├── posecheck.py   # posecheck_clashes, _strain, _interactions
│       └── compute.py     # compute_metrics orchestrator
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

- **Standalone metrics**: Add to `omtra/eval/metrics/` (pure functions, standard inputs)
- **Train-time evals**: Add to `omtra/eval/evals.py` (operates on `SampledSystem` lists)
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

---

## Research Notes

Development logs, research notes, and experiment documentation are stored in `research_notes/` (gitignored).

### Format

**Naming convention**: `YYYY-MM-DD_HHMMSS_<slug>.md`
- Date and time provide chronological sorting
- Slug is a short kebab-case descriptor

**Frontmatter** (YAML):
```yaml
---
name: Human-readable title
description: Brief one-line description
tags: [tag1, tag2, tag3]
date: YYYY-MM-DD
time: "HH:MM"
branch: optional-git-branch
status: active | partial | complete | abandoned
---
```

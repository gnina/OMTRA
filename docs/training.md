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

### Resuming Training
To resume training from a checkpoint, use the `+checkpoint` argument with the path to the checkpoint file:

```console
python routines/train.py \
    +checkpoint=outputs/2026-01-26/my_run_2026-01-26_12-35-166011/checkpoints/last.ckpt \
    max_steps=100000
```

When resuming:
- The original config is loaded from the checkpoint's run directory (`.hydra/config.yaml`)
- The W&B run is automatically resumed with the same run ID
- Checkpoints continue to be saved to the original run directory
- CLI overrides (like `max_steps` above) are applied on top of the original config

You can resume multiple times from the same run - each resume will continue from where the last one left off. 

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

## Noisy Paths: Discrete Corruption Training

Noisy paths training addresses train-test mismatch in discrete flow matching by introducing structured corruption of unmasked tokens during training. This teaches the model to correct incorrect tokens, improving robustness at inference time.

### Stage 1: Uniform Corruption

Train with uniform corruption of discrete tokens (atom types, bond types):

```console
python routines/train.py \
    name=noisy_stage1 \
    task_group=pharmit5050_cond_a \
    model/conditional_paths=noisy \
    max_steps=200000
```

The `noise_alpha` parameter controls corruption level (default 0.15 → ~3.75% max corruption at t=0.5).

### Stage 2: Data-Marginal Corruption

Uses empirical token distribution instead of uniform for more realistic corruption:

```console
python routines/train.py \
    name=noisy_stage2 \
    task_group=pharmit5050_cond_a \
    model/conditional_paths=noisy_marginal \
    max_steps=200000
```

**Prerequisite**: Compute marginal distributions first:
```console
python omtra_pipelines/compute_marginals/compute_marginals.py \
    --pharmit_path data/pharmit \
    --split train \
    --output_path data/pharmit/train_marginals.npz
```

### Stage 4: Corruption Classification Head

Train the model to predict which tokens were corrupted, enabling corruption-aware inference:

```console
python routines/train.py \
    name=noisy_stage4 \
    task_group=pharmit5050_cond_a \
    model/conditional_paths=noisy \
    model/aux_losses=corruption \
    max_steps=200000
```

The `corruption` aux_loss config automatically enables `model.vector_field.enable_corruption_heads=true`.

**Loss parameters** (in `configs/model/aux_losses/corruption.yaml`):
- `weight`: Loss weight relative to main denoising loss (default: 1.0)
- `pos_weight`: BCE positive class weight for class imbalance (default: 10.0)

### Stage 5: Corruption-Aware Sampling

After training with Stage 4, use corruption predictions during inference to remask likely errors:

```python
sampled_systems = model.sample(
    task_name="fixed_protein_ligand_denovo_condensed",
    g_list=graphs,
    corruption_remasking=True,      # Enable remasking
    corruption_threshold=0.5,        # P(corrupted) threshold
)
```

### Combining Stages

Example: Train with marginal corruption + corruption classification head:

```console
python routines/train.py \
    name=noisy_stage2_stage4 \
    task_group=pharmit5050_cond_a \
    model/conditional_paths=noisy_marginal \
    model/aux_losses=corruption \
    max_steps=200000
```

### Configuration Files

| Config | Description |
|--------|-------------|
| `model/conditional_paths=noisy` | Stage 1: uniform corruption (α=0.15) |
| `model/conditional_paths=noisy_marginal` | Stage 2: data-marginal corruption |
| `model/conditional_paths=noisy_alpha32` | Higher corruption rate (α=0.32, 8% max) |
| `model/aux_losses=corruption` | Stage 4: corruption classification head |

For detailed documentation, see [docs/noisy_paths.md](noisy_paths.md).
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
To resume training, pass the checkpoint with `+checkpoint`:

```console
python routines/train.py +checkpoint=<run_dir>/checkpoints/last.ckpt
```

Runs of the validation-controlled recipe write step checkpoints named `step_<N>.ckpt`; pass one of those instead.

Resume is strict. It reuses the training configuration saved in the run directory (`training_config.yaml`, which is also stored inside every checkpoint) and refuses to change it:
- Only `+checkpoint`, `experiment`, `resume_expected_task_group` and `hydra.*` overrides are accepted. Any other override, for example `max_steps=100000` or `task_group=...`, is rejected with an error.
- The checkpoint must be a full training checkpoint: global step, optimizer state, learning-rate schedulers, loop state and the training sampler state. The task allocation and model settings stored in it must match the saved configuration.
- The optimizer, the learning-rate schedule, the step counter and the sampler state are restored. With per-rank sampler states, the number of ranks must be the same as in the original run.
- The W&B run is resumed with the same run ID (read from `resume_info.yaml`).
- Each resume writes its Hydra metadata to a new directory under `resume_attempts/` in the run directory, so the original metadata is never overwritten.
- Starting `train.py` in a run directory that already holds a training run, without `+checkpoint`, is refused.

You can resume multiple times from the same run.

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

## The v1 training recipe
The recipe used for the v1 model is defined by four experiment configs in `configs/experiment/`, each building on the previous one. Select one with `+experiment=<name>`:

| Config | What it adds |
|--------|--------------|
| `final_multitask_m6` | Shared base: training settings, pocket crop, input noise, partial-conditioning options and the leakage exclusion lists. |
| `final_multitask_balanced_chiral` | The balanced 13-task allocation (`task_group=final_multitask_balanced`) and the chirality fix: requested-stereo labels from matched geometry, 25% of molecules trained with all labels hidden, the stereo-correction feature and the margin handedness loss. The fix applies to the seven tasks that supply the ligand identity. |
| `final_chiral_data` | Random initialization, an explicit global budget of 220,000 edges per update (4 ranks x 55,000), and the assets in `recipe_assets/`. No controller. |
| `final_chiral_vc_scratch` | The validation-controlled run: one learning-rate owner (base rate x warmup x controller multiplier), a persistent EMA of the weights, and a frozen control policy. `max_steps` is a resource ceiling only. |

The 13 tasks and their shares of optimizer steps (`configs/task_group/final_multitask_balanced.yaml`):

| Task | Share |
|------|-------|
| `rigid_docking_condensed`, `fixed_protein_ligand_denovo_condensed` | 15% each |
| `rigid_docking_pharmacophore_condensed`, `fixed_protein_pharmacophore_ligand_denovo_condensed`, `partial_rigid_docking_condensed`, `partial_fixed_protein_ligand_denovo_condensed`, `flexible_docking_condensed` | 12.3% each |
| `denovo_ligand_condensed`, `ligand_conformer_condensed` | 2% each |
| `denovo_ligand_from_pharmacophore_condensed`, `ligand_conformer_from_pharmacophore_condensed` | 1.25% each |
| `partial_denovo_ligand_condensed`, `partial_ligand_conformer_condensed` | 1% each |

### Data and assets
The recipe trains on Pharmit, PLINDER and CrossDocked. Set the data locations on the command line:
```console
python routines/train.py +experiment=final_chiral_data name=my_run \
    pharmit_path=<PHARMIT_DIR> plinder_path=<PLINDER_DIR> crossdocked_path=<CROSSDOCKED_DIR>
```
The v1 model was trained on the PLINDER and CrossDocked versions stored in directories named `plinder_12A_final` and `crossdocked/External_Split_Pocket12`; the recipe's default locations are `data/plinder_12A_final` and `data/crossdocked/External_Split_Pocket12` under the repository root. Two files in `recipe_assets/leakage_analysis/` list PLINDER systems and CrossDocked PDB codes to leave out of training, and `recipe_assets/masks_v2/` holds the matched-stereo masks (`ASSET_SHA256.txt` has checksums).

### The controlled run
`final_chiral_vc_scratch` reads these environment variables:

| Variable | Meaning |
|----------|---------|
| `OMTRA_CONTROL_POLICY` | Path to the frozen control-policy JSON. The run refuses to start without a frozen policy. |
| `OMTRA_RUN_UUID` | Identifier of the run. |
| `OMTRA_EVIDENCE_DIR` | Directory where an external evaluator publishes evidence bundles. |
| `OMTRA_RUN_COMMAND`, `OMTRA_RESOURCE_FLAG` | Optional paths for manual commands and a resource-limit flag. |

The controller (`omtra/utils/validation_control.py`, `omtra/utils/training_control.py`) only consumes evidence bundles. The policy file and the evaluator that builds the bundles are not part of this repository, so use `final_chiral_data` if you want the recipe without the controller.

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
# One Model to Rule Them All

A multi-task generative model for small-molecule structure-based drug design. 

# Building the Environment

For now:
```bash
git clone https://github.com/gnina/OMTRA.git
cd OMTRA
mamba create -n omtra python=3.11
mamba activate omtra
chmod +x build_env.sh
./build_env.sh
```

# TODO:
- [ ] residue type as node feature?
- [ ] need to apply masking on node vec feature loss
- [ ] need to consider permutation invariance for vector feature prediction?
- [ ] CCD code frequency weighting in plinder dataset
- [ ] there is massive utiltiy in an on-the-fly (batched) addition of radius edges overlaid on top of edges that contain bond order as an edge feature
- [ ] what to do with pharm interaction feature
- [ ] pharmit dataset chemical space conditoning + predictor?
- [ ] when training omtra with ligand encoder, inject encoder config from encoder checkpoint
- [ ] sample counts are really large when training?
- [ ] task-specific losses
- [ ] wandb metric groups? how does that work?
- [ ] need to isolate/replicate dataloader breaking with num_workers > 2
- [ ] methods for evaluating conformer quality?
- [ ] smooth task distribution for validation set
- [ ] fix molecular stability metric
- [ ] protein-ligand interaction metric? pose check! any others?
- [ ] add posebusters
- [ ] don't need flowmol validity stuff for conformer generation / docking
- [ ] alternative time sampling methods for training (semlaflow, foldflow2)
- [ ] verify apo-holo alignment


# How to train?

The training script is `routines/train.py`. By default this script will use hydra to read the config starting at the top level `configs/config.yaml`. You can override any arguments using the standard hydra command line syntax. 

## training modes

So importantly, we pre-train encoder/decoder pairs to obtain latent representations of ligands and protein pockets. You can use the same training script to train these, you just need to specify the `mode` argument. Currently the available modes are `omtra` and `ligand_encoder`. The config for the ligand encoder is stored in `cfg.ligand_encoder`; the specific hydra config groups is `configs/ligand_encoder`. Currently the default config has the ligand_encoder set to an empty yaml file; that is, there is no ligand encoder, and there is not latent ligand genearation. When training a omtra for latent ligand generation, you need to set `mode=omtra` and you need to set `cfg.model.ligand_encoder_checkpoint` to the checkpoint for a trained ligand encoder. 


# Specifying task_group

What tasks hydra supports / how it is trained on them is entirely specified by the `task_group` config. You can see an example in `configs/task_group/no_protein.yaml` which would create a version of omtra
that does a variety of tasks that do not involve protein structures. The genreal structure of the task group is as follows:

1. `task_phases` this is essentiall a list. Each item of the list describes a "phase" of training omtra; the idea is that different phases can have different task mixtures. For example, maybe you want to focus heavily on unconditional ligand generation intitally and then start to incorporate pocket-conditioned ligand generation in a second phase. Each phase has a duration (measured in minibatches) and a list of tasks + the probability of training on each task. The probabilities need not sum to one; they will be normalized. In otherwords, this specicies for each phase of trainnig, what is p(task) for each training batch.


2. `datset_task_coupling`; this is a dictionary where each key is a task and the value is a list specifying the dataset we will use for that task, along with the probability of using the dataset for that task. In other words, the dataset task coupling is directly specifying the probability distribution p(dataset|task).

Now, what are the tasks and datasets supported? We have defined registers of supported tasks and datasets. The registers are located in `omtra.tasks` and `omtra.datasets` respectively. Every task and datset is associated with a unique string; if your config file specifies a task/dataset not in the register, the training script will tell you so. There are utility functions for printing out the tasks/dataset names supported. I don't know where they are off the top of my head but I'll add them here eventually. 
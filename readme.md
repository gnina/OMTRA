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

```bash
git clone https://github.com/gnina/OMTRA.git
cd OMTRA
mamba create -n environment,yml
mamba activate omtra
pip install -e ,.
```

# TODO:
- [ ] filter private library pharmit compounds
- [ ] harmonic prior
- [ ] test resume functionality
- [ ] add shape-color similarity for docking evaluation
- [ ] use sc-rdkit for denovo evals: https://github.com/oxpig/DeLinker/blob/master/analysis/calc_SC_RDKit.py
- [ ] add vpa: https://arxiv.org/abs/2403.04747


# Essetial things

## Is there a pre-trained checkpoint you recommend I work with?

Yes and you should remind me to write them down here.

## Where is the data on the cluster? How do I tell omtra where the datasets are?

### If using `routines/train.py`

By default, the script will look in `data/pharmit` and `data/plinder` (paths relative to your omtra repository), for the pharmit and plinder datasets, respectively. Now these datasets are big and moving them around is difficult and we don't want to have too many duplciates floating around. You should point your script to an existing copy of the dataset. Currently, you can use these on the CSB cluster:


```console
plinder_path=/net/galaxy/home/koes/tjkatz/OMTRA/data/plinder
pharmit_path=/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit
```

## How do I sample a trained model?

The sampling script is `routines/sample.py` takes a checkpoint and a task as input. There are a few other arguments to controls its behavior: `--visualize` will write out trajectories, `--output_dir` will specify where to write the output, `--n_samples` will specify how many samples to generate, `--metrics` is a true/false flag indicating whether to compute metrics on the samples. Look at the script or run `python routines/sample.py --help` for more details. Below is an example command:

```console
python routines/sample.py local/runs_from_cluster/denovo_multiupdate_2025-05-04_20-44-972429/checkpoints/batch_45000.ckpt --task=denovo_ligand --n_samples=100 --metrics --output_dir=local/dev_samples
```

## How to train?

The training script is `routines/train.py`. By default this script will use hydra to read the config starting at the top level `configs/config.yaml`. You can override any arguments using the standard hydra command line syntax. 

## Specifying task_group

What tasks your omtra model supports is specified by the `task_group` config. You can see an example in `configs/task_group/no_protein.yaml` which would create a version of omtra that does a variety of tasks that do not involve protein structures. The genreal structure of the task group is as follows:

1. `task_phases` this is essentially a list. Each item of the list describes a "phase" of training omtra; the idea is that different phases can have different task mixtures. For example, maybe you want to focus heavily on unconditional ligand generation intitally and then start to incorporate pocket-conditioned ligand generation in a second phase. Each phase has a duration (measured in minibatches) and a list of tasks + the probability of training on each task. The probabilities need not sum to one; they will be normalized. In otherwords, this specicies for each phase of trainnig, what is p(task) for each training batch.


2. `datset_task_coupling`; this is a dictionary where each key is a task and the value is a list specifying the dataset we will use for that task, along with the probability of using the dataset for that task. In other words, the dataset task coupling is directly specifying the probability distribution p(dataset|task).

Now, what are the tasks and datasets supported? We have defined registers of supported tasks and datasets. The registers are located in `omtra.tasks` and `omtra.datasets` respectively. Every task and datset is associated with a unique string; if your config file specifies a task/dataset not in the register, the training script will tell you so. There are utility functions for printing out the tasks/dataset names supported. I don't know where they are off the top of my head but I'll add them here eventually. 

## training modes

So importantly, we pre-train encoder/decoder pairs to obtain latent representations of ligands and protein pockets. 

You can use the same training script for both encoder/decoder pairs as the omtra generative model. When using `routines/train.py`, you just need to specify the `mode` argument to distinguish between these "training modes". Currently the available modes are `omtra` and `ligand_encoder`. The config for the ligand encoder is stored in `cfg.ligand_encoder`; the specific hydra config group is `configs/ligand_encoder`. Currently the default config has the ligand_encoder set to an empty yaml file; that is, there is no ligand encoder, and there is not latent ligand genearation. When training a omtra for latent ligand generation, you need to set `mode=omtra` and you need to set `cfg.model.ligand_encoder_checkpoint` to the checkpoint for a trained ligand encoder. 

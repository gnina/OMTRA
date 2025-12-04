
# Evaluations
A pipeline for generating evaluation metrics for conditional tasks can be found in `omtra_pipelines/docking_eval/docking_eval.py`. The script can be used to sample an OMTRA model from a checkpoint and subsequently compute metrics, or compute metrics on pre-obtained samples. A detailed usage description can be found in `omtra_pipelines/docking_eval/readme.md`.


-----------------------------------------------------------------------------------------------------
# Results
The following section describes how to recreate the published results.

#### Plinder De Novo Design
```console
python ./omtra_pipelines/docking_eval/docking_eval.py \
    --ckpt_path <PATH>/last.ckpt \
    --task fixed_protein_ligand_denovo_condensed \
    --dataset plinder \
    --split test \
    --n_samples 100 \
    --n_replicates 100 \
    --max_batch_size 100 \
    --timeout 2700 \
    --sys_idx_file omtra_pipelines/docking_eval/plinder_eval_sys_idxs.csv
```

#### Crossdocked De Novo Design
```console
python ./omtra_pipelines/docking_eval/docking_eval.py \
    --ckpt_path <LAST>/last.ckpt \
    --task fixed_protein_ligand_denovo_condensed \
    --dataset crossdocked \
    --split val \
    --n_samples 86 \
    --n_replicates 100 \
    --max_batch_size 100 \
    --timeout 2700
```

#### Plinder Docking
```console
python ./omtra_pipelines/docking_eval/docking_eval.py \
    --ckpt_path <PATH>/last.ckpt \
    --task rigid_docking_condensed \
    --dataset plinder \
    --split test \
    --n_samples 100 \
    --n_replicates 100 \
    --max_batch_size 100 \
    --timeout 2700 \
    --sys_idx_file omtra_pipelines/docking_eval/plinder_eval_sys_idxs.csv
```

#### Posebusters Docking 
```console
python ./omtra_pipelines/docking_eval/docking_eval.py \
    --ckpt_path <PATH>/last.ckpt \
    --task rigid_docking_condensed \
    --dataset crossdocked \
    --split val \
    --n_samples 409 \
    --n_replicates 40 \
    --max_batch_size 100 \
    --timeout 2700 \
```

#### Pharmacophore-Conditioned De Novo Design
```console
python ./omtra_pipelines/docking_eval/docking_eval.py \
    --ckpt_path <PATH>/last.ckpt \
    --task fixed_protein_pharmacophore_ligand_denovo_condensed \
    --dataset plinder \
    --split test \
    --n_samples 100 \
    --n_replicates 100 \
    --max_batch_size 100 \
    --timeout 2700 \
    --sys_idx_file omtra_pipelines/docking_eval/plinder_eval_sys_idxs.csv
```

#### Pharmacophore-Conditioned Docking
```console
python ./omtra_pipelines/docking_eval/docking_eval.py \
    --ckpt_path <PATH>/last.ckpt \
    --task rigid_docking_pharmacophore_condensed \
    --dataset plinder \
    --split test \
    --n_samples 100 \
    --n_replicates 100 \
    --max_batch_size 100 \
    --timeout 2700 \
    --sys_idx_file omtra_pipelines/docking_eval/plinder_eval_sys_idxs.csv
```


# rountines/sample.py
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
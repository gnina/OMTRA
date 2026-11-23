
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
    --sys_idx_file omtra_pipelines/docking_eval/eval_idx_files/plinder_eval_sys_idxs.csv
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
    --sys_idx_file omtra_pipelines/docking_eval/eval_idx_files/plinder_eval_sys_idxs.csv
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
    --sys_idx_file omtra_pipelines/docking_eval/eval_idx_files/plinder_eval_sys_idxs.csv
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
    --sys_idx_file omtra_pipelines/docking_eval/eval_idx_files/plinder_eval_sys_idxs.csv
```


-----------------------------------------------------------------------------------------------------
# Sensitivity to the supplied pocket center and other inputs
By default `docking_eval.py` gives the model the center of the reference ligand (an oracle input) and the size of the reference ligand. The options below change what the model is given, so you can test how much a result depends on it. They are off unless stated.

| Option | Effect |
|--------|--------|
| `--com_offset_magnitude <Å>` | Apply a random offset of this magnitude to the ground-truth ligand center of mass that defines the pocket, before sampling. `0` disables it. |
| `--center_source {ligand,pocket}` | `ligand` (default) uses the reference ligand center (oracle). `pocket` uses the supplied pocket centroid. |
| `--additional_prot_crop <Å>` | Evaluation-only fixed pocket crop around the ligand, applied regardless of the crop settings stored in the checkpoint. |
| `--sample_ligand_size` / `--oracle_ligand_size` | Infer the de novo ligand size from the dataset prior, or use the reference ligand size. |
| `--pharm_input_noise_std`, `--fixed_coord_max_std`, `--fixed_token_max_prob` | Simulate errors in the supplied pharmacophore, partial ligand coordinates, and partial ligand chemistry. |
| `--seed <int>` | Make the conditioning selection, crop, perturbations and sampling reproducible. |

Example: Plinder docking with the pocket center shifted by 4 Å.
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
    --com_offset_magnitude 4.0 \
    --seed 0 \
    --sys_idx_file omtra_pipelines/docking_eval/eval_idx_files/plinder_eval_sys_idxs.csv
```
Run `python omtra_pipelines/docking_eval/docking_eval.py --help` for the full list of options.

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
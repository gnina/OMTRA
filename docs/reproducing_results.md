
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



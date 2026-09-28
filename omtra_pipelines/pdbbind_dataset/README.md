# PDBBind → OMTRA zarr pipeline

Processes DynamicBind PDBBind v11 `pocket_aligned_fill_missing` structures into
Plinder-compatible **pred** zarr stores for `predapo_conditioned_ligand_docking_condensed`.

## Prerequisites

1. Extract `pdbbind_v11_pocket_aligned_fill_missing.zip` so data lives at
   `~/.local/share/dynamicbind_data/pdbbind_v11/pocket_aligned_fill_missing/{uid}/`.
   If the zip preserves NAS paths (`mnt/nas/...`), move `pdbbind_v11` up:

   ```bash
   mv ~/.local/share/dynamicbind_data/mnt/nas/research-data/luwei/dynamicbind_data/pdbbind_v11 \
      ~/.local/share/dynamicbind_data/
   rm -rf ~/.local/share/dynamicbind_data/mnt
   ```

2. Place `d3_with_clash_info.csv` at `local/d3_with_clash_info.csv`.

`PDBBIND_RAW_ROOT` defaults to `~/.local/share/dynamicbind_data`. CSV NAS paths
(`/mnt/nas/research-data/luwei/dynamicbind_data/...`) are remapped to
`$PDBBIND_RAW_ROOT/pdbbind_v11/...`.

## Steps

```bash
# Run from repo root with PYTHONPATH=.
export PYTHONPATH=.

# 1. Build manifest (CSV → parquet; uses pdbFile/af2File as-is, valid→val)
python omtra_pipelines/pdbbind_dataset/build_manifest.py \
  --csv local/d3_with_clash_info.csv \
  --output data/pdbbind_12A/manifest.parquet \
  --require-files

# 2. Store pred zarr per split (12Å pocket)
python omtra_pipelines/pdbbind_dataset/store_pred_structures.py \
  --manifest data/pdbbind_12A/manifest.parquet \
  --split train \
  --output data/pdbbind_12A/pred/train.zarr \
  --pocket_cutoff 12.0 \
  --num_cpus 8

# Repeat for val and test splits.

# 3. Spot-check zarr vs reprocessed SystemData
python omtra_pipelines/pdbbind_dataset/validate_store.py \
  --zarr data/pdbbind_12A/pred/train.zarr \
  --split train \
  --n-systems 5

# 4. Add ligand/extra_feats for condensed atom typing (required for training)
for split in train val test; do
  python omtra_pipelines/plinder_ligand_properties/phase1.py \
    --plinder_path data/pdbbind_12A/pred \
    --store_name $split \
    --flat

  python omtra_pipelines/plinder_ligand_properties/run_phase2.py \
    --dataset pdbbind \
    --flat \
    --plinder_path data/pdbbind_12A/pred \
    --pdbbind_path data/pdbbind_12A \
    --store_name $split \
    --n_cpus 8 \
    --output_dir outputs/phase2_pdbbind/$split
done

# 5. Verify condensed atom tuple coverage
python omtra_pipelines/plinder_ligand_properties/count_lig_atom_feat_tuples.py \
  --flat \
  --plinder_path data/pdbbind_12A/pred \
  --splits train val test \
  --check_vocab \
  --output_dir outputs/phase2_pdbbind/count_lig_feats
```

## Training

```bash
python routines/train.py task_group=pdbbind_predapo name=pdbbind_predapo_test max_steps=10
```

## Slurm (full zarr build)

```bash
sbatch ~/scripts/pdbbind_store_zarr.sh
```

Array job `0-2` → `train`, `val`, `test` (32 CPUs each, `dept_cpu`, 48h). Logs:
`data/pdbbind_12A/logs/store_{split}.log` and `~/scripts/slurm-pdbbind-zarr/`.

Output layout: `data/pdbbind_12A/pred/{train,val,test}.zarr`

Manifest policy: `holo_pdb` ← `pdbFile`, `pred_pdb` ← `af2File` (includes
`_refined` / `_mutated` when CSV specifies them). `raw_af2File` is stored for
debugging only and is never used as pred input.

Pred/holo alignment: DynamicBind releases pocket-aligned structures in a common
coordinate frame, but chain IDs and residue numbering differ from holo. The
pipeline maps AF2 coords onto holo atom order via sequence alignment (not exact
PDB atom keys).

NPNDEs (cofactors, ions, metals) are extracted from holo PDB HETATM groups via
the same `filter_npndes` logic as CrossDocked and stored in the zarr `npnde`
group. DynamicBind preprocessed holo files are typically protein-only, so
`npndes` will often be empty.

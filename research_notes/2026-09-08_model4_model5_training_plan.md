# Model 4 & Model 5 Training Plan (leakage-filtered multitask + dynamic crop)

Date: 2026-09-08
Branch: `leakage_filtering` (cherry-pick dynamic crop onto it; do NOT full-merge)

## Goal
Train two new models that combine Juhi's multitask recipe (PLINDER + CrossDocked + Pharmit,
dynamic cropping, 12Å pockets, ligand-COM offset) with OUR leakage filtering applied to
**both** train sets, at `max_steps=400000`.

- **Model 4** = "Model 2 logic" (lighter PB filter)
- **Model 5** = "Model 3 logic" (strict PB filter)

## Filter design (FINAL)
| Filter applied to... | Model 4 | Model 5 |
|---|---|---|
| PLINDER train vs CD test | 30% identity | 30% identity |
| PLINDER train vs PB test | exact match | 30% identity |
| CrossDocked train vs PLINDER test | 30% identity | 30% identity |
| CrossDocked train vs PB test | exact match | 30% identity |

Retained after filtering (system-level, against the 12Å zarrs):
| | CD retained | PLINDER retained |
|---|---|---|
| Model 4 | 74,189 / 91,609 (81.0%) | 158,396 / 164,564 (96.3%) |
| Model 5 | 38,421 / 91,609 (41.9%) | 133,184 / 164,564 (80.9%) |

## Dataset coupling (FINAL)
Original coupling `CD 0.33 / PLINDER 0.67` ≈ natural pooled size proportion (35.8%).
Adapt to filtered sizes:
- **Model 4**: keep **CD 0.33 / PLINDER 0.67** (filtered proportion 0.319 ≈ unchanged).
- **Model 5**: **CD 0.25 / PLINDER 0.75** (1/4; filtered proportion 0.224, rounded to clean 1/4).
Applies to the two heavy tasks (rigid_docking_condensed, fixed_protein_ligand_denovo_condensed).
Pharmit tasks (denovo_ligand_condensed, ligand_conformer_condensed) stay at 0.025 each, pharmit 1.0.

---

## PHASE 0 — Data staging (copy locally)
**Deliverables:** local copies under our OMTRA_ROOT/data of:
- `plinder_12A_final/no_links/{train,val,test}.zarr` (train ~14G) from `/net/galaxy/home/koes/jmgupta/omtra_2/data/plinder_12A_final`
- `crossdocked/External_Split_Pocket12/{train,val}.zarr` (12Å, standard split) from Juhi's omtra_2

**Issues / confirm:**
- [ ] Disk space: plinder train 14G + CD ~several G. Confirm quota on our storage.
- [ ] The pharmacophore/pred/exp link versions of plinder_12A_final only have test.zarr — but the
      multitask tasks here use `no_links` only, so train/val/test no_links is sufficient. CONFIRM the
      four tasks in the coupling all resolve to `no_links` (plinder_link_version).
- [ ] plinder_12A_final val.zarr/test.zarr exist and are non-empty (verified: 52M / 71M).
- [ ] Do NOT use `Resplit_External_Split_Pocket12` — we use the standard `External_Split_Pocket12`
      so CD/PB test sets stay comparable to other tools.
- [ ] Verify copy integrity (zarr opens, system counts match: PL 164,564 / CD 91,609).

## PHASE 1 — Build the 2 new CrossDocked exclusion files
(PLINDER exclusion files already exist: `plinder_train_exclude_cd30_pb_exact.txt` 6,167;
`plinder_train_exclude_cd30_pb30.txt` 31,380.)

**Deliverables (in research_notes/leakage_analysis/):**
- `crossdocked_train_exclude_plinder30_pbexact.txt` — Model 4 CD side.
  Union of: CD-train ≥30% id vs PLINDER test  ∪  CD-train exact-PDB vs PB test (=0 exact).
- `crossdocked_train_exclude_plinder30_pb30.txt` — Model 5 CD side (7,052 PDB codes).
  Union of: CD-train ≥30% id vs PLINDER test  ∪  CD-train ≥30% id vs PB test.

**Format decision:** files are lists of **4-char uppercase PDB codes** (NOT PLINDER-style system IDs),
because the CrossDocked zarr identifies structures by `rec_pdb` (e.g. `5dxt_A_rec.pdb`), matched on
`rec_pdb[:4].upper()`. This differs from the PLINDER exclusion files (full system IDs) — the
crossdocked.py filter must implement PDB-code matching, not exact string match.

**Issues / confirm:**
- [ ] Source hit-sets already computed: `mmseqs_cd_train_full_vs_plinder_test/result.tsv`,
      `mmseqs_cd_train_full_vs_pb/result.tsv` (min-seq-id 0.3, cov 0.5). Reuse, don't rerun.
- [ ] Union must include exact matches (coverage-cutoff artifact drops some real exact hits from
      the identity search — always union exact ∪ id30).
- [ ] Model 4 uses exact-only for PB (0 codes) → its CD file = just the PLINDER-test 30% hits + the
      153 exact PLINDER-test matches. CONFIRM Model 4 CD file != Model 5 CD file.

## PHASE 2 — CrossDocked exclusion mechanism (code)
**Deliverables:**
- `omtra/dataset/crossdocked.py`: add `exclude_pdb_codes: Optional[set] = None` to `__init__`;
  after `system_lookup` built (~line 79) and BEFORE the min-atom filter reset, drop rows where
  `system_lookup['rec_pdb'].str[:4].str.upper().isin(exclude_pdb_codes)`.
- `omtra/dataset/multitask.py`: in the non-plinder branch (currently line 85-91), add
  crossdocked_exclude_file loading (mirror the plinder block at 68-83), pass `exclude_pdb_codes`.
- `configs/config.yaml`: add `crossdocked_exclude_file: null`.
- `configs/task_group/datamodule/single_datasets/crossdocked.yaml`: add
  `crossdocked_exclude_file: ${crossdocked_exclude_file}`.

**Issues / confirm:**
- [ ] The non-plinder branch in multitask.py also handles PHARMIT — must only pop/pass the exclude
      arg for crossdocked, not pharmit (pharmit has no such param). Guard on `dataset_name == 'crossdocked'`.
- [ ] `single_dataset_config.pop(...)` mutates a deepcopy (safe) — confirm pharmit config isn't given
      a stray crossdocked_exclude_file key that its dataset class rejects.
- [ ] rec_pdb values confirmed to be like "5dxt_A_rec.pdb" (first 4 = pdb code). Verified.
- [ ] Filtering happens before the reset_index at line 89 — keep index bookkeeping consistent
      (system_idx column, .iloc access at line 173). Re-verify iloc indexing still valid after filter.

## PHASE 3 — Cherry-pick dynamic cropping (code) — BE CAREFUL
Bring dynamic crop from `dynamic-crop` onto `leakage_filtering` WITHOUT reverting leakage work.

**Files to bring in (from dynamic-crop):**
- `omtra/dataset/crop_utils.py` (NEW, 410 lines) — clean add, no conflict.
- `omtra/dataset/plinder.py` (+123) — crop params + logic. MERGE with our exclude_system_ids additions.
- `omtra/dataset/crossdocked.py` (+71) — dynamic crop logic. MERGE with Phase 2 exclude additions.
- `omtra/models/omtra.py` (+14) — lig_com_offset_std.
- `omtra/dataset/multitask.py` (+9) — MERGE with our plinder_exclude + Phase 2 crossdocked_exclude.
- `omtra/load/quick.py` (+22) — crop param plumbing.
- `configs/config.yaml` — add crop_min_distance, crop_max_distance, alpha, beta,
  additional_prot_crop, lig_com_offset_std; set prot_pos_std 0.5→0.0.
  DO NOT adopt dynamic-crop's `plinder_path: plinder_12A_final` default blindly — set paths via CLI.

**Eval-side (optional now, needed for Phase 6):**
- `omtra_pipelines/docking_eval/docking_eval.py` (+37), `sampleblaster.py` (+67) — crop/offset CLI args.

**DO NOT bring in:**
- dynamic-crop's DELETION of the exclusion txt files (git diff shows -31381/-6168 because those were
  added on leakage_filtering, not on dynamic-crop). A cherry-pick/merge must PRESERVE them.
- plinder_pipeline.py / store_*.py / plinder_links_zarr.py changes (data regeneration, not training).
- The test suite (tests/**) unless we want it — large, optional.

**Issues / confirm:**
- [ ] crossdocked.py and multitask.py and plinder.py each receive changes from BOTH Phase 2 and
      Phase 3 — do Phase 3 first via careful manual merge, then re-apply/verify Phase 2, OR do them
      together file-by-file. Highest conflict risk in the repo.
- [ ] Confirm dynamic-crop's plinder.py crop logic and our exclude_system_ids logic don't touch the
      same lines (exclude is in __init__ filtering; crop is in __getitem__/graph build — likely disjoint).
- [ ] omtra.py lig_com_offset_std: confirm it applies offset to x_0 (prior) only, not x_1_true (target).
- [ ] After merge: `prot_pos_std` default becomes 0.0 — confirm this is what we want for Model 4/5
      (Juhi's runs used 0.0). The running Model 2/3 used the old default (0.5?) — note the divergence.
- [ ] gnn_partial_mod_cond had an unresolved conflict in omtra.py between partial-mod-cond and
      lig_com_offset_std — we are NOT touching that branch; make sure our omtra.py stays clean.
- [ ] Run a 1-step smoke train after merge before committing to full runs.

## PHASE 4 — Task group configs
**Deliverables:**
- Copy `fixed_protein_cond_a_crossdocked_plinder.yaml` from Juhi's omtra_2 to
  `configs/task_group/` (does NOT exist on any of our branches). → Model 4 uses this (CD 0.33/PL 0.67).
- Create `fixed_protein_cond_a_crossdocked_plinder_q.yaml` (or similar) = copy with the two heavy
  tasks' coupling changed to CD 0.25 / PLINDER 0.75. → Model 5 uses this.

**Issues / confirm:**
- [ ] The copied config references `datamodule: default` and task names — confirm those task classes
      (rigid_docking_condensed, fixed_protein_ligand_denovo_condensed, denovo_ligand_condensed,
      ligand_conformer_condensed) all exist on our branch's task register.
- [ ] Confirm the copied config's `dataset_task_coupling` dataset names ('crossdocked','plinder',
      'pharmit') match our single_datasets configs.
- [ ] Model 5 variant: only change the two heavy tasks' [crossdocked, x]/[plinder, y]; leave pharmit
      tasks untouched. Renormalization happens in build_td_coupling (per row), so 0.25/0.75 is fine.

## PHASE 5 — Training commands & submission
**Deliverables:** two commands (Model 4, Model 5), 400k steps, seeded from our uncond ckpt.
Template (fill exclude files + task_group + coupling per model):
```
python routines/train.py name=<gnn_prot_cond_cd_dyncrop | gnn_prot_cond_clean_dyncrop> \
  +partial_ckpt=outputs/2026-02-21/gnn_uncond_2026-02-21_21-52-121789/checkpoints/last.ckpt \
  task_group=<fixed_protein_cond_a_crossdocked_plinder | ..._q> \
  plinder_path=<local>/plinder_12A_final \
  crossdocked_path=<local>/External_Split_Pocket12 \
  pharmit_path=<local>/data/pharmit \
  plinder_exclude_file=research_notes/leakage_analysis/<plinder M2|M3 file> \
  crossdocked_exclude_file=research_notes/leakage_analysis/<crossdocked M4|M5 file> \
  crop_min_distance=8.0 crop_max_distance=12.0 alpha=1.61 beta=0.91 lig_com_offset_std=4.0 \
  max_steps=400000 edges_per_batch=220000 trainer=distributed trainer.devices=2 \
  trainer.val_check_interval=800 trainer.limit_val_batches=5 graph=more_prot \
  plinder_pskip_factor=1.0 model.train_t_dist=beta model.t_alpha=1.8 \
  model.cat_loss_weight=0.8 model/aux_losses=pairs model.time_scaled_loss=true num_workers=6
```

**Issues / confirm:**
- [ ] edges_per_batch: Juhi reduced hers (GPU OOM with 12Å pockets — bigger pockets = more edges).
      Our Model 2/3 use 220000 at 8Å. 12Å pockets are larger → MAY OOM at 220000. Consider lowering
      (Juhi's mt_crossdocked_ex_resplit used 100000). CONFIRM on a smoke run before full submit.
- [ ] GPU/partition: same L40/M46 koes_gpu setup as job 57302614. Confirm availability; do not
      preempt the running Model 2/3.
- [ ] DO NOT submit until Model 2/3 finish (user directive) and all confirms above are green.
- [ ] Checkpoint dir naming / wandb run naming distinct from Model 2/3.

## PHASE 6 — Evaluation
Per docs/reproducing_results.md, evaluate Model 4/5 (and re-use finished Model 2/3 for head-to-head):
- PLINDER test: de novo, docking, pharmacophore-conditioned de novo + docking
- CrossDocked test: de novo
- PoseBusters: docking validity
- Offset-robustness sweep (Juhi's dyn_crop_offset_eval methodology) — confirms the de novo
  robustness gain survives leakage filtering.

**Comparison structure:** Model 2 ↔ Model 4 (same leakage logic, ±CD-multitask+dyncrop);
Model 3 ↔ Model 5. Not Model 4 ↔ Model 5 directly (different coupling by design).

**Issues / confirm:**
- [ ] Eval must run on STANDARD test sets (not resplit) for cross-tool comparability.
- [ ] Need eval-side crop/offset CLI args (Phase 3 eval files) for correct sampling of dyncrop models.
- [ ] additional_prot_crop / com_offset_magnitude at inference: Juhi used additional_prot_crop=10.0,
      com_offset_magnitude=0.0 for docking eval. Decide eval-time crop settings.

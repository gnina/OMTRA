# OMTRA recipe findings to discuss with David / team

Date: 2026-09-08
Author: Marios
Context: While porting Juhi's dynamic-cropping + multitask recipe onto the `leakage_filtering`
branch (for the leakage-filtered paper models "Model 4/5"), two things surfaced about how the
models we're building on were *actually* trained. Neither breaks anything — but both are
recipe-level choices the team should be aware of and sign off on.

---

## Finding 1 — Dynamic cropping is applied to PLINDER only, NOT CrossDocked

### What we found
- The dynamic-cropping augmentation (beta-distributed pocket crop, 8–12 A) is wired into the
  **PLINDER** dataset config, but the **CrossDocked** dataset config never passes crop
  parameters. In `omtra_2` (Juhi's working copy that produced the Aug multitask models):
  - `configs/.../single_datasets/plinder.yaml` -> has `crop_min_distance`, `crop_max_distance`,
    `alpha`, `beta`, `additional_prot_crop`
  - `configs/.../single_datasets/crossdocked.yaml` -> only `processed_data_dir` (no crop params)
- Therefore in every recent multitask model (mt_plinder, mt_crossdocked,
  mt_plinder_crossdocked_ex_resplit, etc.), **CrossDocked pockets were used at their full static
  12 A cutoff; only PLINDER pockets were dynamically cropped during training.**
- Additionally, the CrossDocked dynamic-crop code path contained a **latent bug**
  (`sample_crop_distance` called with 2 args when it requires 4: `alpha`, `beta`). It never
  errored in practice — which is independent confirmation the CrossDocked crop path was never
  executed.

### Why it might be fine (the case for leaving it)
- Dynamic cropping mainly buys **robustness to imprecise pocket definition**, which matters most
  for de novo generation where the pocket isn't precisely known. PLINDER is the diverse,
  experimental-structure dataset where that augmentation helps generalization.
- CrossDocked is docked poses around already-defined pockets (more redundant); cropping adds less.
- This is the exact configuration that produced the current results — CrossDocked eval numbers
  were acceptable.

### Why it's worth a look (the case for changing)
- CrossDocked trains on full 12 A pockets but may be evaluated at tighter crops -> a mild
  train/inference pocket-size mismatch specific to CrossDocked.
- If the team believes crop robustness should apply uniformly, enabling CrossDocked cropping is a
  deliberate experiment (requires fixing the 2-arg bug and deciding uniform vs beta sampling).

### Decision for the team
- For the leakage-filtered Model 4/5 we are **replicating faithfully**: PLINDER cropped,
  CrossDocked static. We removed the broken/unused CrossDocked crop code rather than ship it.
- **Open question:** is CrossDocked-static intentional and something we keep in the paper, or do
  we want a follow-up run with CrossDocked cropping enabled to test it?

---

## Finding 2 — `prot_pos_std` (protein-position noising) is a dead setting

### What we found
- `prot_pos_std` is intended to add Gaussian noise to protein atom positions during training
  (a robustness augmentation). The code for it exists in `omtra/models/omtra.py`
  (`if has_protein and self.prot_pos_std > 0.0: ...`).
- BUT `prot_pos_std` is **not wired into the model config** in either our branch or `omtra_2`:
  it appears only as a top-level key in `configs/config.yaml` (value 0.5) and is never
  interpolated into `configs/model/default.yaml`. Nothing passes it to the model.
- Result: `self.prot_pos_std` always takes the code default `0.0`. Protein-position noising has
  been **off** in every model — Juhi's multitask models, our currently-running Model 2/3, and the
  planned Model 4/5.

### Why it's not a problem right now
- It's **consistent across every model** — no model is advantaged or disadvantaged, so all
  comparisons remain fair.

### Why it's worth flagging
- The `config.yaml` value `prot_pos_std: 0.5` is **misleading** — it looks active but does
  nothing. Anyone reading the config would reasonably assume protein noising is on at 0.5. This
  is a footgun for reproducibility / future work.
- Whether protein-position noising would actually *help* is an untested, open question.

### Decision for the team
- **Open question 1:** was protein-position noising supposed to be on? If yes, it's been silently
  disabled across all recent models and should be wired + re-evaluated.
- **Open question 2:** if we intend it to stay off, we should remove/zero the misleading
  `prot_pos_std: 0.5` from `config.yaml` to avoid confusion.

---

## Summary for the meeting
| # | Finding | Status in our Model 4/5 | Needs a team decision? |
|---|---|---|---|
| 1 | Dynamic crop = PLINDER only, CrossDocked static (+ dead buggy CD crop code) | Replicated faithfully (CD static); removed dead code | Yes — keep CD static, or test CD cropping? |
| 2 | `prot_pos_std` never wired -> protein noising off everywhere | Same as all models (off) | Yes — intended? and clean up misleading config value |

Neither affects the validity of the leakage-filtering comparison (both are constant across all
models being compared). Both are recipe-level choices worth an explicit team sign-off before the
paper.

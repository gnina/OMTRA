# Distributed Sampling Pipeline

Distributes OMTRA sampling and metrics evaluation across a SLURM cluster. Handles chunking, SLURM job submission with dependency chaining, status tracking, resume, and result aggregation.

## Sampling Interfaces

OMTRA has three scripts that can run sampling. Two of them are integrated into this distributed pipeline:

| Script | Distributed? | Input | Use Case |
|--------|-------------|-------|----------|
| `omtra` CLI | Yes (CLI mode) | Files (PDB, SDF, pharmacophore) | Generate many replicates for a single target |
| `docking_eval.py` | Yes (dataset mode) | Checkpoint + dataset indices | Benchmark evaluation across many systems |
| `routines/sample.py` | No (standalone) | Checkpoint + dataset indices | Interactive development, visualization, quick tests |

**CLI mode** parallelizes _replicates_ across GPU jobs. You provide input files (protein PDB, reference ligand SDF, etc.) and the pipeline distributes N total replicates across multiple SLURM array tasks.

**Dataset mode** parallelizes _systems_ across GPU jobs. You provide a checkpoint and dataset system indices, and the pipeline distributes systems across SLURM array tasks. Each task runs `docking_eval.py` which handles both sampling and metrics.

**`routines/sample.py`** is not part of this pipeline. Use it for interactive sampling during development — it supports visualization (`--visualize`), inline metrics (`--eval`), and unconditional generation. See [Using routines/sample.py](#using-routinessamplepy) below.

In both pipeline modes, metrics are always computed by `docking_eval.py --samples_dir`.

## Setup

### Prerequisites

1. **Conda env** — The `omtra` conda environment must be available on all cluster nodes.

2. **OMTRA installed** — The repo must be on a shared filesystem. The `omtra` CLI command must be on `$PATH`.

3. **gnina** — For gnina-based metrics, the binary must be at `{repo_root}/gnina.1.3.2`. If unavailable, disable with `metrics.disable_gnina=true`.

4. **Dataset paths** (dataset mode) — Set in a site config or your job config under `paths:`.

5. **SLURM partitions** — GPU partition for sampling, CPU partition for metrics. Defaults: `dept_gpu` / `dept_cpu`. Override via `--site` or inline.


## CLI Mode

Use CLI mode when you have specific input files (a protein structure, a reference ligand, etc.) and want to generate many replicates of a single target.

### Which task do I use?

| I want to... | Task name | Required input files |
|---|---|---|
| Design a new ligand for a protein pocket | `fixed_protein_ligand_denovo_condensed` | `protein_file`, pocket definition |
| Dock a known ligand into a protein | `rigid_docking_condensed` | `protein_file`, `ligand_file`, pocket definition |
| Generate a ligand conformer | `ligand_conformer_condensed` | `ligand_file` |
| Design a ligand from a pharmacophore | `denovo_ligand_from_pharmacophore_condensed` | `pharmacophore_file` |
| Design a ligand from protein + pharmacophore | `fixed_protein_pharmacophore_ligand_denovo_condensed` | `protein_file`, `pharmacophore_file`, pocket definition |
| Generate molecules unconditionally | `denovo_ligand_condensed` | _(none)_ |

**Pocket definition** — exactly one of:
- `pocket_ligand`: path to SDF file defining the binding pocket location (most common)
- `pocket_center`: comma-separated coordinates, e.g. `"1.0,2.0,3.0"`
- `pocket_residues`: residue specification, e.g. `"A:123-125,B:200"`

### Step-by-step: de novo ligand design for a protein target

**1. Write a config file** (`my_target.yaml`):

```yaml
task: fixed_protein_ligand_denovo_condensed
output_dir: /scratch/user/my_denovo_run

input_files:
  protein_file: /data/structures/1J3J.pdb
  ligand_file: /data/structures/CP6.sdf        # optional — enables RMSD metrics
  pocket_ligand: /data/structures/CP6.sdf       # defines the binding pocket

n_replicates_total: 1000
```

That's it — only Tier 1 fields. Everything else has defaults.

**2. Dry run** — inspect what would be submitted:

```bash
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_target.yaml --site cluster --dry-run
```

This creates the output directory with all generated SLURM scripts and command files, but doesn't submit anything. Inspect the files:

```bash
# See the exact omtra commands that will run:
cat /scratch/user/my_denovo_run/work/commands/sampling_commands.txt

# See the SLURM scripts:
cat /scratch/user/my_denovo_run/work/scripts/sampling.slurm
```

**3. Submit:**

```bash
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_target.yaml --site cluster
```

This submits three SLURM jobs with dependency chaining:
1. **Sampling** (GPU array job) — each task runs `omtra --task ... --n_samples {chunk_size}`
2. **Metrics** (CPU array job, waits for sampling) — each task runs `docking_eval.py --samples_dir ...`
3. **Aggregation** (CPU single job, waits for metrics) — merges all per-chunk results

**4. Monitor:**

```bash
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_target.yaml --status
```

### Step-by-step: rigid docking

```yaml
# docking_job.yaml
task: rigid_docking_condensed
output_dir: /scratch/user/docking_run

input_files:
  protein_file: /data/structures/receptor.pdb
  ligand_file: /data/structures/ligand.sdf
  pocket_ligand: /data/structures/ligand.sdf

n_replicates_total: 500
```

### Step-by-step: pharmacophore-conditioned design

```yaml
# pharm_job.yaml
task: fixed_protein_pharmacophore_ligand_denovo_condensed
output_dir: /scratch/user/pharm_run

input_files:
  protein_file: /data/structures/receptor.pdb
  pharmacophore_file: /data/pharmacophores/my_pharm.json   # Pharmit JSON, XYZ, or SDF
  pocket_ligand: /data/structures/ref_ligand.sdf

n_replicates_total: 1000
```

### Step-by-step: unconditional de novo generation

```yaml
# denovo_job.yaml
task: denovo_ligand_condensed
output_dir: /scratch/user/denovo_run

n_replicates_total: 5000
```

No `input_files` needed — the model generates molecules from scratch.

Pharmacophore files can be:
- **Pharmit JSON** — exported from [Pharmit](https://pharmit.csb.pitt.edu). Automatically converted to XYZ format for metrics.
- **XYZ format** — atom type + coordinates
- **SDF file** — pharmacophore features extracted from a ligand

### How chunking works in CLI mode

With `n_replicates_total: 1000` and `replicates_per_job: 100` (the default), the pipeline creates 10 SLURM array tasks. Each task generates 100 molecules. The last chunk may be smaller if the total isn't evenly divisible.

To change chunk size:
```bash
# Fewer, larger jobs:
... replicates_per_job=500

# More, smaller jobs:
... replicates_per_job=50
```

### Using a specific checkpoint

By default, the `omtra` CLI auto-resolves the checkpoint from the task name (using the mapping in `omtra/utils/checkpoints.py`). To use a specific checkpoint:

```yaml
checkpoint: /path/to/my_checkpoint.ckpt
```


## Dataset Mode

Use dataset mode when you want to evaluate a model checkpoint across many systems from a dataset (plinder or crossdocked). This is the standard benchmark evaluation workflow.

### Step-by-step: evaluate rigid docking on plinder test set

**1. Prepare system indices.** You need a CSV file listing which dataset indices to evaluate:

```bash
# Example: evaluate systems 0-99
python -c "print(','.join(str(i) for i in range(100)))" > eval_sys_idxs.csv
```

Or use a pre-existing index file.

**2. Write a config file** (`my_eval.yaml`):

```yaml
checkpoint: /path/to/checkpoints/last.ckpt
task: rigid_docking_condensed
dataset: plinder
output_dir: /scratch/user/plinder_eval

sys_idx_file: /data/eval_sys_idxs.csv
replicates_per_system: 10                       # optional, default: 1
```

**3. Dry run, then submit:**

```bash
# Dry run
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_eval.yaml --site cluster --dry-run

# Submit
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_eval.yaml --site cluster
```

### Alternative: specify system count instead of index file

Instead of a CSV file, you can specify a count and start index:

```yaml
checkpoint: /path/to/last.ckpt
task: rigid_docking_condensed
dataset: plinder
output_dir: /scratch/user/plinder_eval

n_systems: 100
dataset_start_idx: 0     # optional, default: 0
```

### How chunking works in dataset mode

With 100 systems and `systems_per_job: 10` (the default), the pipeline creates:
- A `work/chunks/` directory with 10 CSV files, each listing 10 system indices
- 10 SLURM array tasks for sampling, each running `docking_eval.py --ckpt_path ... --sample_only` on its chunk
- 10 SLURM array tasks for metrics, each running `docking_eval.py --samples_dir ...` on the sampling output

### What docking_eval.py does under the hood

In dataset mode, the pipeline calls `docking_eval.py` (at `omtra_pipelines/docking_eval/docking_eval.py`) in two phases:

**Sampling phase** (`--ckpt_path ... --sample_only`):
- Loads the checkpoint and dataset
- Samples the specified systems from the dataset
- Writes generated molecules, ground truth ligands, proteins, and pharmacophores to the output directory
- Does NOT compute metrics (that's the next phase)

**Metrics phase** (`--samples_dir ...`):
- Reads the sampling output directory
- Computes all enabled metrics: PoseBusters validity, GNINA scores, RMSD, PoseCheck (clashes, strain, interactions), pharmacophore matching
- Writes `eval_metrics.csv` and `sys_info.csv`


## Using routines/sample.py

`routines/sample.py` is a standalone sampling script for interactive use. It is **not** part of the distributed pipeline — use it for development, debugging, and quick experiments on a single GPU.

### When to use routines/sample.py

- **Visualization**: `--visualize` saves trajectory SDFs showing the sampling process
- **Quick tests**: sample a few systems without pipeline overhead
- **Inline metrics**: `--eval` computes metrics in the same process (no separate stage)
- **Unconditional generation**: generate molecules without any conditioning

### Examples

```bash
# Sample 10 systems from plinder, 5 replicates each
python routines/sample.py /path/to/checkpoint.ckpt \
  --task rigid_docking_condensed \
  --dataset plinder \
  --n_samples 10 \
  --n_replicates 5 \
  --output_dir /scratch/test_samples

# Sample with visualization (saves trajectory)
python routines/sample.py /path/to/checkpoint.ckpt \
  --task rigid_docking_condensed \
  --dataset plinder \
  --n_samples 3 \
  --n_replicates 1 \
  --visualize \
  --output_dir /scratch/vis_samples

# Sample and compute metrics inline
python routines/sample.py /path/to/checkpoint.ckpt \
  --task rigid_docking_condensed \
  --dataset plinder \
  --n_samples 10 \
  --n_replicates 5 \
  --output_dir /scratch/test_samples \
  --eval                    # all applicable metrics
  # or: --eval posebusters gnina rmsd

# Unconditional ligand generation
python routines/sample.py /path/to/checkpoint.ckpt \
  --task denovo_ligand_condensed \
  --dataset pharmit \
  --n_samples 100 \
  --output_dir /scratch/denovo
```

### Key differences from the distributed pipeline

| | `routines/sample.py` | Distributed pipeline |
|---|---|---|
| Parallelism | Single GPU | SLURM array jobs across cluster |
| Metrics | Inline (`--eval`) or via old-style `--metrics` | Separate CPU stage via `docking_eval.py` |
| Visualization | `--visualize` flag | Not supported |
| Unconditional tasks | Supported | Supported (CLI mode) |
| Scale | 10s of systems | 100s-1000s of systems/replicates |

### n_samples meaning

Note that `--n_samples` means different things depending on context:
- In `routines/sample.py` (dataset mode): number of _systems_ to sample from the dataset
- In `omtra` CLI (file mode): number of _molecules_ to generate (internally converted to replicates)

The distributed pipeline config uses unambiguous names (`n_replicates_total`, `replicates_per_system`, `systems_per_job`) to avoid this confusion.


## Configuration System

The pipeline uses a **three-tier config** with layered merging via OmegaConf.

### Config loading order (later overrides earlier)

```
1. defaults/default.yaml                  (always — Tier 2+3 defaults)
2. defaults/site/{name}.yaml              (if --site specified — site settings)
3. User --config YAML file                (your job-specific settings)
4. CLI positional overrides               (key=value — highest priority)
```

### Three tiers

| Tier | What | Where |
|------|------|-------|
| **1** | Must specify every run — no sensible default | Your `--config` file |
| **2** | Reasonable defaults, override when needed | `defaults/default.yaml` |
| **3** | Site-specific, rarely change | `defaults/site/{name}.yaml` |

**Your config file only needs Tier 1 fields.** Everything else has defaults.

### Full config reference

#### Tier 1 — Required

| Parameter | Mode | Description |
|-----------|------|-------------|
| `task` | Both | Task name (e.g. `rigid_docking_condensed`) |
| `output_dir` | Both | Path for pipeline output |
| `checkpoint` | Dataset | Path to model checkpoint |
| `dataset` | Dataset | `plinder` or `crossdocked` |
| `sys_idx_file` | Dataset | Path to CSV of system indices (or use `n_systems`) |
| `n_systems` | Dataset | Number of systems (alternative to `sys_idx_file`) |
| `input_files` | CLI | Dict of input file paths (see below) |
| `n_replicates_total` | CLI | Total molecules to generate |

**`input_files` keys:**

| Key | Required | Description |
|-----|----------|-------------|
| `protein_file` | For protein tasks | Path to protein PDB |
| `ligand_file` | No | Reference ligand SDF (enables RMSD metrics) |
| `pharmacophore_file` | For pharm tasks | Pharmit JSON, XYZ, or SDF |
| `pocket_ligand` | One pocket method | SDF defining pocket location |
| `pocket_center` | One pocket method | Comma-separated coordinates |
| `pocket_residues` | One pocket method | Residue specification |
| `bbox_length` | No | Bounding box size for pocket |

#### Tier 2 — Defaults (often overridden)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `split` | `test` | Dataset split |
| `systems_per_job` | `10` | Systems per GPU job (dataset mode) |
| `replicates_per_system` | `1` | Replicates per system (dataset mode) |
| `replicates_per_job` | `100` | Replicates per GPU job (CLI mode) |
| `dataset_start_idx` | `0` | Starting index when using `n_systems` |
| `sampling.n_timesteps` | `250` | Integration steps |
| `sampling.stochastic_sampling` | `false` | Enable stochastic sampling |
| `sampling.bs_per_gbmem` | `5` | Batch size scaling (dataset mode) |
| `sampling.max_batch_size` | `300` | Max batch size (dataset mode) |
| `sampling.noise_scaler` | `null` | Noise scaling for stochastic sampling |
| `sampling.eps` | `null` | g(t) param for stochastic sampling |
| `metrics.timeout` | `2700` | Max seconds per metric computation |
| `metrics.disable_gnina` | `false` | Skip GNINA scoring |
| `metrics.disable_pb_valid` | `false` | Skip PoseBusters validity |
| `metrics.disable_posecheck` | `false` | Skip PoseCheck suite |
| `metrics.disable_rmsd` | `false` | Skip RMSD computation |
| `metrics.disable_strain` | `false` | Skip strain energy |
| `metrics.disable_interaction_recovery` | `true` | Skip interaction recovery |
| `metrics.disable_pharm_match` | `false` | Skip pharmacophore matching |
| `metrics.disable_ground_truth_metrics` | `false` | Skip ground truth metrics |

#### Tier 3 — Site-specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slurm.partition_gpu` | `dept_gpu` | GPU SLURM partition |
| `slurm.partition_cpu` | `dept_cpu` | CPU SLURM partition |
| `slurm.cpus_per_task` | `4` | CPUs per SLURM task |
| `slurm.mem` | `32G` | Memory per task |
| `slurm.time_sampling` | `4:00:00` | Sampling time limit |
| `slurm.time_metrics` | `8:00:00` | Metrics time limit |
| `slurm.time_aggregate` | `0:30:00` | Aggregation time limit |
| `slurm.conda_env` | `omtra` | Conda environment name |
| `slurm.extra_sbatch_args` | `""` | Additional sbatch arguments |
| `paths.plinder` | `null` | Path to plinder dataset |
| `paths.crossdocked` | `null` | Path to crossdocked dataset |

### Override examples

```bash
# Change sampling steps and disable gnina
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_job.yaml --site cluster \
  sampling.n_timesteps=500 metrics.disable_gnina=true

# Use a different SLURM partition with more time
... slurm.partition_gpu=long_gpu slurm.time_sampling=12:00:00

# Change chunk size for dataset mode
... systems_per_job=20

# Enable stochastic sampling
... sampling.stochastic_sampling=true sampling.noise_scaler=0.5
```

### Backward compatibility

Old-style configs that include all settings (sampling, metrics, slurm sections) continue to work unchanged. Legacy key names are also accepted:

| Legacy name | New name |
|------------|----------|
| `sampling_chunk_size` | `systems_per_job` |
| `n_replicates` | `replicates_per_system` |
| `replicates_per_chunk` | `replicates_per_job` |


## Pipeline Stages

Three stages run as SLURM jobs with dependency chaining (each waits for the previous):

### 1. Sampling (GPU, array job)

- **Dataset mode**: Each array task runs `docking_eval.py --ckpt_path ... --sample_only` on a chunk of systems.
- **CLI mode**: Each array task runs `omtra --task ... --n_samples {chunk_replicates}`.

### 2. Metrics (CPU, array job)

Both modes run `docking_eval.py --samples_dir` on each chunk's output. Computes PoseBusters validity, GNINA scores, RMSD, PoseCheck (clashes, strain, interactions), pharmacophore matching.

### 3. Aggregation (CPU, single job)

Concatenates `eval_metrics.csv` and `sys_info.csv` from all chunks into `results/`. In CLI mode, also merges `gen_ligands.sdf` files.


## Operations

### Check status

```bash
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_job.yaml --status
```

### Resume failed tasks

Resubmits only tasks whose `.done` marker is missing:

```bash
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_job.yaml --resume
```

### Run a single stage

```bash
# Only sampling
... --stage sampling

# Only metrics (assumes sampling is already complete)
... --stage metrics

# Only aggregation (assumes metrics is already complete)
... --stage aggregate
```

### Run a single chunk locally (debugging)

After a dry run, inspect and run a specific chunk's command directly:

```bash
# Look at the generated commands
cat output_dir/work/commands/sampling_commands.txt

# Run chunk 0's sampling command directly
sed -n '1p' output_dir/work/commands/sampling_commands.txt | bash

# Run chunk 0's metrics command directly
sed -n '1p' output_dir/work/commands/metrics_commands.txt | bash
```

### Run aggregation manually

```bash
python -m omtra_pipelines.distributed_sampling.aggregate_results \
  --manifest output_dir/manifest.json
```


## Output Directory Structure

```
output_dir/
├── manifest.json              # Full pipeline specification
├── pipeline_config.yaml       # Resolved config (all tiers merged)
├── status/                    # Completion markers
│   ├── sampling_0.done
│   ├── metrics_0.done
│   └── aggregate.done
├── logs/                      # SLURM stdout/stderr
├── samples/                   # Sampling output
│   ├── chunk_0/
│   │   └── sys_0_gt/
│   │       ├── protein_0.pdb
│   │       ├── ligand.sdf
│   │       └── gen_ligands.sdf
│   └── chunk_1/
├── metrics/                   # Per-chunk metrics
│   ├── chunk_0/
│   │   └── eval_metrics.csv
│   └── chunk_1/
├── results/                   # Aggregated results
│   ├── eval_metrics_all.csv
│   ├── sys_info_all.csv
│   ├── gen_ligands_all.sdf    # CLI mode only
│   └── summary.json
└── work/                      # Generated pipeline files
    ├── chunks/                # Chunk CSVs (dataset mode only)
    ├── commands/
    │   ├── sampling_commands.txt
    │   └── metrics_commands.txt
    └── scripts/
        ├── sampling.slurm
        ├── metrics.slurm
        └── aggregate.slurm
```


## Tuning Parallelism

- **Dataset mode** (`systems_per_job`): 10-20 is a good starting point. More = fewer jobs but longer per-job runtime.
- **CLI mode** (`replicates_per_job`): 100 is reasonable. Increase for fast tasks (rigid docking), decrease for expensive tasks (de novo).
- **Replicates per system** (`replicates_per_system`): Only relevant in dataset mode. Set >1 when you need multiple samples per system (e.g. for diversity analysis).


## Troubleshooting

**Sampling jobs fail immediately**: Check `logs/sampling_*.err`. Common causes:
- Conda env not found (check `slurm.conda_env`)
- Missing dataset paths (set `paths.plinder` or `paths.crossdocked`)
- Checkpoint not found

**Metrics fail with "Missing directory for system"**: Sampling didn't write output. Check `status/sampling_N.done` exists for the corresponding chunk.

**CLI mode metrics fail with "Missing ground truth ligand"**: Add `ligand_file` to `input_files` or set `metrics.disable_rmsd=true`.

**GNINA fails with missing CUDA libraries**: GNINA requires CUDA libraries even for scoring. Options:
- Run metrics on GPU nodes (change `slurm.partition_cpu` to a GPU partition)
- Disable GNINA: `metrics.disable_gnina=true`

**Resume resubmits everything**: Status markers are in `output_dir/status/`. If deleted, all tasks appear incomplete.

**Low RDKit validity**: This indicates model quality at the given checkpoint, not a pipeline problem. Check that you're using a sufficiently trained checkpoint.

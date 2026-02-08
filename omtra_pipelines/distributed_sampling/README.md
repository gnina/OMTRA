# Distributed Sampling Pipeline

Distributes OMTRA sampling and metrics evaluation across a SLURM cluster. Handles chunking, SLURM job submission with dependency chaining, status tracking, resume, and result aggregation.

## Two Modes

| | Dataset mode | CLI mode |
|---|---|---|
| **Input** | System indices from plinder/crossdocked | Files (PDB, SDF, pharmacophore) |
| **What gets chunked** | Systems across jobs | Replicates across jobs |
| **Sampling command** | `docking_eval.py --ckpt_path ... --sample_only` | `omtra --task ... --protein_file ...` |
| **When to use** | Benchmark evaluation across many systems | Blast replicates for a single target |

## Setup

### Prerequisites

1. **Conda env** — The `omtra` conda environment must be available on all cluster nodes. The SLURM templates do `conda activate $conda_env` (default: `omtra`).

2. **OMTRA installed** — The repo must be on a filesystem accessible from all nodes (e.g. `/net/galaxy/home/koes/icd3/moldiff/OMTRA`). Either `pip install -e .` in the conda env, or ensure `PYTHONPATH` includes the repo root. The `omtra` CLI command must be on `$PATH` (installed via the package entry point).

3. **gnina** — For gnina-based metrics, the binary must be at `{repo_root}/gnina.1.3.2`. If you don't have it, set `disable_gnina: true` in the metrics config.

4. **Dataset paths** (dataset mode only) — Plinder and/or CrossDocked data must be accessible. Set their paths in the config under `paths:`.

5. **SLURM partitions** — You need a GPU partition for sampling and a CPU partition for metrics. Defaults are `dept_gpu` and `dept_cpu`.

### Cluster directory layout

```
/net/galaxy/home/koes/icd3/moldiff/OMTRA/   # repo root
├── omtra_pipelines/distributed_sampling/     # this pipeline
├── omtra_pipelines/docking_eval/             # docking_eval.py (metrics)
├── gnina.1.3.2                               # gnina binary
└── omtra/trained_models/                     # checkpoints (CLI mode auto-resolves)
```

## Quick Start

### Dataset mode — evaluate across many systems

```bash
# 1. Create your config (copy and edit the example)
cp omtra_pipelines/distributed_sampling/example_config.yaml my_eval.yaml
# Edit: set checkpoint, task, dataset, sys_idx_file or n_systems, paths, output_dir

# 2. Dry run — inspect generated commands without submitting
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_eval.yaml --dry-run

# 3. Submit
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_eval.yaml
```

### CLI mode — blast replicates for a single target

```bash
# 1. Create your config
cp omtra_pipelines/distributed_sampling/example_config_cli.yaml my_target.yaml
# Edit: set task, input_files (protein, ligand, pocket), n_replicates_total, output_dir

# 2. Dry run
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_target.yaml --dry-run

# 3. Submit
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_target.yaml
```

## Configuration Reference

### Dataset mode (`example_config.yaml`)

```yaml
# Required
checkpoint: /path/to/last.ckpt
task: rigid_docking_condensed
dataset: plinder              # or crossdocked
split: test
output_dir: /path/to/output

# Systems — one of:
sys_idx_file: /path/to/indices.csv   # single-line CSV of system indices
# OR:
# n_systems: 100
# dataset_start_idx: 0

n_replicates: 10              # replicates per system
sampling_chunk_size: 10       # systems per SLURM task

# Dataset paths
paths:
  plinder: /path/to/plinder
  crossdocked: /path/to/crossdocked
```

### CLI mode (`example_config_cli.yaml`)

```yaml
mode: cli                     # or omit — inferred from presence of input_files
task: fixed_protein_ligand_denovo_condensed
output_dir: /path/to/output

input_files:
  protein_file: /path/to/protein.pdb
  ligand_file: /path/to/ligand.sdf          # optional (enables RMSD)
  pharmacophore_file: /path/to/pharm.json    # optional (Pharmit JSON or XYZ)
  pocket_ligand: /path/to/pocket_ref.sdf     # exactly one pocket method required
  # pocket_center: "1.0,2.0,3.0"
  # pocket_residues: "A:123-125,B:200"

# checkpoint: /path/to/ckpt   # optional — auto-resolved from task

n_replicates_total: 1000      # total replicates to generate
replicates_per_chunk: 100     # replicates per SLURM task (→ 10 jobs)
```

### Shared config sections

```yaml
sampling:
  n_timesteps: 250
  stochastic_sampling: false
  # bs_per_gbmem: 5           # dataset mode only
  # max_batch_size: 300        # dataset mode only
  # noise_scaler: 1.0
  # eps: 0.01

metrics:
  timeout: 2700
  disable_gnina: false
  disable_pb_valid: false
  disable_posecheck: false
  disable_rmsd: false
  disable_strain: false
  disable_interaction_recovery: true
  disable_pharm_match: false
  disable_ground_truth_metrics: false

slurm:
  partition_gpu: dept_gpu
  partition_cpu: dept_cpu
  cpus_per_task: 4
  mem: 32G
  time_sampling: "4:00:00"
  time_metrics: "8:00:00"
  time_aggregate: "0:30:00"
  conda_env: omtra
  extra_sbatch_args: ""       # e.g. "#SBATCH --account=myaccount"
```

## Pipeline Stages

The pipeline runs three stages with SLURM dependency chaining (each waits for the previous to finish):

### 1. Sampling (GPU, array job)

- **Dataset mode**: Each array task runs `docking_eval.py --sample_only` on a chunk of systems.
- **CLI mode**: Each array task runs `omtra --task ... --n_samples {chunk_replicates}` on the same input files.

### 2. Metrics (CPU, array job)

Both modes run `docking_eval.py --samples_dir` on each chunk's output. Computes PoseBusters validity, gnina scores, RMSD, posecheck (clashes, strain, interactions), pharmacophore matching.

### 3. Aggregation (CPU, single job)

Concatenates `eval_metrics.csv` and `sys_info.csv` from all chunks. In CLI mode, also merges `gen_ligands.sdf` files into `results/gen_ligands_all.sdf`.

## Output Directory Structure

```
output_dir/
├── manifest.json              # full pipeline manifest
├── pipeline_config.yaml       # copy of input config
├── status/                    # completion markers
│   ├── sampling_0.done
│   ├── metrics_0.done
│   └── aggregate.done
├── logs/                      # SLURM stdout/stderr
├── samples/                   # sampling output
│   ├── chunk_0/
│   │   └── sys_0_gt/          # per-system directories
│   │       ├── protein_0.pdb
│   │       ├── ligand.sdf
│   │       └── gen_ligands.sdf
│   └── chunk_1/
├── metrics/                   # per-chunk metrics
│   ├── chunk_0/
│   │   └── eval_metrics.csv
│   └── chunk_1/
├── results/                   # aggregated results
│   ├── eval_metrics_all.csv
│   ├── sys_info_all.csv
│   ├── gen_ligands_all.sdf    # CLI mode only
│   └── summary.json
└── work/                      # generated pipeline files
    ├── chunks/                # chunk CSVs (dataset mode)
    ├── commands/
    │   ├── sampling_commands.txt
    │   └── metrics_commands.txt
    └── scripts/
        ├── sampling.slurm
        ├── metrics.slurm
        └── aggregate.slurm
```

## Operations

### Check status

```bash
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_eval.yaml --status
```

### Resume failed tasks

Resubmits only tasks whose `.done` marker is missing:

```bash
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_eval.yaml --resume
```

### Run a single stage

```bash
# Only sampling (skip metrics + aggregate)
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_eval.yaml --stage sampling

# Only metrics (assumes sampling is done)
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_eval.yaml --stage metrics

# Only aggregate
python -m omtra_pipelines.distributed_sampling.launch_pipeline \
  --config my_eval.yaml --stage aggregate
```

### Inspect generated commands

After a dry run, check the generated command files:

```bash
cat output_dir/work/commands/sampling_commands.txt
cat output_dir/work/commands/metrics_commands.txt
```

### Run a single chunk locally (for debugging)

```bash
# Simulate SLURM array task 1 (line numbers are 1-indexed in the commands file)
SLURM_ARRAY_TASK_ID=1 bash output_dir/work/scripts/sampling.slurm
```

## CLI Mode: Ground Truth Files

In CLI mode, the pipeline copies your input files into each chunk's `sys_0_gt/` directory at launch time, before SLURM submission. This is necessary because `docking_eval.py` expects ground truth files in a specific layout:

- `protein_file` → `sys_0_gt/protein_0.pdb`
- `ligand_file` → `sys_0_gt/ligand.sdf`
- `pharmacophore_file` → `sys_0_gt/pharmacophore.xyz` (Pharmit JSON is converted to XYZ)

If `ligand_file` is not provided, RMSD-based metrics will be unavailable.

## Tuning Chunk Sizes

- **Dataset mode** (`sampling_chunk_size`): Systems per GPU job. Larger chunks = fewer jobs but longer per-job runtime. 10-20 is a good starting point.
- **CLI mode** (`replicates_per_chunk`): Replicates per GPU job. Balance between job overhead and GPU utilization. 100 replicates is reasonable for most tasks; increase for fast tasks (like rigid docking) or decrease for expensive tasks (like de novo).

## Troubleshooting

**Sampling jobs fail immediately**: Check `logs/sampling_*.err`. Common causes: conda env not found on compute node, missing dataset paths, checkpoint not found.

**Metrics fail with "Missing directory for system"**: The sampling step didn't write output to the expected location. Check that sampling completed for that chunk (`status/sampling_N.done` exists).

**Resume resubmits everything**: Status markers live in `output_dir/status/`. If the directory was deleted, all tasks appear incomplete.

**CLI mode metrics fail with "Missing ground truth ligand"**: You didn't provide `ligand_file` in the config. Either add it or disable RMSD (`disable_rmsd: true`).

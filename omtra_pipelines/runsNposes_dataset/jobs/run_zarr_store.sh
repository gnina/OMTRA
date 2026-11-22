#!/bin/bash
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  SLURM script for building Runs N' Poses Zarr stores (no_links only)
#
#  Usage:
#    sbatch --export=SPLIT=test,CUTOFF=8.0,OUTDIR=runs_n_poses,GROUND_TRUTH=/path/to/ground_truth run_zarr_store.sh
#
#  Environment variables (pass via --export):
#    SPLIT         : test (RnP is test-only)
#    CUTOFF        : pocket cutoff in Angstroms (8.0 or 12.0)
#    OUTDIR        : output subdirectory name (e.g. runs_n_poses, runs_n_poses_wide)
#    GROUND_TRUTH  : path to extracted ground_truth directory
#    NUM_CPUS      : number of CPUs (default: 16)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=90G
#SBATCH --time=28-00:00:00
#SBATCH --output=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/runsNposes_dataset/jobs/logs/rnp_%j_%x.out
#SBATCH --error=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/runsNposes_dataset/jobs/logs/rnp_%j_%x.err

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────
NUM_CPUS="${NUM_CPUS:-16}"
SPLIT="${SPLIT:?ERROR: SPLIT not set (e.g. test)}"
CUTOFF="${CUTOFF:?ERROR: CUTOFF not set (e.g. 8.0 or 12.0)}"
OUTDIR="${OUTDIR:?ERROR: OUTDIR not set (e.g. runs_n_poses or runs_n_poses_wide)}"
GROUND_TRUTH="${GROUND_TRUTH:?ERROR: GROUND_TRUTH not set (path to ground_truth dir)}"

# ── Paths ─────────────────────────────────────────────────────────
OMTRA_DIR="/net/galaxy/home/koes/nduaka1/OMTRA"
PARQUET="${OMTRA_DIR}/omtra_pipelines/runsNposes_dataset/plinder_runsnposes.parquet"
OUTPUT_BASE="${OMTRA_DIR}/data/${OUTDIR}"
LOG_DIR="${OMTRA_DIR}/omtra_pipelines/runsNposes_dataset/logs"

mkdir -p "${OUTPUT_BASE}/no_links"
mkdir -p "${LOG_DIR}"

OUTPUT_ZARR="${OUTPUT_BASE}/no_links/${SPLIT}.zarr"

export LOG_FILE_PATH="${LOG_DIR}/rnp_no_links_${SPLIT}_${CUTOFF}.log"

# ── Fix multiprocessing semaphore issue on SLURM nodes ────────────
export OMTRA_MP_START_METHOD="fork"

# ── Prevent PyTorch thread oversubscription (1 thread per worker) ──
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# ── Use cached HuggingFace models (avoid 16 workers hitting HF simultaneously) ──
export HF_HUB_OFFLINE=1

# ── Activate environment ──────────────────────────────────────────
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate omtra

cd "${OMTRA_DIR}"

# ── Log job info ──────────────────────────────────────────────────
echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Split:        ${SPLIT}"
echo "Cutoff:       ${CUTOFF} Å"
echo "Output:       ${OUTPUT_ZARR}"
echo "Parquet:      ${PARQUET}"
echo "Ground truth: ${GROUND_TRUTH}"
echo "CPUs:         ${NUM_CPUS}"
echo "Started:      $(date)"
echo "============================================"

# ── Run ───────────────────────────────────────────────────────────
CUTOFF_INT="${CUTOFF%.*}"

python omtra_pipelines/runsNposes_dataset/store_unlinked_structures.py \
    --data "${PARQUET}" \
    --ground_truth "${GROUND_TRUTH}" \
    --split "${SPLIT}" \
    --output "${OUTPUT_ZARR}" \
    --pocket_cutoff "${CUTOFF_INT}" \
    --num_cpus "${NUM_CPUS}"

echo "Finished: $(date)"
exit 0

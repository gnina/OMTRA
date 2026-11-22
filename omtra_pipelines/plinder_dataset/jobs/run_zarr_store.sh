#!/bin/bash
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  Parameterized SLURM script for building Plinder time-split Zarr stores
#
#  Usage:
#    sbatch --export=STORE_TYPE=no_links,SPLIT=train,CUTOFF=8.0,OUTDIR=plinder_time run_zarr_store.sh
#    sbatch --export=STORE_TYPE=apo,SPLIT=val,CUTOFF=12.0,OUTDIR=plinder_time_wide run_zarr_store.sh
#    sbatch --export=STORE_TYPE=pred,SPLIT=test,CUTOFF=8.0,OUTDIR=plinder_time run_zarr_store.sh
#
#  Environment variables (pass via --export):
#    STORE_TYPE  : no_links | apo | pred
#    SPLIT       : train | val | test
#    CUTOFF      : pocket cutoff in Angstroms (8.0 or 12.0)
#    OUTDIR      : output subdirectory name (plinder_time, plinder_time_wide)
#    NUM_CPUS    : number of CPUs (default: 16)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=90G
#SBATCH --time=28-00:00:00
#SBATCH --output=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_dataset/jobs/logs/rnp_%j_%x.out
#SBATCH --error=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_dataset/jobs/logs/rnp_%j_%x.err

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────
NUM_CPUS="${NUM_CPUS:-16}"
STORE_TYPE="${STORE_TYPE:?ERROR: STORE_TYPE not set (no_links|apo|pred)}"
SPLIT="${SPLIT:?ERROR: SPLIT not set (train|val|test)}"
CUTOFF="${CUTOFF:?ERROR: CUTOFF not set (e.g. 8.0 or 12.0)}"
OUTDIR="${OUTDIR:?ERROR: OUTDIR not set (e.g. runs_n_poses or runs_n_poses_wide)}"

# ── Paths ─────────────────────────────────────────────────────────
OMTRA_DIR="/net/galaxy/home/koes/nduaka1/OMTRA"
PARQUET="${PARQUET:-${OMTRA_DIR}/omtra_pipelines/plinder_dataset/plinder_filtered_time.parquet}"
OUTPUT_BASE="${OMTRA_DIR}/data/${OUTDIR}"
LOG_DIR="${OMTRA_DIR}/omtra_pipelines/plinder_dataset/logs"

mkdir -p "${OUTPUT_BASE}/${STORE_TYPE}"
mkdir -p "${LOG_DIR}"

OUTPUT_ZARR="${OUTPUT_BASE}/${STORE_TYPE}/${SPLIT}.zarr"

export LOG_FILE_PATH="${LOG_DIR}/rnp_${STORE_TYPE}_${SPLIT}_${CUTOFF}.log"

# ── Fix multiprocessing semaphore issue on SLURM nodes ────────────
# Python multiprocessing needs /dev/shm for POSIX semaphores.
# On nodes where it's missing/broken, create a user-owned fallback
# and bind-mount or symlink isn't possible, so we use the 'fork'
# start method instead of 'spawn' to avoid the SemLock pickling issue.
export OMTRA_MP_START_METHOD="fork"

# ── Prevent PyTorch thread oversubscription (1 thread per worker) ──
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# ── Use cached HuggingFace models (avoid workers hitting HF simultaneously) ──
export HF_HUB_OFFLINE=1

# ── Use Plinder 2024-04/v1 to match plinder-time.parquet ──
export PLINDER_RELEASE=2024-04
export PLINDER_ITERATION=v1

# ── Activate environment ──────────────────────────────────────────
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate omtra

cd "${OMTRA_DIR}"

# ── Log job info ──────────────────────────────────────────────────
echo "============================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Store type:   ${STORE_TYPE}"
echo "Split:        ${SPLIT}"
echo "Cutoff:       ${CUTOFF} Å"
echo "Output:       ${OUTPUT_ZARR}"
echo "Parquet:      ${PARQUET}"
echo "CPUs:         ${NUM_CPUS}"
echo "Started:      $(date)"
echo "============================================"

# ── Run ───────────────────────────────────────────────────────────
# store_unlinked_structures.py expects --pocket_cutoff as int;
# store_linked_structures.py expects it as float.
CUTOFF_INT="${CUTOFF%.*}"

if [ "${STORE_TYPE}" = "no_links" ]; then
    python omtra_pipelines/plinder_dataset/store_unlinked_structures.py \
        --data "${PARQUET}" \
        --split "${SPLIT}" \
        --output "${OUTPUT_ZARR}" \
        --pocket_cutoff "${CUTOFF_INT}" \
        --num_cpus "${NUM_CPUS}"
else
    # apo or pred
    python omtra_pipelines/plinder_dataset/store_linked_structures.py \
        --data "${PARQUET}" \
        --split "${SPLIT}" \
        --type "${STORE_TYPE}" \
        --output "${OUTPUT_ZARR}" \
        --pocket_cutoff "${CUTOFF}" \
        --num_cpus "${NUM_CPUS}"
fi

echo "Finished: $(date)"
exit 0

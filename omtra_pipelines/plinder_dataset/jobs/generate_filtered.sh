#!/bin/bash
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  Generate plinder_filtered.parquet for plinder-time.parquet systems
#
#  Usage:
#    sbatch generate_filtered.sh
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#SBATCH --job-name=gen_filtered
#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --output=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_dataset/jobs/logs/gen_filtered_%j.out
#SBATCH --error=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_dataset/jobs/logs/gen_filtered_%j.err

set -euo pipefail

OMTRA_DIR="/net/galaxy/home/koes/nduaka1/OMTRA"

# Use Plinder 2024-04/v1
export PLINDER_RELEASE=2024-04
export PLINDER_ITERATION=v1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export HF_HUB_OFFLINE=1
export PLINDER_OFFLINE=1

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate omtra

cd "${OMTRA_DIR}"

echo "============================================"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "============================================"

python -u omtra_pipelines/plinder_dataset/generate_filtered_parquet.py \
    --input omtra_pipelines/plinder_dataset/plinder-time.parquet \
    --output omtra_pipelines/plinder_dataset/plinder_filtered_time.parquet \
    --num_cpus 8

echo "Finished: $(date)"
exit 0

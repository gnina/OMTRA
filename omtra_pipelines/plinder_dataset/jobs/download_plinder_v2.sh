#!/bin/bash
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  SLURM script to pre-download all Plinder 2024-06/v2 system zips
#
#  Usage:
#    sbatch download_plinder_v2.sh
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#SBATCH --job-name=plinder_v2_download
#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_dataset/jobs/logs/plinder_v2_download_%j.out
#SBATCH --error=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_dataset/jobs/logs/plinder_v2_download_%j.err

set -euo pipefail

OMTRA_DIR="/net/galaxy/home/koes/nduaka1/OMTRA"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate omtra

cd "${OMTRA_DIR}"

echo "============================================"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "============================================"

python omtra_pipelines/plinder_dataset/jobs/predownload_plinder.py --workers 8

echo "Finished: $(date)"
exit 0

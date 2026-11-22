#!/bin/bash
#SBATCH -J fp_block
#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -o /net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_clustering/logs/fp_%A_%a.out
#SBATCH -e /net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_clustering/logs/fp_%A_%a.err

set -euo pipefail

: "${CMD_FILE:?ERROR: CMD_FILE not set}"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate omtra

cd /net/galaxy/home/koes/nduaka1/OMTRA

cmd=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CMD_FILE}")
echo "Running: ${cmd}"
eval "${cmd}"

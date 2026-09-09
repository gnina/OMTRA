#!/bin/bash
#SBATCH -J dynamic_crop_models_train
#SBATCH --partition=koes_gpu
#SBATCH -o slurm_output/single_runs/%A_%a.out
#SBATCH -e slurm_output/single_runs/%A_%a.err
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH -C M46

# Model 4 (task 1) and Model 5 (task 2): leakage-filtered multitask + dynamic crop.
# Submit with: sbatch --array=1-2 scripts/train_model45.slurm
# NOTE: edges_per_batch=150000 is conservative for 12A pockets (CrossDocked is static 12A,
#       so batches are heavier than the 8A Model 2/3 runs). Watch the first run for OOM and
#       adjust down if needed.

hostname
nvidia-smi

source /net/galaxy/home/koes/mag1037/miniforge3/etc/profile.d/conda.sh
conda activate /net/galaxy/home/koes/mag1037/miniforge3/envs/omtra

cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" train_omtra_gnn_model45_cmds.txt`

eval $cmd
exit

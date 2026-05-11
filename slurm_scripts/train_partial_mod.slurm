#!/bin/bash
#SBATCH --job-name=omtra_partial_mod
#SBATCH --partition=koes_gpu
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --gres=gpu:2
#SBATCH --constraint=M46
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mag1037@pitt.edu
#SBATCH --chdir=/net/galaxy/home/koes/mag1037/work/thesis/OMTRA

hostname
nvidia-smi
source /net/galaxy/home/koes/mag1037/miniforge3/etc/profile.d/conda.sh
conda activate /net/galaxy/home/koes/mag1037/miniforge3/envs/omtra
cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" partial_mod_resume_commands.txt`
eval $cmd
exit

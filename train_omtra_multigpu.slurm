#!/bin/bash
#SBATCH -J sr
#SBATCH --partition=koes_gpu
#SBATCH -o slurm_output/single_runs/%A_%a.out
#SBATCH -e slurm_output/single_runs/%A_%a.out
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH -C "M46|M40"
#  |M24|M20

hostname

nvidia-smi

source ~/.bashrc
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate omtra

cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" train_omtra_gnn_cmds.txt`


eval $cmd

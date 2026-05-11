#!/bin/bash
#SBATCH --job-name=omtra_train
#SBATCH --partition=koes_gpu
#SBATCH --output=/net/galaxy/home/mscbio/mag1037/work/thesis/OMTRA/logs/%A_%a.out
#SBATCH --error=/net/galaxy/home/mscbio/mag1037/work/thesis/OMTRA/logs/%A_%a.err
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:2
#SBATCH --constraint=M46
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mag1037@pitt.edu
#SBATCH --chdir=/net/galaxy/home/mscbio/mag1037/work/thesis/OMTRA

hostname
nvidia-smi
source /net/galaxy/home/mscbio/mag1037/miniforge3/etc/profile.d/conda.sh
conda activate omtra

# Setup scratch
mkdir -p /scr/${SLURM_JOB_ID}/torchinductor
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/scr/${SLURM_JOB_ID}/torchinductor

cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" train_omtra_gnn_cmds.txt`
eval $cmd
exit

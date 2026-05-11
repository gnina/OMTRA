#!/bin/bash
#SBATCH --job-name=omtra_sample
#SBATCH --partition=koes_gpu
#SBATCH --output=/net/galaxy/home/mscbio/mag1037/work/thesis/OMTRA/logs/%A_%a.out
#SBATCH --error=/net/galaxy/home/mscbio/mag1037/work/thesis/OMTRA/logs/%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=M24|M40|M46
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mag1037@pitt.edu
#SBATCH --chdir=/net/galaxy/home/mscbio/mag1037/work/thesis/OMTRA

hostname
nvidia-smi
source /net/galaxy/home/mscbio/mag1037/miniforge3/etc/profile.d/conda.sh
conda activate omtra
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" sample_cmds.txt`
eval $cmd
exit

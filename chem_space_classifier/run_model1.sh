#!/bin/bash
#SBATCH --job-name=model1_best
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g005,g006,g007,g009
#SBATCH --time=27-23:00:00
#SBATCH --output=logs/model1.out
#SBATCH --error=logs/model1.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mag1037@pitt.edu

# Model 1: Best performance - lr=1e-4, inverse_freq, 3 layers, 128/64
source /net/dali/home/mscbio/mag1037/miniforge3/etc/profile.d/conda.sh
conda activate /net/dali/home/mscbio/mag1037/miniforge3/envs/omtra_env

cd /net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/chem_space_classifier

python train_multitask.py \
  --exp_name model1_best \
  --exp_id 1 \
  --hidden_dim 128 \
  --edge_dim 64 \
  --n_vec_channels 8 \
  --num_layers 3 \
  --shared_repr_dim 128 \
  --task_hidden_dim 64 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --lr 1e-4 \
  --pos_weight_strategy inverse_freq \
  --molport_task_weight 1.0 \
  --num_workers 2 \
  --max_epochs 740 \
  --output_dir final_models/model1_best \
  --wandb_project omtra-final-models \
  --resume_from_checkpoint last

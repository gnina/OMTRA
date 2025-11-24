#!/bin/bash
#SBATCH --job-name=model4_small_fast_rerun
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g006
#SBATCH --time=27-23:00:00
#SBATCH --output=logs/model4_rerun.out
#SBATCH --error=logs/model4_rerun.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mag1037@pitt.edu

# Model 4: Small+fast - lr=2e-4, inverse_freq, 3 layers, 96/48
source /net/dali/home/mscbio/mag1037/miniforge3/etc/profile.d/conda.sh
conda activate /net/dali/home/mscbio/mag1037/miniforge3/envs/omtra_env

cd /net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/chem_space_classifier

python train_multitask.py \
  --exp_name model4_small_fast \
  --exp_id 4 \
  --hidden_dim 96 \
  --edge_dim 48 \
  --n_vec_channels 8 \
  --num_layers 3 \
  --shared_repr_dim 96 \
  --task_hidden_dim 48 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --lr 2e-4 \
  --pos_weight_strategy inverse_freq \
  --molport_task_weight 1.0 \
  --num_workers 2 \
  --max_epochs 740 \
  --output_dir final_models/model4_small_fast \
  --wandb_project omtra-final-models \
  --resume_from_checkpoint last

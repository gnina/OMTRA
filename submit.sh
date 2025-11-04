#!/bin/bash
#SBATCH -J sample
#SBATCH -t 01:00:00
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=1
#SBATCH --output=sample_%j.out
#SBATCH --error=sample_%j.err

export OMTRA_DEBUG=1
python omtra/cli.py sample \
  /net/galaxy/home/koes/icd3/moldiff/OMTRA/local/mlsb_runs_/mt_plinder/prot_cond_2025-09-11_18-31-088649/checkpoints/last.ckpt \
  --task fixed_protein_ligand_denovo_condensed \
  --protein_file /net/galaxy/home/koes/pengq/OMTRA1/omtra/tools/tools/pocket2.pdb \
  --n_samples 1 \
  --n_timesteps 250 \
  --output_dir out/staple_try4 \
  --anchor1 "20:CB" \
  --anchor2 "27:CB" 

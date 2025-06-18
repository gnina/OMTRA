#!/bin/bash

#job name
#SBATCH --job struct_rep
#SBATCH --partition koes_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --constraint L40
#SBATCH --mail-user=jmc530@pitt.edu
#SBATCH --mail-type=ALL



source activate omtra

#python ./train_pharmnn.py --train_data data/chemsplit_train0.pkl --test_data data/chemsplit_test0.pkl  --wandb_name default_chemsplit0_large_256 --grid_dimension 15.5  --expand_width 0 --model models/default_chemsplit0_large_256_last_model.pkl --lr 0.00001
#python ./train_pharmnn.py --train_data data/chemsplit_train2_with_ligand.pkl --test_data data/chemsplit_test2_with_ligand.pkl  --wandb_name obabel_chemsplit2_2 --negative_data data/obabel_chemsplit_2_negatives_train.txt --batch_size 256 --model models/obabel_chemsplit2_last_model.pkl --lr 0.00001
python routines/train.py num_workers=16 name=ph5050_fullpharm_no_cond task_group=pharmit5050 edges_per_batch=400000 trainer.val_check_interval=600 max_steps=300000 trainer.limit_val_batches=2 model.vector_field.convs_per_update=1 model.vector_field.n_molecule_updates=4 pharmit_path=/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/pharmit pharmit_library_conditioning=False plinder_path=/net/galaxy/home/koes/tjkatz/OMTRA/data/plinder 


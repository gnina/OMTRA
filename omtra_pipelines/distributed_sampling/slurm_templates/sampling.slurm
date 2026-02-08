#!/bin/bash
#SBATCH -J omtra_sampling
#SBATCH --partition=$partition_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --mem=$mem
#SBATCH --time=$time_sampling
#SBATCH -o $log_dir/sampling_%A_%a.out
#SBATCH -e $log_dir/sampling_%A_%a.err
$extra_sbatch_args

source ~/.bashrc
conda activate $conda_env

TASK_ID=$$SLURM_ARRAY_TASK_ID
LINE_NUM=$$((TASK_ID + 1))
CMD=$$(sed -n "$${LINE_NUM}p" $sampling_commands_file)

echo "=== OMTRA Sampling Pipeline ==="
echo "Job ID: $$SLURM_JOB_ID | Array Task: $$SLURM_ARRAY_TASK_ID"
echo "Node: $$(hostname) | GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Command: $$CMD"
echo "==============================="

eval $$CMD
EXIT_CODE=$$?

if [ $$EXIT_CODE -eq 0 ]; then
    mkdir -p $status_dir
    touch $status_dir/sampling_$${TASK_ID}.done
    echo "Task $${TASK_ID} completed successfully."
else
    echo "Task $${TASK_ID} FAILED with exit code $$EXIT_CODE."
    exit $$EXIT_CODE
fi

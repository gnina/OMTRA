#!/bin/bash
#SBATCH -J omtra_metrics
#SBATCH --partition=$partition_cpu
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --mem=$mem
#SBATCH --time=$time_metrics
#SBATCH -o $log_dir/metrics_%A_%a.out
#SBATCH -e $log_dir/metrics_%A_%a.err
#SBATCH --exclude=g023
$extra_sbatch_args

source ~/.bashrc
conda activate $conda_env

TASK_ID=$$SLURM_ARRAY_TASK_ID
LINE_NUM=$$((TASK_ID + 1))
CMD=$$(sed -n "$${LINE_NUM}p" $metrics_commands_file)

echo "=== OMTRA Metrics Pipeline ==="
echo "Job ID: $$SLURM_JOB_ID | Array Task: $$SLURM_ARRAY_TASK_ID"
echo "Node: $$(hostname)"
echo "Command: $$CMD"
echo "==============================="

eval $$CMD
EXIT_CODE=$$?

if [ $$EXIT_CODE -eq 0 ]; then
    mkdir -p $status_dir
    touch $status_dir/metrics_$${TASK_ID}.done
    echo "Task $${TASK_ID} completed successfully."
else
    echo "Task $${TASK_ID} FAILED with exit code $$EXIT_CODE."
    exit $$EXIT_CODE
fi

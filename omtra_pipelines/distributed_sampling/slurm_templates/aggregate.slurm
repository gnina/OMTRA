#!/bin/bash
#SBATCH -J omtra_aggregate
#SBATCH --partition=$partition_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=$time_aggregate
#SBATCH -o $log_dir/aggregate_%j.out
#SBATCH -e $log_dir/aggregate_%j.err
$extra_sbatch_args

source ~/.bashrc
conda activate $conda_env

echo "=== OMTRA Aggregate Pipeline ==="
echo "Job ID: $$SLURM_JOB_ID"
echo "Node: $$(hostname)"
echo "================================"

python -m omtra_pipelines.distributed_sampling.aggregate_results --manifest $manifest_path

EXIT_CODE=$$?
if [ $$EXIT_CODE -eq 0 ]; then
    mkdir -p $status_dir
    touch $status_dir/aggregate.done
    echo "Aggregation completed successfully."
else
    echo "Aggregation FAILED with exit code $$EXIT_CODE."
    exit $$EXIT_CODE
fi

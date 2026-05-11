#!/bin/bash
#SBATCH -J omtra_eval
#SBATCH --partition=koes_gpu,dept_gpu,bahar_gpu
#SBATCH -o slurm_output/omtra_eval/%A_%a.out
#SBATCH -e slurm_output/omtra_eval/%A_%a.out
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH -C M12|M16|M20|M24
#SBATCH --exclude=g011,g013
#SBATCH --chdir=/net/galaxy/home/koes/mag1037/work/thesis/OMTRA

# CMD_FILE=outputs/eval/mixed_whole_frag_plus_atoms/partial_rigid_docking_condensed_whole_fragments/eval_commands.txt \
#   sbatch --array=1-100 slurm_scripts/omtra_eval.slurm


hostname

source /net/galaxy/home/koes/mag1037/miniforge3/etc/profile.d/conda.sh
conda activate /net/galaxy/home/koes/mag1037/miniforge3/envs/omtra

NVIDIA_LIBS=$(find $CONDA_PREFIX/lib/python3.11/site-packages/nvidia -maxdepth 2 -name "lib" -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=${NVIDIA_LIBS}${LD_LIBRARY_PATH}

nvidia-smi
echo "Job started at: $(date)"

# Pre-check: allocate 1 GB on GPU to detect zombie CUDA contexts.
# Zombie contexts fragment virtual address space, causing large allocations to fail
# even with plenty of free memory. Requeue to find a clean node.
python -c "import torch; torch.zeros(256, 1024, 1024, device='cuda'); torch.cuda.empty_cache()" 2>/dev/null || {
    echo "CUDA pre-check failed on $(hostname), requeuing ${SLURM_JOB_ID}"
    scontrol requeue ${SLURM_JOB_ID}
    exit 1
}

cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" $CMD_FILE`
eval $cmd 2> >(grep -v "lazyInitCUDA" >&2)
exit_code=$?

echo "Job ended at: $(date) (exit $exit_code)"
exit $exit_code

#!/bin/bash
#SBATCH --job-name=plinder_lig_props
#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_ligand_properties/outputs/run_pipeline_%j.out
#SBATCH --error=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_ligand_properties/outputs/run_pipeline_%j.err

set -euo pipefail

OMTRA_DIR="/net/galaxy/home/koes/nduaka1/OMTRA"
N_CPUS=16

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate omtra
cd "${OMTRA_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT_BASE="${OMTRA_DIR}/omtra_pipelines/plinder_ligand_properties/outputs"
mkdir -p "${OUT_BASE}"

run_one() {
  local PLINDER_PATH=$1
  local STORE=$2          # train | val | test
  local TAG=$3            # short id used for output dir

  echo "=========================================================="
  echo "[$(date)] $TAG :: ${PLINDER_PATH} / no_links / ${STORE}.zarr"
  echo "=========================================================="

  python omtra_pipelines/plinder_ligand_properties/phase1.py \
    --plinder_path "${PLINDER_PATH}" \
    --store_name "${STORE}" \
    --versions no_links

  python omtra_pipelines/plinder_ligand_properties/run_phase2.py \
    --plinder_path "${PLINDER_PATH}" \
    --store_name "${STORE}" \
    --versions no_links \
    --task_group st_prot_denovo \
    --n_cpus "${N_CPUS}" \
    --output_dir "${OUT_BASE}/${TAG}_${STORE}"
}

# Plinder time-split (8 A graphs)
for SPLIT in train val test; do
  run_one "${OMTRA_DIR}/data/plinder_time"      "${SPLIT}" "plinder_time"
done

# Plinder time-split (12 A graphs)
for SPLIT in train val test; do
  run_one "${OMTRA_DIR}/data/plinder_time_wide" "${SPLIT}" "plinder_time_wide"
done

# Runs N' Poses (test only)
run_one "${OMTRA_DIR}/data/runs_n_poses"      test "runs_n_poses"
run_one "${OMTRA_DIR}/data/runs_n_poses_wide" test "runs_n_poses_wide"

echo "[$(date)] All stores done."

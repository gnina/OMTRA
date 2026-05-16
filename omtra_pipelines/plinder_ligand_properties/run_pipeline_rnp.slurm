#!/bin/bash
#SBATCH --job-name=plinder_lig_props_rnp
#SBATCH --partition=dept_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_ligand_properties/outputs/run_pipeline_rnp_%j.out
#SBATCH --error=/net/galaxy/home/koes/nduaka1/OMTRA/omtra_pipelines/plinder_ligand_properties/outputs/run_pipeline_rnp_%j.err

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

# Wipe stale failed_ligands.txt from prior failed run
rm -f "${OUT_BASE}/runs_n_poses_test/failed_ligands.txt" \
      "${OUT_BASE}/runs_n_poses_wide_test/failed_ligands.txt"

run_one() {
  local PLINDER_PATH=$1
  local TAG=$2

  echo "=========================================================="
  echo "[$(date)] $TAG :: ${PLINDER_PATH} / no_links / test.zarr"
  echo "=========================================================="

  python omtra_pipelines/plinder_ligand_properties/phase1.py \
    --plinder_path "${PLINDER_PATH}" \
    --store_name test \
    --versions no_links

  python omtra_pipelines/plinder_ligand_properties/run_phase2.py \
    --plinder_path "${PLINDER_PATH}" \
    --store_name test \
    --versions no_links \
    --task_group st_prot_denovo \
    --n_cpus "${N_CPUS}" \
    --output_dir "${OUT_BASE}/${TAG}_test"
}

run_one "${OMTRA_DIR}/data/runs_n_poses"      "runs_n_poses"
run_one "${OMTRA_DIR}/data/runs_n_poses_wide" "runs_n_poses_wide"

echo "[$(date)] Done."

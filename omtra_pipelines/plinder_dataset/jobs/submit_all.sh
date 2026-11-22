#!/bin/bash
#
# Submit Zarr store jobs for Plinder time-split data.
#
# Output structure:
#   data/plinder_time/           (8.0 Å pocket crop)
#     ├── no_links/{train,val,test}.zarr
#     ├── apo/{train,val,test}.zarr
#     └── pred/{train,val,test}.zarr
#   data/plinder_time_wide/      (12.0 Å pocket crop)
#     ├── no_links/{train,val,test}.zarr
#     ├── apo/{train,val,test}.zarr
#     └── pred/{train,val,test}.zarr
#
# Usage:
#   bash submit_all.sh          # submit all 18 jobs
#   bash submit_all.sh --dry    # dry run (print commands without submitting)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMTRA_DIR="/net/galaxy/home/koes/nduaka1/OMTRA"
JOB_SCRIPT="${SCRIPT_DIR}/run_zarr_store.sh"
PARQUET="${OMTRA_DIR}/omtra_pipelines/plinder_dataset/plinder-time.parquet"
NUM_CPUS="${NUM_CPUS:-16}"
DRY_RUN=false

if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN (not submitting) ==="
    echo ""
fi

SPLITS=("train" "val" "test")
STORE_TYPES=("no_links" "apo" "pred")

# 8.0 Å crop → plinder_time
# 12.0 Å crop → plinder_time_wide
declare -A CUTOFF_MAP
CUTOFF_MAP[plinder_time]=8.0
CUTOFF_MAP[plinder_time_wide]=12.0

count=0
for OUTDIR in plinder_time plinder_time_wide; do
    CUTOFF="${CUTOFF_MAP[$OUTDIR]}"
    for STORE_TYPE in "${STORE_TYPES[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            JOB_NAME="plt_${STORE_TYPE}_${SPLIT}_${CUTOFF}"
            CMD="sbatch \
                --job-name=${JOB_NAME} \
                --export=STORE_TYPE=${STORE_TYPE},SPLIT=${SPLIT},CUTOFF=${CUTOFF},OUTDIR=${OUTDIR},PARQUET=${PARQUET},NUM_CPUS=${NUM_CPUS} \
                --cpus-per-task=${NUM_CPUS} \
                ${JOB_SCRIPT}"

            if [ "$DRY_RUN" = true ]; then
                echo "${CMD}"
            else
                echo "Submitting: ${JOB_NAME}"
                eval "${CMD}"
            fi
            count=$((count + 1))
        done
    done
done

echo ""
echo "Total jobs: ${count}"

#!/bin/bash
# Setup script to enable running 'omtra' CLI commands via Docker

# Default image name
OMTRA_CLI_IMAGE="${OMTRA_CLI_IMAGE:-omtra/cli:latest}"


omtra() {
    local IMAGE_NAME="${OMTRA_CLI_IMAGE}"
    local GPU_FLAG=""
    
    # Check if image exists locally, if not try to build it
    if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
        echo "Image ${IMAGE_NAME} not found locally. Attempting to pull from registry..."
        if ! docker pull "${IMAGE_NAME}" 2>/dev/null; then
            echo "Image not found in registry. Building locally..."
            local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE_NAME}" "${SCRIPT_DIR}"
        fi
    fi
    
    # GPU support (set OMTRA_NO_GPU=1 to disable)
    if [ -z "${OMTRA_NO_GPU}" ]; then
        GPU_FLAG="--gpus all"
    fi
    
    TTY_FLAG=""
    if [ -t 0 ]; then
        TTY_FLAG="-it"
    else
        TTY_FLAG="-i"
    fi
    
    USER_ID=$(id -u)
    GROUP_ID=$(id -g)
    
    local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local OMTRA_DIR="$SCRIPT_DIR"
    
    local FILE_ARGS=("--protein_file" "--ligand_file" "--pharmacophore_file" "--output_dir" "--checkpoint")
    declare -A MOUNT_DIRS
    local PREV_ARG=""
    
    for arg in "$@"; do
        local is_file_arg=false
        for file_arg in "${FILE_ARGS[@]}"; do
            [[ "$PREV_ARG" == "$file_arg" ]] && is_file_arg=true && break
        done
        
        # If absolute path, mount its parent directory
        if [[ "$is_file_arg" == true ]] && [[ "$arg" == /* ]] && [[ "$arg" != "$OMTRA_DIR"/* ]]; then
            local parent_dir="$(dirname "$arg")"
            # Mark as writable if output_dir, else read-only
            if [[ "$PREV_ARG" == "--output_dir" ]]; then
                MOUNT_DIRS["$parent_dir"]="rw"
            else
                MOUNT_DIRS["$parent_dir"]="ro"
            fi
        fi
        PREV_ARG="$arg"
    done
    
    # Build volume mounts
    local VOLUME_ARGS=(-v "$OMTRA_DIR:/workspace")
    for dir in "${!MOUNT_DIRS[@]}"; do
        VOLUME_ARGS+=(-v "$dir:$dir:${MOUNT_DIRS[$dir]}")
    done
    
    # Run the command in Docker
    docker run ${GPU_FLAG} --rm ${TTY_FLAG} \
      --user ${USER_ID}:${GROUP_ID} \
      -e DGLBACKEND=pytorch \
      -e HOME=/tmp \
      "${VOLUME_ARGS[@]}" \
      -w /workspace \
      ${IMAGE_NAME} \
      "$@"
}

# Export the function so it's available in the shell
export -f omtra

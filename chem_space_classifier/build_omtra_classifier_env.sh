#!/bin/bash
# Build conda environment for OMTRA Multitask Classifier
# Environment name: omtra_classifier_env

set -e # Stop on first error

echo "========================================================================"
echo "Building OMTRA Classifier Environment"
echo "========================================================================"
echo ""

# Check if mamba is available
if ! command -v mamba &> /dev/null; then
    echo "ERROR: mamba not found. Please install miniforge/mambaforge first."
    exit 1
fi

ENV_NAME="omtra_classifier_env"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "WARNING: Environment '${ENV_NAME}' already exists."
    echo "Options:"
    echo "  1. Remove existing: conda env remove -n ${ENV_NAME}"
    echo "  2. Update existing: activate it and re-run this script"
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Aborting."
        exit 0
    fi
fi

echo "Creating environment: ${ENV_NAME}"
mamba create -n ${ENV_NAME} python=3.11 -y

echo ""
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

echo ""
echo "========================================================================"
echo "Installing PyTorch and CUDA packages"
echo "========================================================================"
mamba install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo ""
echo "========================================================================"
echo "Installing PyTorch Geometric packages"
echo "========================================================================"
mamba install -c pyg pytorch-scatter=2.1.2=py311_torch_2.4.0_cu121 pytorch-cluster -y

echo ""
echo "========================================================================"
echo "Installing DGL (Deep Graph Library)"
echo "========================================================================"
mamba install -c dglteam/label/th24_cu121 dgl -y

echo ""
echo "========================================================================"
echo "Installing PyTorch Lightning and Hydra"
echo "========================================================================"
mamba install -c conda-forge hydra-core pytorch-lightning -y

echo ""
echo "========================================================================"
echo "Installing scientific packages"
echo "========================================================================"
mamba install -c conda-forge \
    rdkit=2023.09.4 \
    pystow \
    einops \
    zarr=3 \
    jupyterlab \
    rich \
    matplotlib \
    biotite \
    pyarrow \
    -y

echo ""
echo "========================================================================"
echo "Installing pip packages"
echo "========================================================================"
pip install wandb useful_rdkit_utils py3Dmol tqdm peppr --no-input

echo ""
echo "========================================================================"
echo "Installing OMTRA package (development mode)"
echo "========================================================================"
OMTRA_DIR="/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA"
if [ -d "${OMTRA_DIR}" ]; then
    cd "${OMTRA_DIR}"
    pip install -e ./
    echo "✓ OMTRA installed from ${OMTRA_DIR}"
else
    echo "WARNING: OMTRA directory not found at ${OMTRA_DIR}"
    echo "You may need to install it manually with: pip install -e /path/to/OMTRA"
fi

echo ""
echo "========================================================================"
echo "Environment build complete!"
echo "========================================================================"
echo ""
echo "Environment name: ${ENV_NAME}"
echo ""
echo "To activate:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
echo "  python -c 'import pytorch_lightning; print(f\"Lightning: {pytorch_lightning.__version__}\")'"
echo "  python -c 'import omtra; print(\"OMTRA imported successfully\")'"
echo ""
echo "========================================================================"
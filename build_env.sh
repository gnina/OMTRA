#!/bin/bash
set -e # this will stop the script on first error

# Check if we are in a virtual environment
if [[ "$CONDA_DEFAULT_ENV" == "base" ]] || [[ -z "$VIRTUAL_ENV" && -z "$CONDA_PREFIX" ]]; then
    echo "WARNING: It looks like you are not in a virtual environment or you are in the base Conda environment."
    echo "It is recommended to create a new environment before installing."
    echo "Press Ctrl+C to abort, or wait 5 seconds to continue..."
    sleep 5
fi

echo "Installing uv package manager..."
pip install uv

echo "Installing CUDA-enabled dependencies..."
uv pip install -r requirements-cuda.txt

echo "Installing OMTRA and remaining dependencies..."
uv pip install -e .

echo "✔ Installation complete!"
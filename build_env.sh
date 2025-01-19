#!/bin/bash
set -e # this will stop the script on first error

# get the name of the current conda environment
ENV_NAME=$(basename "$CONDA_PREFIX")

# print the name of the current conda environment to the terminal
echo "Building omtra into the environment '$ENV_NAME'"

mamba install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
mamba install -c pyg pytorch-scatter=2.1.2=py311_torch_2.4.0_cu121 pytorch-cluster -y
mamba install -c dglteam/label/th24_cu121 dgl -y
mamba install -c conda-forge pytorch-lightning -y
mamba install -c conda-forge rdkit=2023.09.4 pystow einops zarr jupyterlab rich matplotlib -y

pip install wandb useful_rdkit_utils py3Dmol --no-input
pip install -e ./
# Report 3: Installation and Usage Guide

## Overview

This document provides complete instructions for setting up the environment, installing dependencies, and running the OMTRA Multitask Classifier training pipeline.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Training the Models](#training-the-models)
5. [Monitoring Training](#monitoring-training)
6. [Checkpoint Management](#checkpoint-management)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## System Requirements

### Hardware Requirements

| Resource | Minimum | Recommended | Notes |
|----------|---------|-------------|-------|
| **GPU** | 1x NVIDIA GPU | 1x A100/V100/A6000 | CUDA 12.1 support required |
| **VRAM** | 16 GB | 24+ GB | Model 1 needs more than Model 4 |
| **RAM** | 32 GB | 64 GB | For data loading and caching |
| **CPUs** | 4 cores | 8+ cores | For parallel data loading |
| **Storage** | 100 GB | 500 GB | For datasets, cache, checkpoints |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| **Linux** | Any modern distro | Ubuntu 20.04+ tested |
| **CUDA** | 12.1 | GPU acceleration |
| **Python** | 3.11 | Runtime environment |
| **Conda/Mamba** | Latest | Package management |
| **SLURM** | Any version | Job scheduling (optional) |

### Tested Configurations

**Model 1 (Best)**:
- GPU: 1x NVIDIA GPU
- RAM: 64 GB
- CPUs: 4 cores
- Time: Up to 28 days

**Model 4 (Small+Fast)**:
- GPU: 1x NVIDIA GPU
- RAM: 32 GB
- CPUs: 4 cores
- Time: Up to 28 days

---

## Environment Setup

### Method 1: Automated Installation (Recommended)

#### Step 1: Run the Environment Build Script

```bash
cd /net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/tuning_experiments

# Make script executable
chmod +x build_omtra_classifier_env.sh

# Run the installer
./build_omtra_classifier_env.sh
```

The script will:
1. Check if mamba is available
2. Create environment `omtra_classifier_env`
3. Install all dependencies
4. Install OMTRA package in development mode

**Expected output**:
```
========================================================================
Building OMTRA Classifier Environment
========================================================================

Creating environment: omtra_classifier_env
Installing PyTorch and CUDA packages
Installing PyTorch Geometric packages
Installing DGL (Deep Graph Library)
Installing PyTorch Lightning and Hydra
Installing scientific packages
Installing pip packages
Installing OMTRA package (development mode)

========================================================================
Environment build complete!
========================================================================

Environment name: omtra_classifier_env

To activate:
  conda activate omtra_classifier_env
```

#### Step 2: Activate the Environment

```bash
conda activate omtra_classifier_env
```

#### Step 3: Verify Installation

```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Expected: PyTorch: 2.4.0+cu121

# Test CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Expected: CUDA available: True

# Test PyTorch Lightning
python -c "import pytorch_lightning; print(f'Lightning: {pytorch_lightning.__version__}')"
# Expected: Lightning: 2.x.x

# Test OMTRA
python -c "import omtra; print('OMTRA imported successfully')"
# Expected: OMTRA imported successfully
```

---

### Method 2: Manual Installation

If the automated script fails, install manually:

#### Step 1: Create Conda Environment

```bash
mamba create -n omtra_classifier_env python=3.11 -y
conda activate omtra_classifier_env
```

#### Step 2: Install PyTorch with CUDA

```bash
mamba install pytorch=2.4.0 torchvision torchaudio pytorch-cuda=12.1 \
  -c pytorch -c nvidia -y
```

#### Step 3: Install PyTorch Geometric

```bash
mamba install -c pyg \
  pytorch-scatter=2.1.2=py311_torch_2.4.0_cu121 \
  pytorch-cluster -y
```

#### Step 4: Install DGL

```bash
mamba install -c dglteam/label/th24_cu121 dgl -y
```

#### Step 5: Install Core ML Libraries

```bash
mamba install -c conda-forge \
  hydra-core \
  pytorch-lightning \
  -y
```

#### Step 6: Install Scientific Libraries

```bash
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
```

#### Step 7: Install Pip Packages

```bash
pip install wandb useful_rdkit_utils py3Dmol tqdm peppr --no-input
```

#### Step 8: Install OMTRA

```bash
cd /net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA
pip install -e ./
```

---

## Data Preparation

### Dataset Location

The Pharmit dataset should be located at:
```
/net/dali/home/mscbio/mag1037/work/rotations/koes/datasets/pharmit_by_pattern
```

### Dataset Structure

```
pharmit_by_pattern/
├── pattern_A/
│   ├── train.zarr/
│   │   ├── lig/
│   │   │   ├── node/
│   │   │   │   ├── a_1_true/      # Atom types
│   │   │   │   ├── c_1_true/      # Charges
│   │   │   │   ├── x_1_true/      # Coordinates
│   │   │   │   └── graph_lookup/  # Node-to-graph mapping
│   │   │   └── edge/
│   │   │       └── e_1_true/      # Bond types
│   │   └── db/
│   │       └── db/                # Database labels [N, 5]
│   └── val.zarr/
│       └── ...
└── pattern_B/
    └── ...
```

### Cache Directory

Training uses cached molecular data for faster loading:

```bash
# Default cache location
/net/dali/home/mscbio/mag1037/work/rotations/koes/multitask_cache/

# Cache structure (auto-generated)
multitask_cache/
├── multitask_streaming_fixed_<hash>.npz  # Sample indices and edge counts
└── ...
```

**Cache benefits**:
- Faster startup (no dataset scanning)
- Persistent across runs
- Automatically invalidated if dataset changes

---

## Training the Models

### Directory Structure

```
chem_space_classifier/
├── train_multitask.py              # Main training script
├── run_model1.sh                   # SLURM script for Model 1
├── run_model4_rerun.sh             # SLURM script for Model 4
├── build_omtra_classifier_env.sh   # Environment setup
├── SELECTED_MODELS.md              # Model configurations
├── logs/                           # Training logs
│   ├── model1.out
│   ├── model1.err
│   ├── model4_rerun.out
│   └── model4_rerun.err
└── final_models/                   # Model outputs
    ├── model1_best/
    │   └── exp_1/
    │       ├── checkpoints/
    │       ├── hyperparams.json
    │       └── results.json
    └── model4_small_fast/
        └── exp_4/
            ├── checkpoints/
            ├── hyperparams.json
            └── results.json
```

---

### Training Model 1 (Best Performance)

#### Using SLURM (Recommended)

```bash
cd /net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/tuning_experiments

# Submit job
sbatch run_model1.sh

# Check job status
squeue -u $USER

# View job details
scontrol show job <job_id>
```

**SLURM Configuration** (from run_model1.sh):
```bash
#SBATCH --job-name=model1_best
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=27-23:00:00         # ~28 days
#SBATCH --output=logs/model1.out
#SBATCH --error=logs/model1.err
```

#### Manual Training (Without SLURM)

```bash
cd /net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/tuning_experiments

conda activate omtra_classifier_env

python train_multitask.py \
  --exp_name model1_best \
  --exp_id 1 \
  --hidden_dim 128 \
  --edge_dim 64 \
  --n_vec_channels 8 \
  --num_layers 3 \
  --shared_repr_dim 128 \
  --task_hidden_dim 64 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --lr 1e-4 \
  --pos_weight_strategy inverse_freq \
  --molport_task_weight 1.0 \
  --num_workers 2 \
  --max_epochs 740 \
  --output_dir final_models/model1_best \
  --wandb_project omtra-final-models \
  --resume_from_checkpoint last
```

---

### Training Model 4 (Small and Fast)

#### Using SLURM

```bash
sbatch run_model4_rerun.sh
```

**SLURM Configuration**:
```bash
#SBATCH --job-name=model4_small_fast_rerun
#SBATCH --partition=dept_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  # Less RAM than Model 1
#SBATCH --gres=gpu:1
#SBATCH --time=27-23:00:00
#SBATCH --output=logs/model4_rerun.out
#SBATCH --error=logs/model4_rerun.err
```

#### Manual Training

```bash
python train_multitask.py \
  --exp_name model4_small_fast \
  --exp_id 4 \
  --hidden_dim 96 \
  --edge_dim 48 \
  --n_vec_channels 8 \
  --num_layers 3 \
  --shared_repr_dim 96 \
  --task_hidden_dim 48 \
  --dropout 0.2 \
  --weight_decay 1e-5 \
  --lr 2e-4 \
  --pos_weight_strategy inverse_freq \
  --molport_task_weight 1.0 \
  --num_workers 2 \
  --max_epochs 740 \
  --output_dir final_models/model4_small_fast \
  --wandb_project omtra-final-models \
  --resume_from_checkpoint last
```

---

### Training Parameters Explained

| Parameter | Model 1 | Model 4 | Description |
|-----------|---------|---------|-------------|
| `--exp_name` | model1_best | model4_small_fast | Experiment identifier |
| `--exp_id` | 1 | 4 | Numeric ID |
| `--hidden_dim` | 128 | 96 | Node feature dimension |
| `--edge_dim` | 64 | 48 | Edge feature dimension |
| `--n_vec_channels` | 8 | 8 | Vector channels (geometry) |
| `--num_layers` | 3 | 3 | GVP layers |
| `--shared_repr_dim` | 128 | 96 | Shared representation size |
| `--task_hidden_dim` | 64 | 48 | Task-specific hidden size |
| `--dropout` | 0.2 | 0.2 | Dropout probability |
| `--weight_decay` | 1e-5 | 1e-5 | L2 regularization |
| `--lr` | 1e-4 | 2e-4 | Learning rate |
| `--pos_weight_strategy` | inverse_freq | inverse_freq | Class balancing |
| `--molport_task_weight` | 1.0 | 1.0 | MolPort task weight |
| `--num_workers` | 2 | 2 | Data loading workers |
| `--max_epochs` | 740 | 740 | Maximum epochs |

**Fixed parameters** (same for both):
```bash
--edges_per_batch 200000        # Edges per batch
--accumulate_grad_batches 2     # Gradient accumulation
--max_batches_train 10000       # Training batches/epoch
--max_batches_val 1000          # Validation batches/epoch
--pharmit_path /net/dali/.../pharmit_by_pattern
--cache_dir /net/dali/.../multitask_cache
--wandb_api_key <your_key>
```

---

## Monitoring Training

### 1. Live Log Monitoring

#### Watch Training Output

```bash
# Model 1
tail -f logs/model1.out

# Model 4
tail -f logs/model4_rerun.out
```

**Expected output**:
```
================================================================================
EXPERIMENT: model1_best (ID: 1)
================================================================================

Class Balancing:
  Strategy: inverse_freq
  Pos weights: ['4.02', '2.34', '0.99', '3.79', '23.56']
  Task weights: [1.0, 1.0, 1.0, 1.0, 1.0]
  MolPort effective weight: 23.56x

================================================================================
LOADING DATASETS
================================================================================
Train dataset: 1,234,567 molecules
Val dataset: 123,456 molecules

================================================================================
CREATING DATALOADERS
================================================================================
Train batches: 10,000
Val batches: 1,000

================================================================================
CREATING MODEL
================================================================================
Total parameters: 463,251

================================================================================
SETUP TRAINER
================================================================================
Output directory: final_models/model1_best/exp_001_model1_best
Wandb run: https://wandb.ai/.../runs/...

================================================================================
STARTING TRAINING
================================================================================

Epoch 0: 100%|██████████| 10000/10000 [1:23:45<00:00, 2.01it/s, loss=5.234]

================================================================================
VALIDATION EPOCH SUMMARY
================================================================================
       CSC: AUROC=0.6234 | AUPRC=0.5123 | F1=0.6012 | MCC=0.4123
     MCULE: AUROC=0.6543 | AUPRC=0.5876 | F1=0.6234 | MCC=0.4567
   PubChem: AUROC=0.7123 | AUPRC=0.6987 | F1=0.6876 | MCC=0.5234
      ZINC: AUROC=0.6345 | AUPRC=0.5432 | F1=0.6123 | MCC=0.4321
   MolPort: AUROC=0.5987 | AUPRC=0.2134 | F1=0.3456 | MCC=0.1987
================================================================================

Epoch 1: 100%|██████████| 10000/10000 [1:22:15<00:00, 2.03it/s, loss=4.876]
...
```

#### Check for Errors

```bash
# Model 1 errors
tail -f logs/model1.err

# Model 4 errors
tail -f logs/model4_rerun.err
```

Common errors are logged here (OOM, CUDA errors, data issues)

---

### 2. SLURM Job Management

#### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# Output:
#  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
# 123456 dept_gpu  model1_b  mag1037  R   12:34:56      1 g001
```

**Job states**:
- `PD`: Pending (waiting for resources)
- `R`: Running
- `CG`: Completing
- `CD`: Completed
- `F`: Failed

#### View Job Details

```bash
scontrol show job <job_id>
```

#### Cancel Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel by name
scancel --name=model1_best
```

#### Check Resource Usage

SSH to the compute node and check GPU usage:

```bash
# Find node name from squeue
squeue -u $USER

# SSH to node (e.g., g001)
ssh g001

# Check GPU usage
nvidia-smi

# Check CPU/RAM usage
htop
```

---

### 3. Weights & Biases (Wandb) Dashboard

#### Access Wandb

```
https://wandb.ai/mag1037-university-of-pittsburgh/omtra-final-models
```

#### Key Metrics to Monitor

**Training Tab**:
- `train_loss`: Should decrease steadily
- `train_avg_auroc`: Should increase
- Per-task losses: All should decrease

**Validation Tab**:
- `val_avg_auroc`: **Primary metric** - should increase
- `val_loss`: Should decrease
- Per-task AUROCs: Monitor individually

**System Tab**:
- GPU utilization: Should be >90%
- GPU memory: Should be stable (not increasing)
- CPU utilization: 50-80% with num_workers=2

#### Creating Custom Plots

1. Click "Add panel"
2. Select metrics (e.g., all task AUROCs)
3. Choose plot type (line chart)
4. Save to workspace

**Recommended plots**:
- All 5 task AUROCs on one plot
- Train vs validation loss
- Learning rate schedule (if using scheduler)

---

## Checkpoint Management

### Checkpoint Structure

```
final_models/model1_best/exp_001_model1_best/checkpoints/
├── epoch_12_auroc_0.7234.ckpt
├── epoch_23_auroc_0.7456.ckpt      # 2nd best
├── epoch_47_auroc_0.7623.ckpt      # Best model
├── epoch_52_auroc_0.7598.ckpt      # 3rd best
└── last.ckpt                        # Most recent (for resumption)
```

### Resuming Training

Training automatically resumes from `last.ckpt` if found:

```bash
# Automatic resumption (default)
sbatch run_model1.sh
```

The script includes `--resume_from_checkpoint last`, which:
1. Checks for `checkpoints/last.ckpt`
2. If found: Resumes training from that epoch
3. If not found: Starts training from scratch

**Manual resumption from specific checkpoint**:

```bash
python train_multitask.py \
  ... \
  --resume_from_checkpoint final_models/model1_best/exp_001/checkpoints/epoch_47_auroc_0.7623.ckpt
```

### Loading Best Model

```python
from multitask_lightning_module import MultitaskLightningModule

# Load from checkpoint
model = MultitaskLightningModule.load_from_checkpoint(
    'final_models/model1_best/exp_001/checkpoints/epoch_47_auroc_0.7623.ckpt'
)

# Use for inference
model.eval()
with torch.no_grad():
    logits = model(graph)
    probs = torch.sigmoid(logits)
```

### Checkpoint File Size

| Model | Checkpoint Size |
|-------|----------------|
| Model 1 | ~5.5 MB |
| Model 4 | ~3.1 MB |

**Storage requirements** (740 epochs):
- Top 3 checkpoints: ~16-17 MB
- Last checkpoint: ~5-6 MB
- Total: ~22 MB per model

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Error

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:

a. **Reduce batch size** (edit train_multitask.py):
```python
--edges_per_batch 150000  # Reduce from 200000
```

b. **Reduce model size** (use Model 4 instead of Model 1)

c. **Enable gradient checkpointing** (edit model):
```python
# In train_multitask.py, add:
trainer = pl.Trainer(
    ...
    gradient_clip_val=1.0,
    precision='16-mixed',  # Already enabled
)
```

d. **Reduce num_workers**:
```bash
--num_workers 0  # Disable multiprocessing
```

#### 2. Slow Training

**Symptoms**:
- Training slower than expected
- GPU utilization < 90%

**Solutions**:

a. **Check GPU usage**:
```bash
nvidia-smi
```

b. **Increase num_workers** (if CPU bottleneck):
```bash
--num_workers 4  # Increase from 2
```

c. **Enable pin_memory** (faster GPU transfer):
```python
# Edit train_multitask.py line 209
'pin_memory': True,  # Change from False
```

d. **Verify batch size is adequate**:
```bash
# Check batch formation in logs
# Should see consistent batch sizes, not tiny batches
```

#### 3. NaN Loss

**Symptoms**:
```
train_loss: nan
```

**Solutions**:

a. **Reduce learning rate**:
```bash
--lr 5e-5  # Reduce from 1e-4 or 2e-4
```

b. **Check for invalid inputs**:
```python
# Add assertions in train_multitask.py
assert not torch.isnan(logits).any()
assert not torch.isinf(logits).any()
```

c. **Increase gradient clipping**:
```python
# Edit trainer config
gradient_clip_val=0.5  # Reduce from 1.0
```

#### 4. Data Loading Errors

**Symptoms**:
```
FileNotFoundError: pharmit_by_pattern not found
RuntimeError: Error loading zarr chunk
```

**Solutions**:

a. **Verify dataset path**:
```bash
ls -la /net/dali/home/mscbio/mag1037/work/rotations/koes/datasets/pharmit_by_pattern
```

b. **Check permissions**:
```bash
# Ensure read access to dataset
chmod -R +r pharmit_by_pattern/
```

c. **Clear cache and rebuild**:
```bash
rm -rf /net/dali/.../multitask_cache/*
# Restart training (will rebuild cache)
```

#### 5. Wandb Login Issues

**Symptoms**:
```
wandb: ERROR Unable to authenticate
```

**Solutions**:

a. **Login manually**:
```bash
wandb login b32b90b1572db9356cdfe74709ba49e3c21dfad7
```

b. **Set API key in script** (already done in train_multitask.py:98)

c. **Check internet connection** from compute node

#### 6. SLURM Job Stuck in Queue

**Symptoms**:
- Job state: `PD` (Pending) for long time

**Solutions**:

a. **Check queue status**:
```bash
sinfo  # View partition availability
squeue  # View all jobs
```

b. **Remove node exclusions** (if not critical):
```bash
# Edit run_model1.sh, remove or modify:
#SBATCH --exclude=g005,g006,g007,g009
```

c. **Reduce resource requirements**:
```bash
#SBATCH --mem=32G  # Reduce from 64G (for Model 1)
```

d. **Check SLURM limits**:
```bash
sacctmgr show qos
```

---

## Advanced Usage

### Custom Hyperparameter Sweeps

Create a new experiment configuration:

```python
# custom_experiment.py
python train_multitask.py \
  --exp_name custom_lr_3e4 \
  --exp_id 99 \
  --lr 3e-4 \
  --dropout 0.15 \
  --hidden_dim 112 \
  --output_dir experiments/custom_lr_3e4 \
  --wandb_project omtra-experiments
```

### Multi-GPU Training (Future)

To scale to multiple GPUs:

```python
# Edit trainer in train_multitask.py
trainer = pl.Trainer(
    accelerator='gpu',
    devices=4,  # Use 4 GPUs
    strategy='ddp',  # Distributed data parallel
    ...
)
```

**Note**: Requires larger batch sizes to benefit from multiple GPUs.

### Evaluating on Test Set

After training, evaluate on held-out test set:

```python
# Load best model
model = MultitaskLightningModule.load_from_checkpoint(
    'checkpoints/epoch_47_auroc_0.7623.ckpt'
)

# Load test dataset
test_dataset = datamodule.load_dataset("test")
test_loader = create_multitask_streaming_dataloader_fixed(...)

# Evaluate
trainer.test(model, test_loader)
```

### Inference on New Molecules

```python
import torch
from multitask_lightning_module import MultitaskLightningModule
from omtra.load.quick import datamodule_from_config

# Load model
model = MultitaskLightningModule.load_from_checkpoint('best_model.ckpt')
model.eval()

# Load your molecule as DGL graph
# ... (molecule loading code)

# Predict
with torch.no_grad():
    logits = model(molecule_graph)
    probs = torch.sigmoid(logits)

# Interpret
task_names = ['CSC', 'MCULE', 'PubChem', 'ZINC', 'MolPort']
for i, task in enumerate(task_names):
    print(f"{task}: {probs[0, i]:.3f}")
```

---

## Quick Reference

### Start Training

```bash
# Model 1
sbatch run_model1.sh

# Model 4
sbatch run_model4_rerun.sh
```

### Monitor Progress

```bash
# Logs
tail -f logs/model1.out

# SLURM status
squeue -u $USER

# Wandb
open https://wandb.ai/.../omtra-final-models
```

### Stop Training

```bash
scancel <job_id>
```

### Resume Training

```bash
# Automatic (via last.ckpt)
sbatch run_model1.sh
```

### Check Results

```bash
# View final results
cat final_models/model1_best/exp_001/results.json

# List checkpoints
ls -lh final_models/model1_best/exp_001/checkpoints/
```

---

## File Locations Reference

| File | Path | Purpose |
|------|------|---------|
| Training script | [train_multitask.py](train_multitask.py) | Main training code |
| Model 1 runner | [run_model1.sh](run_model1.sh) | SLURM script for Model 1 |
| Model 4 runner | [run_model4_rerun.sh](run_model4_rerun.sh) | SLURM script for Model 4 |
| Environment setup | [build_omtra_classifier_env.sh](build_omtra_classifier_env.sh) | Conda environment builder |
| Model configs | [SELECTED_MODELS.md](SELECTED_MODELS.md) | Hyperparameter documentation |
| Architecture | [multitask_gvp_classifier.py](models/multitask_gvp_classifier.py) | Model architecture |
| Lightning module | [multitask_lightning_module.py](models/multitask_lightning_module.py) | Training logic |
| Data sampler | [multitask_streaming_sampler_fixed.py](models/multitask_streaming_sampler_fixed.py) | Data loading |

---

## Summary Checklist

### Setup ✓
- [ ] Install mamba/conda
- [ ] Run `build_omtra_classifier_env.sh`
- [ ] Activate `omtra_classifier_env`
- [ ] Verify PyTorch CUDA availability

### Data ✓
- [ ] Confirm dataset at correct path
- [ ] Check cache directory exists
- [ ] Verify read permissions

### Training ✓
- [ ] Submit job: `sbatch run_model1.sh` or `run_model4_rerun.sh`
- [ ] Monitor logs: `tail -f logs/model*.out`
- [ ] Check Wandb dashboard
- [ ] Verify GPU utilization

### Results ✓
- [ ] Check best validation AUROC
- [ ] Locate best checkpoint
- [ ] Review per-task performance
- [ ] Save final results

---

## Getting Help

**Issues with**:
- SLURM: Contact HPC support
- CUDA/GPU: Check `nvidia-smi`, update drivers
- Data: Verify paths and permissions
- Model: Check logs, reduce batch size
- Wandb: Re-login, check API key

**Contact**: mag1037@pitt.edu

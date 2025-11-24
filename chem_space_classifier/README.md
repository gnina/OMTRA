# OMTRA Multitask Classifier - Documentation

## Overview

This directory contains the complete training pipeline for the OMTRA Multitask GVP Classifier, which predicts molecular membership across 5 chemical databases (CSC, MCULE, PubChem, ZINC, and MolPort).

---

## Quick Start

### 1. Setup Environment
```bash
./build_omtra_classifier_env.sh
conda activate omtra_classifier_env
```

### 2. Train Models
```bash
# Model 1 (Best Performance)
sbatch run_model1.sh

# Model 4 (Small and Fast)
sbatch run_model4_rerun.sh
```

### 3. Monitor Training
```bash
tail -f logs/model1.out
```

See [REPORT_3_INSTALLATION_AND_USAGE.md](REPORT_3_INSTALLATION_AND_USAGE.md) for complete instructions.

---

## Documentation

### Core Reports

| Report | Description | File |
|--------|-------------|------|
| **Report 1** | Model Architecture and Input Processing | [REPORT_1_MODEL_ARCHITECTURE.md](REPORT_1_MODEL_ARCHITECTURE.md) |
| **Report 2** | Loss Functions and Metrics | [REPORT_2_LOSS_AND_METRICS.md](REPORT_2_LOSS_AND_METRICS.md) |
| **Report 3** | Installation and Usage Guide | [REPORT_3_INSTALLATION_AND_USAGE.md](REPORT_3_INSTALLATION_AND_USAGE.md) |
| **Model Info** | Selected Model Configurations | [models_description.md](models_description.md) |

### Report 1: Model Architecture

**Topics covered**:
- Molecular graph representation (atoms, bonds, geometry)
- GVP (Geometric Vector Perceptron) architecture
- Layer-by-layer description
- Embedding layers, projections, and readout
- Multitask head architecture
- Parameter counts and model configurations

**Read this if you want to understand**:
- How molecules are converted to graphs
- What features are used (atom types, charges, coordinates, bonds)
- How GVP layers process geometric information
- Why the model is SE(3)-equivariant
- The difference between Model 1 (463k params) and Model 4 (260k params)

### Report 2: Loss Functions and Metrics

**Topics covered**:
- Multitask learning framework
- Binary cross-entropy loss with class weights
- Two-level weighting system (pos_weight and task_weight)
- Class balancing strategies (inverse_freq vs sqrt_inverse_freq)
- All evaluation metrics (AUROC, AUPRC, F1, MCC, etc.)
- Training monitoring and checkpoint selection

**Read this if you want to understand**:
- How class imbalance is handled (MolPort has only 4% positives!)
- Why inverse_freq weighting was chosen
- What metrics are used to evaluate performance
- How the loss function works
- Why AUROC is the primary metric

### Report 3: Installation and Usage

**Topics covered**:
- System requirements (hardware and software)
- Environment setup (automated and manual)
- Data preparation and caching
- Training both models (SLURM and manual)
- Monitoring training (logs, SLURM, Wandb)
- Checkpoint management and resumption
- Troubleshooting common issues
- Advanced usage (custom experiments, inference)

**Read this if you want to**:
- Set up the environment from scratch
- Run the training pipeline
- Monitor training progress
- Debug common errors (OOM, NaN loss, data loading)
- Resume interrupted training
- Load trained models for inference

---

## Directory Structure

```
chem_space_classifier/
├── README.md                           # This file
├── REPORT_1_MODEL_ARCHITECTURE.md      # Architecture documentation
├── REPORT_2_LOSS_AND_METRICS.md        # Loss and metrics documentation
├── REPORT_3_INSTALLATION_AND_USAGE.md  # Installation and usage guide
├── models_description.md               # Selected model configurations
│
├── train_multitask.py                  # Main training script
├── run_model1.sh                       # SLURM script for Model 1
├── run_model4_rerun.sh                 # SLURM script for Model 4
├── build_omtra_classifier_env.sh       # Environment setup script
│
├── logs/                               # Training logs
│   ├── model1.out
│   ├── model1.err
│   ├── model4_rerun.out
│   └── model4_rerun.err
│
└── final_models/                       # Model outputs
    ├── model1_best/
    │   └── exp_1/
    │       ├── checkpoints/
    │       │   ├── epoch_XX_auroc_0.XXXX.ckpt
    │       │   └── last.ckpt
    │       ├── hyperparams.json
    │       └── results.json
    └── model4_small_fast/
        └── exp_4/
            ├── checkpoints/
            ├── hyperparams.json
            └── results.json
```

---

## Selected Models

### Model 1: Best Performance Baseline

**Hyperparameters**:
- Architecture: 128/64 dims, 3 layers (~463k parameters)
- Learning rate: 1e-4
- Loss: inverse_freq pos_weights, task_weights = [1, 1, 1, 1, 1]
- Regularization: dropout=0.2, weight_decay=1e-5

**Expected Performance**:
- Validation AUROC: 0.70-0.75
- Best tuning loss: 3.932
- Strategy: Optimal performance

**Use when**: You want the best possible accuracy

### Model 4: Small and Fast

**Hyperparameters**:
- Architecture: 96/48 dims, 3 layers (~260k parameters, 44% smaller)
- Learning rate: 2e-4 (faster convergence)
- Loss: inverse_freq pos_weights, task_weights = [1, 1, 1, 1, 1]
- Regularization: dropout=0.2, weight_decay=1e-5

**Expected Performance**:
- Tuning loss: 4.552
- Faster training per epoch
- Lower memory usage

**Use when**: You want faster training or have memory constraints

See [models_description.md](models_description.md) for complete details.

---

## Key Features

### Multitask Learning
- **5 simultaneous tasks**: CSC, MCULE, PubChem, ZINC, MolPort
- **Shared backbone**: GVP layers learn common molecular features
- **Task-specific heads**: Each database gets specialized decision boundaries

### Class Imbalance Handling
- **Positive class weighting**: Upweight rare positives (MolPort: 23.56x)
- **inverse_freq strategy**: Standard and effective
- **Task-level weighting**: Equal weights across all tasks

### Geometric Deep Learning
- **GVP architecture**: SE(3)-equivariant (rotation/translation invariant)
- **Complete graphs**: N×(N-1) edges capture all atom interactions
- **RBF encoding**: Smooth distance features (0-20 Å)
- **Vector features**: Geometric information (8 channels × 3D)

### Efficient Data Loading
- **Streaming sampler**: Edge-based batching (200k edges/batch)
- **Caching**: Persistent indices for faster startup
- **Zarr format**: Efficient compressed storage
- **Mixed precision**: 16-bit training for speed

---

## Performance Metrics

### Primary Metrics

| Metric | Description | Range | Use |
|--------|-------------|-------|-----|
| **AUROC** | Area under ROC curve | [0.5, 1.0] | **Model selection** (threshold-independent) |
| **AUPRC** | Area under PR curve | [0, 1] | Imbalanced tasks (focus on positives) |
| **MCC** | Matthews correlation | [-1, 1] | Balanced metric (all confusion matrix) |
| **F1** | Harmonic mean of P/R | [0, 1] | Classification quality (threshold-dependent) |

### Model Selection Criteria

**Primary**: Average AUROC across 5 tasks (`val_avg_auroc`)
- Saved as best model checkpoint
- Used for hyperparameter comparison
- Threshold-independent evaluation

**Secondary**: Per-task AUROCs
- Identify weak tasks (typically MolPort)
- Ensure all tasks are learning

---

## Training Details

### Hardware Requirements

| Resource | Model 1 | Model 4 |
|----------|---------|---------|
| GPU | 1x (24GB VRAM) | 1x (16GB VRAM) |
| RAM | 64 GB | 32 GB |
| CPUs | 4 cores | 4 cores |
| Time | ~28 days | ~28 days |

### Data Statistics

| Task | Database | Positive Ratio | Pos Weight | Samples |
|------|----------|----------------|------------|---------|
| 1 | CSC | 19.91% | 4.02 | ~250k |
| 2 | MCULE | 29.95% | 2.34 | ~375k |
| 3 | PubChem | 50.22% | 0.99 | ~630k |
| 4 | ZINC | 20.87% | 3.79 | ~262k |
| 5 | MolPort | **4.07%** | **23.56** | **~51k** |

**Total molecules**: ~1.25M training, ~125k validation

---

## Hyperparameter Tuning Summary

### Experiments Performed

1. **Class Balancing** (4 experiments) - COMPLETED
   - Tested inverse_freq vs sqrt_inverse_freq
   - Tested MolPort task weights: 1.0x, 2.0x, 3.0x
   - **Result**: inverse_freq with 1.0x selected

2. **Model Architecture** (2 experiments) - COMPLETED
   - Tested smaller models (96/48 dims)
   - Tested fewer layers (2 vs 3)
   - **Result**: 3-layer baseline better, but small model selected for speed

3. **Learning Rate** (3 experiments) - COMPLETED
   - Tested 5e-5, 1e-4, 2e-4
   - **Result**: 1e-4 best performance, 2e-4 faster convergence

### Best Configurations

| Rank | Config | Loss | Stability | Selected |
|------|--------|------|-----------|----------|
| 1 | lr=1e-4, 128/64 dims | 3.932 | Moderate | ✅ Model 1 |
| 2 | lr=2e-4, 128/64 dims | 4.045 | Lower | - |
| 3 | lr=2e-4, 96/48 dims | 4.552 | Moderate | ✅ Model 4 |

---

## Monitoring and Logging

### Wandb Dashboard
```
https://wandb.ai/mag1037-university-of-pittsburgh/omtra-final-models
```

**Tracked metrics**:
- Total loss (train/val)
- Per-task losses (5 tasks)
- Per-task AUROCs (5 tasks)
- Per-task AUPRC, F1, MCC, accuracy, precision, recall
- Average AUROC (model selection)
- Learning rate, GPU memory, system stats

### Log Files

```bash
# Training output
tail -f logs/model1.out

# Errors
tail -f logs/model1.err

# SLURM status
squeue -u $USER
```

---

## Citation and References

### OMTRA Framework
- Repository: [GitHub](https://github.com/drorlab/OMTRA)
- Paper: (Add publication if available)

### GVP Architecture
- Paper: "Learning from Protein Structure with Geometric Vector Perceptrons"
- Authors: Jing et al., ICLR 2021

### Datasets
- **Pharmit**: Chemical library from multiple sources
- **Sources**: CSC, MCULE, PubChem, ZINC, MolPort databases

---

## Contact and Support

**Author**: mag1037@pitt.edu

**Issues**:
- Environment setup: See Report 3, Troubleshooting section
- Training errors: Check logs, reduce batch size
- SLURM issues: Contact HPC support
- Model questions: See Reports 1 & 2

---

## Quick Reference Card

### Setup
```bash
./build_omtra_classifier_env.sh
conda activate omtra_classifier_env
```

### Train
```bash
sbatch run_model1.sh      # Best performance
sbatch run_model4_rerun.sh # Small and fast
```

### Monitor
```bash
tail -f logs/model1.out    # Logs
squeue -u $USER            # SLURM status
nvidia-smi                 # GPU usage
```

### Resume
```bash
# Automatic via last.ckpt
sbatch run_model1.sh
```

### Stop
```bash
scancel <job_id>
```

### Results
```bash
cat final_models/model1_best/exp_1/results.json
ls final_models/model1_best/exp_1/checkpoints/
```

---

## License

(Add license information as appropriate)

---

**Last Updated**: 2025-11-23
**Version**: 1.0
**Status**: Production Ready

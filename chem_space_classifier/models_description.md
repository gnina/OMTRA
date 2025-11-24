# Selected Models for Final Training

## Overview
After hyperparameter tuning experiments, two models were selected for final training based on their performance and characteristics.

---

## Model 1: Best Performance Baseline
**Name**: `model1_best`
**Job Script**: [run_model1.sh](run_model1.sh)
**Output Directory**: `final_models/model1_best/`

### Hyperparameters

#### Model Architecture
- **Hidden dimension**: 128
- **Edge dimension**: 64
- **Vector channels**: 8
- **Number of layers**: 3
- **Shared representation dimension**: 128
- **Task-specific hidden dimension**: 64
- **Total parameters**: ~463k

#### Loss Configuration
- **Pos weight strategy**: `inverse_freq`
  - Formula: pos_weight = (1-r)/r where r is positive ratio
  - MolPort pos_weight: ~23.5x (due to 4% positive class)
- **MolPort task weight**: 1.0x
- **Effective MolPort weight**: ~23.5x (pos_weight × task_weight)

#### Regularization
- **Dropout**: 0.2
- **Weight decay**: 1e-5

#### Optimization
- **Learning rate**: 1e-4
- **Optimizer**: Adam
- **Gradient clipping**: 1.0
- **Gradient accumulation**: 2 batches
- **Precision**: Mixed (16-bit)

#### Training Configuration
- **Max epochs**: 740
- **Edges per batch**: 200,000
- **Number of workers**: 2
- **Max batches (train)**: 10,000
- **Max batches (val)**: 1,000
- **Early stopping**: Disabled (let model see all data)

### Performance Characteristics
- **Best tuning performance**: Late-stage mean loss of 3.932 (best among all experiments)
- **Stability**: Moderate (std: 0.598)
- **Convergence**: Good balance between speed and final performance
- **Architecture**: Standard 3-layer network provides sufficient capacity

### Rationale for Selection
This configuration achieved the best overall performance during hyperparameter tuning. The learning rate of 1e-4 provided the optimal balance between convergence speed and stability. The baseline architecture (128/64 dims, 3 layers) was superior to smaller alternatives.

---

## Model 4 Rerun: Small and Fast
**Name**: `model4_small_fast`
**Job Script**: [run_model4_rerun.sh](run_model4_rerun.sh)
**Output Directory**: `final_models/model4_small_fast/`

### Hyperparameters

#### Model Architecture
- **Hidden dimension**: 96 (↓ from 128)
- **Edge dimension**: 48 (↓ from 64)
- **Vector channels**: 8
- **Number of layers**: 3
- **Shared representation dimension**: 96 (↓ from 128)
- **Task-specific hidden dimension**: 48 (↓ from 64)
- **Total parameters**: ~260k (~44% fewer than Model 1)

#### Loss Configuration
- **Pos weight strategy**: `inverse_freq`
- **MolPort task weight**: 1.0x
- **Effective MolPort weight**: ~23.5x

#### Regularization
- **Dropout**: 0.2
- **Weight decay**: 1e-5

#### Optimization
- **Learning rate**: 2e-4 (↑ 2x from Model 1)
- **Optimizer**: Adam
- **Gradient clipping**: 1.0
- **Gradient accumulation**: 2 batches
- **Precision**: Mixed (16-bit)

#### Training Configuration
- **Max epochs**: 740
- **Edges per batch**: 200,000
- **Number of workers**: 2
- **Max batches (train)**: 10,000
- **Max batches (val)**: 1,000
- **Early stopping**: Disabled

### Performance Characteristics
- **Tuning performance**: Late-stage mean loss of 4.552
- **Model size**: 44% smaller than Model 1 (260k vs 463k parameters)
- **Training speed**: Faster due to smaller model and higher learning rate
- **Memory efficiency**: Lower GPU memory footprint

### Rationale for Selection
This model offers a speed-memory tradeoff. The smaller architecture (96/48 dims) combined with a higher learning rate (2e-4) enables faster training and lower memory usage. While it showed slightly lower performance during tuning compared to Model 1, it provides valuable diversity for ensemble approaches and may generalize differently.

---

## Comparison Summary

| Aspect | Model 1 (Best) | Model 4 (Small+Fast) |
|--------|----------------|----------------------|
| **Hidden/Edge dims** | 128/64 | 96/48 |
| **Parameters** | ~463k | ~260k |
| **Learning rate** | 1e-4 | 2e-4 |
| **Tuning loss** | 3.932 (best) | 4.552 |
| **Training speed** | Moderate | Fast |
| **Memory usage** | Higher | Lower |
| **Strategy** | Best performance | Speed/efficiency |

---

## Training Details

### Common Settings (Both Models)
- **Dataset**: Pharmit5050 condition A (5 tasks: CSC, MCULE, PubChem, ZINC, MolPort)
- **Positive class ratios**: [19.91%, 29.95%, 50.22%, 20.87%, 4.07%]
- **Loss function**: Binary cross-entropy with logits (BCEWithLogitsLoss)
- **Class balancing**: Inverse frequency pos_weights
- **Validation metric**: Average AUROC across all 5 tasks
- **Checkpointing**: Save top 3 models + last checkpoint
- **Logging**: Weights & Biases (project: `omtra-final-models`)

### Hardware Requirements
- **GPU**: 1 GPU per job
- **Memory**: 64GB RAM (Model 1), 32GB RAM (Model 4)
- **CPUs**: 4 cores
- **Time limit**: 27 days, 23 hours per job
- **Excluded nodes**: g005, g006, g007, g009 (Model 1); g006 (Model 4)

### Checkpoint Resumption
Both models are configured with `--resume_from_checkpoint last`, allowing them to:
- Resume training from the last saved checkpoint if interrupted
- Continue from previous runs without losing progress
- Auto-find `checkpoints/last.ckpt` in their respective output directories

---

## Running the Models

### Submit both models
```bash
cd /net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/tuning_experiments

# Submit Model 1
sbatch run_model1.sh

# Submit Model 4 rerun
sbatch run_model4_rerun.sh
```

### Monitor progress
```bash
# Check job status
squeue -u $USER

# Watch Model 1 output
tail -f logs/model1.out

# Watch Model 4 rerun output
tail -f logs/model4_rerun.out

# Check Wandb
# https://wandb.ai/mag1037-university-of-pittsburgh/omtra-final-models
```

### Cancel jobs if needed
```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

---

## Expected Outcomes

### Model 1
- Higher validation AUROC (expected: 0.70-0.75)
- Lower training loss convergence
- More stable training dynamics
- Better overall performance

### Model 4
- Faster epoch completion time
- Lower memory footprint
- Potentially different generalization behavior
- Good for ensemble diversity

---

## Next Steps

1. Monitor both models during training via Wandb
2. Compare final validation AUROC scores
3. Analyze per-task performance breakdown
4. Consider ensemble predictions combining both models
5. Evaluate on held-out test set

---

**Date Created**: 2025-11-23
**Experiment Directory**: `/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/chem_space_classifier/`
**Wandb Project**: `omtra-final-models`
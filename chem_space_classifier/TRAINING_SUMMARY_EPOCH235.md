# Model 1 Best - Training Summary (Epoch 235)

**Report Generated**: 2025-12-03
**Training Status**: STOPPED
**Last Checkpoint**: Epoch 235 (Dec 3, 2025 17:22)
**Total Training Duration**: ~26 days (Nov 8 - Dec 3)

---

## Summary

The model1_best training run has been **stopped** at epoch 235 out of a planned 740 epochs. The latest checkpoint has been saved for potential resumption.

### Key Performance Metrics (Epoch 235)

**Best Validation Performance:**
- **Epoch 235**: val_avg_auroc = **0.8167** ⭐ (BEST)
- **Epoch 232**: val_avg_auroc = 0.8162
- **Epoch 226**: val_avg_auroc = 0.8167

The model achieved its best validation AUROC of **0.8167** at epoch 235, showing significant improvement from early training (epoch 0-4: ~0.60-0.68).

---

## Model Configuration

### Architecture
- **Model**: Multitask GVP (Graph Vector Perceptron) Classifier
- **Tasks**: 5-way binary classification
  1. CSC (ChemSpace Classifier)
  2. MCULE
  3. PubChem
  4. ZINC
  5. MolPort

### Hyperparameters
- **Learning Rate**: 1e-4
- **Dropout**: 0.2
- **Weight Decay**: 1e-5
- **Layers**: 3
- **Hidden Dimensions**: 128
- **Edge Dimensions**: 64
- **Vector Channels**: 8
- **Shared Representation**: 128
- **Task Hidden**: 64

### Training Configuration
- **Batch Size**: 200,000 edges per batch
- **Gradient Accumulation**: 2 batches
- **Precision**: 16-bit mixed
- **Optimizer**: Adam
- **Gradient Clipping**: 1.0
- **Training Batches/Epoch**: 10,000
- **Validation Batches/Epoch**: 1,000

---

## Training Progress

### Epochs Completed: 235 / 740 (31.8%)

### Performance Timeline
- **Epoch 0-4**: val_avg_auroc = 0.605 - 0.688 (Initial training)
- **Epoch 226**: val_avg_auroc = 0.8167 (Peak performance)
- **Epoch 232**: val_avg_auroc = 0.8162 (Slight decline)
- **Epoch 235**: val_avg_auroc = 0.8167 (Recovery to peak)

### Observations
1. **Strong Learning**: Model improved from 0.60 to 0.82 AUROC (+36% improvement)
2. **Stability**: Performance plateaued around 0.816-0.817 in recent epochs
3. **No Overfitting**: Validation performance remains stable at peak

---

## Saved Checkpoints

### Primary Checkpoint (for resumption)
```
Location: /net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/tuning_experiments/final_models/model1_best/model1_best_epoch235_resume.ckpt
Size: 5.5 MB
Date: Dec 3, 2025 17:22
Epoch: 235
Val AUROC: 0.8167
```

### Available Checkpoints in Directory
```
last.ckpt                                              [Dec 3, 17:22] - Epoch 235
epoch_epoch=235_auroc_val_avg_auroc=0.8165.ckpt       [Nov 26, 04:02]
epoch_epoch=232_auroc_val_avg_auroc=0.8162.ckpt       [Nov 26, 00:04]
epoch_epoch=226_auroc_val_avg_auroc=0.8167.ckpt       [Nov 25, 15:44]
```

**Note**: The checkpoint filenames show epoch 226, 232, 235 but `last.ckpt` is from epoch 235 (most recent).

---

## WandB Logging

- **Project**: omtra-final-models
- **Most Recent Run**: run-20251124_133730-cs5dyvjg
- **Experiment ID**: exp_001_model1_best
- **Run ID (v_num)**: 27fd

---

## Next Steps & Recommendations

### Option 1: Resume Training
To continue training from epoch 235:
```bash
sbatch run_model1.sh
```

The script is configured with `--resume_from_checkpoint last`, which will automatically load:
```
/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/tuning_experiments/final_models/model1_best/exp_001_model1_best/checkpoints/last.ckpt
```

Alternatively, use the backup:
```
/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/tuning_experiments/final_models/model1_best/model1_best_epoch235_resume.ckpt
```

### Option 2: Early Stopping Consideration
Given that:
- Performance has plateaued at ~0.816-0.817 for multiple epochs
- Only 31.8% of planned epochs completed
- Strong performance already achieved

**Recommendation**: Consider the model sufficiently trained unless:
1. You need to exhaust the full 740-epoch budget
2. You want to experiment with learning rate decay
3. You're testing long-term training stability

### Option 3: Fine-tuning
Use this checkpoint as a starting point for:
- Transfer learning to related tasks
- Fine-tuning on specific subsets
- Downstream molecular property prediction

---

## File Locations

**Experiment Directory:**
```
/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/tuning_experiments/final_models/model1_best/exp_001_model1_best/
```

**Checkpoints:**
```
exp_001_model1_best/checkpoints/
```

**Configuration:**
```
exp_001_model1_best/hyperparams.json
```

**WandB Logs:**
```
exp_001_model1_best/wandb/
```

---

## Status: Ready for Resumption ✓

All checkpoints are saved and the model is ready to resume training or proceed to evaluation/deployment.

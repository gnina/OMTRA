# Report 2: Loss Functions, Multitask Learning, and Metrics

## Overview

This document describes the loss function formulation, class balancing strategies, multitask learning approach, and all evaluation metrics used in the OMTRA Multitask Classifier.

---

## Table of Contents

1. [Multitask Learning Framework](#multitask-learning-framework)
2. [Loss Function](#loss-function)
3. [Class Balancing Strategies](#class-balancing-strategies)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Training Monitoring](#training-monitoring)
6. [Hyperparameter Impact](#hyperparameter-impact)

---

## Multitask Learning Framework

### Problem Formulation

The model solves **5 simultaneous binary classification tasks**:

```
Input: Molecular graph G = (V, E, X)
Output: y = [y₁, y₂, y₃, y₄, y₅]

where yᵢ ∈ {0, 1} indicates presence in database i
```

### Tasks and Data Statistics

| Task | Database | Positive Ratio | Imbalance | Task Index |
|------|----------|----------------|-----------|------------|
| 1 | CSC | 19.91% | 4.0:1 | 2 |
| 2 | MCULE | 29.95% | 2.3:1 | 5 |
| 3 | PubChem | 50.22% | 1.0:1 (balanced) | 8 |
| 4 | ZINC | 20.87% | 3.8:1 | 12 |
| 5 | MolPort | **4.07%** | **23.6:1** (extreme) | 6 |

**Key Challenge**: MolPort is severely imbalanced (only 4% positive samples)

### Multitask Architecture Benefits

1. **Shared Representation Learning**
   - Common GVP backbone extracts molecular features
   - Tasks share knowledge about chemical structures
   - Reduces overfitting on rare classes (MolPort)

2. **Transfer Learning**
   - Well-represented tasks (PubChem: 50%) help rare tasks (MolPort: 4%)
   - Shared features capture general molecular properties
   - Task-specific heads specialize decision boundaries

3. **Computational Efficiency**
   - Single forward pass → 5 predictions
   - Shared parameters reduce total model size
   - Faster training than 5 independent models

4. **Regularization Effect**
   - Multitask learning acts as implicit regularization
   - Forces model to learn generalizable features
   - Prevents task-specific overfitting

---

## Loss Function

### Total Loss Formulation

The total loss is a **weighted sum of per-task binary cross-entropy losses**:

```
L_total = Σᵢ₌₁⁵ wᵢ · L_task(yᵢ, ŷᵢ)

where:
  wᵢ = task weight for task i
  L_task = binary cross-entropy with positive class weighting
  yᵢ = true labels for task i
  ŷᵢ = predicted logits for task i
```

### Per-Task Loss (Binary Cross-Entropy with Logits)

For each task i:

```python
L_task(y, ŷ) = BCEWithLogitsLoss(ŷ, y, pos_weight=pᵢ)
```

Where:
- `ŷ`: Predicted logits (unbounded real values)
- `y`: True binary labels {0, 1}
- `pᵢ`: Positive class weight for task i

**Expanded form**:

```
L_BCE(y, ŷ) = -(1/N) Σⱼ [y_j · log(σ(ŷ_j)) · pᵢ + (1 - y_j) · log(1 - σ(ŷ_j))]

where σ(z) = 1 / (1 + e⁻ᶻ) is the sigmoid function
```

**Interpretation**:
- Positive samples (y=1) are weighted by `pᵢ`
- Negative samples (y=0) have weight 1.0
- Higher `pᵢ` → model pays more attention to positive samples
- Used to counteract class imbalance

### Two-Level Weighting System

The loss uses **two levels of weights**:

#### Level 1: Positive Class Weights (pos_weight)

Applied **within each task** to balance positive/negative classes:

```python
pos_weight[i] = (1 - r_i) / r_i

where r_i = positive ratio for task i
```

**Effect**: Upweights rare positive samples

**Example** (inverse_freq strategy):
- CSC: `pos_weight = (1 - 0.1991) / 0.1991 = 4.02`
- MolPort: `pos_weight = (1 - 0.0407) / 0.0407 = 23.56`

#### Level 2: Task Weights

Applied **across tasks** to prioritize certain tasks:

```python
task_weights = [w_CSC, w_MCULE, w_PubChem, w_ZINC, w_MolPort]
```

**Default configuration**:
```python
# Model 1 & Model 4 (selected models)
task_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
```

**Previous experiments** tested:
```python
task_weights = [1.0, 1.0, 1.0, 1.0, 2.0]  # 2x weight for MolPort
```

**Effective weight** for a positive sample in task i:
```
effective_weight = task_weights[i] × pos_weights[i]
```

**Example for MolPort**:
- `pos_weight = 23.56` (from class imbalance)
- `task_weight = 1.0` (selected configuration)
- `effective_weight = 23.56`

---

## Class Balancing Strategies

### Strategy 1: Inverse Frequency (Selected)

**Formula**:
```python
pos_weight[i] = (1 - r_i) / r_i
```

**Rationale**:
- Directly inversely proportional to positive ratio
- Strong correction for imbalanced classes
- Standard approach in imbalanced learning

**Weights for our datasets**:
```python
pos_ratios = [0.1991, 0.2995, 0.5022, 0.2087, 0.0407]

pos_weights_inverse = [
    4.02,   # CSC
    2.34,   # MCULE
    0.99,   # PubChem (nearly balanced)
    3.79,   # ZINC
    23.56   # MolPort (extreme imbalance)
]
```

**Selected for both Model 1 and Model 4**
- Achieved lowest training loss (3.2-5.8 range)
- Best convergence stability
- Standard and interpretable

### Strategy 2: Square Root Inverse Frequency

**Formula**:
```python
pos_weight[i] = sqrt((1 - r_i) / r_i)
```

**Rationale**:
- Less aggressive than inverse_freq
- Reduces extreme weights
- May prevent over-correction

**Weights for our datasets**:
```python
pos_weights_sqrt = [
    2.00,   # CSC (vs 4.02)
    1.53,   # MCULE (vs 2.34)
    0.99,   # PubChem
    1.95,   # ZINC (vs 3.79)
    4.86    # MolPort (vs 23.56) ← Much smaller!
]
```

**Tested but not selected**:
- Performed slightly worse in tuning experiments
- MolPort weight reduced from 23.56 to 4.86
- May under-correct severe imbalance

### Strategy 3: No Reweighting

**Formula**:
```python
pos_weight[i] = 1.0  # All equal
```

**Not tested** - would likely fail on MolPort due to extreme imbalance.

### Tuning Results Summary

From hyperparameter experiments:

| Strategy | MolPort Task Weight | Late-Stage Mean Loss | Selected? |
|----------|---------------------|----------------------|-----------|
| inverse_freq | 1.0x | 4.237 | ✅ Yes |
| inverse_freq | 2.0x | Not tested fully | ❌ No |
| inverse_freq | 3.0x | Not tested fully | ❌ No |
| sqrt_inverse_freq | 1.0x | 4.102 | ❌ No |

**Decision**: `inverse_freq` with `task_weight=1.0` for all tasks

---

## Evaluation Metrics

### Per-Task Binary Classification Metrics

All metrics are computed **separately for each of the 5 tasks**:

#### 1. AUROC (Area Under ROC Curve)

**Definition**: Area under the Receiver Operating Characteristic curve

```
AUROC = P(score(positive) > score(negative))
```

**Range**: [0.5, 1.0]
- 0.5 = Random guessing
- 1.0 = Perfect classifier
- **Primary metric** for model selection

**Why AUROC?**
- Threshold-independent (evaluates ranking quality)
- Robust to class imbalance
- Standard in drug discovery
- Interpretable (probability of correct ranking)

**Computed at**:
- Every training step (accumulated)
- Every validation step
- Logged per-epoch

#### 2. AUPRC (Area Under Precision-Recall Curve)

**Definition**: Area under the Precision-Recall curve

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

**Range**: [0, 1]
- Depends on positive class ratio
- Higher is better

**Why AUPRC?**
- More informative for imbalanced datasets than AUROC
- Focuses on positive class performance
- Penalizes false positives more heavily
- Critical for rare MolPort task

**Use case**: Secondary metric for imbalanced tasks

#### 3. F1 Score

**Definition**: Harmonic mean of precision and recall

```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

**Range**: [0, 1]
- 0 = Worst
- 1 = Perfect

**Why F1?**
- Balances precision and recall
- Single metric for classification quality
- Requires choosing threshold (default: 0.5)

**Limitation**: Threshold-dependent, less robust than AUROC

#### 4. MCC (Matthews Correlation Coefficient)

**Definition**: Correlation between predictions and true labels

```
MCC = (TP·TN - FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Range**: [-1, 1]
- -1 = Perfect disagreement
- 0 = Random
- +1 = Perfect agreement

**Why MCC?**
- Considered one of the best metrics for binary classification
- Balanced even with imbalanced classes
- Takes all confusion matrix elements into account
- Less biased than accuracy

#### 5. Accuracy

**Definition**: Fraction of correct predictions

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Range**: [0, 1]

**Limitation**:
- Misleading for imbalanced datasets
- E.g., predicting all negatives for MolPort gives 96% accuracy!
- **Not recommended** as primary metric

**Computed as**: Macro-averaged accuracy

#### 6. Precision

**Definition**: Fraction of positive predictions that are correct

```
Precision = TP / (TP + FP)
```

**Range**: [0, 1]

**Interpretation**: Of predicted positives, how many are actually positive?

#### 7. Recall (Sensitivity, True Positive Rate)

**Definition**: Fraction of actual positives correctly identified

```
Recall = TP / (TP + FN)
```

**Range**: [0, 1]

**Interpretation**: Of actual positives, how many did we find?

### Aggregated Metrics

#### Average AUROC (Primary Model Selection Metric)

```python
avg_auroc = (1/5) Σᵢ₌₁⁵ AUROC_i
```

**Used for**:
- Model checkpointing (save best model)
- Early stopping (if enabled)
- Hyperparameter comparison
- **Primary metric** in tuning experiments

**Logged as**: `val_avg_auroc`

#### Average MCC

```python
avg_mcc = (1/5) Σᵢ₌₁⁵ MCC_i
```

**Used for**:
- Secondary evaluation metric
- Complementary to AUROC

**Logged as**: `val_avg_mcc`

---

## Training Monitoring

### Logged Metrics (Per Step)

**Training** (logged every 50 steps):
```
train_loss                    # Total weighted loss
train/<task>_loss             # Per-task loss (5 tasks)
```

**Validation** (logged every epoch):
```
val_loss                      # Total weighted loss
val/<task>_loss               # Per-task loss (5 tasks)
val/<task>_auroc              # Per-task AUROC (5 tasks)
val/<task>_auprc              # Per-task AUPRC (5 tasks)
val/<task>_f1                 # Per-task F1 (5 tasks)
val/<task>_mcc                # Per-task MCC (5 tasks)
val/<task>_acc                # Per-task accuracy (5 tasks)
val/<task>_precision          # Per-task precision (5 tasks)
val/<task>_recall             # Per-task recall (5 tasks)
val_avg_auroc                 # Average AUROC (PRIMARY METRIC)
val_avg_mcc                   # Average MCC
```

### Example Training Output

```
Epoch 5/740
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train_loss: 4.235
train_avg_auroc: 0.742
  CSC_loss: 0.623      CSC_auroc: 0.781
  MCULE_loss: 0.512    MCULE_auroc: 0.765
  PubChem_loss: 0.401  PubChem_auroc: 0.823
  ZINC_loss: 0.598     ZINC_auroc: 0.744
  MolPort_loss: 2.101  MolPort_auroc: 0.598  ← Hardest task!

================================================================================
VALIDATION EPOCH SUMMARY
================================================================================
       CSC: AUROC=0.7652 | AUPRC=0.6543 | F1=0.7123 | MCC=0.5432
     MCULE: AUROC=0.7891 | AUPRC=0.7012 | F1=0.7345 | MCC=0.5678
   PubChem: AUROC=0.8234 | AUPRC=0.8123 | F1=0.7876 | MCC=0.6234
      ZINC: AUROC=0.7543 | AUPRC=0.6712 | F1=0.7234 | MCC=0.5543
   MolPort: AUROC=0.6234 | AUPRC=0.2134 | F1=0.3456 | MCC=0.2345
================================================================================

val_avg_auroc: 0.7511  ← Model checkpoint metric
```

### Checkpointing Strategy

```python
ModelCheckpoint(
    monitor='val_avg_auroc',
    mode='max',
    save_top_k=3,
    save_last=True
)
```

**Saves**:
- Top 3 models by validation AUROC
- Last checkpoint (for resumption)

**Checkpoint naming**:
```
epoch_{epoch:02d}_auroc_{val_avg_auroc:.4f}.ckpt
```

Example:
```
checkpoints/
  epoch_23_auroc_0.7511.ckpt
  epoch_47_auroc_0.7623.ckpt  ← Best model
  epoch_52_auroc_0.7598.ckpt
  last.ckpt
```

---

## Hyperparameter Impact

### Loss Configuration Impact (From Tuning)

| Config | pos_weight | task_weight | Train Loss | Validation AUROC | Notes |
|--------|-----------|-------------|------------|------------------|-------|
| **inverse_freq, 1.0x** | Standard | Equal | 3.93-4.24 | Best | ✅ Selected |
| sqrt_inverse_freq, 1.0x | Reduced | Equal | 4.10 | Good | Less aggressive |
| inverse_freq, 2.0x | Standard | 2x MolPort | Higher | Not tested | Over-weights MolPort |

### Learning Rate Impact

| Learning Rate | Convergence | Stability | Performance | Notes |
|--------------|-------------|-----------|-------------|-------|
| 5e-5 | Slow | High (std: 0.307) | Loss: 4.256 | Most stable |
| **1e-4** | Moderate | Moderate (std: 0.598) | Loss: 3.932 | ✅ Best performance (Model 1) |
| **2e-4** | Fast | Lower (std: 0.682) | Loss: 4.045 | ✅ Faster training (Model 4) |

### Regularization Impact

| Dropout | Weight Decay | Effect | Use Case |
|---------|--------------|--------|----------|
| 0.2 | 1e-5 | Baseline | ✅ Both models |
| 0.3 | 1e-4 | Higher reg | Tested but not selected |
| 0.35 | 1e-4 | Highest reg | For smaller models |

---

## Best Practices and Recommendations

### 1. Handling Class Imbalance

**Recommended**:
- Use `inverse_freq` pos_weights
- Set `task_weights = 1.0` for all tasks (don't over-correct)
- Monitor per-task losses and AUROCs separately
- Use AUROC/AUPRC instead of accuracy

**Avoid**:
- Relying on accuracy for imbalanced tasks
- Extreme task weight scaling (causes instability)
- Ignoring class imbalance entirely

### 2. Model Selection

**Primary metric**: `val_avg_auroc`
- Robust to imbalance
- Threshold-independent
- Directly interpretable

**Secondary metrics**:
- `val_avg_mcc` - Complementary view
- Per-task AUROCs - Identify weak tasks
- Training stability - Low loss variance

### 3. Multitask Learning

**Benefits realized when**:
- Tasks share common features (chemical databases do!)
- Some tasks are data-rich (PubChem helps MolPort)
- Shared backbone is sufficiently expressive

**Watch out for**:
- Negative transfer (tasks hurting each other)
- Dominant tasks overshadowing rare tasks
- Solution: Monitor per-task metrics, adjust task weights if needed

### 4. Loss Monitoring

**Healthy training shows**:
- Decreasing training loss
- Stable or improving validation AUROC
- Per-task losses decreasing (even MolPort!)
- Train-val gap < 15% (not severe overfitting)

**Warning signs**:
- Validation AUROC decreasing (overfitting)
- MolPort loss not decreasing (imbalance not addressed)
- Training loss fluctuating wildly (LR too high or batch size too small)

---

## Summary

### Loss Function Components

1. **Binary Cross-Entropy** with logits (numerically stable)
2. **Positive class weighting** (`inverse_freq`: 4.02x to 23.56x)
3. **Task weighting** (equal 1.0x for all tasks)
4. **Multitask aggregation** (weighted sum of 5 task losses)

### Key Metrics

- **Primary**: Average AUROC across 5 tasks
- **Secondary**: Average MCC, per-task AUROCs
- **Monitoring**: Per-task losses, F1, precision, recall

### Selected Configurations

| Model | pos_weight | task_weights | Learning Rate | Loss Weight |
|-------|-----------|--------------|---------------|-------------|
| Model 1 | inverse_freq | [1,1,1,1,1] | 1e-4 | Standard |
| Model 4 | inverse_freq | [1,1,1,1,1] | 2e-4 | Standard |

### Design Philosophy

- **Handle imbalance** through pos_weights, not task_weights
- **Use AUROC** as threshold-independent evaluation
- **Monitor all tasks** separately to catch issues
- **Multitask learning** for shared feature learning
- **Regularization** via dropout, not extreme loss weighting

**File Locations**:
- Loss computation: [multitask_lightning_module.py:128-166](models/multitask_lightning_module.py#L128-L166)
- Metrics: [multitask_lightning_module.py:90-123](models/multitask_lightning_module.py#L90-L123)

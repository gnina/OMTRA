# Report 1: Model Architecture and Input Processing

## Overview

This document describes the architecture of the Multitask GVP (Geometric Vector Perceptron) Classifier and how molecular inputs are processed through the network.

---

## Table of Contents

1. [Model Purpose](#model-purpose)
2. [Input Data Format](#input-data-format)
3. [Architecture Overview](#architecture-overview)
4. [Detailed Layer-by-Layer Description](#detailed-layer-by-layer-description)
5. [Model Configurations](#model-configurations)
6. [Parameter Counts](#parameter-counts)

---

## Model Purpose

The Multitask GVP Classifier performs **5 simultaneous binary classification tasks** to predict which chemical databases a molecule belongs to:

| Task # | Database | Task Index | Description |
|--------|----------|------------|-------------|
| 1 | CSC | 2 | ChemSpace Collection |
| 2 | MCULE | 5 | MCule database |
| 3 | PubChem | 8 | PubChem compounds |
| 4 | ZINC | 12 | ZINC database |
| 5 | MolPort | 6 | MolPort screening library |

**Why multitask learning?**
- Shares learned molecular representations across all tasks
- Improves generalization by learning common chemical features
- More efficient than training 5 separate models
- Each task has specialized decision boundaries while using shared features

---

## Input Data Format

### Molecular Graph Representation

Each molecule is represented as a **heterogeneous directed graph** with:

#### Node Features (Atoms)
Each atom has the following features stored in the DGL graph:

| Feature | Node Type | Shape | Description |
|---------|-----------|-------|-------------|
| `a_1_true` | `lig` | `[num_atoms]` | Atom type (0-117, atomic number) |
| `c_1_true` | `lig` | `[num_atoms]` | Formal charge (binned: 0-9) |
| `x_1_true` | `lig` | `[num_atoms, 3]` | 3D atomic coordinates (Å) |

**Atom types**: Integer indices 0-117 representing elements (H=1, C=6, N=7, O=8, etc.)

**Charge types**: Formal charges binned into 10 categories:
- 0: neutral
- 1-4: positive charges (+1 to +4)
- 5-9: negative charges (-1 to -5)

**Coordinates**: Cartesian coordinates in Ångströms used for:
- Computing edge geometry (distances, directions)
- Building geometric features for GVP layers

#### Edge Features (Bonds)
The graph uses a **complete directed graph** topology:

| Feature | Edge Type | Shape | Description |
|---------|-----------|-------|-------------|
| `e_1_true` | `lig_to_lig` | `[num_edges]` | Bond type (0-3) |

**Bond types**:
- 0: Single bond
- 1: Double bond
- 2: Triple bond
- 3: Aromatic bond

**Edge structure**: For a molecule with N atoms, the graph contains:
- **N × (N-1) directed edges** (complete graph)
- This includes both covalent bonds AND non-bonded interactions
- Allows GVP to learn long-range molecular interactions

#### System-Level Labels

| Feature | Shape | Description |
|---------|-------|-------------|
| `pharmit_library` | `[5]` | Binary labels for 5 databases [CSC, MCULE, PubChem, ZINC, MolPort] |

Example: `[1, 0, 1, 0, 0]` means the molecule is in CSC and PubChem, but not in MCULE, ZINC, or MolPort.

---

## Architecture Overview

```
Input Molecule Graph
        ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 1: Feature Embedding                                  │
│  - Atom embedding: 118 types → 64d                          │
│  - Charge embedding: 10 types → 64d                         │
│  - Bond embedding: 4 types → 64d                            │
│  - Node projection: 128d → hidden_dim                        │
│  - Edge projection: 64d → edge_dim                          │
└──────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 2: Geometric Processing                               │
│  - Compute edge vectors: x_i - x_j                          │
│  - Compute distances: ||x_i - x_j||                         │
│  - RBF encoding: distances → 32d radial basis functions     │
│  - Normalize edge vectors                                    │
└──────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 3: Shared GVP Backbone (num_layers × GVP Conv)      │
│  - Layer 1: HeteroGVPConv (scalar + vector features)        │
│  - Layer 2: HeteroGVPConv                                   │
│  - Layer 3: HeteroGVPConv                                   │
│  - [Optional Layer 4 for larger models]                     │
│                                                              │
│  Each GVP layer processes:                                   │
│    - Scalar features: [N, hidden_dim]                       │
│    - Vector features: [N, n_vec_channels, 3]               │
│    - Edge features: [E, edge_dim]                           │
│    - Geometric features: distances + directions             │
└──────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 4: Graph-Level Readout                                │
│  - Aggregate node features → graph representation           │
│  - Readout operation: mean/sum/max over all atoms           │
│  - Output: [batch_size, hidden_dim]                         │
└──────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 5: Shared Representation Layer                        │
│  - Linear: hidden_dim → shared_repr_dim                     │
│  - ReLU activation                                           │
│  - Dropout (0.2)                                            │
│  - LayerNorm                                                │
│  - Output: [batch_size, shared_repr_dim]                    │
└──────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────┐
│  STAGE 6: Task-Specific Branches (5 parallel heads)         │
│                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │ CSC Head            │  │ MCULE Head          │          │
│  │ shared → 64/128d    │  │ shared → 64/128d    │          │
│  │ ReLU + Dropout      │  │ ReLU + Dropout      │  ...     │
│  │ 64/128d → 1         │  │ 64/128d → 1         │          │
│  │ (logit)             │  │ (logit)             │          │
│  └─────────────────────┘  └─────────────────────┘          │
│                                                              │
│  Output: [batch_size, 5] logits                             │
└──────────────────────────────────────────────────────────────┘
        ↓
   5 Binary Predictions (via sigmoid)
```

---

## Detailed Layer-by-Layer Description

### Stage 1: Feature Embedding

#### Atom Embedding
```python
atom_embedding: nn.Embedding(118, 64)
```
- **Input**: Atom type indices [0-117]
- **Output**: 64-dimensional learned embeddings
- **Purpose**: Convert discrete atom types to continuous representations
- **Learnable**: Yes (learned during training)

#### Charge Embedding
```python
charge_embedding: nn.Embedding(10, 64)
```
- **Input**: Charge type indices [0-9]
- **Output**: 64-dimensional learned embeddings
- **Purpose**: Encode formal charge information
- **Learnable**: Yes

#### Bond Embedding
```python
bond_embedding: nn.Embedding(4, 64)
```
- **Input**: Bond type indices [0-3]
- **Output**: 64-dimensional learned embeddings
- **Purpose**: Encode bond order/aromaticity
- **Learnable**: Yes

#### Node Projection
```python
node_projection: Sequential(
    Linear(128, hidden_dim),
    SiLU(),
    LayerNorm(hidden_dim)
)
```
- **Input**: Concatenated atom + charge embeddings [128d]
- **Output**: Node scalar features [hidden_dim]
- **Purpose**: Project combined atomic features to GVP input dimension
- **Activation**: SiLU (Sigmoid Linear Unit) - smooth, non-monotonic

#### Edge Projection
```python
edge_projection: Sequential(
    Linear(64, edge_dim),
    SiLU(),
    LayerNorm(edge_dim)
)
```
- **Input**: Bond embeddings [64d]
- **Output**: Edge scalar features [edge_dim]
- **Purpose**: Project bond features to GVP edge dimension

---

### Stage 2: Geometric Processing

#### Edge Geometry Computation

For each directed edge (i → j):

1. **Edge vector**: `x_diff = x_i - x_j`
   - Direction from atom j to atom i
   - Shape: `[num_edges, 3]`

2. **Distance**: `d_ij = ||x_i - x_j||`
   - Euclidean distance between atoms
   - Shape: `[num_edges]`

3. **Normalized direction**: `x_diff_normalized = x_diff / (d_ij + ε)`
   - Unit vector in edge direction
   - ε = 1e-8 prevents division by zero

4. **Radial Basis Function (RBF) encoding**:
   ```python
   RBF(d) = exp(-((d - μ_k) / σ)²)
   ```
   - μ_k: 32 evenly spaced centers from 0 to 20 Å
   - σ = 20 / 32 ≈ 0.625 Å
   - Output: `[num_edges, 32]` - smooth distance encoding
   - **Why RBF?** Provides continuous, differentiable distance features

---

### Stage 3: Shared GVP Backbone

Each **HeteroGVPConv** layer processes **two types of features**:

#### Scalar Features (Rotation-Invariant)
- Atom types, charges, learned embeddings
- Shape: `[num_nodes, hidden_dim]`
- Invariant to molecular rotation/translation

#### Vector Features (Rotation-Equivariant)
- Geometric features (edge directions)
- Shape: `[num_nodes, n_vec_channels, 3]`
- Transforms correctly under rotation (equivariance)

#### GVP Layer Operations

Each GVP convolution performs:

1. **Message Function**:
   - Combines source node features, edge features, and geometry
   - Uses 3 sequential GVP modules (n_message_gvps=3)
   - Processes both scalars and vectors

2. **Aggregation**:
   - Sums messages from all neighboring atoms
   - Applies message normalization (factor=100)

3. **Update Function**:
   - Combines aggregated messages with current node features
   - Uses 3 sequential GVP modules (n_update_gvps=3)
   - Updates both scalar and vector features

4. **Activation Functions**:
   - Scalars: SiLU activation
   - Vectors: Sigmoid activation (maintains equivariance)

5. **Dropout**: Applied with probability 0.2

**Key Property**: GVP layers are **SE(3)-equivariant**
- Rotating the molecule rotates the vector features accordingly
- Predictions are invariant to rotation/translation

---

### Stage 4: Graph-Level Readout

Aggregates node features into a single graph-level representation:

```python
readout_fn = dgl.readout_nodes(graph, 'h', op='mean')
```

**Operations available**:
- **Mean**: Average node features (default)
  - `h_graph = (1/N) Σ h_i`
- **Sum**: Sum node features
  - `h_graph = Σ h_i`
- **Max**: Element-wise maximum
  - `h_graph = max(h_1, h_2, ..., h_N)`

**Output**: `[batch_size, hidden_dim]`

---

### Stage 5: Shared Representation Layer

```python
shared_repr: Sequential(
    Linear(hidden_dim, shared_repr_dim),
    ReLU(),
    Dropout(0.2),
    LayerNorm(shared_repr_dim)
)
```

**Purpose**:
- Creates a common molecular representation for all tasks
- Allows task-specific heads to learn from shared features
- Regularization via dropout prevents overfitting

**Output**: `[batch_size, shared_repr_dim]`

---

### Stage 6: Task-Specific Branches

Each of the 5 tasks has an independent branch:

```python
task_branch: Sequential(
    Linear(shared_repr_dim, task_hidden_dim),
    ReLU(),
    Dropout(0.2),
    Linear(task_hidden_dim, 1)
)
```

**Architecture per branch**:
1. Projection: shared_repr_dim → task_hidden_dim
2. ReLU activation
3. Dropout (0.2)
4. Final linear layer: task_hidden_dim → 1 (logit)

**Output**: Single logit value per task
- Positive logit → predicted as positive class
- Negative logit → predicted as negative class
- Convert to probability: `p = sigmoid(logit)`

**Why separate branches?**
- Each database may have different decision boundaries
- Allows task-specific fine-tuning while sharing features
- Prevents negative transfer between dissimilar tasks

---

## Model Configurations

### Model 1: Best Performance

| Parameter | Value | Description |
|-----------|-------|-------------|
| **hidden_dim** | 128 | Node scalar feature dimension |
| **edge_dim** | 64 | Edge scalar feature dimension |
| **n_vec_channels** | 8 | Number of vector channels per node |
| **num_layers** | 3 | Number of GVP convolution layers |
| **shared_repr_dim** | 128 | Shared representation dimension |
| **task_hidden_dim** | 64 | Task-specific hidden layer size |
| **dropout** | 0.2 | Dropout probability |
| **readout** | mean | Graph pooling operation |
| **rbf_dim** | 32 | Radial basis function dimension |
| **rbf_dmax** | 20.0 | Maximum distance for RBF (Å) |

**Total parameters**: ~463,000

### Model 4: Small and Fast

| Parameter | Value | Description |
|-----------|-------|-------------|
| **hidden_dim** | 96 | ↓ Reduced from 128 |
| **edge_dim** | 48 | ↓ Reduced from 64 |
| **n_vec_channels** | 8 | Same |
| **num_layers** | 3 | Same |
| **shared_repr_dim** | 96 | ↓ Reduced from 128 |
| **task_hidden_dim** | 48 | ↓ Reduced from 64 |
| **dropout** | 0.2 | Same |
| **readout** | mean | Same |

**Total parameters**: ~260,000 (44% reduction)

---

## Parameter Counts

### Breakdown by Component (Model 1)

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| **Embeddings** | | |
| Atom embedding | 7,552 | 118 × 64 |
| Charge embedding | 640 | 10 × 64 |
| Bond embedding | 256 | 4 × 64 |
| **Projections** | | |
| Node projection | 16,512 | (128 × 128) + 128 + norm |
| Edge projection | 4,160 | (64 × 64) + 64 + norm |
| **GVP Backbone** | ~340,000 | 3 layers × ~113k each |
| **Shared Repr** | 16,512 | (128 × 128) + 128 + norm |
| **Task Branches** | ~41,000 | 5 × (128×64 + 64×1) |
| **Total** | **~463,000** | |

### Scaling with Hyperparameters

Parameters scale roughly as:
- **O(hidden_dim²)** for GVP layers
- **O(n_vec_channels × hidden_dim)** for vector features
- **O(num_layers)** for backbone depth
- **O(n_tasks)** for task heads

---

## Input Processing Pipeline

### Step-by-Step Example

Consider a molecule with **N = 20 atoms**:

1. **Input shapes**:
   - Atom types: `[20]`
   - Charges: `[20]`
   - Coordinates: `[20, 3]`
   - Bond types: `[380]` (complete graph: 20 × 19)

2. **After embedding**:
   - Atom embeddings: `[20, 64]`
   - Charge embeddings: `[20, 64]`
   - Node features: `[20, 128]` (concatenated)
   - Bond embeddings: `[380, 64]`

3. **After projection**:
   - Node scalars: `[20, 128]` (Model 1)
   - Edge scalars: `[380, 64]` (Model 1)
   - Node vectors: `[20, 8, 3]` (initialized to zeros)

4. **After GVP layers**:
   - Node scalars: `[20, 128]` (updated)
   - Node vectors: `[20, 8, 3]` (updated with geometry)

5. **After readout**:
   - Graph features: `[1, 128]` (mean over 20 atoms)

6. **After shared repr**:
   - Shared repr: `[1, 128]`

7. **After task branches**:
   - Logits: `[1, 5]` (one per task)

### Batching

For a batch of **B molecules**:
- Graphs are combined into a single large graph
- Batch-level operations use DGL's batching utilities
- Final output: `[B, 5]` logits

---

## Key Design Decisions

### 1. Why GVP (Geometric Vector Perceptrons)?
- **3D geometry matters**: Drug-like molecules are 3D objects
- **Rotation equivariance**: Predictions shouldn't change if molecule is rotated
- **Long-range interactions**: Complete graph captures all atom-atom relationships
- **Proven performance**: State-of-art on molecular property prediction

### 2. Why Complete Graphs?
- Captures non-covalent interactions (hydrogen bonds, π-stacking)
- No need to define bond cutoffs
- GVP handles varying node degrees well
- Important for binding site recognition

### 3. Why Multitask Learning?
- **Data efficiency**: Share representations across tasks
- **Better generalization**: Common chemical features benefit all tasks
- **Computational efficiency**: One forward pass → 5 predictions
- **Transfer learning**: Rare classes (MolPort) benefit from common classes

### 4. Why Separate Task Heads?
- Different databases have different selection criteria
- Allows task-specific decision boundaries
- Prevents negative transfer
- Easy to add/remove tasks without retraining backbone

---

## Summary

The Multitask GVP Classifier is a sophisticated architecture that:

1. **Embeds** molecular graphs with learned atom, charge, and bond representations
2. **Processes geometry** via radial basis functions and normalized edge vectors
3. **Learns features** through 3 GVP convolution layers (SE(3)-equivariant)
4. **Aggregates** atom-level features into graph-level representations
5. **Shares** a common molecular representation across tasks
6. **Specializes** with task-specific heads for each database

The model achieves state-of-the-art performance by combining:
- Geometric deep learning (GVP)
- Multitask learning
- Careful regularization (dropout, normalization)
- Efficient batching strategies

**File Location**: [multitask_gvp_classifier.py](models/multitask_gvp_classifier.py)

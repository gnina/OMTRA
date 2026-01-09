# Testing

OMTRA uses [pytest](https://docs.pytest.org/) as its testing framework. Tests are organized into unit tests (fast, isolated tests with mock data) and integration tests (tests that require real data or external dependencies).

## Quick Start

```bash
# Run all unit tests (no data required)
pytest tests/unit/

# Run all tests except those requiring data
pytest -m "not requires_data"

# Run all tests (requires data to be available)
pytest tests/

# Run tests with verbose output
pytest -v tests/

# Run tests with custom data paths
pytest tests/ --plinder-path=/path/to/plinder --pharmit-path=/path/to/pharmit
```

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── unit/                 # Fast, isolated unit tests
│   ├── __init__.py
│   ├── test_crop_utils.py
│   ├── test_interpolant.py
│   └── test_tasks.py
└── integration/          # Tests requiring real data or external dependencies
    ├── __init__.py
    ├── test_cli.py
    ├── test_data_module.py
    ├── test_graph_construction.py
    ├── test_omtra_forward.py
    ├── test_pharmacophores.py
    ├── test_plinder_dataset.py
    ├── test_vector_field.py
    ├── test_lig.sdf        # Test ligand structure
    ├── test_rec.pdb        # Test protein structure
    └── test_pharmacophore.xyz  # Test pharmacophore file
```

## Test Categories

### Unit Tests (`tests/unit/`)

Unit tests are fast and don't require any external data. They use mock objects and synthetic data to test individual functions and classes in isolation.

**`test_crop_utils.py`** — Tests for the dynamic protein cropping utilities:
- `TestSampleCropDistance` — Tests for sampling crop distances within specified ranges
- `TestComputeCloseResidues` — Tests for finding protein residues close to reference coordinates
- `TestCropStructureData` — Tests for cropping protein structure data based on distance
- `TestFilterNpndesByDistance` — Tests for filtering non-protein non-DNA entities by distance

**`test_interpolant.py`** — Tests for the interpolant scheduler and conditional paths:
- `TestInterpolantScheduler` — Tests for linear interpolant weights (alpha, beta) at various time points
- `TestContinuousInterpolant` — Tests for continuous feature interpolation (positions)
- `TestCTMCMask` — Tests for categorical feature masking/unmasking (atom types)

**`test_tasks.py`** — Tests for the task system:
- `TestTaskRegistry` — Tests for task registration and retrieval
- `TestTaskModalities` — Tests for task modality properties (fixed vs generated)
- `TestTaskProperties` — Tests for computed task properties (unconditional, has_protein, etc.)
- `TestModalityRegistry` — Tests for modality registration and retrieval
- `TestModalityProperties` — Tests for modality dataclass properties

### Integration Tests (`tests/integration/`)

Integration tests verify that different components work correctly together. Some require real data.

**`test_cli.py`** — Tests for the command-line interface:
- Validates CLI help output and argument parsing
- Tests error handling for missing required inputs
- Validates task-specific input requirements (protein files, ligand files, pharmacophore files)
- Uses mock checkpoints and models to avoid requiring actual model weights

**`test_data_module.py`** — Tests for the data loading pipeline:
- Tests `MultiTaskDataModule` instantiation and configuration
- Tests loading train and validation `MultitaskDataSet` instances
- Tests the `__getitem__` interface for retrieving graphs
- Tests the multi-task indexing format `(task_idx, dataset_idx, local_idx)`
- **Requires**: Plinder data to be available

**`test_graph_construction.py`** — Tests for graph structure and features:
- `TestGraphNodeTypes` — Tests that graphs have correct node types for each task
- `TestGraphEdgeTypes` — Tests that graphs have correct edge types
- `TestGraphLigandFeatures` — Tests for prior/target ligand features (positions, atom types)
- `TestGraphProteinFeatures` — Tests for protein node features
- `TestGraphPriorSampling` — Tests that priors are correctly sampled
- `TestGraphBatching` — Tests for correct graph batching behavior
- **Requires**: Plinder and Pharmit data

**`test_omtra_forward.py`** — Tests for OMTRA model forward pass:
- `TestOMTRAInstantiation` — Tests model instantiation from config
- `TestOMTRAForwardFixedProtein` — Tests forward pass for `fixed_protein_ligand_denovo_condensed`
- `TestOMTRAForwardRigidDocking` — Tests forward pass for `rigid_docking_condensed`
- `TestOMTRAForwardDeNovoLigand` — Tests forward pass for `denovo_ligand_condensed`
- `TestOMTRAConditionalPath` — Tests conditional path sampling
- `TestOMTRATrainingStep` — Tests training step returns valid loss with gradients
- **Requires**: Plinder and Pharmit data

**`test_pharmacophores.py`** — Tests for pharmacophore extraction:
- Compares OMTRA's pharmacophore extraction against the reference Pharmit implementation
- Tests both ligand-only and protein-ligand pharmacophore extraction
- **Requires**: `pharmit` command-line tool to be installed

**`test_plinder_dataset.py`** — Tests for the Plinder dataset:
- Tests dataset initialization and configuration
- Tests dynamic cropping functionality
- Validates graph structure and node types
- **Requires**: Plinder data to be available (see [Data Requirements](#data-requirements))

**`test_vector_field.py`** — Tests for VectorField forward pass and integration:
- `TestVectorFieldInstantiation` — Tests VectorField exists with expected components
- `TestVectorFieldForwardFixedProtein` — Tests forward pass output shapes and values
- `TestVectorFieldForwardRigidDocking` — Tests forward for rigid docking task
- `TestVectorFieldDenoiseGraph` — Tests the denoising graph method
- `TestVectorFieldIntegrate` — Tests ODE integration (sampling)
- `TestVectorFieldStep` — Tests single integration step
- `TestVectorFieldVectorFieldMethod` — Tests ODE velocity computation
- **Requires**: Plinder data

## Data Requirements

Some integration tests require real data to be available:

### Plinder Data

Tests marked with `@pytest.mark.requires_data` require Plinder data. You can:

1. **Use the default location**: Place data at `{OMTRA_ROOT}/data/plinder/`

2. **Use a command-line option**:
   ```bash
   pytest tests/ --plinder-path=/path/to/plinder/data
   ```

3. **Set an environment variable**:
   ```bash
   OMTRA_PLINDER_PATH=/path/to/plinder/data pytest tests/
   ```

4. **Skip data-dependent tests**:
   ```bash
   pytest -m "not requires_data" tests/
   ```

### Pharmit

The `test_pharmacophores.py` tests require the `pharmit` command-line tool to be installed and available in your PATH. See [http://pharmit.csb.pitt.edu/](http://pharmit.csb.pitt.edu/) for installation instructions.

### Pharmit Data

Tests for unconditional tasks like `denovo_ligand_condensed` require the Pharmit dataset. You can:

1. **Use the default location**: Place data at `{OMTRA_ROOT}/data/pharmit/`

2. **Use a command-line option**:
   ```bash
   pytest tests/ --pharmit-path=/path/to/pharmit/data
   ```

3. **Set an environment variable**:
   ```bash
   OMTRA_PHARMIT_PATH=/path/to/pharmit/data pytest tests/
   ```

## Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

### Data Path Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `plinder_path` | session | Path to Plinder data (skips if unavailable) |
| `pharmit_path` | session | Path to Pharmit data (skips if unavailable) |

### Configuration Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `hydra_cfg` | session | Hydra config for `fixed_protein` task group |
| `hydra_cfg_pharmit` | session | Hydra config for `pharmit5050_cond_a` task group |
| `graph_config` | session | Graph configuration extracted from Hydra config |
| `prior_config` | session | Prior configuration extracted from Hydra config |

### Dataset Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `plinder_dataset_factory` | function | Factory for creating `PlinderDataset` instances |
| `plinder_dataset` | function | Default `PlinderDataset` instance (train split, no cropping) |
| `plinder_dataset_with_cropping` | function | `PlinderDataset` with dynamic cropping enabled |
| `datamodule_plinder` | session | `MultiTaskDataModule` for plinder-based tasks |
| `datamodule_pharmit` | session | `MultiTaskDataModule` for pharmit-based tasks |
| `train_dataset_plinder` | session | Train `MultitaskDataSet` for plinder |
| `train_dataset_pharmit` | session | Train `MultitaskDataSet` for pharmit |

### Model Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `omtra_model_plinder` | session | OMTRA model for plinder tasks (CPU) |
| `omtra_model_pharmit` | session | OMTRA model for pharmit tasks (CPU) |

### Sample Batch Fixtures

| Fixture | Description |
|---------|-------------|
| `sample_batch_fixed_protein` | Single graph for `fixed_protein_ligand_denovo_condensed` |
| `sample_batch_rigid_docking` | Single graph for `rigid_docking_condensed` |
| `sample_batch_denovo_ligand` | Single graph for `denovo_ligand_condensed` |
| `sample_batch_multi_fixed_protein` | Batch of 4 graphs for `fixed_protein_ligand_denovo_condensed` |
| `sample_batch_multi_rigid_docking` | Batch of 4 graphs for `rigid_docking_condensed` |

### Task Name Fixtures

| Fixture | Description |
|---------|-------------|
| `test_task_name` | Default task name: `fixed_protein_ligand_denovo_condensed` |
| `plinder_task_name` | Parameterized: `fixed_protein_ligand_denovo_condensed`, `rigid_docking_condensed` |
| `pharmit_task_name` | Task name: `denovo_ligand_condensed` |

## Configuration

Test configuration is managed in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "requires_data: marks tests that require real dataset files (deselect with '-m \"not requires_data\"')",
]
```

### Custom Markers

- **`@pytest.mark.requires_data`** — Tests that require real dataset files. Use `-m "not requires_data"` to skip these tests when data is not available.

## Running Specific Tests

```bash
# Run a specific test file
pytest tests/unit/test_crop_utils.py

# Run a specific test class
pytest tests/unit/test_crop_utils.py::TestCropStructureData

# Run a specific test method
pytest tests/unit/test_crop_utils.py::TestCropStructureData::test_crops_to_close_residues

# Run tests matching a pattern
pytest -k "crop" tests/

# Run tests with coverage report
pytest --cov=omtra tests/
```

## Writing New Tests

### Adding Unit Tests

1. Create a new file in `tests/unit/` named `test_<module_name>.py`
2. Import the functions/classes you want to test
3. Use mock data or synthetic data to test functionality in isolation

```python
"""Unit tests for omtra.module_name"""
import pytest
import numpy as np
from omtra.module_name import function_to_test


class TestFunctionToTest:
    def test_basic_functionality(self):
        """Test the basic case."""
        result = function_to_test(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test an edge case."""
        result = function_to_test(edge_case_input)
        assert result is None
```

### Adding Integration Tests

1. Create a new file in `tests/integration/` named `test_<feature>.py`
2. Use fixtures from `conftest.py` when possible
3. Mark tests that require data with `@pytest.mark.requires_data`

```python
"""Integration tests for feature X"""
import pytest


@pytest.mark.requires_data
class TestFeatureX:
    def test_with_real_data(self, plinder_dataset):
        """Test feature X with real Plinder data."""
        result = feature_x(plinder_dataset)
        assert result is not None
```

### Test Data Files

For tests that need sample input files, place them in `tests/integration/`:
- `test_lig.sdf` — Sample ligand structure
- `test_rec.pdb` — Sample protein structure  
- `test_pharmacophore.xyz` — Sample pharmacophore file

## Continuous Integration

Tests are designed to run in CI environments. For CI pipelines without access to full datasets:

```bash
# Run only tests that don't require data
pytest -m "not requires_data" tests/
```

For full test coverage, ensure the CI environment has:
- Access to Plinder data
- The `pharmit` command-line tool installed

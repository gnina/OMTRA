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
```

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── unit/                 # Fast, isolated unit tests
│   ├── __init__.py
│   └── test_crop_utils.py
└── integration/          # Tests requiring real data or external dependencies
    ├── __init__.py
    ├── test_cli.py
    ├── test_data_module.py
    ├── test_pharmacophores.py
    ├── test_plinder_dataset.py
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

**`test_pharmacophores.py`** — Tests for pharmacophore extraction:
- Compares OMTRA's pharmacophore extraction against the reference Pharmit implementation
- Tests both ligand-only and protein-ligand pharmacophore extraction
- **Requires**: `pharmit` command-line tool to be installed

**`test_plinder_dataset.py`** — Tests for the Plinder dataset:
- Tests dataset initialization and configuration
- Tests dynamic cropping functionality
- Validates graph structure and node types
- **Requires**: Plinder data to be available (see [Data Requirements](#data-requirements))

## Data Requirements

Some integration tests require real data to be available:

### Plinder Data

Tests marked with `@pytest.mark.requires_data` require Plinder data. You can:

1. **Use the default location**: Place data at `{OMTRA_ROOT}/data/plinder/`

2. **Specify a custom location**: Set the `OMTRA_PLINDER_PATH` environment variable:
   ```bash
   OMTRA_PLINDER_PATH=/path/to/plinder/data pytest tests/
   ```

3. **Skip data-dependent tests**:
   ```bash
   pytest -m "not requires_data" tests/
   ```

### Pharmit

The `test_pharmacophores.py` tests require the `pharmit` command-line tool to be installed and available in your PATH. See [http://pharmit.csb.pitt.edu/](http://pharmit.csb.pitt.edu/) for installation instructions.

## Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

| Fixture | Scope | Description |
|---------|-------|-------------|
| `plinder_path` | session | Path to Plinder data (skips if unavailable) |
| `hydra_cfg` | session | Loaded Hydra configuration with Plinder path override |
| `graph_config` | session | Graph configuration extracted from Hydra config |
| `prior_config` | session | Prior configuration extracted from Hydra config |
| `plinder_dataset_factory` | function | Factory for creating `PlinderDataset` instances |
| `plinder_dataset` | function | Default `PlinderDataset` instance (train split, no cropping) |
| `plinder_dataset_with_cropping` | function | `PlinderDataset` with dynamic cropping enabled |
| `test_task_name` | function | The task name used for testing `__getitem__` |

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

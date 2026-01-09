"""
Shared pytest fixtures for OMTRA tests.

To run tests with a custom data path, use command-line options:
    pytest tests/ --plinder-path=/path/to/plinder --pharmit-path=/path/to/pharmit

Or set environment variables:
    OMTRA_PLINDER_PATH=/path/to/plinder/data pytest tests/
    OMTRA_PHARMIT_PATH=/path/to/pharmit/data pytest tests/

Otherwise, tests will use the default paths:
    - Plinder: {omtra_root}/data/plinder
    - Pharmit: {omtra_root}/data/pharmit
"""

import os
import pytest
from pathlib import Path
import torch
import dgl


def pytest_addoption(parser):
    """Add command-line options for data paths."""
    parser.addoption(
        "--plinder-path",
        action="store",
        default=None,
        help="Path to Plinder data directory"
    )
    parser.addoption(
        "--pharmit-path", 
        action="store",
        default=None,
        help="Path to Pharmit data directory"
    )


def get_plinder_path(config=None):
    """Get plinder data path from CLI option, env var, or default location."""
    from omtra.utils import omtra_root
    
    # Check CLI option first
    if config is not None:
        cli_path = config.getoption("--plinder-path")
        if cli_path:
            return Path(cli_path)
    
    # Then check env var
    env_path = os.environ.get("OMTRA_PLINDER_PATH")
    if env_path:
        return Path(env_path)
    
    # Default path relative to omtra root
    return Path(omtra_root()) / "data" / "plinder"


def get_pharmit_path(config=None):
    """Get pharmit data path from CLI option, env var, or default location."""
    from omtra.utils import omtra_root
    
    # Check CLI option first
    if config is not None:
        cli_path = config.getoption("--pharmit-path")
        if cli_path:
            return Path(cli_path)
    
    # Then check env var
    env_path = os.environ.get("OMTRA_PHARMIT_PATH")
    if env_path:
        return Path(env_path)
    
    # Default path relative to omtra root
    return Path(omtra_root()) / "data" / "pharmit"


@pytest.fixture(scope="session")
def plinder_path(request):
    """
    Fixture that returns the path to plinder data.
    Skips test if data is not available.
    """
    path = get_plinder_path(request.config)
    if not path.exists():
        pytest.skip(f"Plinder data not available at {path}. Use --plinder-path or set OMTRA_PLINDER_PATH env var.")
    return path


@pytest.fixture(scope="session")
def pharmit_path(request):
    """
    Fixture that returns the path to pharmit data.
    Skips test if data is not available.
    """
    path = get_pharmit_path(request.config)
    if not path.exists():
        pytest.skip(f"Pharmit data not available at {path}. Use --pharmit-path or set OMTRA_PHARMIT_PATH env var.")
    return path


@pytest.fixture(scope="session")
def hydra_cfg(plinder_path, pharmit_path):
    """
    Load the default Hydra config with plinder and pharmit paths.
    Uses fixed_protein_cond_a task group which includes:
    - fixed_protein_ligand_denovo_condensed
    - rigid_docking_condensed
    - denovo_ligand_condensed
    - ligand_conformer_condensed
    """
    from omtra.load.quick import load_cfg
    
    cfg = load_cfg(
        config_name="config.yaml",
        plinder_path=str(plinder_path),
        pharmit_path=str(pharmit_path),
        overrides=[
            "task_group=fixed_protein_cond_a",
        ]
    )
    return cfg


@pytest.fixture(scope="session")
def hydra_cfg_pharmit(plinder_path, pharmit_path):
    """
    Load Hydra config for pharmit tasks (denovo_ligand_condensed).
    Uses pharmit5050_cond_a task group.
    """
    from omtra.load.quick import load_cfg
    
    cfg = load_cfg(
        config_name="config.yaml",
        plinder_path=str(plinder_path),
        pharmit_path=str(pharmit_path),
        overrides=[
            "task_group=pharmit5050_cond_a",
        ]
    )
    return cfg


@pytest.fixture(scope="session")
def graph_config(hydra_cfg):
    """Extract graph config from Hydra config."""
    return hydra_cfg.graph


@pytest.fixture(scope="session")
def prior_config(hydra_cfg):
    """Extract prior config from Hydra config."""
    return hydra_cfg.prior


@pytest.fixture(scope="function")
def plinder_dataset_factory(plinder_path, graph_config, prior_config):
    """
    Factory fixture for creating PlinderDataset instances.
    """
    from omtra.dataset.plinder import PlinderDataset
    
    def _create(
        split: str = "train",
        link_version: str = None,  # None for tasks that don't use linked structures
        crop_min_distance: float = None,
        crop_max_distance: float = None,
        fake_atom_p: float = 0.0,
        **kwargs
    ):
        return PlinderDataset(
            link_version=link_version,
            split=split,
            processed_data_dir=str(plinder_path),
            graph_config=graph_config,
            prior_config=prior_config,
            fake_atom_p=fake_atom_p,
            crop_min_distance=crop_min_distance,
            crop_max_distance=crop_max_distance,
            **kwargs
        )
    
    return _create


@pytest.fixture(scope="function")
def plinder_dataset(plinder_dataset_factory):
    """Default PlinderDataset instance (train, no cropping)."""
    return plinder_dataset_factory(split="train")


@pytest.fixture(scope="function")
def plinder_dataset_with_cropping(plinder_dataset_factory):
    """PlinderDataset with dynamic cropping enabled."""
    return plinder_dataset_factory(
        split="train",
        crop_min_distance=4.0,
        crop_max_distance=8.0,
    )


TEST_TASK_NAME = "fixed_protein_ligand_denovo_condensed"


@pytest.fixture
def test_task_name():
    """The task name to use for testing __getitem__."""
    return TEST_TASK_NAME


# =============================================================================
# Task names for testing different task types
# =============================================================================

TASK_FIXED_PROTEIN = "fixed_protein_ligand_denovo_condensed"
TASK_RIGID_DOCKING = "rigid_docking_condensed"
TASK_DENOVO_LIGAND = "denovo_ligand_condensed"


@pytest.fixture(params=[TASK_FIXED_PROTEIN, TASK_RIGID_DOCKING])
def plinder_task_name(request):
    """Parameterized fixture for plinder-based tasks."""
    return request.param


@pytest.fixture
def pharmit_task_name():
    """Task name for pharmit-based testing (denovo_ligand_condensed)."""
    return TASK_DENOVO_LIGAND


# =============================================================================
# DataModule fixtures
# =============================================================================

@pytest.fixture(scope="session")
def datamodule_plinder(hydra_cfg):
    """
    MultiTaskDataModule for plinder-based tasks.
    Uses edges_per_batch=20000 for reasonable batch sizes.
    """
    from omtra.load.quick import datamodule_from_config
    
    datamodule = datamodule_from_config(hydra_cfg, edges_per_batch=20000)
    return datamodule


@pytest.fixture(scope="session")
def datamodule_pharmit(hydra_cfg_pharmit):
    """
    MultiTaskDataModule for pharmit-based tasks.
    Uses edges_per_batch=20000 for reasonable batch sizes.
    """
    from omtra.load.quick import datamodule_from_config
    
    datamodule = datamodule_from_config(hydra_cfg_pharmit, edges_per_batch=20000)
    return datamodule


# =============================================================================
# OMTRA Model fixtures
# =============================================================================

@pytest.fixture(scope="session")
def omtra_model_plinder(hydra_cfg, pharmit_path):
    """
    OMTRA model instantiated from config for plinder-based tasks.
    Runs on CPU for testing.
    """
    import hydra as hydra_lib
    from omtra.models.omtra import OMTRA
    from pathlib import Path
    
    dists_file = Path(pharmit_path) / 'train_dists.npz'
    
    model = hydra_lib.utils.instantiate(
        hydra_cfg.model,
        task_phases=hydra_cfg.task_group.task_phases,
        task_dataset_coupling=hydra_cfg.task_group.dataset_task_coupling,
        graph_config=hydra_cfg.graph,
        dists_file=str(dists_file),
        ligand_encoder=hydra_cfg.ligand_encoder,
        _recursive_=False,
        prior_config=hydra_cfg.prior,
        og_run_dir=None,
        eval_config=hydra_cfg.eval,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def omtra_model_pharmit(hydra_cfg_pharmit, pharmit_path):
    """
    OMTRA model instantiated from config for pharmit-based tasks.
    Runs on CPU for testing.
    """
    import hydra as hydra_lib
    from omtra.models.omtra import OMTRA
    from pathlib import Path
    
    dists_file = Path(pharmit_path) / 'train_dists.npz'
    
    model = hydra_lib.utils.instantiate(
        hydra_cfg_pharmit.model,
        task_phases=hydra_cfg_pharmit.task_group.task_phases,
        task_dataset_coupling=hydra_cfg_pharmit.task_group.dataset_task_coupling,
        graph_config=hydra_cfg_pharmit.graph,
        dists_file=str(dists_file),
        ligand_encoder=hydra_cfg_pharmit.ligand_encoder,
        _recursive_=False,
        prior_config=hydra_cfg_pharmit.prior,
        og_run_dir=None,
        eval_config=hydra_cfg_pharmit.eval,
    )
    model.eval()
    return model


# =============================================================================
# Sample batch fixtures
# =============================================================================

@pytest.fixture(scope="session")
def train_dataset_plinder(datamodule_plinder):
    """Load the train MultitaskDataSet for plinder."""
    return datamodule_plinder.load_dataset('train')


@pytest.fixture(scope="session")
def train_dataset_pharmit(datamodule_pharmit):
    """Load the train MultitaskDataSet for pharmit."""
    return datamodule_pharmit.load_dataset('train')


def get_sample_graph(dataset, task_name: str, idx: int = 0):
    """Helper to get a sample graph from a MultitaskDataSet."""
    from omtra.tasks.register import task_name_to_class
    
    task_class = task_name_to_class(task_name)
    
    # Determine which dataset to use based on task
    if 'denovo_ligand' in task_name and 'protein' not in task_name:
        # Pharmit task
        dataset_name = 'pharmit'
        link_version = None
    else:
        # Plinder task
        dataset_name = 'plinder'
        link_version = task_class.plinder_link_version
    
    if dataset_name == 'plinder':
        inner_dataset = dataset.datasets[dataset_name][link_version]
    else:
        inner_dataset = dataset.datasets[dataset_name]
    
    graph = inner_dataset[(task_name, idx)]
    return graph


@pytest.fixture
def sample_batch_fixed_protein(train_dataset_plinder):
    """Single graph batch for fixed_protein_ligand_denovo_condensed task."""
    task_name = TASK_FIXED_PROTEIN
    g = get_sample_graph(train_dataset_plinder, task_name, idx=0)
    return dgl.batch([g]), task_name


@pytest.fixture
def sample_batch_rigid_docking(train_dataset_plinder):
    """Single graph batch for rigid_docking_condensed task."""
    task_name = TASK_RIGID_DOCKING
    g = get_sample_graph(train_dataset_plinder, task_name, idx=0)
    return dgl.batch([g]), task_name


@pytest.fixture
def sample_batch_denovo_ligand(train_dataset_pharmit):
    """Single graph batch for denovo_ligand_condensed task."""
    task_name = TASK_DENOVO_LIGAND
    g = get_sample_graph(train_dataset_pharmit, task_name, idx=0)
    return dgl.batch([g]), task_name


@pytest.fixture
def sample_batch_multi_fixed_protein(train_dataset_plinder):
    """Multiple graph batch (4 graphs) for fixed_protein_ligand_denovo_condensed task."""
    task_name = TASK_FIXED_PROTEIN
    graphs = [get_sample_graph(train_dataset_plinder, task_name, idx=i) for i in range(4)]
    return dgl.batch(graphs), task_name


@pytest.fixture
def sample_batch_multi_rigid_docking(train_dataset_plinder):
    """Multiple graph batch (4 graphs) for rigid_docking_condensed task."""
    task_name = TASK_RIGID_DOCKING
    graphs = [get_sample_graph(train_dataset_plinder, task_name, idx=i) for i in range(4)]
    return dgl.batch(graphs), task_name

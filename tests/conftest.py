"""
Shared pytest fixtures for OMTRA tests.

To run tests with a custom data path, set the environment variable:
    OMTRA_PLINDER_PATH=/path/to/plinder/data pytest tests/

Otherwise, tests will use the default path: {omtra_root}/data/plinder
"""

import os
import pytest
from pathlib import Path


def get_plinder_path():
    """Get plinder data path from env var or default location."""
    from omtra.utils import omtra_root
    
    env_path = os.environ.get("OMTRA_PLINDER_PATH")
    if env_path:
        return Path(env_path)
    
    # Default path relative to omtra root
    return Path(omtra_root()) / "data" / "plinder"


@pytest.fixture(scope="session")
def plinder_path():
    """
    Fixture that returns the path to plinder data.
    Skips test if data is not available.
    """
    path = get_plinder_path()
    if not path.exists():
        pytest.skip(f"Plinder data not available at {path}. Set OMTRA_PLINDER_PATH env var.")
    return path


@pytest.fixture(scope="session")
def hydra_cfg(plinder_path):
    """
    Load the default Hydra config with plinder path override.
    """
    from omtra.load.quick import load_cfg
    
    cfg = load_cfg(
        config_name="config.yaml",
        plinder_path=str(plinder_path),
        overrides=[
            "task_group=fixed_protein",
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

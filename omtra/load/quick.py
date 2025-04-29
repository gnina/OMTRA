from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, DictConfig
from omtra.utils import omtra_root
from omtra.load.conf import merge_task_spec
from omtra.dataset.data_module import MultiTaskDataModule
from omtra.models.omtra import OMTRA
import hydra
from pathlib import Path
from typing import List

OmegaConf.register_new_resolver("omtra_root", omtra_root, replace=True)


def load_cfg(config_dir: str = None, config_name: str = "config.yaml", pharmit_path: str = None, plinder_path: str = None, overrides: List[str] = []):
    """
    Load the configuration from the specified directory and file name.
    
    Args:
        config_dir (str): The directory containing the configuration files.
        config_name (str): The name of the configuration file (without extension).
    
    Returns:
        cfg: The loaded configuration object.
    """

    # Absolute or relative path to your config directory
    if config_dir is None:
        config_dir = Path(omtra_root()) / 'configs'
    else:
        config_dir = Path(config_dir)


    if pharmit_path is not None:
        overrides.append(f'pharmit_path={pharmit_path}')
    if plinder_path is not None:
        overrides.append(f'plinder_path={plinder_path}')
    
    if len(overrides) == 0:
        overrides = None

    # Initialize Hydra and compose the config
    # can odd overrides = [ list of string overrides]
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides)

    cfg = merge_task_spec(cfg)

    return cfg


def datamodule_from_config(cfg: DictConfig) -> MultiTaskDataModule:

    print(f"⚛ Instantiating datamodule <{cfg.task_group.datamodule._target_}>")

    # load data module
    datamodule: MultiTaskDataModule = hydra.utils.instantiate(
        cfg.task_group.datamodule, 
    )

    return datamodule

def omtra_from_config(cfg: DictConfig) -> OMTRA:

    print(f"⚛ Instantiating model <{cfg.model._target_}>")

    dists_file = Path(cfg.pharmit_path) / 'train_dists.npz'

    # TODO: there is al ligand_encoder and we are training OMTRA, we need a ligand_encoder_checkpoint argument
    # somewhere we need to check that this is the case and throw an error if not, perhaps in the train script?

    # load model
    model = hydra.utils.instantiate(
        cfg.model,
        task_phases=cfg.task_group.task_phases,
        task_dataset_coupling=cfg.task_group.dataset_task_coupling,
        graph_config=cfg.graph,
        dists_file=dists_file,
        ligand_encoder=cfg.ligand_encoder,
        _recursive_=False,
        prior_config=cfg.prior,
    )

    return model

def lig_encoder_from_config(cfg: DictConfig):

    # check that ligand_encoder is set
    if cfg.ligand_encoder is None:
        raise ValueError("Ligand encoder is not set in the config. Please set it to a valid model.")

    print(f"⚛ Instantiating Ligand Encoder <{cfg.ligand_encoder._target_}>")

    # load model
    lig_encoder = hydra.utils.instantiate(cfg.ligand_encoder)

    return lig_encoder
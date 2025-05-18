from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf, DictConfig
from omtra.utils import omtra_root
from omtra.load.conf import merge_task_spec
from omtra.dataset.data_module import MultiTaskDataModule
from omtra.models.omtra import OMTRA
import hydra
from pathlib import Path
from typing import List, Optional
import torch

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

def load_trained_model_cfg(cfg_path: str):

    cfg = OmegaConf.load(cfg_path)
    cfg = merge_task_spec(cfg)
    return cfg


def datamodule_from_config(cfg: DictConfig, **kwargs) -> MultiTaskDataModule:

    print(f"⚛ Instantiating datamodule <{cfg.task_group.datamodule._target_}>")

    # load data module
    datamodule: MultiTaskDataModule = hydra.utils.instantiate(
        cfg.task_group.datamodule, 
        **kwargs,
    )

    return datamodule

def omtra_from_config(cfg: DictConfig) -> OMTRA:

    print(f"⚛ Instantiating model <{cfg.model._target_}>")

    dists_file = Path(cfg.pharmit_path) / 'train_dists.npz'

    # TODO: there is al ligand_encoder and we are training OMTRA, we need a ligand_encoder_checkpoint argument
    # somewhere we need to check that this is the case and throw an error if not, perhaps in the train script?

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']

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
        og_run_dir=log_dir,
    )

    return model

def omtra_from_checkpoint(ckpt_path: str) -> OMTRA:
    """
    Load the OMTRA model from a checkpoint.
    
    Args:
        ckpt_path (str): Path to the checkpoint file.
    
    Returns:
        model: The loaded OMTRA model.
    """

    # load model
    model = OMTRA.load_from_checkpoint(ckpt_path)

    return model

def lig_encoder_from_config(cfg: DictConfig):

    # check that ligand_encoder is set
    if cfg.ligand_encoder is None:
        raise ValueError("Ligand encoder is not set in the config. Please set it to a valid model.")

    print(f"⚛ Instantiating Ligand Encoder <{cfg.ligand_encoder._target_}>")

    # load model
    lig_encoder = hydra.utils.instantiate(cfg.ligand_encoder)

    return lig_encoder

def omtra_from_partial_checkpoint(cfg: DictConfig, ckpt_path: str, secondary_ckpt_path: Optional[str] = None) -> OMTRA:
    """
    Instantiate an OMTRA model from config, and load compatible weights from primary and optional secondary checkpoint.
    Warning: vector field config probably has to be the same across checkpoints
    
    Args:
        cfg: Full config including all tasks.
        ckpt_path: Path to the primary checkpoint.
        secondary_ckpt_path: Optional path to a secondary checkpoint to fill in other missing weights.
    
    Returns:
        model: The frankenstein OMTRA model.
    """
    model = omtra_from_config(cfg)

    ckpt1 = torch.load(ckpt_path, map_location="cpu")
    state_dict1 = ckpt1["state_dict"] if "state_dict" in ckpt1 else ckpt1

    excluded_keys = {"vector_field.task_embedding.weight"}
    state_dict1 = {k: v for k, v in state_dict1.items() if k not in excluded_keys}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict1, strict=False)
    
    print(f"Loaded from primary checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing_keys)}")

    if secondary_ckpt_path is not None:
        ckpt2 = torch.load(secondary_ckpt_path, map_location="cpu")
        state_dict2 = ckpt2["state_dict"] if "state_dict" in ckpt2 else ckpt2

        supplemental_state_dict = {
            k: v for k, v in state_dict2.items()
            if k in missing_keys and k not in excluded_keys
        }

        still_missing, still_unexpected = model.load_state_dict(supplemental_state_dict, strict=False)

        print(f"Loaded from secondary checkpoint: {secondary_ckpt_path}")
        print(f"Remaining missing keys: {len(still_missing)}")

    return model

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


def load_cfg(config_dir: str = None, config_name: str = "config.yaml", pharmit_path: str = None, plinder_path: str = None, crossdocked_path: str = None, overrides: List[str] = []):
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
    if crossdocked_path is not None:
        overrides.append(f'crossdocked_path={crossdocked_path}')
    
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

    # Use og_run_dir from config if set (for resume), otherwise use hydra output dir
    if cfg.get('og_run_dir') is not None:
        log_dir = cfg.og_run_dir
    else:
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
        eval_config=cfg.eval,
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
    # Force weights_only=False regardless of what Lightning passes — checkpoints
    # contain OmegaConf types that are not in PyTorch 2.6's safe globals list.
    _orig = torch.load
    def _load_unsafe(*args, **kwargs):
        kwargs['weights_only'] = False
        return _orig(*args, **kwargs)
    torch.load = _load_unsafe
    try:
        model = OMTRA.load_from_checkpoint(ckpt_path)
    finally:
        torch.load = _orig

    return model

def lig_encoder_from_config(cfg: DictConfig):

    # check that ligand_encoder is set
    if cfg.ligand_encoder is None:
        raise ValueError("Ligand encoder is not set in the config. Please set it to a valid model.")

    print(f"⚛ Instantiating Ligand Encoder <{cfg.ligand_encoder._target_}>")

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    log_dir = hydra_cfg['runtime']['output_dir']

    # load model
    lig_encoder = hydra.utils.instantiate(cfg.ligand_encoder, og_run_dir=log_dir)

    return lig_encoder


def is_primary_process():
    """True in the process that should write run-level files.

    Called while the model is built, i.e. before Lightning attaches a trainer, so `model.global_rank` is 0 in EVERY DDP process
    (this made all four ranks write initialization_manifest.json and ranks 1-3 crash with FileExistsError). The subprocess DDP
    launcher gives every child process its LOCAL_RANK; the original process has none.
    """
    import os
    for key in ('LOCAL_RANK', 'RANK'):
        if os.environ.get(key) not in (None, ''):
            return int(os.environ[key]) == 0
    return True

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

    ckpt1 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict1 = ckpt1["state_dict"] if "state_dict" in ckpt1 else ckpt1

    from omtra.load.weights import initialize_weights
    import json
    manifest_path = cfg.get('partial_load_expected_manifest')
    if cfg.get('strict_partial_load', False) and not manifest_path:
        raise ValueError('Strict initialization requires partial_load_expected_manifest; provide a reviewed manifest')
    expected = json.loads(Path(manifest_path).read_text()) if manifest_path else None
    manifest = initialize_weights(model, state_dict1, expected=expected,
        allow_mismatch=cfg.get('partial_load_allow_mismatch', []),
        excluded=tuple(cfg.get('partial_load_exclude', ['vector_field.task_embedding.weight'])))
    hash_path=cfg.get('partial_load_expected_tensor_hashes')
    if hash_path:
        hashes=json.loads(Path(hash_path).read_text())
        for key in ('loaded_tensor_sha256','new_initialization_sha256'):
            if manifest[key]!=hashes[key]:raise ValueError(f'Frozen common initialization differs: {key}')
    if secondary_ckpt_path is not None:
        ckpt2 = torch.load(secondary_ckpt_path, map_location='cpu', weights_only=False)
        state2 = ckpt2.get('state_dict', ckpt2)
        supplement = {k: v for k, v in state2.items() if k in manifest['new'] and k != 'vector_field.task_embedding.weight'}
        if any(model.state_dict()[k].shape != v.shape for k, v in supplement.items()):
            raise ValueError('Secondary checkpoint shape mismatch')
        model.load_state_dict(supplement, strict=False)
        model.initialization_manifest['secondary_loaded'] = sorted(supplement)
    if is_primary_process() and model.og_run_dir:
        output = Path(model.og_run_dir) / 'initialization_manifest.json'
        if output.exists():
            raise FileExistsError(output)
        with output.open('x') as stream:
            json.dump(model.initialization_manifest, stream, indent=2)
    print(f"Loaded {len(manifest['loaded'])} exact keys; {len(manifest['new'])} target keys initialized afresh")
    return model

from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
from omtra.utils import omtra_root
from typing import List
import hydra

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

def merge_task_spec(cfg: DictConfig) -> DictConfig:
    """
    Load the single dataset configurations for the single datasets that are specified in the task_group configuration.
    """
    
    # Folder where single dataset YAML files are stored.
    single_datasets_dir = Path(omtra_root()) / 'configs' / 'task_group' / 'datamodule' / 'single_datasets'
    
    # infer the datasets specified in the task group
    dataset_task_coupling = cfg.task_group.dataset_task_coupling
    datasets = set()
    for task in dataset_task_coupling:
        task_datasets = [ c[0] for c in dataset_task_coupling[task] ]
        datasets.update(task_datasets)


    with open_dict(cfg.task_group.datamodule):
        for ds_name in datasets:
            ds_path = single_datasets_dir / f"{ds_name}.yaml"
            if not ds_path.exists():
                raise ValueError(f"Invalid dataset specified: {ds_path}")
            ds_cfg = OmegaConf.load(ds_path)
            cfg.task_group.datamodule.dataset_config.single_dataset_configs[ds_name] = ds_cfg
    return cfg

def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []
    if rank_zero_only.rank != 0:
        return callbacks

    if not callbacks_cfg:
        print("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            print(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
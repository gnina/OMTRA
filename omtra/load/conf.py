from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
from omtra.utils import omtra_root

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
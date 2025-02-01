import hydra

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from omtra.dataset.data_module import MultiTaskDataModule
from omtra.utils import omtra_root

# register the omtra_root resolver so that anything in a config file
# with ${omtra_root:} will be replaced with the root path of the omtra package
OmegaConf.register_new_resolver("omtra_root", omtra_root, replace=True)

def train(cfg: DictConfig):
    """Trains the model.

    cfg is a DictConfig configuration composed by Hydra.
    """
    # set seed everywhere (pytorch, numpy, python)
    pl.seed_everything(cfg.seed, workers=True)

    print(f"⚛ Instantiating datamodule <{cfg.task_group.data._target_}>")
    datamodule: MultiTaskDataModule = hydra.utils.instantiate(
        cfg.task_group.data, 
        graph_config=cfg.graph)

    datamodule.setup(stage='fit')
    # TODO: load dataloader
    # TODO: turn datamodule instantiation and dataloader test into unit tests
    dataloader = datamodule.train_dataloader()
    n_batches = 2
    for _ in range(n_batches):
        batch = next(iter(dataloader))


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for training.

    cfg is a DictConfig configuration composed by Hydra.
    """
    print("\n=== Training Config ===")
    print(OmegaConf.to_yaml(cfg))

    # train the model
    _ = train(cfg)

if __name__ == "__main__":
    main()
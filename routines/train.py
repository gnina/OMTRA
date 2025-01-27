import hydra

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from omegaconf import DictConfig, OmegaConf

def train(cfg: DictConfig):
    """Trains the model.

    cfg is a DictConfig configuration composed by Hydra.
    """
    # set seed everywhere (pytorch, numpy, python)
    pl.seed_everything(cfg.seed, workers=True)

    print(f"⚛ Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # TODO: Implmenet other instantiation logic


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
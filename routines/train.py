import hydra

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from omtra.dataset.data_module import MultiTaskDataModule
from omtra.load.conf import merge_task_spec
from omtra.utils import omtra_root
import torch.multiprocessing as mp
import multiprocessing
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

multiprocessing.set_start_method('spawn', force=True)
mp.set_start_method("spawn", force=True)

# register the omtra_root resolver so that anything in a config file
# with ${omtra_root:} will be replaced with the root path of the omtra package
OmegaConf.register_new_resolver("omtra_root", omtra_root, replace=True)

def train(cfg: DictConfig):
    """Trains the model.

    cfg is a DictConfig configuration composed by Hydra.
    """
    # set seed everywhere (pytorch, numpy, python)
    pl.seed_everything(cfg.seed, workers=True)

    print(f"⚛ Instantiating datamodule <{cfg.task_group.datamodule._target_}>")
    datamodule: MultiTaskDataModule = hydra.utils.instantiate(
        cfg.task_group.datamodule, 
        graph_config=cfg.graph)

    
    print(f"⚛ Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    wandb_config = cfg.wandb
    # TODO: flowmol sets save dir in wandb config
    # there is some complicated logic around instantiation of the output dir for flowmol
    # this logic has to do with gpu rank as well
    # need to remember what this logic is and whether we need it here
    # our situation here may differ slightly because we don't need to come up with a logging directory
    # hydra should create it for us?? how do we get access to the logging directory anyways?
    wandb_logger = WandbLogger(
        config=cfg,
        save_dir=cfg.hydra.run.dir,  # ensures logs are stored with the Hydra output dir
        **wandb_config
    )


    trainer = pl.Trainer(
        logger=wandb_logger, 
        **cfg.trainer, 
        callbacks=[checkpoint_callback, pbar_callback])


    # datamodule.setup(stage='fit')
    # dataloader = datamodule.train_dataloader()
    # dataloader_iter = iter(dataloader)
    # n_batches = 5
    # for _ in range(n_batches):
    #     g, task_name, dataset_name = next(dataloader_iter)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for training.

    cfg is a DictConfig configuration composed by Hydra.
    """
    cfg = merge_task_spec(cfg)


    print("\n=== Training Config ===")
    print(OmegaConf.to_yaml(cfg))

    # train the model
    _ = train(cfg)

if __name__ == "__main__":
    # TODO: do we need both of these? perhaps not
    main()
import hydra
import os
from typing import List
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from omtra.dataset.data_module import MultiTaskDataModule
from omtra.load.conf import merge_task_spec, instantiate_callbacks
from omtra.utils import omtra_root
import torch.multiprocessing as mp
import multiprocessing
from pathlib import Path
import wandb

import pytorch_lightning as pl
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

    # figure out if we are resuming a previous run
    resume = cfg.get("ckpt_path") is not None


    wandb_config = cfg.wandb
    if resume:
        # if we are resuming, we need to read the run_id from the resume_info.yaml file
        resume_info_file = Path(cfg.hydra.run.dir) / 'resume_info.yaml'
        with open(resume_info_file, 'r') as f:
            resume_info = OmegaConf.load(f)
        run_id = resume_info.run_id
        wandb_config.resume = 'must'
    else:
        # otherwise, we generate a new run_id
        run_id = wandb.util.generate_id()
        

    wandb_logger = WandbLogger(
        config=cfg,
        save_dir=cfg.hydra.run.dir,  # ensures logs are stored with the Hydra output dir
        run_id=run_id,
        **wandb_config
    )

    if not resume and rank_zero_only.rank == 0:
        # if this is a fresh run, we need to get the run_id and name from wandb
        # we use this info to create the resume_info.yaml file
        # and also to create a symlink in the symlink_dir that will
        # make it easy for us to lookup output directories just from wandb names
        wandb_logger.experiment # this triggers the creation of the wandb run
        run_id = wandb_logger.experiment.id
        resume_info = {}
        resume_info["run_id"] = run_id
        resume_info["name"] = wandb_logger.experiment.name

        # write resume info as yaml file to run directory
        resume_info_file = Path(cfg.hydra.run.dir) / "resume_info.yaml"
        with open(resume_info_file, "w") as f:
            OmegaConf.save(resume_info, f)

        # create symlink in symlink_dir
        symlink_dir = Path(cfg.symlink_dir)
        symlink_path = symlink_dir / f"{wandb_logger.experiment.name}_{run_id}"
        os.symlink(cfg.hydra.run.dir, symlink_path)

    # instantiate callbacks
    callbacks: List[pl.Callback] = instantiate_callbacks(cfg.callbacks)


    trainer = pl.Trainer(
        logger=wandb_logger, 
        **cfg.trainer, 
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


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

    resume = cfg.get('ckpt_path') is not None
    if resume:
        ckpt_path = Path(cfg.ckpt_path)
        run_dir = ckpt_path.parent.parent
        original_cfg_path = run_dir / 'config.yaml'
        cfg = OmegaConf.load(original_cfg_path)
        cfg.ckpt_path = str(ckpt_path)
    else:
        cfg = merge_task_spec(cfg)


    print("\n=== Training Config ===")
    print(OmegaConf.to_yaml(cfg))

    # train the model
    _ = train(cfg)

if __name__ == "__main__":
    # TODO: do we need both of these? perhaps not
    main()
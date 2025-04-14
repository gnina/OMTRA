import hydra
import os
from typing import List
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig

from omtra.dataset.data_module import MultiTaskDataModule
from omtra.load.conf import merge_task_spec, instantiate_callbacks
from omtra.utils import omtra_root
import torch.multiprocessing as mp
import multiprocessing
from pathlib import Path
import wandb

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy

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
        # graph_config=cfg.graph,
        # prior_config=cfg.prior
    )

    # get dists file from pharmit dir
    # TODO: this is bad as it requires pharmit dataset to be in place
    dists_file = Path(cfg.pharmit_path) / 'train_dists.npz'

    
    print(f"⚛ Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model,
                                    task_phases=cfg.task_group.task_phases,
                                    task_dataset_coupling=cfg.task_group.dataset_task_coupling,
                                    graph_config=cfg.graph,
                                    dists_file=dists_file,
                                )
    
    # figure out if we are resuming a previous run
    resume = cfg.get("ckpt_path") is not None


    wandb_config = cfg.wandb_conf
    if resume:
        # if we are resuming, we need to read the run_id from the resume_info.yaml file
        # we also set the run dir to be the previous run directory
        run_dir = Path(cfg.og_run_dir)
        resume_info_file = run_dir / 'resume_info.yaml'
        with open(resume_info_file, 'r') as f:
            resume_info = OmegaConf.load(f)
        run_id = resume_info.run_id
        wandb_config.resume = 'must'
    else:
        # get the run directory from hydra
        run_dir = HydraConfig.get().runtime.output_dir
        run_dir = Path(run_dir)
        # generate a new run_id
        run_id = wandb.util.generate_id()
        

    wandb_logger = WandbLogger(
        name=cfg['name'],
        config=OmegaConf.to_container(cfg, resolve=True),
        # save_dir=run_dir,  # ensures logs are stored with the Hydra output dir
        id=run_id,
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
        resume_info_file = run_dir / "resume_info.yaml"
        with open(resume_info_file, "w") as f:
            OmegaConf.save(resume_info, f)

        # create symlink in symlink_dir
        symlink_dir = Path(cfg.symlink_dir)
        if not symlink_dir.exists():
            symlink_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = symlink_dir / f"{wandb_logger.experiment.name}_{run_id}"
        os.symlink(run_dir, symlink_path)

    # instantiate callbacks
    if resume:
        override_dir = run_dir
    else:
        override_dir = None
    callbacks: List[pl.Callback] = instantiate_callbacks(cfg.callbacks, override_dir=override_dir)

    if cfg.trainer.get("devices", 1) > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"
        
    trainer = pl.Trainer(
        logger=wandb_logger,
        strategy=strategy,
        **cfg.trainer,
        callbacks=callbacks,
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
        cfg.og_run_dir = str(run_dir)
    else:
        cfg = merge_task_spec(cfg)


    print("\n=== Training Config ===")
    print(OmegaConf.to_yaml(cfg))

    # train the model
    _ = train(cfg)

if __name__ == "__main__":
    # TODO: do we need both of these? perhaps not
    main()
import hydra
import os
from typing import List
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig

from omtra.dataset.data_module import MultiTaskDataModule
from omtra.load.conf import merge_task_spec, instantiate_callbacks
import omtra.load.quick as quick_load
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
torch.multiprocessing.set_sharing_strategy('file_system')

# def configure_tensor_cores(precision: str = 'medium'):
#     """
#     Checks if a CUDA device with Tensor Cores (compute capability >= 8.0)
#     is available, and if so, sets the float32 matmul precision accordingly.
#     """
#     if torch.cuda.is_available():
#         props = torch.cuda.get_device_properties(torch.cuda.current_device())
#         if props.major >= 8:
#             torch.set_float32_matmul_precision(precision)
#             print(f"Enabled TF32 on device (compute capability {props.major}.{props.minor})")
#         else:
#             print(f"TF32 not supported (compute capability {props.major}.{props.minor})")
#     else:
#         print("CUDA not available; skipping TF32 configuration")

def train(cfg: DictConfig):
    """Trains the model.

    cfg is a DictConfig configuration composed by Hydra.
    """
    # set seed everywhere (pytorch, numpy, python)

    # Run at start
    # configure_tensor_cores()

    pl.seed_everything(cfg.seed, workers=True)


    mode = cfg.get("mode", None)
    if mode is None:
        raise ValueError("mode must specify a training mode (e.g. 'omtra', 'ligand_encoder', 'protein_encoder', etc.)")
    
    # load datamodule
    datamodule = quick_load.datamodule_from_config(cfg)

    # load model
    if mode == 'omtra':
        lig_encoder_empty = cfg.ligand_encoder.is_empty()
        lig_enc_ckpt_specified = cfg.model.get('ligand_encoder_checkpoint') is not None
        if not lig_encoder_empty and not lig_enc_ckpt_specified:
            raise ValueError("ligand_encoder_checkpoint must be specified if omtra is doing latent ligand generation")
        
        partial_ckpt = cfg.get('partial_ckpt')
        if partial_ckpt:
            model = quick_load.omtra_from_partial_checkpoint(cfg, partial_ckpt)
        else:
            model = quick_load.omtra_from_config(cfg)
    elif mode == 'ligand_encoder':
        model = quick_load.lig_encoder_from_config(cfg)
    else:
        raise ValueError(f"mode {mode} not recognized, must be 'omtra' or 'ligand_encoder'")
    
    # figure out if we are resuming a previous run
    resume = cfg.get("checkpoint") is not None

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
        # name=cfg['name'],
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
        resume_info["url"] = wandb_logger.experiment.url

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
        
    trainer = pl.Trainer(
        logger=wandb_logger,
        **cfg.trainer,
        callbacks=callbacks,
    )
    
    torch.cuda.empty_cache()
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("checkpoint"))

    # right after training ends:
    if trainer.is_global_zero:
        log_dir = trainer.lightning_module.og_run_dir
        checkpoint_dir = Path(log_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_dir / "last.ckpt"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for training.

    cfg is a DictConfig configuration composed by Hydra.
    """

    resume = cfg.get("checkpoint") is not None
    if resume:
        ckpt_path = Path(cfg.ckpt_path)
        run_dir = ckpt_path.parent.parent
        original_cfg_path = run_dir / "config.yaml"
        original_cfg = OmegaConf.load(original_cfg_path)
        # Only apply CLI overrides to the original config
        overrides = HydraConfig.get().overrides.task
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(original_cfg, cli_cfg)
        cfg.ckpt_path = str(ckpt_path)
        cfg.og_run_dir = str(run_dir)
    else:
        cfg = merge_task_spec(cfg)

    # train the model
    _ = train(cfg)

if __name__ == "__main__":
    # TODO: do we need both of these? perhaps not
    main()
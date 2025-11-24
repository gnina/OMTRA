#!/usr/bin/env python
"""
Multitask GVP Classifier Training Script
Accepts hyperparameters as command-line arguments for SLURM job arrays.
"""

import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np

# Add OMTRA to path
sys.path.insert(0, '/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA')
sys.path.insert(0, '/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/chem_space_classifier/models')
sys.path.insert(0, '/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/chem_space_classifier/data_loading')

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    TQDMProgressBar,
    EarlyStopping
)

# OMTRA
import omtra.load.quick as quick_load
from omtra.load.quick import datamodule_from_config

# Multitask modules
from multitask_lightning_module import MultitaskLightningModule
from multitask_streaming_sampler_fixed import create_multitask_streaming_dataloader_fixed

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train multitask GVP classifier')

    # Experiment identification
    parser.add_argument('--exp_name', type=str, required=True,
                        help='Experiment name (e.g., "lr_1e-4_dropout_0.2")')
    parser.add_argument('--exp_id', type=int, required=True,
                        help='Experiment ID (for SLURM array task ID)')

    # Model architecture (simplified - no changes)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--n_vec_channels', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--shared_repr_dim', type=int, default=128)
    parser.add_argument('--task_hidden_dim', type=int, default=64)

    # Regularization (KEY PARAMETER TO TUNE)
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (0.1, 0.2, 0.3, 0.4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (1e-6, 1e-5, 1e-4)')

    # Learning rate (KEY PARAMETER TO TUNE)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (5e-5, 1e-4, 2e-4)')

    # Class balancing (KEY PARAMETER TO TUNE)
    parser.add_argument('--pos_weight_strategy', type=str, default='inverse_freq',
                        choices=['inverse_freq', 'sqrt_inverse_freq', 'none'],
                        help='Pos weight strategy: inverse_freq=(1-r)/r, sqrt_inverse_freq=sqrt((1-r)/r), none=1.0')
    parser.add_argument('--molport_task_weight', type=float, default=2.0,
                        help='Task weight for MolPort (1.0, 2.0, 3.0)')

    # Training settings
    parser.add_argument('--edges_per_batch', type=int, default=200_000)
    parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--max_batches_train', type=int, default=10_000)
    parser.add_argument('--max_batches_val', type=int, default=1_000)

    # Hardware
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)  # Changed to 0 to avoid RAM exhaustion with pin_memory

    # Paths
    parser.add_argument('--pharmit_path', type=str,
                        default='/net/dali/home/mscbio/mag1037/work/rotations/koes/datasets/pharmit_by_pattern')
    parser.add_argument('--cache_dir', type=str,
                        default='/net/dali/home/mscbio/mag1037/work/rotations/koes/multitask_cache')
    parser.add_argument('--output_dir', type=str,
                        default='/net/dali/home/mscbio/mag1037/work/rotations/koes/OMTRA/chem_space_classifier/outputs')

    # Wandb
    parser.add_argument('--wandb_project', type=str, default='omtra-multitask-tuning')
    parser.add_argument('--wandb_api_key', type=str,
                        default='b32b90b1572db9356cdfe74709ba49e3c21dfad7')
    parser.add_argument('--wandb_entity', type=str, default=None)

    # Checkpoint resumption
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from (use "last" to auto-find last.ckpt)')

    return parser.parse_args()


def compute_pos_weights(pos_ratios, strategy='inverse_freq'):
    """
    Compute pos_weights for BCEWithLogitsLoss.

    Args:
        pos_ratios: List of positive class ratios for each task
        strategy: 'inverse_freq', 'sqrt_inverse_freq', or 'none'

    Returns:
        List of pos_weight values
    """
    pos_ratios = np.array(pos_ratios)

    if strategy == 'inverse_freq':
        # Original: (1 - r) / r
        pos_weights = (1 - pos_ratios) / pos_ratios
    elif strategy == 'sqrt_inverse_freq':
        # Less aggressive: sqrt((1 - r) / r)
        pos_weights = np.sqrt((1 - pos_ratios) / pos_ratios)
    elif strategy == 'none':
        # No reweighting
        pos_weights = np.ones_like(pos_ratios)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return pos_weights.tolist()


def main():
    """Main training function."""
    args = parse_args()

    # SLURM handles GPU assignment via CUDA_VISIBLE_DEVICES automatically
    # Don't override it!

    # Set Wandb API key
    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    import wandb
    wandb.login()

    # Create output directory
    exp_dir = Path(args.output_dir) / f"exp_{args.exp_id:03d}_{args.exp_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"EXPERIMENT: {args.exp_name} (ID: {args.exp_id})")
    print("="*80)

    # ============================================================================
    # Compute hyperparameters
    # ============================================================================

    # Positive class ratios (from data analysis)
    pos_ratios = [0.1991, 0.2995, 0.5022, 0.2087, 0.0407]  # CSC, MCULE, PubChem, ZINC, MolPort

    # Compute pos_weights based on strategy
    pos_weights = compute_pos_weights(pos_ratios, args.pos_weight_strategy)

    # Task weights (only MolPort is variable)
    task_weights = [1.0, 1.0, 1.0, 1.0, args.molport_task_weight]

    print(f"\nClass Balancing:")
    print(f"  Strategy: {args.pos_weight_strategy}")
    print(f"  Pos weights: {[f'{w:.2f}' for w in pos_weights]}")
    print(f"  Task weights: {task_weights}")
    print(f"  MolPort effective weight: {pos_weights[4] * task_weights[4]:.2f}x")

    # Build full hyperparams dict
    hyperparams = {
        # Model architecture (fixed)
        'n_atom_types': 118,
        'n_charge_types': 10,
        'n_bond_types': 4,
        'hidden_dim': args.hidden_dim,
        'edge_dim': args.edge_dim,
        'n_vec_channels': args.n_vec_channels,
        'num_layers': args.num_layers,
        'rbf_dim': 32,
        'rbf_dmax': 20.0,
        'shared_repr_dim': args.shared_repr_dim,
        'task_hidden_dim': args.task_hidden_dim,
        'readout': 'mean',

        # Regularization (TUNING TARGET)
        'dropout': args.dropout,
        'weight_decay': args.weight_decay,

        # Learning rate (TUNING TARGET)
        'lr': args.lr,

        # Loss weights (TUNING TARGET)
        'task_weights': task_weights,
        'pos_ratios': pos_ratios,
        'pos_weight_strategy': args.pos_weight_strategy,
        'molport_task_weight': args.molport_task_weight,

        # Training
        'edges_per_batch': args.edges_per_batch,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'epochs': args.max_epochs,
        'num_workers': args.num_workers,
        'pin_memory': False,  # Disabled to avoid RAM exhaustion
        'max_batches_train': args.max_batches_train,
        'max_batches_val': args.max_batches_val,
        'cache_dir': args.cache_dir,
    }

    # Override pos_weights if using sqrt or none
    if args.pos_weight_strategy != 'inverse_freq':
        hyperparams['pos_weights_override'] = pos_weights

    # Save hyperparams
    with open(exp_dir / 'hyperparams.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)

    print(f"\nHyperparameters saved to: {exp_dir / 'hyperparams.json'}")

    # ============================================================================
    # Load datasets
    # ============================================================================

    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)

    cfg = quick_load.load_cfg(
        overrides=['task_group=pharmit5050_cond_a'],
        pharmit_path=args.pharmit_path
    )
    datamodule = datamodule_from_config(cfg)
    train_dataset = datamodule.load_dataset("train")
    val_dataset = datamodule.load_dataset("val")

    pharmit_train_dataset = train_dataset.datasets['pharmit']
    pharmit_val_dataset = val_dataset.datasets['pharmit']

    print(f"Train dataset: {len(pharmit_train_dataset):,} molecules")
    print(f"Val dataset: {len(pharmit_val_dataset):,} molecules")

    # ============================================================================
    # Create dataloaders
    # ============================================================================

    print("\n" + "="*80)
    print("CREATING DATALOADERS")
    print("="*80)

    train_loader = create_multitask_streaming_dataloader_fixed(
        dataset=pharmit_train_dataset,
        edges_per_batch=args.edges_per_batch,
        max_batches=args.max_batches_train,
        shuffle=True,
        num_workers=args.num_workers,
        min_capacity_utilization=0.8,
        use_cache=True,
        cache_dir=args.cache_dir
    )

    val_loader = create_multitask_streaming_dataloader_fixed(
        dataset=pharmit_val_dataset,
        edges_per_batch=args.edges_per_batch,
        max_batches=args.max_batches_val,
        shuffle=False,
        num_workers=args.num_workers,
        min_capacity_utilization=0.8,
        use_cache=True,
        cache_dir=args.cache_dir
    )

    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")

    # ============================================================================
    # Create model
    # ============================================================================

    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)

    model = MultitaskLightningModule(hyperparams)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # ============================================================================
    # Setup trainer
    # ============================================================================

    print("\n" + "="*80)
    print("SETUP TRAINER")
    print("="*80)

    # Initialize Wandb
    wandb_tags = [
        f'exp_{args.exp_id:03d}',
        f'pos_weight_{args.pos_weight_strategy}',
        f'molport_tw_{args.molport_task_weight}',
        f'lr_{args.lr}',
        f'dropout_{args.dropout}',
        f'wd_{args.weight_decay}'
    ]

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.exp_name,
        config=hyperparams,
        tags=wandb_tags,
        dir=str(exp_dir),
    )

    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.exp_name,
        save_dir=str(exp_dir),
        log_model='all',
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir / 'checkpoints',
            filename='epoch_{epoch:02d}_auroc_{val_avg_auroc:.4f}',
            monitor='val_avg_auroc',
            mode='max',
            save_top_k=3,
            save_last=True,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=10),
        # EarlyStopping disabled - dataset is huge, let model see all data
        # EarlyStopping(
        #     monitor='val_avg_auroc',
        #     patience=5,
        #     mode='max',
        #     verbose=True,
        # )
    ]

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        precision='16-mixed',
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        default_root_dir=str(exp_dir),
    )

    print(f"Output directory: {exp_dir}")
    print(f"Wandb run: {wandb.run.url}")

    # ============================================================================
    # Handle checkpoint resumption
    # ============================================================================

    ckpt_path = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "last":
            # Auto-find last.ckpt in the experiment directory
            last_ckpt = exp_dir / 'checkpoints' / 'last.ckpt'
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
                print(f"\n{'='*80}")
                print(f"RESUMING FROM LAST CHECKPOINT")
                print(f"Checkpoint: {ckpt_path}")
                print(f"{'='*80}\n")
            else:
                print(f"\n{'='*80}")
                print(f"WARNING: last.ckpt not found at {last_ckpt}")
                print(f"Starting training from scratch")
                print(f"{'='*80}\n")
        else:
            ckpt_path = args.resume_from_checkpoint
            print(f"\n{'='*80}")
            print(f"RESUMING FROM CHECKPOINT")
            print(f"Checkpoint: {ckpt_path}")
            print(f"{'='*80}\n")

    # ============================================================================
    # Train
    # ============================================================================

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # ============================================================================
    # Save results
    # ============================================================================

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

    results = {
        'exp_id': args.exp_id,
        'exp_name': args.exp_name,
        'best_model_path': trainer.checkpoint_callback.best_model_path,
        'best_val_auroc': float(trainer.checkpoint_callback.best_model_score) if trainer.checkpoint_callback.best_model_score else None,
        'hyperparams': hyperparams,
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Best model: {results['best_model_path']}")
    print(f"Best val AUROC: {results['best_val_auroc']:.4f}" if results['best_val_auroc'] else "N/A")
    print(f"Results saved to: {exp_dir / 'results.json'}")

    wandb.finish()

    return results


if __name__ == '__main__':
    main()

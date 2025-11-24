"""
PyTorch Lightning module for Multitask GVP Classifier

Handles:
- 5-task multitask learning (CSC, MCULE, PubChem, ZINC, MolPort)
- Weighted loss (both per-task and per-class weights)
- Per-task metrics (AUROC, AUPRC, F1, MCC, etc.)
- Automatic mixed precision training
- Checkpointing and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import (
    AUROC,
    AveragePrecision,
    F1Score,
    MatthewsCorrCoef,
    Accuracy,
    Precision,
    Recall,
    MetricCollection,
)
from typing import Dict, List, Optional

from multitask_gvp_classifier import MultitaskGVPClassifier


class MultitaskLightningModule(pl.LightningModule):
    """
    Lightning wrapper for multitask GVP classifier.

    Supports weighted loss at two levels:
    1. Per-task weights: Weight each classification task differently
    2. Per-class weights: Weight positive/negative samples within each task
    """

    def __init__(self, hyperparams):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(hyperparams)

        # Create the multitask model
        self.model = MultitaskGVPClassifier(
            n_atom_types=hyperparams['n_atom_types'],
            n_charge_types=hyperparams['n_charge_types'],
            n_bond_types=hyperparams['n_bond_types'],
            hidden_dim=hyperparams['hidden_dim'],
            edge_dim=hyperparams['edge_dim'],
            n_vec_channels=hyperparams['n_vec_channels'],
            num_layers=hyperparams['num_layers'],
            rbf_dim=hyperparams['rbf_dim'],
            rbf_dmax=hyperparams['rbf_dmax'],
            shared_repr_dim=hyperparams['shared_repr_dim'],
            task_hidden_dim=hyperparams['task_hidden_dim'],
            dropout=hyperparams['dropout'],
            readout=hyperparams['readout'],
        )

        # Task information
        self.task_names = ['CSC', 'MCULE', 'PubChem', 'ZINC', 'MolPort']
        self.task_indices = [2, 5, 8, 12, 6]  # Original indices in pharmit_library
        self.n_tasks = 5

        # ================================================================
        # LOSS WEIGHTS
        # ================================================================

        # Per-task weights (how much to weight each task in the total loss)
        # Default: all equal, but MolPort gets 2x due to extreme rarity
        task_weights = hyperparams.get('task_weights', [1.0, 1.0, 1.0, 1.0, 2.0])
        self.task_weights = torch.tensor(task_weights, dtype=torch.float32)

        # Per-class weights (positive weight for BCEWithLogitsLoss)
        # These are calculated as: (1 - positive_ratio) / positive_ratio
        # to balance the classes
        pos_ratios = hyperparams.get('pos_ratios', [0.1991, 0.2995, 0.5022, 0.2087, 0.0407])

        # Calculate pos_weight for BCEWithLogitsLoss
        # pos_weight = (1 - pos_ratio) / pos_ratio
        # This upweights positive samples to balance the classes
        self.pos_weights = torch.tensor([
            (1 - r) / r for r in pos_ratios
        ], dtype=torch.float32)

        # ================================================================
        # METRICS (PER TASK)
        # ================================================================

        # Create metric collections for each task
        metric_params = {'task': 'binary'}

        # Training metrics (per task)
        self.train_metrics = nn.ModuleList([
            MetricCollection({
                'auroc': AUROC(**metric_params),
                'auprc': AveragePrecision(**metric_params),
                'f1': F1Score(**metric_params),
                'mcc': MatthewsCorrCoef(**metric_params, num_classes=2),
                'acc': Accuracy(**metric_params, average='macro'),
                'precision': Precision(**metric_params),
                'recall': Recall(**metric_params),
            }, prefix=f'train/{self.task_names[i]}_')
            for i in range(self.n_tasks)
        ])

        # Validation metrics (per task)
        self.val_metrics = nn.ModuleList([
            MetricCollection({
                'auroc': AUROC(**metric_params),
                'auprc': AveragePrecision(**metric_params),
                'f1': F1Score(**metric_params),
                'mcc': MatthewsCorrCoef(**metric_params, num_classes=2),
                'acc': Accuracy(**metric_params, average='macro'),
                'precision': Precision(**metric_params),
                'recall': Recall(**metric_params),
            }, prefix=f'val/{self.task_names[i]}_')
            for i in range(self.n_tasks)
        ])

    def forward(self, g):
        """Forward pass through the model."""
        return self.model(g)

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted multitask loss.

        Args:
            logits: [batch_size, 5] - model predictions
            labels: [batch_size, 5] - ground truth labels

        Returns:
            total_loss: Weighted sum of per-task losses
        """
        # Move weights to correct device
        task_weights = self.task_weights.to(logits.device)
        pos_weights = self.pos_weights.to(logits.device)

        total_loss = 0.0
        task_losses = []

        for task_idx in range(self.n_tasks):
            # Extract task-specific logits and labels
            task_logits = logits[:, task_idx]  # [batch_size]
            task_labels = labels[:, task_idx].float()  # [batch_size]

            # Compute BCE loss with positive class weighting
            task_loss = F.binary_cross_entropy_with_logits(
                task_logits,
                task_labels,
                pos_weight=pos_weights[task_idx],
            )

            task_losses.append(task_loss)

            # Weight and accumulate
            total_loss += task_weights[task_idx] * task_loss

        # Store individual task losses for logging
        self.last_task_losses = task_losses

        return total_loss

    def training_step(self, batch, batch_idx):
        """Training step - called for each batch."""
        # Extract graph and labels
        g = batch['graph']
        labels = batch['system_features']['pharmit_library']  # [B, 5]

        # Forward pass
        logits = self(g)  # [B, 5]
        loss = self.compute_loss(logits, labels)

        # Compute predictions
        probs = torch.sigmoid(logits)  # [B, 5]

        # Update per-task metrics
        for task_idx in range(self.n_tasks):
            task_probs = probs[:, task_idx]
            task_labels = labels[:, task_idx].int()

            self.train_metrics[task_idx].update(task_probs, task_labels)

        # Log total loss (both per-step and epoch average)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log per-task losses (both per-step and epoch average)
        for task_idx, task_name in enumerate(self.task_names):
            self.log(
                f'train/{task_name}_loss',
                self.last_task_losses[task_idx],
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True
            )

        # Log per-task metrics (at epoch end only, to show accumulated metrics)
        # Computing at every step causes memory issues with 740 epochs
        # Metrics will be automatically logged at epoch end by PyTorch Lightning
        task_aurocs = []
        task_mccs = []
        for task_idx in range(self.n_tasks):
            # Compute current accumulated metrics (not logged per-step to save memory)
            metrics = self.train_metrics[task_idx].compute()

            # Collect AUROC and MCC for averaging
            for metric_name, metric_value in metrics.items():
                # Log at epoch end only
                self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=False)

                if 'auroc' in metric_name:
                    task_aurocs.append(metric_value)
                if 'mcc' in metric_name:
                    task_mccs.append(metric_value)

        # Log averaged AUROC and MCC across all tasks (epoch end only)
        if task_aurocs:
            avg_auroc = torch.stack(task_aurocs).mean()
            self.log('train_avg_auroc', avg_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if task_mccs:
            avg_mcc = torch.stack(task_mccs).mean()
            self.log('train_avg_mcc', avg_mcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - called for each validation batch."""
        # Extract graph and labels
        g = batch['graph']
        labels = batch['system_features']['pharmit_library']  # [B, 5]

        # Forward pass
        logits = self(g)  # [B, 5]
        loss = self.compute_loss(logits, labels)

        # Compute predictions
        probs = torch.sigmoid(logits)  # [B, 5]

        # Update per-task metrics
        for task_idx in range(self.n_tasks):
            task_probs = probs[:, task_idx]
            task_labels = labels[:, task_idx].int()

            self.val_metrics[task_idx].update(task_probs, task_labels)

        # Log total loss
        self.log('val_loss', loss, prog_bar=True, logger=True)

        # Log per-task losses
        for task_idx, task_name in enumerate(self.task_names):
            self.log(
                f'val/{task_name}_loss',
                self.last_task_losses[task_idx],
                prog_bar=False,
                logger=True
            )

        # Log per-task metrics (at epoch end)
        for task_idx in range(self.n_tasks):
            metrics = self.val_metrics[task_idx].compute()
            for metric_name, metric_value in metrics.items():
                self.log(metric_name, metric_value, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def on_train_epoch_end(self):
        """Reset training metrics at the end of each epoch to prevent memory accumulation."""
        for task_idx in range(self.n_tasks):
            self.train_metrics[task_idx].reset()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch - print summary."""
        print(f"\n{'='*80}")
        print("VALIDATION EPOCH SUMMARY")
        print(f"{'='*80}")

        for task_idx, task_name in enumerate(self.task_names):
            metrics = self.val_metrics[task_idx].compute()

            auroc = metrics.get(f'val/{task_name}_auroc', 0.0)
            auprc = metrics.get(f'val/{task_name}_auprc', 0.0)
            f1 = metrics.get(f'val/{task_name}_f1', 0.0)
            mcc = metrics.get(f'val/{task_name}_mcc', 0.0)

            print(f"{task_name:>10}: AUROC={auroc:.4f} | AUPRC={auprc:.4f} | "
                  f"F1={f1:.4f} | MCC={mcc:.4f}")

        print(f"{'='*80}\n")

    def configure_optimizers(self):
        """Configure optimizer - constant learning rate for large dataset exploration."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # LR scheduler disabled - dataset is huge, model needs to see all data
        # without premature LR reduction. Keep constant LR for full exploration.
        # # ReduceLROnPlateau scheduler
        # # Monitor average validation AUROC across all tasks
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode='max',
        #     factor=0.5,
        #     patience=3,  # Reduce LR if no improvement for 3 epochs
        #     verbose=True,
        # )
        #
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_avg_auroc',  # Monitor average AUROC
        #         'interval': 'epoch',
        #         'frequency': 1
        #     }
        # }

        # Return just optimizer for constant LR
        return optimizer

    def on_validation_epoch_end(self):
        """Compute and log average metrics across all tasks."""
        super().on_validation_epoch_end()

        # Compute average AUROC across all tasks
        aurocs = []
        for task_idx, task_name in enumerate(self.task_names):
            metrics = self.val_metrics[task_idx].compute()
            auroc = metrics.get(f'val/{task_name}_auroc', 0.0)
            if auroc > 0:
                aurocs.append(auroc)

        if aurocs:
            avg_auroc = torch.stack(aurocs).mean()
            self.log('val_avg_auroc', avg_auroc, prog_bar=True, logger=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_hyperparams(
    # Model architecture
    n_atom_types: int = 118,
    n_charge_types: int = 10,
    n_bond_types: int = 4,
    hidden_dim: int = 256,
    edge_dim: int = 128,
    n_vec_channels: int = 16,
    num_layers: int = 4,
    rbf_dim: int = 32,
    rbf_dmax: float = 20.0,
    shared_repr_dim: int = 256,
    task_hidden_dim: int = 128,
    dropout: float = 0.2,
    readout: str = 'mean',

    # Loss weights
    task_weights: Optional[List[float]] = None,
    pos_ratios: Optional[List[float]] = None,

    # Optimization
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
) -> Dict:
    """
    Create hyperparameter dictionary for MultitaskLightningModule.

    Args:
        Model architecture params (same as MultitaskGVPClassifier)
        task_weights: Weights for each task in total loss [CSC, MCULE, PubChem, ZINC, MolPort]
        pos_ratios: Positive class ratios for each task (for pos_weight calculation)
        lr: Learning rate
        weight_decay: Weight decay for Adam optimizer

    Returns:
        Dictionary of hyperparameters
    """
    # Default task weights (give MolPort 2x weight due to extreme rarity)
    if task_weights is None:
        task_weights = [1.0, 1.0, 1.0, 1.0, 2.0]

    # Default positive ratios (from your dataset analysis)
    if pos_ratios is None:
        pos_ratios = [0.1991, 0.2995, 0.5022, 0.2087, 0.0407]

    return {
        # Model architecture
        'n_atom_types': n_atom_types,
        'n_charge_types': n_charge_types,
        'n_bond_types': n_bond_types,
        'hidden_dim': hidden_dim,
        'edge_dim': edge_dim,
        'n_vec_channels': n_vec_channels,
        'num_layers': num_layers,
        'rbf_dim': rbf_dim,
        'rbf_dmax': rbf_dmax,
        'shared_repr_dim': shared_repr_dim,
        'task_hidden_dim': task_hidden_dim,
        'dropout': dropout,
        'readout': readout,

        # Loss weights
        'task_weights': task_weights,
        'pos_ratios': pos_ratios,

        # Optimization
        'lr': lr,
        'weight_decay': weight_decay,
    }

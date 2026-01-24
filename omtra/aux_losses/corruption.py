"""
Corruption Classification Auxiliary Loss (Stage 4 of Noisy Paths Experiment)

This loss trains the model to identify which tokens were corrupted during the
three-way conditional path sampling. At inference time, the corruption predictions
can be used to identify likely errors and trigger remasking/resampling.

The loss is only active when noisy paths are enabled (noise_alpha > 0), which
causes corruption masks to be stored in the graph during sample_conditional_path().
"""

from omtra.aux_losses.register import register_aux_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from typing import Dict
from omtra.utils.graph import g_local_scope
from omtra.tasks.tasks import Task
from omtra.tasks.modalities import name_to_modality


@register_aux_loss(name='corruption_classification')
class CorruptionClassification(nn.Module):
    """
    Binary classification loss for predicting which tokens were corrupted.

    For each discrete modality (lig_cond_a, lig_e_condensed), computes BCE loss
    between the corruption classification head predictions and the ground truth
    corruption masks stored in the graph.

    Args:
        weight: Loss weight multiplier
        pos_weight: Weight for positive class (corrupted tokens) to handle class
                   imbalance. Higher values encourage the model to better detect
                   corrupted tokens at the cost of more false positives.
        modalities: List of modality names to apply this loss to. If None, applies
                   to all discrete modalities that have corruption masks.
    """

    def __init__(
        self,
        weight: float = 1.0,
        pos_weight: float = 10.0,
        modalities: list = None,
    ):
        super().__init__()
        self.weight = weight
        self.pos_weight = pos_weight
        self.modalities = modalities  # None means all available

    @g_local_scope
    def forward(
        self,
        g: dgl.DGLHeteroGraph,
        dst_dict: Dict[str, torch.Tensor],
        task: Task,
        node_batch_idxs: Dict[str, torch.Tensor],
        lig_ue_mask: torch.Tensor,
        time_weights: torch.Tensor,
    ):
        """
        Compute corruption classification loss.

        Args:
            g: Graph containing corruption masks in node/edge data
            dst_dict: Model predictions, including "{modality}_corruption_logits"
            task: Task definition
            node_batch_idxs: Batch indices for nodes
            lig_ue_mask: Upper edge mask for symmetric edge handling
            time_weights: Optional time-dependent loss weights

        Returns:
            Scalar loss value
        """
        total_loss = torch.tensor(0.0, device=g.device, requires_grad=True)
        n_modalities = 0

        for modality in task.modalities_generated:
            if not modality.is_categorical:
                continue

            # Check if this modality should be processed
            if self.modalities is not None and modality.name not in self.modalities:
                continue

            # Check if corruption logits are available
            corruption_key = f"{modality.name}_corruption_logits"
            if corruption_key not in dst_dict:
                continue

            # Get ground truth corruption mask from graph
            dk = modality.data_key
            data_src = g.nodes if modality.is_node else g.edges
            corruption_mask_key = f"{dk}_is_corrupted"

            if corruption_mask_key not in data_src[modality.entity_name].data:
                # No corruption mask means noisy paths not enabled for this modality
                continue

            is_corrupted = data_src[modality.entity_name].data[corruption_mask_key]
            pred_logits = dst_dict[corruption_key]

            # For edge features, we only have predictions for upper edges
            if not modality.is_node:
                is_corrupted = is_corrupted[lig_ue_mask]

            # Ensure shapes match
            assert pred_logits.shape == is_corrupted.shape, (
                f"Shape mismatch for {modality.name}: pred {pred_logits.shape} vs target {is_corrupted.shape}"
            )

            # Compute BCE loss with positive class weighting
            pos_weight_tensor = torch.tensor(
                self.pos_weight, device=pred_logits.device
            )

            time_scaled = time_weights is not None
            reduction = 'none' if time_scaled else 'mean'

            mod_loss = F.binary_cross_entropy_with_logits(
                pred_logits,
                is_corrupted.float(),
                pos_weight=pos_weight_tensor,
                reduction=reduction,
            )

            if time_scaled:
                # Expand time weights to match the predictions
                if modality.is_node:
                    batch_idxs = node_batch_idxs[modality.entity_name]
                else:
                    # For edges, use source node batch indices
                    src_nodes, _ = g.edges(etype=modality.entity_name)
                    src_nodes = src_nodes[lig_ue_mask]
                    batch_idxs = node_batch_idxs['lig'][src_nodes]

                time_weights_expanded = time_weights[batch_idxs]
                mod_loss = (mod_loss * time_weights_expanded).mean()

            total_loss = total_loss + mod_loss
            n_modalities += 1

        # Average over modalities if we processed any
        if n_modalities > 0:
            total_loss = total_loss / n_modalities

        return total_loss * self.weight

    def supports_task(self, task: Task) -> bool:
        """
        This loss supports any task that generates discrete modalities.
        The actual activation depends on whether corruption masks are present
        in the graph (determined by noisy paths config).
        """
        # Check if any generated modality is categorical
        for modality in task.modalities_generated:
            if modality.is_categorical:
                return True
        return False

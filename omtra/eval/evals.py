from omtra.eval.register import register_eval
from omtra.eval.utils import compute_validity, compute_stability, reos_and_rings
from omtra.data.graph.utils import SampledSystem
from typing import Dict, Any, Optional, List


##
# tasks with ligand only
##
@register_eval("denovo_ligand")
def denovo_ligand(sampled_systems: List[SampledSystem]):
    metrics = compute_validity(sampled_systems)
    metrics.update(compute_stability(sampled_systems))
    metrics.update(reos_and_rings(sampled_systems))
    return metrics


@register_eval("ligand_conformer")
def ligand_conformer(sampled_systems: List[SampledSystem]):
    metrics = compute_validity(sampled_systems)
    metrics.update(compute_stability(sampled_systems))
    metrics.update(reos_and_rings(sampled_systems))
    return metrics


##
# tasks with ligand + pharmacophore
##
@register_eval("denovo_ligand_pharmacophore")
def denovo_ligand_pharmacophore(sampled_systems: List[SampledSystem]):
    metrics = compute_validity(sampled_systems)
    metrics.update(compute_stability(sampled_systems))
    metrics.update(reos_and_rings(sampled_systems))

    # TODO: add pharm metrics
    return metrics


@register_eval("denovo_ligand_from_pharmacophore")
def denovo_ligand_from_pharmacophore(sampled_systems: List[SampledSystem]):
    metrics = compute_validity(sampled_systems)
    metrics.update(compute_stability(sampled_systems))
    metrics.update(reos_and_rings(sampled_systems))

    # TODO: add pharm metrics
    return metrics


##
# tasks with ligand+protein and no pharmacophore
##
@register_eval("pred_apo_conditioned_denovo_ligand")
@register_eval("exp_apo_conditioned_denovo_ligand")
@register_eval("protein_ligand_denovo")
def protein_ligand_denovo(sampled_systems: List[SampledSystem]):
    metrics = compute_validity(sampled_systems)
    metrics.update(compute_stability(sampled_systems))
    metrics.update(reos_and_rings(sampled_systems))

    # TODO: add system level metrics
    return metrics


@register_eval("predapo_conditioned_ligand_docking")
@register_eval("expapo_conditioned_ligand_docking")
@register_eval("flexible_docking")
def flexible_docking(sampled_systems: List[SampledSystem]):
    return {}


##
# Tasks with ligand+protein+pharmacophore
##
@register_eval("protein_ligand_pharmacophore_denovo")
def protein_ligand_pharmacophore_denovo(sampled_systems: List[SampledSystem]):
    metrics = compute_validity(sampled_systems)
    metrics.update(compute_stability(sampled_systems))
    metrics.update(reos_and_rings(sampled_systems))
    
    # TODO: add system level metrics, pharm metrics
    
    return metrics


##
# Tasks with protein+pharmacophore and no ligand
##
@register_eval("protein_pharmacophore")
@register_eval("expapo_conditioned_protein_pharmacophore")
def protein_pharmacophore(sampled_systems: List[SampledSystem]):
    return {}


##
# Tasks with protein only
##

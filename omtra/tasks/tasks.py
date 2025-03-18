import torch
import functools
from omtra.utils.misc import classproperty
from copy import deepcopy
from typing import List

import omtra.tasks.prior_collections as pc
import omtra.tasks.cond_path_collections as cpc

from omtra.tasks.register import register_task
import omtra.tasks.modalities as modal
from omtra.tasks.modalities import Modality, name_to_modality
   
class Task:

    @classproperty
    def groups_present(self):
        return self.groups_fixed + self.groups_generated

    @classproperty
    def groups_absent(self):
        return list(modal.GROUP_SPACE - set(self.groups_present))
    
    @classproperty
    def modalities_generated(self) -> List[Modality]:
        modalities = []
        for modality_name in modal.MODALITY_ORDER:
            modality = name_to_modality(modality_name)
            if modality.group in self.groups_generated:
                modalities.append(modality)
        return modalities
    
    @classproperty
    def modalities_fixed(self) -> List[Modality]:
        modalities = []
        for modality in modal.MODALITY_ORDER:
            if modality.group in self.groups_fixed:
                modalities.append(modality)
        return modalities

##
# tasks with ligand only
##
@register_task("denovo_ligand")
class DeNovoLigand(Task):
    groups_fixed = []
    groups_generated = ['ligand_identity', 'ligand_structure']

    priors = pc.denovo_ligand

    conditional_paths = cpc.denovo_ligand

@register_task("ligand_conformer")
class LigandConformer(Task):
    groups_fixed = ['ligand_identity']
    groups_generated = ['ligand_structure']

    priors = pc.ligand_conformer
    conditional_paths = cpc.ligand_conformer

## 
# tasks with ligand + pharmacophore
##
@register_task("denovo_ligand_pharmacophore")
class DeNovoLigandPharmacophore(Task):
    groups_fixed = []
    groups_generated = ['ligand_identity', 'ligand_structure', 'pharmacophore']

    priors = dict(**pc.denovo_ligand, **pc.denovo_pharmacophore)
    conditional_paths = dict(**cpc.denovo_ligand, **cpc.denovo_pharmacophore)

@register_task("denovo_ligand_from_pharmacophore")
class DeNovoLigandFromPharmacophore(Task):
    groups_fixed = ['pharmacophore']
    groups_generated = ['ligand_identity', 'ligand_structure']

    priors = pc.denovo_ligand
    conditional_paths = cpc.denovo_ligand

##
# tasks with ligand+protein and no pharmacophore
##
@register_task("protein_ligand_denovo")
class ProteinLigandDeNovo(Task):
    groups_fixed = []
    groups_generated = ['protein_structure', 'ligand_identity', 'ligand_structure']

    priors = deepcopy(pc.denovo_ligand)
    priors['prot_atom'] = {
        'type': 'target_dependent_gaussian',
    }
    


@register_task("exp_apo_conditioned_denovo_ligand")
class ExpApoDeNovoLigand(Task):
    groups_fixed = []
    groups_generated = ['ligand_identity', 'ligand_structure', 'protein_structure']

    priors = deepcopy(pc.denovo_ligand)
    priors['prot_atom'] = {
        'type': 'apo_exp', # in this case the prior is an actual apo structure itself; a sample from a data distribution
    }

@register_task("pred_apo_conditioned_denovo_ligand")
class PredApoDeNovoLigand(ExpApoDeNovoLigand):
    priors = deepcopy(pc.denovo_ligand)
    priors['prot_atom'] = {
        'type': 'apo_pred'
    }

@register_task("flexible_docking")
class FlexibleDocking(Task):
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0"""
    groups_fixed = ['ligand_identity']
    groups_generated = ['ligand_structure', 'protein_structure']

    priors = deepcopy(pc.ligand_conformer)
    priors['prot_atom'] = {
        'type': 'target_dependent_gaussian',
    }

@register_task("expapo_conditioned_ligand_docking")
class ExpApoConditionedLigandDocking(Task):
    """Docking a ligand into the protein structure, protein structure is an experimentally determined apo structure at t=0."""
    groups_fixed = ['ligand_identity']
    groups_generated = ['ligand_structure', 'protein_structure']

    priors = deepcopy(pc.ligand_conformer)
    priors['prot_atom'] = {
        'type': 'apo_exp',
    }

@register_task("predapo_conditioned_ligand_docking")
class PredApoConditionedLigandDocking(Task):
    """Docking a ligand into the protein structure, protein structure is a predicted apo structure at t=0."""
    groups_fixed = ['ligand_identity']
    groups_generated = ['ligand_structure', 'protein_structure']

    priors = deepcopy(pc.ligand_conformer)
    priors['prot_atom'] = {
        'type': 'apo_pred',
    }


##
# Tasks with ligand+protein+pharmacophore
##
@register_task("protein_ligand_pharmacophore_denovo")
class ProteinLigandPharmacophoreDeNovo(Task):
    groups_fixed = []
    groups_generated = ['protein_structure', 'ligand_identity', 'ligand_structure', 'pharmacophore']

    priors = dict(**pc.denovo_ligand, **pc.denovo_pharmacophore)
    priors['prot_atom'] = {
        'type': 'target_dependent_gaussian',
    }


# TODO: there could be more protein+ligand+pharmacophore tasks but that is a future decision

## 
# Tasks with protein+pharmacophore and no ligand
##
@register_task("protein_pharmacophore")
class ProteinPharmacophore(Task):
    groups_fixed = []
    groups_generated = ['protein_structure', 'pharmacophore']

    priors = deepcopy(pc.denovo_pharmacophore)
    priors['prot_atom'] = {
        'type': 'target_dependent_gaussian',
    }

@register_task("expapo_conditioned_protein_pharmacophore")
class ExpApoConditionedProteinPharmacophore(Task):
    """Generate a pharmacophore from an experimentally determined apo protein structure at t=0"""
    groups_fixed = []
    groups_generated = ['protein_structure', 'pharmacophore']

    priors = deepcopy(pc.denovo_pharmacophore)
    priors['protein'] = {
        'type': 'apo_exp',
    }


##
# Tasks with protein only
## 
# @register_task("apo_protein_sampling")
# class ApoProteinSampling(Task):
#     """Sampling apo protein conformations, starting from noise for the protein at t=0"""
#     groups_fixed = []
#     groups_generated = ['protein_structure']

# @register_task("apo_to_holo_protein")
# class ApotoHoloProtein(Task):
#     """Predicting the holo protein structure, starting from the apo protein structure at t=0"""
#     groups_fixed = []
#     groups_generated = ['protein_structure']



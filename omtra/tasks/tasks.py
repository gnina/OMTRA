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
        for modality_name in modal.MODALITY_ORDER:
            modality = name_to_modality(modality_name)
            if modality.group in self.groups_fixed:
                modalities.append(modality)
        return modalities
    
    @classproperty
    def modalities_present(self) -> List[Modality]:
        return self.modalities_fixed + self.modalities_generated
    
    @classproperty
    def plinder_link_version(self) -> str:
        prot_atom_prior = self.priors.get('prot_atom_x', None)
        if prot_atom_prior is None:
            return 'no_links'
        if prot_atom_prior['type'] == 'apo_exp':
            return 'exp'
        elif prot_atom_prior['type'] == 'apo_pred':
            return 'pred'
        else:
            return 'no_links'
        
    @classproperty
    def node_modalities_present(self) -> List[Modality]:
        """Returns the node modalities for this task. This is a subset of the modalities present in the task."""
        return [ m for m in self.modalities_present if m.is_node ]
    
    @classproperty
    def edge_modalities_present(self) -> List[Modality]:
        """Returns the edge modalities for this task. This is a subset of the modalities present in the task."""
        return [ m for m in self.modalities_present if not m.is_node ]
    
    @classproperty
    def unconditional(self) -> bool:
        """Returns True if the task is fully unconditional, i.e., all groups are generated and none are fixed."""
        return set(self.groups_generated) == set(self.groups_present)

    @classproperty
    def has_protein(self) -> bool:
        return 'protein_identity' in self.groups_present
    
    @classproperty
    def has_pharmacophore(self) -> bool:
        return 'pharmacophore' in self.groups_present

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


@register_task("denovo_ligand_extra_feats")
class DeNovoLigandExtraFeats(Task):
    groups_fixed = []
    groups_generated = ['ligand_identity', 'ligand_structure', 'ligand_identity_extra']

    priors = pc.denovo_ligand
    conditional_paths = cpc.denovo_ligand

    for modality in ['impl_H', 'aro', 'hyb', 'ring', 'chiral']:
       priors[f'lig_{modality}'] = dict(type='masked')
       conditional_paths[f'lig_{modality}'] = dict(type='ctmc_mask')


@register_task("ligand_conformer_extra_feats")
class LigandConformerExtraFeats(Task):
    groups_fixed = ['ligand_identity', 'ligand_identity_extra']
    groups_generated = ['ligand_structure']

    priors = pc.ligand_conformer
    conditional_paths = cpc.ligand_conformer


@register_task("denovo_ligand_condensed")
class DeNovoLigandCondensed(Task):
    groups_fixed = []
    groups_generated = ['ligand_identity_condensed', 'ligand_structure']

    priors = {'lig_x': {'type': 'gaussian', 'params': {'ot': True}},
              'lig_e_condensed': dict(type='masked'),
              'lig_cond_a': dict(type='masked')}
    conditional_paths = {'lig_x': dict(type='continuous_interpolant'),
                         'lig_e_condensed': dict(type='ctmc_mask'),
                         'lig_cond_a': dict(type='ctmc_mask')}


@register_task("ligand_conformer_condensed")
class LigandConformerCondensed(Task):
    groups_fixed = ['ligand_identity_condensed']
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
    groups_fixed = ['protein_identity']
    groups_generated = ['protein_structure', 'ligand_identity', 'ligand_structure']

    priors = deepcopy(pc.denovo_ligand)
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.denovo_ligand, **cpc.protein)

@register_task("protein_ligand_denovo_condensed")
class ProteinLigandDeNovoCondensed(Task):
    groups_fixed = ['protein_identity']
    groups_generated = ['protein_structure', 'ligand_identity_condensed', 'ligand_structure']

    priors = {'lig_x': {'type': 'gaussian', 'params': {'ot': True}},
              'lig_e_condensed': dict(type='masked'),
              'lig_cond_a': dict(type='masked')}
    
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    
    conditional_paths = dict({'lig_x': dict(type='continuous_interpolant'),
                         'lig_e_condensed': dict(type='ctmc_mask'),
                         'lig_cond_a': dict(type='ctmc_mask')}, **cpc.protein)

@register_task("fixed_protein_ligand_denovo")
class FixedProteinLigandDeNovo(Task):
    groups_fixed = ['protein_identity', 'protein_structure']
    groups_generated = ['ligand_identity', 'ligand_structure']

    priors = deepcopy(pc.denovo_ligand)

    conditional_paths = dict(**cpc.denovo_ligand, **cpc.protein) # can probably remove protein from this

@register_task("fixed_protein_ligand_denovo_condensed")
class FixedProteinLigandDeNovoCondensed(Task):
    groups_fixed = ['protein_identity', 'protein_structure']
    groups_generated = ['ligand_identity_condensed', 'ligand_structure']

    priors = {'lig_x': {'type': 'gaussian', 'params': {'ot': True}},
              'lig_e_condensed': dict(type='masked'),
              'lig_cond_a': dict(type='masked')}

    conditional_paths = dict({'lig_x': dict(type='continuous_interpolant'),
                         'lig_e_condensed': dict(type='ctmc_mask'),
                         'lig_cond_a': dict(type='ctmc_mask')}, **cpc.protein) # can probably remove protein from this

@register_task("exp_apo_conditioned_denovo_ligand")
class ExpApoDeNovoLigand(Task):
    groups_fixed = ['protein_identity']
    groups_generated = ['ligand_identity', 'ligand_structure', 'protein_structure']

    priors = deepcopy(pc.denovo_ligand)
    priors['prot_atom_x'] = {
        'type': 'apo_exp', # in this case the prior is an actual apo structure itself; a sample from a data distribution
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.denovo_ligand, **cpc.protein)

@register_task("pred_apo_conditioned_denovo_ligand")
class PredApoDeNovoLigand(ExpApoDeNovoLigand):
    priors = deepcopy(pc.denovo_ligand)
    priors['prot_atom_x'] = {
        'type': 'apo_pred'
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.denovo_ligand, **cpc.protein)

@register_task("flexible_docking")
class FlexibleDocking(Task):
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0"""
    groups_fixed = ['ligand_identity','protein_identity']
    groups_generated = ['ligand_structure', 'protein_structure']

    priors = deepcopy(pc.ligand_conformer)
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)

@register_task("flexible_docking_condensed")
class FlexibleDockingCondensed(Task):
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0"""
    groups_fixed = ['ligand_identity_condensed','protein_identity']
    groups_generated = ['ligand_structure', 'protein_structure']

    priors = deepcopy(pc.ligand_conformer)
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)
    

@register_task("rigid_docking")
class RigidDocking(Task):
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0"""

    groups_fixed = ["ligand_identity", "protein_identity", "protein_structure"]
    groups_generated = ["ligand_structure"]

    priors = deepcopy(pc.ligand_conformer)
    priors["npnde_x"] = {
        "type": "target_dependent_gaussian",
    }
    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)

@register_task("rigid_docking_condensed")
class RigidDockingCondensed(Task):
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0 with condensed atom typing for ligands"""

    groups_fixed = ["ligand_identity_condensed", "protein_identity", "protein_structure"]
    groups_generated = ["ligand_structure"]

    priors = deepcopy(pc.ligand_conformer)
    priors["npnde_x"] = {
        "type": "target_dependent_gaussian",
    }
    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)

@register_task("expapo_conditioned_ligand_docking")
class ExpApoConditionedLigandDocking(Task):
    """Docking a ligand into the protein structure, protein structure is an experimentally determined apo structure at t=0."""
    groups_fixed = ['ligand_identity','protein_identity']
    groups_generated = ['ligand_structure', 'protein_structure']

    priors = deepcopy(pc.ligand_conformer)
    priors['prot_atom_x'] = {
        'type': 'apo_exp',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)

@register_task("predapo_conditioned_ligand_docking")
class PredApoConditionedLigandDocking(Task):
    """Docking a ligand into the protein structure, protein structure is a predicted apo structure at t=0."""
    groups_fixed = ['ligand_identity','protein_identity']
    groups_generated = ['ligand_structure', 'protein_structure']

    priors = deepcopy(pc.ligand_conformer)
    priors['prot_atom_x'] = {
        'type': 'apo_pred',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)


##
# Tasks with ligand+protein+pharmacophore
##
@register_task("protein_ligand_pharmacophore_denovo")
class ProteinLigandPharmacophoreDeNovo(Task):
    groups_fixed = ['protein_identity']
    groups_generated = ['protein_structure', 'ligand_identity', 'ligand_structure', 'pharmacophore']

    priors = dict(**pc.denovo_ligand, **pc.denovo_pharmacophore)
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }

    conditional_paths = dict(**cpc.denovo_ligand, **cpc.denovo_pharmacophore, **cpc.protein)


# TODO: there could be more protein+ligand+pharmacophore tasks but that is a future decision

## 
# Tasks with protein+pharmacophore and no ligand
##
@register_task("protein_pharmacophore")
class ProteinPharmacophore(Task):
    groups_fixed = ['protein_identity']
    groups_generated = ['protein_structure', 'pharmacophore']

    priors = deepcopy(pc.denovo_pharmacophore)
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }

@register_task("expapo_conditioned_protein_pharmacophore")
class ExpApoConditionedProteinPharmacophore(Task):
    """Generate a pharmacophore from an experimentally determined apo protein structure at t=0"""
    groups_fixed = ['protein_identity']
    groups_generated = ['protein_structure', 'pharmacophore']

    priors = deepcopy(pc.denovo_pharmacophore)
    priors['prot_atom_x'] = {
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



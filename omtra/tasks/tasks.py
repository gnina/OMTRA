import torch
import functools
from omtra.utils.misc import classproperty
from copy import deepcopy

import omtra.tasks.prior_collections as pc

from omtra.tasks.register import register_task

# canonical order of modalitiy GROUPS
# these are not modalities themselves, but modalities that are grouped together
# for conceptual purposes
canonical_mg_order = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']
canonical_entity_order = ['ligand', 'protein', 'pharmacophore']

# this maps the modality groups to the raw modalities they contain
canonical_modality_structure = {
    'ligand_identity': list('ace'),
    'ligand_structure': ['x'],
    'pharmacophore': list('xav'),
    'protein': ['atom_x']
}

canonical_modality_order = []
mg_order = []
raw_modality_order = []
for mg, modalities in canonical_modality_structure.items():
    for modality in modalities:
        canonical_modality_order.append(f'{mg}_{modality}')
        mg_order.append(mg)
        raw_modality_order.append(modality)
   
class Task:
    protein_state_t0 = 'noise'

    @classproperty
    def t0_modgroup_arr(self) -> torch.Tensor:
        arr = torch.zeros(len(canonical_mg_order), dtype=bool)
        for i, modality_group in enumerate(canonical_mg_order):
            if modality_group in self.observed_at_t0:
                arr[i] = 1
        return arr
    
    @classproperty
    def t0_modality_arr(self) -> torch.Tensor:
        """Tensor of length len(canonical_modality_order) indicating which modalities are observed at t=0"""
        arr = torch.zeros(len(canonical_modality_order), dtype=bool)
        for modality_idx, mg in enumerate(mg_order):
            if mg in self.observed_at_t0:
                arr[modality_idx] = 1
        return arr
    
    @classproperty
    def t1_modgroup_arr(cls) -> torch.Tensor:
        arr = torch.zeros(len(canonical_mg_order), dtype=bool)
        for i, modality in enumerate(canonical_mg_order):
            if modality in cls.observed_at_t1:
                arr[i] = 1
        return arr
    
    @classproperty
    def t1_modality_arr(cls) -> torch.Tensor:
        """Tensor of length len(canonical_modality_order) indicating which modalities are observed at t=1"""
        arr = torch.zeros(len(canonical_modality_order), dtype=bool)
        for modality_idx, mg in enumerate(mg_order):
            if mg in cls.observed_at_t1:
                arr[modality_idx] = 1
        return arr
    
    
    @classproperty
    def modgroups_present(self):
        present_modgroup_mask = self.t0_modgroup_arr | self.t1_modgroup_arr
        present_modgroup_idxs = torch.where(present_modgroup_mask)[0]
        return [canonical_mg_order[i] for i in present_modgroup_idxs]
    
    @classproperty
    def modalities_present(self):
        present_modality_mask = self.t0_modality_arr | self.t1_modality_arr
        present_modality_idxs = torch.where(present_modality_mask)[0]
        return [canonical_modality_order[i] for i in present_modality_idxs]
    
    @classproperty
    def uses_apo(self):
        # TODO: this logic is subject to change if we ever decide to do apo sampling as a task
        # because then there would be a task where the intial protein state is noise but we still require the apo state (it would be the target)
        # but i think sampling apo states is not a very useful task for the application of sbdd
        return self.protein_state_t0 == 'apo'
    
##
# tasks with ligand only
##
@register_task("denovo_ligand")
class DeNovoLigand(Task):
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure']

    priors = dict(lig=pc.denovo_ligand)

@register_task("ligand_conformer")
class LigandConformer(Task):
    observed_at_t0 = ['ligand_identity']
    observed_at_t1 = ['ligand_identity', 'ligand_structure']

    priors = dict(lig=pc.ligand_conformer)

## 
# tasks with ligand + pharmacophore
##
@register_task("denovo_ligand_pharmacophore")
class DeNovoLigandPharmacophore(Task):
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']

    priors = {
        'lig': pc.denovo_ligand,
        'pharm': pc.denovo_pharmacophore
    }

@register_task("denovo_ligand_from_pharmacophore")
class DeNovoLigandFromPharmacophore(Task):
    observed_at_t0 = ['pharmacophore']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']

    priors = dict(lig=pc.denovo_ligand, pharm=pc.fixed_pharmacophore)

# while technically possible, i don't think this task is very useful?
# @register_task("ligand_conformer_pharmacophore")
# class LigandConformerPharmacophore(Task):
#     observed_at_t0 = ['ligand_identity']
#     observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']

##
# tasks with ligand+protein and no pharmacophore
##
@register_task("protein_ligand_denovo")
class ProteinLigandDeNovo(Task):
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein']

    priors = dict(lig=pc.denovo_ligand)
    priors['protein'] = {
        'type': 'target_dependent_gaussian',
    }


@register_task("apo_conditioned_denovo_ligand")
class ApoConditionedDeNovoLigand(Task):
    observed_at_t0 = ['protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure']
    protein_state_t0 = 'apo'

    priors = {
        'lig': pc.denovo_ligand
    }
    priors['protein'] = {
        'type': 'data_prior', # in this case the prior is an actual apo structure itself; a sample from a data distribution
    }

@register_task("unconditional_ligand_docking")
class UnconditionalLigandDocking(Task):
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0"""
    observed_at_t0 = ['ligand_identity']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein']

    priors = {
        'lig': pc.ligand_conformer
    }
    priors['protein'] = {
        'type': 'target_dependent_gaussian',
    }

@register_task("apo_conditioned_ligand_docking")
class ApoConditionedLigandDocking(Task):
    """Docking a ligand into the protein structure, assuming knowledge of the apo protein structure at t=0"""
    observed_at_t0 = ['ligand_identity', 'protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein']
    protein_state_t0 = 'apo'

    priors = {
        'lig': pc.ligand_conformer
    }
    priors['protein'] = {
        'type': 'data_prior', # in this case the prior is an actual apo structure itself; a sample from a data distribution
    }

##
# Tasks with ligand+protein+pharmacophore
##
@register_task("protein_ligand_pharmacophore_denovo")
class ProteinLigandPharmacophoreDeNovo(Task):
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']

    priors = dict(lig=pc.denovo_ligand, pharm=pc.denovo_pharmacophore)
    priors['protein'] = {
        'type': 'target_dependent_gaussian',
    }

@register_task("apo_conditioned_denovo_ligand_pharmacophore")
class ApoConditionedDeNovoLigandPharmacophore(Task):
    observed_at_t0 = ['protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']
    protein_state_t0 = 'apo'

# TODO: have a think about these guys
@register_task("unconditional_ligand_docking_pharmacophore")
class UnconditionalLigandDockingPharmacophore(Task):
    """Docking a ligand into the protein while generating a pharmacophore, assuming no knowledge of the protein structure at t=0"""
    observed_at_t0 = ['ligand_identity',]
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']

@register_task("apo_conditioned_ligand_docking_pharmacophore")
class ApoConditionedLigandDockingPharmacophore(Task):
    """Docking a ligand into the protein while generating a pharmacophore, assuming knowledge of the apo protein structure at t=0"""
    observed_at_t0 = ['ligand_identity', 'protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']
    protein_state_t0 = 'apo'

## 
# Tasks with protein+pharmacophore and no ligand
##
@register_task("protein_pharmacophore")
class ProteinPharmacophore(Task):
    observed_at_t0 = []
    observed_at_t1 = ['protein', 'pharmacophore']

@register_task("apo_conditioned_protein_pharmacophore")
class ApoConditionedProteinPharmacophore(Task):
    observed_at_t0 = ['protein']
    observed_at_t1 = ['protein', 'pharmacophore']
    protein_state_t0 = 'apo'

##
# Tasks with protein only
## 
@register_task("apo_protein_sampling")
class ApoProteinSampling(Task):
    """Sampling apo protein conformations, starting from noise for the protein at t=0"""
    observed_at_t0 = []
    observed_at_t1 = ['protein']

@register_task("apo_to_holo_protein")
class ApotoHoloProtein(Task):
    """Predicting the holo protein structure, starting from the apo protein structure at t=0"""
    observed_at_t0 = ['protein']
    observed_at_t1 = ['protein']
    protein_state_t0 = 'apo'



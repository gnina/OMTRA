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
@register_task("denovo_ligand_extra_feats")
class DeNovoLigandExtraFeats(Task):
    groups_fixed = []
    groups_generated = ['ligand_identity', 'ligand_structure', 'ligand_identity_extra']

    priors = pc.denovo_ligand_extra_feats
    conditional_paths = cpc.denovo_ligand_extra_feats


@register_task("ligand_conformer_extra_feats")
class LigandConformerExtraFeats(Task):
    groups_fixed = ['ligand_identity', 'ligand_identity_extra']
    groups_generated = ['ligand_structure']

    priors = pc.ligand_conformer
    conditional_paths = cpc.ligand_conformer


##
# tasks with ligand+protein and no pharmacophore
##
@register_task("rigid_docking_extra_feats")
class RigidDockingExtraFeats(Task):
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0"""

    groups_fixed = ["ligand_identity", "ligand_identity_extra", "protein_identity", "protein_structure"]
    groups_generated = ["ligand_structure"]

    priors = deepcopy(pc.ligand_conformer)
    priors["npnde_x"] = {
        "type": "target_dependent_gaussian",
    }
    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)

@register_task("flexible_docking_extra_feats")
class FlexibleDockingExtraFeats(Task):
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0"""
    groups_fixed = ['ligand_identity', 'ligand_identity_extra', 'protein_identity']
    groups_generated = ['ligand_structure', 'protein_structure']

    priors = deepcopy(pc.ligand_conformer)
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)


@register_task("fixed_protein_ligand_denovo_extra_feats")
class FixedProteinLigandDeNovoExtraFeats(Task):
    groups_fixed = ['protein_identity', 'protein_structure']
    groups_generated = ['ligand_identity', 'ligand_identity_extra', 'ligand_structure']

    priors = deepcopy(pc.denovo_ligand_extra_feats)

    conditional_paths = dict(**cpc.denovo_ligand_extra_feats, **cpc.protein) # can probably remove protein from this

@register_task("protein_ligand_denovo_extra_feats")
class ProteinLigandDeNovoExtraFeats(Task):
    groups_fixed = ['protein_identity']
    groups_generated = ['protein_structure', 'ligand_identity', 'ligand_identity_extra', 'ligand_structure']

    priors = deepcopy(pc.denovo_ligand_extra_feats)
    
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    
    conditional_paths = dict(**cpc.denovo_ligand_extra_feats, **cpc.protein)

## 
# tasks with ligand + pharmacophore
##


##
# Tasks with ligand+protein+pharmacophore
##


## 
# Tasks with protein+pharmacophore and no ligand
##



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
@register_task("denovo_ligand_condensed")
class DeNovoLigandCondensed(Task):
    groups_fixed = []
    groups_generated = ['ligand_identity_condensed', 'ligand_structure']

    priors = pc.denovo_ligand_condensed
    conditional_paths = cpc.denovo_ligand_condensed


@register_task("ligand_conformer_condensed")
class LigandConformerCondensed(Task):
    groups_fixed = ['ligand_identity_condensed']
    groups_generated = ['ligand_structure']

    priors = pc.ligand_conformer
    conditional_paths = cpc.ligand_conformer

##
# tasks with ligand+protein and no pharmacophore
##
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

@register_task("fixed_protein_ligand_denovo_condensed")
class FixedProteinLigandDeNovoCondensed(Task):
    groups_fixed = ['protein_identity', 'protein_structure']
    groups_generated = ['ligand_identity_condensed', 'ligand_structure']

    priors = deepcopy(pc.denovo_ligand_condensed)

    conditional_paths = dict(**cpc.denovo_ligand_condensed, **cpc.protein) # can probably remove protein from this


@register_task("protein_ligand_denovo_condensed")
class ProteinLigandDeNovoCondensed(Task):
    groups_fixed = ['protein_identity']
    groups_generated = ['protein_structure', 'ligand_identity_condensed', 'ligand_structure']

    priors = deepcopy(pc.denovo_ligand_condensed)
    
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    
    conditional_paths = dict(**cpc.denovo_ligand_condensed, **cpc.protein)


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

@register_task("exp_apo_conditioned_denovo_ligand_condensed")
class ExpApoDeNovoLigandCondensed(Task):
    groups_fixed = ['protein_identity']
    groups_generated = ['ligand_identity_condensed', 'ligand_structure', 'protein_structure']

    priors = deepcopy(pc.denovo_ligand_condensed)
    priors['prot_atom_x'] = {
        'type': 'apo_exp', # in this case the prior is an actual apo structure itself; a sample from a data distribution
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.denovo_ligand_condensed, **cpc.protein)

@register_task("pred_apo_conditioned_denovo_ligand_condensed")
class PredApoDeNovoLigandCondensed(ExpApoDeNovoLigandCondensed):
    priors = deepcopy(pc.denovo_ligand_condensed)
    priors['prot_atom_x'] = {
        'type': 'apo_pred'
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.denovo_ligand_condensed, **cpc.protein)


@register_task("expapo_conditioned_ligand_docking_condensed")
class ExpApoConditionedLigandDockingCondensed(Task):
    """Docking a ligand with condensed atom types into the protein structure, protein structure is an experimentally determined apo structure at t=0."""
    groups_fixed = ['ligand_identity_condensed','protein_identity']
    groups_generated = ['ligand_structure', 'protein_structure']

    priors = deepcopy(pc.ligand_conformer)
    priors['prot_atom_x'] = {
        'type': 'apo_exp',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }
    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)


@register_task("predapo_conditioned_ligand_docking_condensed")
class PredApoConditionedLigandDockingCondensed(Task):
    """Docking a ligand with condensed atom types into the protein structure, protein structure is a predicted apo structure at t=0."""
    groups_fixed = ['ligand_identity_condensed','protein_identity']
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
# tasks with ligand + pharmacophore
##
@register_task("denovo_ligand_pharmacophore_condensed")
class DeNovoLigandPharmacophoreCondensed(Task):
    groups_fixed = []
    groups_generated = ['ligand_identity_condensed', 'ligand_structure', 'pharmacophore']

    priors = dict(**pc.denovo_ligand_condensed, **pc.denovo_pharmacophore)
    conditional_paths = dict(**cpc.denovo_ligand_condensed, **cpc.denovo_pharmacophore)

@register_task("denovo_ligand_from_pharmacophore_condensed")
class DeNovoLigandFromPharmacophoreCondensed(Task):
    groups_fixed = ['pharmacophore']
    groups_generated = ['ligand_identity_condensed', 'ligand_structure']

    priors = pc.denovo_ligand_condensed
    conditional_paths = cpc.denovo_ligand_condensed

@register_task("ligand_conformer_from_pharmacophore_condensed")
class LigandConformerFromPharmacophoreCondensed(Task):
    groups_fixed = ['ligand_identity_condensed', 'pharmacophore']
    groups_generated = ['ligand_structure']

    priors = pc.ligand_conformer
    conditional_paths = cpc.ligand_conformer


##
# Tasks with ligand+protein+pharmacophore
##
@register_task("rigid_docking_pharmacophore_condensed")
class ProteinLigandPharmacophoreDeNovoCondensed(Task):
    groups_fixed = ['protein_identity', 'protein_structure', 'ligand_identity_condensed', 'pharmacophore']
    groups_generated = ['ligand_structure']

    priors = dict(**pc.ligand_conformer)
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }

    conditional_paths = dict(**cpc.ligand_conformer, **cpc.protein)


@register_task("fixed_protein_ligand_pharmacophore_denovo_condensed")
class ProteinLigandPharmacophoreDeNovoCondensed(Task):
    groups_fixed = ['protein_identity', 'protein_structure', 'pharmacophore']
    groups_generated = ['ligand_identity_condensed', 'ligand_structure']

    priors = dict(**pc.denovo_ligand_condensed)

    conditional_paths = dict(**cpc.denovo_ligand_condensed, **cpc.protein)


@register_task("protein_ligand_pharmacophore_denovo_condensed")
class ProteinLigandPharmacophoreDeNovoCondensed(Task):
    groups_fixed = ['protein_identity']
    groups_generated = ['protein_structure', 'ligand_identity_condensed', 'ligand_structure', 'pharmacophore']

    priors = dict(**pc.denovo_ligand_condensed, **pc.denovo_pharmacophore)
    priors['prot_atom_x'] = {
        'type': 'target_dependent_gaussian',
    }
    priors['npnde_x'] = {
        'type': 'target_dependent_gaussian',
    }

    conditional_paths = dict(**cpc.denovo_ligand_condensed, **cpc.denovo_pharmacophore, **cpc.protein)

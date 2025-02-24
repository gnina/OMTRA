import torch
import functools
from omtra.utils.misc import classproperty

canonical_modality_order = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']
canonical_entity_order = ['ligand', 'protein', 'pharmacophore']

class Task:
    protein_state_t0 = 'noise'

    @classproperty
    def t0_modality_arr(self) -> torch.Tensor:
        arr = torch.zeros(len(canonical_modality_order), dtype=bool)
        for i, modality in enumerate(canonical_modality_order):
            if modality in self.observed_at_t0:
                arr[i] = 1
        return arr
    
    @classproperty
    def t1_modality_arr(cls) -> torch.Tensor:
        arr = torch.zeros(len(canonical_modality_order), dtype=bool)
        for i, modality in enumerate(canonical_modality_order):
            if modality in cls.observed_at_t1:
                arr[i] = 1
        return arr
    
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
class DeNovoLigand(Task):
    name = "denovo_ligand"
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure']

class LigandConformer(Task):
    name = "ligand_conformer"
    observed_at_t0 = ['ligand_identity']
    observed_at_t1 = ['ligand_identity', 'ligand_structure']

## 
# tasks with ligand + pharmacophore
##
class DeNovoLigandPharmacophore(Task):
    name = "denovo_ligand_pharmacophore"
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']

class LigandConformerPharmacophore(Task):
    name = "ligand_conformer_pharmacophore"
    observed_at_t0 = ['ligand_identity']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']

##
# tasks with ligand+protein and no pharmacophore
##
class ProteinLigandDeNovo(Task):
    name = "protein_ligand_denovo"
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein']

class ApoConditionedDeNovoLigand(Task):
    name = "apo_conditioned_denovo_ligand"
    observed_at_t0 = ['protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure']
    protein_state_t0 = 'apo'

class UnconditionalLigandDocking(Task):
    name = "unconditional_ligand_docking"
    """Docking a ligand into the protein structure, assuming no knowledge of the protein structure at t=0"""
    observed_at_t0 = ['ligand_identity']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein']

class ApoConditionedLigandDocking(Task):
    name = "apo_conditioned_ligand_docking"
    """Docking a ligand into the protein structure, assuming knowledge of the apo protein structure at t=0"""
    observed_at_t0 = ['ligand_identity', 'protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein']
    protein_state_t0 = 'apo'

##
# Tasks with ligand+protein+pharmacophore
##
class ProteinLigandPharmacophoreDeNovo(Task):
    name = "protein_ligand_pharmacophore_denovo"
    observed_at_t0 = []
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']

class ApoConditionedDeNovoLigandPharmacophore(Task):
    name = "apo_conditioned_denovo_ligand_pharmacophore"
    observed_at_t0 = ['protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'pharmacophore']
    protein_state_t0 = 'apo'

class UnconditionalLigandDockingPharmacophore(Task):
    name = "unconditional_ligand_docking_pharmacophore"
    """Docking a ligand into the protein while generating a pharmacophore, assuming no knowledge of the protein structure at t=0"""
    observed_at_t0 = ['ligand_identity']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']

class ApoConditionedLigandDockingPharmacophore(Task):
    name = "apo_conditioned_ligand_docking_pharmacophore"
    """Docking a ligand into the protein while generating a pharmacophore, assuming knowledge of the apo protein structure at t=0"""
    observed_at_t0 = ['ligand_identity', 'protein']
    observed_at_t1 = ['ligand_identity', 'ligand_structure', 'protein', 'pharmacophore']
    protein_state_t0 = 'apo'

## 
# Tasks with protein+pharmacophore and no ligand
##
class ProteinPharmacophore(Task):
    name = "protein_pharmacophore"
    observed_at_t0 = []
    observed_at_t1 = ['protein', 'pharmacophore']

class ApoConditionedProteinPharmacophore(Task):
    name = "apo_conditioned_protein_pharmacophore"
    observed_at_t0 = ['protein']
    observed_at_t1 = ['protein', 'pharmacophore']
    protein_state_t0 = 'apo'

##
# Tasks with protein only
## 
class ApoProteinSampling(Task):
    name = "apo_protein_sampling"
    "Sampling apo protein conformations, starting from noise for the protein at t=0"
    observed_at_t0 = []
    observed_at_t1 = ['protein']

class ApotoHoloProtein(Task):
    name = "apo_to_holo_protein"
    "Predicting the holo protein structure, starting from the apo protein structure at t=0"
    observed_at_t0 = ['protein']
    observed_at_t1 = ['protein']
    protein_state_t0 = 'apo'



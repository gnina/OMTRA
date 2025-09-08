import functools
from omtra.utils.misc import classproperty
import omtra.tasks.modalities as modal
from omtra.tasks.modalities import Modality, name_to_modality

from typing import List

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
from dataclasses import dataclass
from typing import Dict, Optional
from omtra.utils.misc import classproperty
from omtra.constants import (
    lig_atom_type_map,
    charge_map,
    npnde_atom_type_map,
    ph_idx_to_type,
    residue_map,
    protein_element_map,
    protein_atom_map,
)


@dataclass(frozen=True)
class Modality:
    name: str
    group: str
    graph_entity: str
    entity_name: str
    data_key: str
    n_categories: Optional[int] = None # None if continuous, else number of categories

    @classmethod
    def register(cls, register: Dict[str, "Modality"], **kwargs):
        """Ensures that the register key matches the name attribute when registering a Modality."""
        key = kwargs['name']

        if key in register:
            raise ValueError(f"Modality with name {key} already exists in register.")

        register[key] = cls(**kwargs)
    
    @property
    def is_categorical(self) -> bool:
        """Checks if the modality is categorical."""
        # TODO: whats going on with n_categories == 0? why we doing that?
        return self.n_categories is not None and self.n_categories > 0
    
    @property
    def is_node(self) -> bool:
        """Checks if the modality is defined on nodes."""
        return self.graph_entity == 'node'
    

MODALITY_REGISTER: Dict[str, Modality] = {}

Modality.register(MODALITY_REGISTER,
    name='lig_x',
    group='ligand_structure',
    graph_entity='node',
    entity_name='lig',
    data_key='x',
)

Modality.register(MODALITY_REGISTER,
    name='lig_a',
    group='ligand_identity',
    graph_entity='node',
    entity_name='lig',
    data_key='a',
    n_categories=len(lig_atom_type_map)
)

Modality.register(MODALITY_REGISTER,
    name='lig_c',
    group='ligand_identity',
    graph_entity='node',
    entity_name='lig',
    data_key='c',
    n_categories=len(charge_map)
)

Modality.register(MODALITY_REGISTER,
    name='lig_e',
    group='ligand_identity',
    graph_entity='edge',
    entity_name='lig_to_lig',
    data_key='e',
    n_categories=4, # TODO: consider adding ligand_bond_types to constants.py and use that
)

Modality.register(MODALITY_REGISTER,
    name='pharm_x',
    group='pharmacophore',
    graph_entity='node',
    entity_name='pharm',
    data_key='x'
)

Modality.register(MODALITY_REGISTER,
    name='pharm_a',
    group='pharmacophore',
    graph_entity='node',
    entity_name='pharm',
    data_key='a',
    n_categories=len(ph_idx_to_type)
)

Modality.register(MODALITY_REGISTER,
    name='pharm_v',
    group='pharmacophore',
    graph_entity='node',
    entity_name='pharm',
    data_key='v'
)

Modality.register(MODALITY_REGISTER,
    name='prot_atom_x',
    group='protein_structure',
    graph_entity='node',
    entity_name='prot_atom',
    data_key='x'
)

Modality.register(MODALITY_REGISTER,
    name='prot_atom_element',
    group='protein_identity',
    graph_entity='node',
    entity_name='prot_atom',
    data_key='e',
    n_categories=len(protein_element_map),
)
Modality.register(MODALITY_REGISTER,
    name='prot_atom_name',
    group='protein_identity',
    graph_entity='node',
    entity_name='prot_atom',
    data_key='a',
    n_categories=len(protein_atom_map),
)

Modality.register(MODALITY_REGISTER,
    name='npnde_x',
    group='protein_structure',
    graph_entity='node',
    entity_name='npnde',
    data_key='x'
)

# TODO: how do we model npndes in our graphs?
Modality.register(MODALITY_REGISTER,
    name='npnde_a',
    group='protein_identity',
    graph_entity='node',
    entity_name='npnde',
    data_key='a',
    n_categories=len(npnde_atom_type_map)
)
Modality.register(MODALITY_REGISTER,
    name='npnde_c',
    group='protein_identity',
    graph_entity='node',
    entity_name='npnde',
    data_key='c',
    n_categories=len(charge_map)
)

# TODO: is it really necessary to model bond orders in npndes? maybe just sparse?
Modality.register(MODALITY_REGISTER,
    name='npnde_e',
    group='protein_identity',
    graph_entity='edge',
    entity_name='npnde_to_npnde',
    data_key='e',
    n_categories=4, # TODO: either use edge types from constants or dont use edge types at all for npndes
)

MODALITY_ORDER = [modality.name for modality in MODALITY_REGISTER.values()]
GROUP_SPACE = set([modality.group for modality in MODALITY_REGISTER.values()])
DESIGN_SPACE = set([modality.name for modality in MODALITY_REGISTER.values() if (modality.group not in ["protein_identity"])])

def name_to_modality(name: str) -> Modality:
    return MODALITY_REGISTER[name]
from dataclasses import dataclass
from typing import Dict


@dataclass
class Modality:
    name: str
    group: str
    graph_entity: str
    entity_name: str
    data_key: str

    @classmethod
    def register(cls, register: Dict[str, "Modality"], **kwargs):
        """Ensures that the register key matches the name attribute when registering a Modality."""
        key = kwargs['name']

        if key in register:
            raise ValueError(f"Modality with name {key} already exists in register.")

        register[key] = cls(**kwargs)

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
    data_key='a'
)

Modality.register(MODALITY_REGISTER,
    name='lig_c',
    group='ligand_identity',
    graph_entity='node',
    entity_name='lig',
    data_key='c'
)

Modality.register(MODALITY_REGISTER,
    name='lig_e',
    group='ligand_identity',
    graph_entity='edge',
    entity_name='lig_to_lig',
    data_key='e'
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
    data_key='a'
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
    data_key='e'
)
Modality.register(MODALITY_REGISTER,
    name='prot_atom_name',
    group='protein_identity',
    graph_entity='node',
    entity_name='prot_atom',
    data_key='a'
)
Modality.register(MODALITY_REGISTER,
    name='prot_atom_e',
    group='protein_identity',
    graph_entity='edge',
    entity_name='prot_atom_to_prot_atom',
    data_key='e'
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
    data_key='a'
)
Modality.register(MODALITY_REGISTER,
    name='npnde_c',
    group='protein_identity',
    graph_entity='node',
    entity_name='npnde',
    data_key='c'
)

# TODO: is it really necessary to model bond orders in npndes? maybe just sparse?
Modality.register(MODALITY_REGISTER,
    name='npnde_e',
    group='protein_identity',
    graph_entity='edge',
    entity_name='npnde_to_npnde',
    data_key='e'
)

MODALITY_ORDER = [modality.name for modality in MODALITY_REGISTER.values()]
GROUP_SPACE = set([modality.group for modality in MODALITY_REGISTER.values()])

def name_to_modality(name: str) -> Modality:
    return MODALITY_REGISTER[name]
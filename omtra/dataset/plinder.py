import dgl
import torch
from omegaconf import DictConfig

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.data.graph import build_complex_graph
from omtra.data.xace_ligand import sparse_to_dense
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from omtra.utils.misc import classproperty
from omtra.data.plinder import LigandData, PharmacophoreData, StructureData, SystemData
from typing import List, Dict
import pandas as pd

class PlinderDataset(ZarrDataset):
    def __init__(self, 
                 split: str,
                 processed_data_dir: str,
                 graphs_per_chunk: int,
                 graph_config: DictConfig,
    ):
        super().__init__(split, processed_data_dir)
        self.graphs_per_chunk = graphs_per_chunk
        self.graph_config = graph_config

        self.system_lookup = pd.DataFrame(self.root.attrs["system_lookup"])
        self.npnde_lookup = pd.DataFrame(self.root.attrs["npnde_lookup"])

    @classproperty
    def name(cls):
        return 'plinder'

    def __len__(self):
        return self.system_lookup.shape[0]
    
    def get_npndes(self, npnde_idxs: List[int]) -> Dict[str, LigandData]:
        npndes = {}
        for idx in npnde_idxs:
            npnde_info = self.npnde_lookup[self.npnde_lookup["npnde_idx"] == idx].iloc[
                0
            ]

            key = npnde_info["npnde_id"]

            atom_start, atom_end = npnde_info["atom_start"], npnde_info["atom_end"]
            bond_start, bond_end = npnde_info["bond_start"], npnde_info["bond_end"]

            is_covalent = False
            if npnde_info["linkages"]:
                is_covalent = True

            npndes[key] = LigandData(
                sdf=npnde_info["lig_sdf"],
                ccd=npnde_info["ccd"],
                is_covalent=is_covalent,
                linkages=npnde_info["linkages"],
                coords=self.slice_array("npnde/coords", atom_start, atom_end),
                atom_types=self.slice_array("npnde/atom_types", atom_start, atom_end),
                atom_charges=self.slice_array("npnde/atom_charges", atom_start, atom_end),
                bond_types=self.slice_array("npnde/bond_types", bond_start, bond_end),
                bond_indices=self.slice_array("npnde/bond_indices", bond_start, bond_end),
            )
        return npndes

    def get_system(self, index: int) -> SystemData:
        system_info = self.system_lookup[
            self.system_lookup["system_idx"] == index
        ].iloc[0]

        rec_start, rec_end = system_info["rec_start"], system_info["rec_end"]
        lig_atom_start, lig_atom_end = (
            system_info["lig_atom_start"],
            system_info["lig_atom_end"],
        )
        lig_bond_start, lig_bond_end = (
            system_info["lig_bond_start"],
            system_info["lig_bond_end"],
        )
        pharm_start, pharm_end = system_info["pharm_start"], system_info["pharm_end"]
        pocket_start, pocket_end = (
            system_info["pocket_start"],
            system_info["pocket_end"],
        )
        link_start, link_end = system_info["link_start"], system_info["link_end"]
        link_type = system_info["link_type"]

        receptor = StructureData(
            coords=self.slice_array("receptor/coords", rec_start, rec_end),
            atom_names=self.slice_array("receptor/atom_names", rec_start, rec_end).astype(str),
            elements=self.slice_array("receptor/elements", rec_start, rec_end).astype(str),
            res_ids=self.slice_array("receptor/res_ids", rec_start, rec_end),
            res_names=self.slice_array("receptor/res_names", rec_start, rec_end).astype(str),
            chain_ids=self.slice_array("receptor/chain_ids", rec_start, rec_end).astype(str),
            cif=system_info["rec_cif"],
        )

        is_covalent = False
        if system_info["linkages"]:
            is_covalent = True

        ligand = LigandData(
            sdf=system_info["lig_sdf"],
            ccd=system_info["ccd"],
            is_covalent=is_covalent,
            linkages=system_info["linkages"],
            coords=self.slice_array("ligand/coords", lig_atom_start, lig_atom_end),
            atom_types=self.slice_array("ligand/atom_types", lig_atom_start, lig_atom_end),
            atom_charges=self.slice_array("ligand/atom_charges", lig_atom_start, lig_atom_end),
            bond_types=self.slice_array("ligand/bond_types", lig_bond_start, lig_bond_end),
            bond_indices=self.slice_array("ligand/bond_indices", lig_bond_start, lig_bond_end),
        )

        pharmacophore = PharmacophoreData(
            coords=self.slice_array("pharmacophore/coords", pharm_start, pharm_end),
            types=self.slice_array("pharmacophore/types", pharm_start, pharm_end),
            vectors=self.slice_array("pharmacophore/vectors", pharm_start, pharm_end),
            interactions=self.slice_array("pharmacophore/interactions", pharm_start, pharm_end),
        )

        pocket = StructureData(
            coords=self.slice_array("pocket/coords", pocket_start, pocket_end),
            atom_names=self.slice_array("pocket/atom_names", pocket_start, pocket_end).astype(str),
            elements=self.slice_array("pocket/elements", pocket_start, pocket_end).astype(str),
            res_ids=self.slice_array("pocket/res_ids", pocket_start, pocket_end),
            res_names=self.slice_array("pocket/res_names", pocket_start, pocket_end).astype(
                str
            ),
            chain_ids=self.slice_array("pocket/chain_ids", pocket_start, pocket_end).astype(
                str
            ),
        )
        npndes = None
        if system_info["npnde_idxs"]:
            npndes = self.get_npndes(system_info["npnde_idxs"])

        apo = None
        pred = None
        if link_type == "apo":
            apo = StructureData(
                coords=self.slice_array("apo/coords", link_start, link_end),
                atom_names=self.slice_array("apo/atom_names", link_start, link_end).astype(
                    str
                ),
                elements=self.slice_array("apo/elements", link_start, link_end).astype(
                    str
                ),
                res_ids=self.slice_array("apo/res_ids", link_start, link_end),
                res_names=self.slice_array("apo/res_names", link_start, link_end).astype(
                    str
                ),
                chain_ids=self.slice_array("apo/chain_ids", link_start, link_end).astype(
                    str
                ),
                cif=system_info["link_cif"],
            )
        elif link_type == "pred":
            pred = StructureData(
                coords=self.slice_array("pred/coords", link_start, link_end),
                atom_names=self.slice_array("pred/atom_names", link_start, link_end).astype(
                    str
                ),
                elements=self.slice_array("pred/elements", link_start, link_end).astype(
                    str
                ),
                res_ids=self.slice_array("pred/res_ids", link_start, link_end),
                res_names=self.slice_array("pred/res_names", link_start, link_end).astype(
                    str
                ),
                chain_ids=self.slice_array("pred/chain_ids", link_start, link_end).astype(
                    str
                ),
                cif=system_info["link_cif"],
            )

        system = SystemData(
            system_id=system_info["system_id"],
            ligand_id=system_info["ligand_id"],
            receptor=receptor,
            ligand=ligand,
            pharmacophore=pharmacophore,
            pocket=pocket,
            npndes=npndes,
            link_type=link_type,
            link_id=system_info["link_id"],
            link=apo if apo else pred,
        )
        return system

    def __getitem__(self, index) -> dgl.DGLHeteroGraph:
        task_name, idx = index
        task_class: Task = task_name_to_class[task_name]

        system = self.get_system(index)

        # TODO: things!
        g = build_complex_graph()

        return g
    
    def retrieve_graph_chunks(self, apo_systems: bool = False):
        """
        This dataset contains len(self) examples. We divide all samples (or, graphs) into separate chunk. 
        We call these "graph chunks"; this is not the same thing as chunks defined in zarr arrays.
        I know we need better terminology; but they're chunks! they're totally chunks. just a different kind of chunk.
        """
        n_graphs = len(self) # this is wrong! n_graphs depends on apo_systems!!!!
        n_even_chunks, n_graphs_in_last_chunk = divmod(n_graphs, self.graphs_per_chunk)

        n_chunks = n_even_chunks + int(n_graphs_in_last_chunk > 0)

        raise NotImplementedError("need to build capability to modify chunks based on whether or not the task uses the apo state")

        # construct a tensor containing the index ranges for each chunk
        chunk_index = torch.zeros(n_chunks, 2, dtype=torch.int64)
        chunk_index[:, 0] = self.graphs_per_chunk*torch.arange(n_chunks)
        chunk_index[:-1, 1] = chunk_index[1:, 0]
        chunk_index[-1, 1] = n_graphs

        return chunk_index
    
    def get_num_nodes(self, task: Task, start_idx, end_idx):
        # here, unlike in other places, start_idx and end_idx are 
        # indexes into the graph_lookup array, not a node/edge data array

        node_types = ['lig']
        if 'pharmacophore' in task.modalities_present:
            node_types.append('pharm')

        node_counts = []
        for ntype in node_types:
            graph_lookup = self.slice_array(f'{ntype}/node/graph_lookup', start_idx, end_idx)
            node_counts.append(graph_lookup[:, 1] - graph_lookup[:, 0])

        node_counts = np.stack(node_counts, axis=0).sum(axis=0)
        node_counts = torch.from_numpy(node_counts)
        return node_counts
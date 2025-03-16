import dgl
import torch
from omegaconf import DictConfig

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.data.graph import build_complex_graph
from omtra.data.graph import edge_builders
from omtra.data.xace_ligand import sparse_to_dense
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from omtra.utils.misc import classproperty
from omtra.data.plinder import LigandData, PharmacophoreData, StructureData, SystemData
from typing import List, Dict
import pandas as pd


class PlinderDataset(ZarrDataset):
    def __init__(
        self,
        split: str,
        processed_data_dir: str,
        graphs_per_chunk: int,
        graph_config: DictConfig,
    ):
        super().__init__(split, processed_data_dir)
        self.graphs_per_chunk = graphs_per_chunk
        self.graph_config = graph_config

        self.include_pharmacophore = False
        self.system_lookup = pd.DataFrame(self.root.attrs["system_lookup"])
        self.npnde_lookup = pd.DataFrame(self.root.attrs["npnde_lookup"])

    @classproperty
    def name(cls):
        return "plinder"

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

            bond_start = (
                npnde_info["bond_start"]
                if not pd.isna(npnde_info["bond_start"])
                else None
            )
            bond_end = (
                npnde_info["bond_end"] if not pd.isna(npnde_info["bond_end"]) else None
            )

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
                atom_charges=self.slice_array(
                    "npnde/atom_charges", atom_start, atom_end
                ),
                bond_types=self.slice_array(
                    "npnde/bond_types", int(bond_start), int(bond_end)
                )
                if bond_start is not None and bond_end is not None
                else None,
                bond_indices=self.slice_array(
                    "npnde/bond_indices", int(bond_start), int(bond_end)
                )
                if bond_start is not None and bond_end is not None
                else None,
            )
        return npndes

    def get_system(self, index: int) -> SystemData:
        system_info = self.system_lookup[
            self.system_lookup["system_idx"] == index
        ].iloc[0]

        rec_start, rec_end = system_info["rec_start"], system_info["rec_end"]
        backbone_start, backbone_end = (
            system_info["backbone_start"],
            system_info["backbone_end"],
        )

        lig_atom_start, lig_atom_end = (
            system_info["lig_atom_start"],
            system_info["lig_atom_end"],
        )
        lig_bond_start, lig_bond_end = (
            system_info["lig_bond_start"],
            system_info["lig_bond_end"],
        )

        pocket_start, pocket_end = (
            system_info["pocket_start"],
            system_info["pocket_end"],
        )
        pocket_bb_start, pocket_bb_end = (
            system_info["pocket_bb_start"],
            system_info["pocket_bb_end"],
        )

        link_start, link_end = system_info["link_start"], system_info["link_end"]
        link_bb_start, link_bb_end = (
            system_info["link_bb_start"],
            system_info["link_bb_end"],
        )
        link_type = system_info["link_type"]

        backbone = BackboneData(
            coords=self.slice_array(
                "receptor/backbone_coords", backbone_start, backbone_end
            ),
            res_ids=self.slice_array(
                "receptor/backbone_res_ids", backbone_start, backbone_end
            ),
            res_names=self.slice_array(
                "receptor/backbone_res_names", backbone_start, backbone_end
            ).astype(str),
            chain_ids=self.slice_array(
                "receptor/backbone_chain_ids", backbone_start, backbone_end
            ).astype(str),
        )

        receptor = StructureData(
            coords=self.slice_array("receptor/coords", rec_start, rec_end),
            atom_names=self.slice_array(
                "receptor/atom_names", rec_start, rec_end
            ).astype(str),
            elements=self.slice_array("receptor/elements", rec_start, rec_end).astype(
                str
            ),
            res_ids=self.slice_array("receptor/res_ids", rec_start, rec_end),
            res_names=self.slice_array("receptor/res_names", rec_start, rec_end).astype(
                str
            ),
            chain_ids=self.slice_array("receptor/chain_ids", rec_start, rec_end).astype(
                str
            ),
            backbone_mask=self.slice_array(
                "receptor/backbone_mask", rec_start, rec_end
            ),
            backbone=backbone,
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
            coords=self.slice_array("ligand/coords", lig_atom_start, lig_atom_end),  # x
            atom_types=self.slice_array(
                "ligand/atom_types", lig_atom_start, lig_atom_end
            ),  # a
            atom_charges=self.slice_array(
                "ligand/atom_charges", lig_atom_start, lig_atom_end
            ),  # c
            bond_types=self.slice_array(
                "ligand/bond_types", lig_bond_start, lig_bond_end
            ),  # e
            bond_indices=self.slice_array(
                "ligand/bond_indices", lig_bond_start, lig_bond_end
            ),  # edge index
        )

        if self.include_pharmacophore:
            pharm_start, pharm_end = (
                system_info["pharm_start"],
                system_info["pharm_end"],
            )
            pharmacophore = PharmacophoreData(
                coords=self.slice_array("pharmacophore/coords", pharm_start, pharm_end),
                types=self.slice_array("pharmacophore/types", pharm_start, pharm_end),
                vectors=self.slice_array(
                    "pharmacophore/vectors", pharm_start, pharm_end
                ),
                interactions=self.slice_array(
                    "pharmacophore/interactions", pharm_start, pharm_end
                ),
            )

        pocket_backbone = BackboneData(
            coords=self.slice_array(
                "pocket/backbone_coords", pocket_bb_start, pocket_bb_end
            ),
            res_ids=self.slice_array(
                "pocket/backbone_res_ids", pocket_bb_start, pocket_bb_end
            ),
            res_names=self.slice_array(
                "pocket/backbone_res_names", pocket_bb_start, pocket_bb_end
            ).astype(str),
            chain_ids=self.slice_array(
                "pocket/backbone_chain_ids", pocket_bb_start, pocket_bb_end
            ).astype(str),
        )

        pocket = StructureData(
            coords=self.slice_array("pocket/coords", pocket_start, pocket_end),
            atom_names=self.slice_array(
                "pocket/atom_names", pocket_start, pocket_end
            ).astype(str),
            elements=self.slice_array(
                "pocket/elements", pocket_start, pocket_end
            ).astype(str),
            res_ids=self.slice_array("pocket/res_ids", pocket_start, pocket_end),
            res_names=self.slice_array(
                "pocket/res_names", pocket_start, pocket_end
            ).astype(str),
            chain_ids=self.slice_array(
                "pocket/chain_ids", pocket_start, pocket_end
            ).astype(str),
            backbone_mask=self.slice_array(
                "pocket/backbone_mask", pocket_start, pocket_end
            ),
            backbone=pocket_backbone,
        )

        npndes = None
        if system_info["npnde_idxs"]:
            npndes = self.get_npndes(system_info["npnde_idxs"])

        apo = None
        pred = None
        if link_type == "apo":
            apo_backbone = BackboneData(
                coords=self.slice_array(
                    "apo/backbone_coords", link_bb_start, link_bb_end
                ),
                res_ids=self.slice_array(
                    "apo/backbone_res_ids", link_bb_start, link_bb_end
                ),
                res_names=self.slice_array(
                    "apo/backbone_res_names", link_bb_start, link_bb_end
                ).astype(str),
                chain_ids=self.slice_array(
                    "apo/backbone_chain_ids", link_bb_start, link_bb_end
                ).astype(str),
            )
            apo = StructureData(
                coords=self.slice_array("apo/coords", link_start, link_end),
                atom_names=self.slice_array(
                    "apo/atom_names", link_start, link_end
                ).astype(str),
                elements=self.slice_array("apo/elements", link_start, link_end).astype(
                    str
                ),
                res_ids=self.slice_array("apo/res_ids", link_start, link_end),
                res_names=self.slice_array(
                    "apo/res_names", link_start, link_end
                ).astype(str),
                chain_ids=self.slice_array(
                    "apo/chain_ids", link_start, link_end
                ).astype(str),
                cif=system_info["link_cif"],
                backbone_mask=self.slice_array(
                    "apo/backbone_mask", link_start, link_end
                ),
                backbone=apo_backbone,
            )
        elif link_type == "pred":
            pred_backbone = BackboneData(
                coords=self.slice_array(
                    "pred/backbone_coords", link_bb_start, link_bb_end
                ),
                res_ids=self.slice_array(
                    "pred/backbone_res_ids", link_bb_start, link_bb_end
                ),
                res_names=self.slice_array(
                    "pred/backbone_res_names", link_bb_start, link_bb_end
                ).astype(str),
                chain_ids=self.slice_array(
                    "pred/backbone_chain_ids", link_bb_start, link_bb_end
                ).astype(str),
            )
            pred = StructureData(
                coords=self.slice_array("pred/coords", link_start, link_end),
                atom_names=self.slice_array(
                    "pred/atom_names", link_start, link_end
                ).astype(str),
                elements=self.slice_array("pred/elements", link_start, link_end).astype(
                    str
                ),
                res_ids=self.slice_array("pred/res_ids", link_start, link_end),
                res_names=self.slice_array(
                    "pred/res_names", link_start, link_end
                ).astype(str),
                chain_ids=self.slice_array(
                    "pred/chain_ids", link_start, link_end
                ).astype(str),
                cif=system_info["link_cif"],
                backbone_mask=self.slice_array(
                    "pred/backbone_mask", link_start, link_end
                ),
                backbone=pred_backbone,
            )

        system = SystemData(
            system_id=system_info["system_id"],
            ligand_id=system_info["ligand_id"],
            receptor=receptor,
            ligand=ligand,
            pharmacophore=pharmacophore if self.include_pharmacophore else None,
            pocket=pocket,
            npndes=npndes,
            link_type=link_type,
            link_id=system_info["link_id"],
            link=apo if apo else pred,
        )
        return system

    def encode_atom_names(self, atom_names: np.ndarray) -> np.ndarray:
        # TODO: encode atom_names
        pass

    def encode_elements(self, elements: np.ndarray) -> np.ndarray:
        # TODO: encode elements
        pass

    def encode_res_names(self, res_names: np.ndarray) -> np.ndarray:
        # TODO: encode res_names
        pass

    def convert_protein(
        self,
        holo: StructureData,
        link: StructureData,
        pocket: StructureData,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        node_data = {}
        edge_idxs = {}
        edge_data = {}

        prot_coords = torch.from_numpy(holo.coords).float()
        prot_atom_names = torch.from_numpy(encode_atom_names(holo.atom_names)).long()
        prot_elements = torch.from_numpy(encode_elements(holo.elements)).long()
        prot_res_ids = torch.from_numpy(holo.res_ids).long()
        prot_res_names = torch.from_numpy(encode_res_names(holo.res_names)).long()
        prot_backbone_mask = torch.from_numpy(holo.backbone_mask).bool()

        link_coords = torch.from_numpy(link.coords).float()

        # TODO: figure out how to store chain ids
        prot_chain_ids = torch.from_numpy(holo.chain_ids.astype(np.int64)).long()

        pocket_res_identifiers = set()
        for i in range(len(pocket.res_ids)):
            chain_id = pocket.chain_ids[i]
            res_id = pocket.res_ids[i]
            pocket_res_identifiers.add((chain_id, res_id))

        pocket_mask = torch.zeros_like(prot_res_ids, dtype=torch.bool)
        for i in range(len(prot_res_ids)):
            chain_id = holo.chain_ids[i]
            res_id = prot_res_ids[i].item()
            if (chain_id, res_id) in pocket_res_identifiers:
                pocket_mask[i] = True

        node_data["prot_atom"] = {
            "x_1_true": prot_coords,
            "x_0": link_coords,
            "a": prot_atom_names,
            "e": prot_elements,
            "res_id": prot_res_ids,
            "res_names": prot_res_names,
            "chain_id": prot_chain_ids,
            "pocket_mask": pocket_mask,
            "backbone_mask": prot_backbone_mask,
        }

        backbone_coords = torch.from_numpy(holo.backbone.coords).float()
        backbone_res_ids = torch.from_numpy(holo.backbone.res_ids).long()
        backbone_res_names = torch.from_numpy(
            encode_res_names(holo.backbone.res_names)
        ).long()

        # TODO: figure out how to store chain ids
        backbone_chain_ids = torch.from_numpy(
            holo.backbone.chain_ids.astype(np.int64)
        ).long()

        backbone_pocket_mask = torch.zeros_like(backbone_res_ids, dtype=torch.bool)
        for i in range(len(backbone_res_ids)):
            chain_id = holo.backbone.chain_ids[i]
            res_id = backbone_res_ids[i].item()
            if (chain_id, res_id) in pocket_res_identifiers:
                backbone_pocket_mask[i] = True

        link_backbone_coords = torch.from_numpy(link.backbone.coords).float()

        node_data["prot_res"] = {
            "x_1_true": backbone_coords,
            "x_0": link_backbone_coords,
            "res_id": backbone_res_ids,
            "res_name": backbone_res_names,
            "chain_id": backbone_chain_ids,
            "pocket_mask": backbone_pocket_mask,
        }

        return node_data, edge_idxs, edge_data

    def convert_ligand(
        self,
        ligand: LigandData,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        coords = torch.from_numpy(ligand.coords).float()
        atom_types = torch.from_numpy(ligand.atom_types).long()
        atom_charges = torch.from_numpy(ligand.atom_charges).long()

        if ligand.bond_types is not None and ligand.bond_indices is not None:
            bond_types = torch.from_numpy(ligand.bond_types).long()
            bond_indices = torch.from_numpy(ligand.bond_indices).long()
        else:
            bond_types = torch.zeros((0,), dtype=torch.long)
            bond_indices = torch.zeros((2, 0), dtype=torch.long)

        lig_x, lig_a, lig_c, lig_e, lig_edge_idxs = sparse_to_dense(
            coords, atom_types, atom_charges, bond_types, bond_indices
        )

        node_data = {
            "lig": {
                "x": lig_x,
                "a": lig_a,
                "c": lig_c,
            }
        }

        edge_data = {
            "lig_to_lig": {
                "e": lig_e,
            }
        }

        edge_idxs = {
            "lig_to_lig": lig_edge_idxs,
        }

        return node_data, edge_idxs, edge_data

    def convert_npndes(
        self, npndes: Optional[Dict[str, LigandData]]
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        node_data = {}
        edge_idxs = {}
        edge_data = {}

        if not npndes:
            return node_data, edge_idxs, edge_data

        all_coords = []
        all_atom_types = []
        all_atom_charges = []
        all_bond_types = []
        all_bond_indices = []

        node_offset = 0

        for npnde_id, ligand_data in npndes.items():
            coords = torch.from_numpy(ligand_data.coords).float()
            atom_types = torch.from_numpy(ligand_data.atom_types).long()
            atom_charges = torch.from_numpy(ligand_data.atom_charges).long()

            all_coords.append(coords)
            all_atom_types.append(atom_types)
            all_atom_charges.append(atom_charges)

            if (
                ligand_data.bond_types is not None
                and ligand_data.bond_indices is not None
            ):
                bond_types = torch.from_numpy(ligand_data.bond_types).long()
                bond_indices = torch.from_numpy(ligand_data.bond_indices).long()

                adjusted_indices = bond_indices.clone()
                adjusted_indices[0, :] += node_offset
                adjusted_indices[1, :] += node_offset

                all_bond_types.append(bond_types)
                all_bond_indices.append(adjusted_indices)

            node_offset += coords.shape[0]

        combined_coords = (
            torch.cat(all_coords, dim=0)
            if all_coords
            else torch.zeros((0, 3), dtype=torch.float)
        )
        combined_atom_types = (
            torch.cat(all_atom_types, dim=0)
            if all_atom_types
            else torch.zeros((0,), dtype=torch.long)
        )
        combined_atom_charges = (
            torch.cat(all_atom_charges, dim=0)
            if all_atom_charges
            else torch.zeros((0,), dtype=torch.long)
        )

        if all_bond_types and all_bond_indices:
            combined_bond_types = torch.cat(all_bond_types, dim=0)
            combined_bond_indices = torch.cat(all_bond_indices, dim=1)

            npnde_x, npnde_a, npnde_c, npnde_e, npnde_edge_idxs = sparse_to_dense(
                combined_coords,
                combined_atom_types,
                combined_atom_charges,
                combined_bond_types,
                combined_bond_indices,
            )

            node_data["npnde"] = {"x": npnde_x, "a": npnde_a, "c": npnde_c}

            edge_data["npnde_to_npnde"] = {"e": npnde_e}

            edge_idxs["npnde_to_npnde"] = npnde_edge_idxs
        else:
            node_data["npnde"] = {
                "x": combined_coords,
                "a": combined_atom_types,
                "c": combined_atom_charges,
            }

        return node_data, edge_idxs, edge_data

    def convert_pharmacophore(
        self, pharmacophore: PharmacophoreData
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        node_data = {}
        edge_idxs = {}
        edge_data = {}

        coords = torch.from_numpy(pharmacophore.coords).float()
        types = torch.from_numpy(pharmacophore.types).long()
        vectors = torch.from_numpy(pharmacophore.vectors).float()
        interactions = torch.from_numpy(pharmacophore.interactions).bool()

        node_data["pharm"] = {"x": coords, "a": types, "v": vectors, "i": interactions}

        assert self.graph_config.edges["pharm_to_pharm"]["type"] == "complete", (
            "the following code assumes complete pharm-pharm graph"
        )

        num_centers = coords.shape[0]
        if num_centers > 1:
            pharm_edge_idxs = edge_builders.complete_graph(coords)
            edge_idxs["pharm_to_pharm"] = pharm_edge_idxs

        return node_data, edge_idxs, edge_data

    def convert_system(
        system: SystemData,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        node_data = {}
        edge_idxs = {}
        edge_data = {}

        prot_node_data, prot_edge_idxs, prot_edge_data = self.convert_protein(
            system.receptor, system.link, system.pocket
        )
        node_data.update(prot_node_data)
        edge_idxs.udpate(prot_edge_idxs)
        edge_data.update(prot_edge_data)

        lig_node_data, lig_edge_idxs, lig_edge_data = self.convert_ligand(system.ligand)
        node_data.update(lig_node_data)
        edge_idxs.udpate(lig_edge_idxs)
        edge_data.update(lig_edge_data)

        npnde_node_data, npnde_edge_idxs, npnde_edge_data = self.convert_npndes(
            system.npndes
        )
        node_data.update(npnde_node_data)
        edge_idxs.udpate(npnde_edge_idxs)
        edge_data.update(npnde_edge_data)

        if self.include_pharmacophore:
            pharm_node_data, pharm_edge_idxs, pharm_edge_data = (
                self.convert_pharmacophore(system.pharmacophore)
            )
            node_data.update(pharm_node_data)
            edge_idxs.udpate(pharm_edge_idxs)
            edge_data.update(pharm_edge_data)

        return node_data, edge_idxs, edge_data

    def __getitem__(self, index) -> dgl.DGLHeteroGraph:
        task_name, idx = index
        task_class: Task = task_name_to_class[task_name]
        self.include_pharmacophore = "pharmacophore" in task_class.modalities_present

        system = self.get_system(index)

        node_data, edge_idxs, edge_data = convert_system(system)

        # TODO: things!
        g = build_complex_graph(node_data, edge_idxs, edge_data)

        return g

    def retrieve_graph_chunks(self, apo_systems: bool = False):
        """
        This dataset contains len(self) examples. We divide all samples (or, graphs) into separate chunk.
        We call these "graph chunks"; this is not the same thing as chunks defined in zarr arrays.
        I know we need better terminology; but they're chunks! they're totally chunks. just a different kind of chunk.
        """
        n_graphs = len(self)  # this is wrong! n_graphs depends on apo_systems!!!!
        n_even_chunks, n_graphs_in_last_chunk = divmod(n_graphs, self.graphs_per_chunk)

        n_chunks = n_even_chunks + int(n_graphs_in_last_chunk > 0)

        raise NotImplementedError(
            "need to build capability to modify chunks based on whether or not the task uses the apo state"
        )

        # construct a tensor containing the index ranges for each chunk
        chunk_index = torch.zeros(n_chunks, 2, dtype=torch.int64)
        chunk_index[:, 0] = self.graphs_per_chunk * torch.arange(n_chunks)
        chunk_index[:-1, 1] = chunk_index[1:, 0]
        chunk_index[-1, 1] = n_graphs

        return chunk_index

    def get_num_nodes(self, task: Task, start_idx, end_idx):
        # here, unlike in other places, start_idx and end_idx are
        # indexes into the graph_lookup array, not a node/edge data array

        node_types = ["lig"]
        if "pharmacophore" in task.modalities_present:
            node_types.append("pharm")

        node_counts = []
        for ntype in node_types:
            graph_lookup = self.slice_array(
                f"{ntype}/node/graph_lookup", start_idx, end_idx
            )
            node_counts.append(graph_lookup[:, 1] - graph_lookup[:, 0])

        node_counts = np.stack(node_counts, axis=0).sum(axis=0)
        node_counts = torch.from_numpy(node_counts)
        return node_counts

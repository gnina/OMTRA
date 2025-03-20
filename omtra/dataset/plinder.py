import dgl
import torch
from omegaconf import DictConfig

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.constants import (
    lig_atom_type_map,
    npnde_atom_type_map,
    ph_idx_to_type,
    aa_substitutions,
    residue_map,
    protein_element_map,
    protein_atom_map,
)
from omtra.data.graph import build_complex_graph
from omtra.data.graph import edge_builders
from omtra.data.xace_ligand import sparse_to_dense
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from omtra.utils.misc import classproperty
from omtra.priors.prior_factory import get_prior
from omtra.tasks.modalities import name_to_modality
from omtra.data.plinder import (
    LigandData,
    PharmacophoreData,
    StructureData,
    SystemData,
    BackboneData,
)
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
import functools


class PlinderDataset(ZarrDataset):
    def __init__(
        self,
        split: str,
        processed_data_dir: str,
        graphs_per_chunk: int = 1,
        graph_config: Optional[DictConfig] = None,
        prior_config: Optional[DictConfig] = None,
        include_pharmacophore: bool = False,
    ):
        super().__init__(split, processed_data_dir)
        self.graphs_per_chunk = graphs_per_chunk
        self.graph_config = graph_config
        self.prior_config = prior_config

        self.include_pharmacophore = include_pharmacophore
        self.system_lookup = pd.DataFrame(self.root.attrs["system_lookup"])
        self.npnde_lookup = pd.DataFrame(self.root.attrs["npnde_lookup"])

        self.encode_element = {
            element: i for i, element in enumerate(protein_element_map)
        }
        self.encode_residue = {res: i for i, res in enumerate(residue_map)}
        self.encode_atom = {atom: i for i, atom in enumerate(protein_atom_map)}
        self.n_categories_dict = {
            'lig_a': len(lig_atom_type_map),
            # 'lig_c': len(lig_c_idx_to_val),
            'lig_c': 10, # TODO: change this later just for debugging
            'lig_e': 4, # hard-coded assumption of 4 bond types (none, single, double, triple)
            'pharm_a': len(ph_idx_to_type),
        }

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
                else np.zeros((0,), dtype=np.int32),
                bond_indices=self.slice_array(
                    "npnde/bond_indices", int(bond_start), int(bond_end)
                )
                if bond_start is not None and bond_end is not None
                else np.zeros((0, 2), dtype=np.int32),
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
            pharmacophore=pharmacophore if self.include_pharmacophore else PharmacophoreData(
                coords=np.zeros((0, 3), dtype=np.float32),
                types=np.zeros((0,), dtype=np.int32),
                vectors=np.zeros((0, 3), dtype=np.float32),
                interactions=np.zeros((0,), dtype=bool),
            ),
            pocket=pocket,
            npndes=npndes,
            link_type=link_type,
            link_id=system_info["link_id"] if link_type else None,
            link=apo if apo else pred,
        )
        return system

    def encode_atom_names(
        self, atom_names: np.ndarray, elements: np.ndarray, res_names: np.ndarray
    ) -> np.ndarray:
        encoded_atom_names = []
        for i, atom_name in enumerate(atom_names):
            if atom_name in self.encode_atom:
                atom_code = self.encode_atom[atom_name]
            else:
                atom_code = self.encode_atom["UNK"]
            encoded_atom_names.append(atom_code)
        return np.array(encoded_atom_names)

    def encode_elements(self, elements: np.ndarray) -> np.ndarray:
        encoded_elements = []
        for element in elements:
            code = self.encode_element[element]
            encoded_elements.append(code)
        return np.array(encoded_elements)

    def encode_res_names(self, res_names: np.ndarray) -> np.ndarray:
        encoded_residues = []
        for res in res_names:
            if res not in self.encode_residue:
                sub = aa_substitutions.get(res, "UNK")
                code = self.encode_residue[sub]
            else:
                code = self.encode_residue[res]
            encoded_residues.append(code)
        return np.array(encoded_residues)
    
    def get_link_coords(
        self,
        link: StructureData,
        modality_name: str,
    ) -> torch.Tensor:
        if modality_name == 'prot_atom_x':
            x_0 = torch.from_numpy(link.coords).float()
        elif modality_name == 'prot_res':
            x_0 = torch.from_numpy(link.backbone.coords).float()
        else:
            raise NotImplementedError(f"{modality_name} does not have linked structure coords")
        return x_0

    def convert_protein(
        self,
        holo: StructureData,
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
        prot_atom_names = torch.from_numpy(
            self.encode_atom_names(holo.atom_names, holo.elements, holo.res_names)
        ).long()
        prot_elements = torch.from_numpy(self.encode_elements(holo.elements)).long()
        prot_res_ids = torch.from_numpy(holo.res_ids).long()
        prot_res_names = torch.from_numpy(self.encode_res_names(holo.res_names)).long()
        prot_backbone_mask = torch.from_numpy(holo.backbone_mask).bool()

        # TODO: figure out how to store chain ids
        offset = ord("A")
        prot_chain_ids = torch.tensor(
            [ord(chain_id) - offset for chain_id in holo.chain_ids], dtype=torch.long
        )

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
            self.encode_res_names(holo.backbone.res_names)
        ).long()

        # TODO: figure out how to store chain ids
        offset = ord("A")
        backbone_chain_ids = torch.tensor(
            [ord(chain_id) - offset for chain_id in holo.backbone.chain_ids],
            dtype=torch.long,
        )

        backbone_pocket_mask = torch.zeros_like(backbone_res_ids, dtype=torch.bool)
        for i in range(len(backbone_res_ids)):
            chain_id = holo.backbone.chain_ids[i]
            res_id = backbone_res_ids[i].item()
            if (chain_id, res_id) in pocket_res_identifiers:
                backbone_pocket_mask[i] = True

        node_data["prot_res"] = {
            "x_1_true": backbone_coords,
            "res_id": backbone_res_ids,
            "a": backbone_res_names,
            "chain_id": backbone_chain_ids,
            "pocket_mask": backbone_pocket_mask,
        }

        # NOTE: change later?
        edge_idxs["prot_atom_to_prot_atom"] = edge_builders.radius_graph(
            prot_coords, 
            radius=4.5, 
            max_num_neighbors=1000
        )

        return node_data, edge_idxs, edge_data

    def convert_ligand(
        self,
        ligand: LigandData,
        ligand_id: str,
        receptor: Optional[StructureData] = None,
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
                "x_1_true": lig_x,
                "a_1_true": lig_a,
                "c_1_true": lig_c,
            }
        }

        edge_data = {
            "lig_to_lig": {
                "e_1_true": lig_e,
            }
        }

        edge_idxs = {
            "lig_to_lig": lig_edge_idxs,
        }
        if ligand.is_covalent and ligand.linkages and receptor is not None:
            prot_atom_to_lig_idxs = []
            prot_res_to_lig_idxs = []
            lig_asym_id = ligand_id.split(".")[1]

            lig_identifier = f"{ligand.ccd}:{lig_asym_id}"

            for linkage in ligand.linkages:
                prtnr1, prtnr2 = linkage.split("__")

                if lig_identifier in prtnr1:
                    lig_part = prtnr1
                    prot_part = prtnr2
                elif lig_identifier in prtnr2:
                    lig_part = prtnr2
                    prot_part = prtnr1
                else:
                    continue

                (
                    prot_auth_resid,
                    prot_resname,
                    prot_asym_id,
                    prot_seq_resid,
                    prot_atom_name,
                ) = prot_part.split(":")
                (
                    lig_auth_resid,
                    lig_resname,
                    lig_asym_id,
                    lig_seq_resid,
                    lig_atom_name,
                ) = lig_part.split(":")

                prot_atom_idx = None
                for i, atom_name in enumerate(receptor.atom_names):
                    if (
                        receptor.res_ids[i] == int(prot_seq_resid)
                        and atom_name == prot_atom_name
                    ):
                        prot_atom_idx = i
                        break

                lig_atom_idx = None
                for i, atom_type in enumerate(ligand.atom_types):
                    atom_name = lig_atom_type_map[atom_type]
                    if atom_name == lig_atom_name[0] and lig_auth_resid == i:
                        lig_atom_idx = i
                        break

                prot_res_idx = None
                for i, res_id in enumerate(receptor.backbone.res_ids):
                    if res_id == int(prot_seq_resid):
                        prot_res_idx = i
                        break

                if prot_atom_idx is not None and lig_atom_idx is not None:
                    prot_atom_to_lig_idxs.append([prot_atom_idx, lig_atom_idx])

                if prot_res_idx is not None and lig_atom_idx is not None:
                    prot_res_to_lig_idxs.append([prot_res_idx, lig_atom_idx])

            if prot_atom_to_lig_idxs:
                prot_atom_to_lig_tensor = torch.tensor(
                    prot_atom_to_lig_idxs, dtype=torch.long
                ).t()
                edge_idxs["prot_atom_to_lig"] = prot_atom_to_lig_tensor
                # TODO: covalent edge feature
                edge_data["prot_atom_to_lig"] = {
                    "e_1_true": torch.ones(
                        (prot_atom_to_lig_tensor.shape[1], 1), dtype=torch.float
                    )
                }

            if prot_res_to_lig_idxs:
                prot_res_to_lig_tensor = torch.tensor(
                    prot_res_to_lig_idxs, dtype=torch.long
                ).t()
                edge_idxs["prot_res_to_lig"] = prot_res_to_lig_tensor
                # TODO: covalent edge feature
                edge_data["prot_res_to_lig"] = {
                    "e_1_true": torch.ones(
                        (prot_res_to_lig_tensor.shape[1], 1), dtype=torch.float
                    )
                }

        return node_data, edge_idxs, edge_data
    
    def connect_ligand_to_pocket(
        self,
        node_data: Dict[str, Dict[str, torch.Tensor]],
        edge_idxs: Dict[str, torch.Tensor],
        edge_data: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        prot_node_data = node_data['prot_atom']
        lig_node_data = node_data["lig"]

        pocket_mask = prot_node_data["pocket_mask"]
        pocket_atom_indices = torch.nonzero(pocket_mask, as_tuple=True)[0]
        
        num_lig_atoms = lig_node_data["x_1_true"].shape[0]
        
        prot_to_lig_edges = []
        
        for lig_idx in range(num_lig_atoms):
            for prot_idx in pocket_atom_indices:
                prot_to_lig_edges.append((prot_idx.item(), lig_idx))
        
        if prot_to_lig_edges:
            prot_to_lig_tensor = torch.tensor(prot_to_lig_edges, dtype=torch.long).t()
            edge_idxs["prot_atom_to_lig"] = prot_to_lig_tensor
            edge_data["prot_atom_to_lig"] = {
                "e": torch.zeros((prot_to_lig_tensor.shape[1], 1), dtype=torch.float)
            }
        
        return node_data, edge_idxs, edge_data

    def convert_npndes(
        self,
        npndes: Dict[str, LigandData],
        receptor: Optional[StructureData] = None,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        node_data, edge_data, edge_idxs = {}, {}, {}
        node_data["npnde"] = {"x_1_true": torch.empty(0), "a_1_true": torch.empty(0), "c_1_true": torch.empty(0)}
        edge_data["npnde_to_npnde"] = {"e_1_true": torch.empty(0)}
        edge_idxs["npnde_to_npnde"] = torch.empty((2, 0), dtype=torch.long)

        if not npndes:
            return node_data, edge_idxs, edge_data

        all_coords = []
        all_atom_types = []
        all_atom_charges = []
        all_bond_types = []
        all_bond_indices = []

        all_prot_atom_to_npnde_idxs = []
        all_prot_res_to_npnde_idxs = []

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

            if (
                ligand_data.is_covalent
                and ligand_data.linkages
                and receptor is not None
            ):
                npnde_asym_id = npnde_id.split(".")[1]
                npnde_identifier = f"{ligand_data.ccd}:{npnde_asym_id}"

                for linkage in ligand_data.linkages:
                    prtnr1, prtnr2 = linkage.split("__")

                    if npnde_identifier in prtnr1:
                        npnde_part = prtnr1
                        prot_part = prtnr2
                    elif npnde_identifier in prtnr2:
                        npnde_part = prtnr2
                        prot_part = prtnr1
                    else:
                        continue

                    (
                        prot_auth_resid,
                        prot_resname,
                        prot_asym_id,
                        prot_seq_resid,
                        prot_atom_name,
                    ) = prot_part.split(":")
                    (
                        npnde_auth_resid,
                        npnde_resname,
                        npnde_asym_id,
                        npnde_seq_resid,
                        npnde_atom_name,
                    ) = npnde_part.split(":")

                    prot_atom_idx = None
                    for i, atom_name in enumerate(receptor.atom_names):
                        if (
                            receptor.res_ids[i] == int(prot_seq_resid)
                            and atom_name == prot_atom_name
                        ):
                            prot_atom_idx = i
                            break

                    npnde_atom_idx = None
                    for i, atom_type in enumerate(ligand_data.atom_types):
                        atom_name = npnde_atom_type_map[atom_type]
                        if atom_name == npnde_atom_name[0] and npnde_auth_resid == i:
                            npnde_atom_idx = i
                            break

                    prot_res_idx = None
                    for i, res_id in enumerate(receptor.backbone.res_ids):
                        if res_id == int(prot_seq_resid):
                            prot_res_idx = i
                            break

                    if prot_atom_idx is not None and npnde_atom_idx is not None:
                        all_prot_atom_to_npnde_idxs.append(
                            [prot_atom_idx, npnde_atom_idx + node_offset]
                        )

                    if prot_res_idx is not None and npnde_atom_idx is not None:
                        all_prot_res_to_npnde_idxs.append(
                            [prot_res_idx, npnde_atom_idx + node_offset]
                        )

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

            node_data["npnde"] = {"x_1_true": npnde_x, "a_1_true": npnde_a, "c_1_true": npnde_c}

            edge_data["npnde_to_npnde"] = {"e_1_true": npnde_e}

            edge_idxs["npnde_to_npnde"] = npnde_edge_idxs
        else:
            node_data["npnde"] = {
                "x_1_true": combined_coords,
                "a_1_true": combined_atom_types,
                "c_1_true": combined_atom_charges,
            }

        if all_prot_atom_to_npnde_idxs:
            prot_atom_to_npnde_tensor = torch.tensor(
                all_prot_atom_to_npnde_idxs, dtype=torch.long
            ).t()
            edge_idxs["prot_atom_to_npnde"] = prot_atom_to_npnde_tensor
            # TODO: covalent edge feature
            edge_data["prot_atom_to_npnde"] = {
                "e_1_true": torch.ones(
                    (prot_atom_to_npnde_tensor.shape[1], 1), dtype=torch.float
                )
            }

        if all_prot_res_to_npnde_idxs:
            prot_res_to_npnde_tensor = torch.tensor(
                all_prot_res_to_npnde_idxs, dtype=torch.long
            ).t()
            edge_idxs["prot_res_to_npnde"] = prot_res_to_npnde_tensor
            # TODO: covalent edge feature
            edge_data["prot_res_to_npnde"] = {
                "e_1_true": torch.ones(
                    (prot_res_to_npnde_tensor.shape[1], 1), dtype=torch.float
                )
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

        node_data["pharm"] = {"x_1_true": coords, "a_1_true": types, "v_1_true": vectors, "i_1_true": interactions}

        # assert self.graph_config.edges["pharm_to_pharm"]["type"] == "complete", (
        #     "the following code assumes complete pharm-pharm graph"
        # )

        num_centers = coords.shape[0]
        if num_centers > 1:
            pharm_edge_idxs = edge_builders.complete_graph(coords)
            edge_idxs["pharm_to_pharm"] = pharm_edge_idxs

        return node_data, edge_idxs, edge_data

    def convert_system(
        self,
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
            system.receptor, system.pocket
        )
        node_data.update(prot_node_data)
        edge_idxs.update(prot_edge_idxs)
        edge_data.update(prot_edge_data)

        lig_node_data, lig_edge_idxs, lig_edge_data = self.convert_ligand(
            system.ligand, system.ligand_id, system.receptor
        )
        node_data.update(lig_node_data)
        edge_idxs.update(lig_edge_idxs)
        edge_data.update(lig_edge_data)

        npnde_node_data, npnde_edge_idxs, npnde_edge_data = self.convert_npndes(
            system.npndes if system.npndes is not None else {}
        )
        node_data.update(npnde_node_data)
        edge_idxs.update(npnde_edge_idxs)
        edge_data.update(npnde_edge_data)

        if self.include_pharmacophore:
            pharm_node_data, pharm_edge_idxs, pharm_edge_data = (
                self.convert_pharmacophore(system.pharmacophore)
            )
            node_data.update(pharm_node_data)
            edge_idxs.update(pharm_edge_idxs)
            edge_data.update(pharm_edge_data)
        
        node_data, edge_idxs, edge_data = self.connect_ligand_to_pocket(node_data, edge_idxs, edge_data)

        return node_data, edge_idxs, edge_data

    def __getitem__(self, index) -> dgl.DGLHeteroGraph:
        task_name, idx = index
        task_class: Task = task_name_to_class(task_name)

        system = self.get_system(idx)

        node_data, edge_idxs, edge_data = self.convert_system(system)

        # TODO: things!
        g = build_complex_graph(node_data, edge_idxs, edge_data)

        priors_fns = get_prior(task_class, self.prior_config, train=True)

        # sample priors
        for modality_name in priors_fns:
            prior_name, prior_func = priors_fns[modality_name] # get prior name and function
            modality = name_to_modality(modality_name) # get the modality object

            # fetch the target data from the graph object
            g_data_loc = g.nodes if modality.graph_entity == 'node' else g.edges
            
            if 'apo' in prior_name:
                if system.link is not None:
                    target_data = self.get_link_coords(system.link, modality_name)
                else:
                    raise ValueError("system.link is None, cannot retrieve link coordinates.")
            else:
                print(modality.entity_name)
                target_data = g_data_loc[modality.entity_name].data[f'{modality.data_key}_1_true']

            # if the prior is masked, we need to pass the number of categories for this modality to the prior function
            if prior_name == 'masked':
                prior_func = functools.partial(prior_func, n_categories=self.n_categories_dict[modality_name])

            # draw a sample from the prior
            prior_sample = prior_func(target_data)

            # for edge features, make sure upper and lower triangle are the same
            # TODO: this logic may change if we decide to do something other fully-connected lig-lig edges
            if modality.graph_entity == 'edge':
                upper_edge_mask = torch.zeros_like(target_data, dtype=torch.bool)
                upper_edge_mask[:target_data.shape[0]//2] = 1
                prior_sample[~upper_edge_mask] = prior_sample[upper_edge_mask]

            # add the prior sample to the graph
            g_data_loc[modality.entity_name].data[f'{modality.data_key}_0'] = prior_sample

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
        # indexes into the system_lookup array, not a node/edge data array

        node_types = ["lig", "prot_atom", "prot_res", "npnde"]
        if "pharmacophore" in task.groups_present:
            node_types.append("pharm")

        node_counts = []
        for ntype in node_types:
            if ntype == "lig":
                counts = np.array(
                    [
                        row["lig_atom_end"] - row["lig_atom_start"]
                        for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                            "records"
                        )
                    ]
                )
            elif ntype == "prot_atom":
                counts = np.array(
                    [
                        row["rec_end"] - row["rec_start"]
                        for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                            "records"
                        )
                    ]
                )
            elif ntype == "prot_res":
                counts = np.array(
                    [
                        row["backbone_end"] - row["backbone_start"]
                        for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                            "records"
                        )
                    ]
                )
            elif ntype == "npnde":
                counts = []
                for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                    "records"
                ):
                    npnde_count = 0
                    if row["npnde_idxs"]:
                        for npnde_idx in row["npnde_idxs"]:
                            npnde_row = self.npnde_lookup.iloc[npnde_idx]
                            npnde_count += (
                                npnde_row["atom_end"] - npnde_row["atom_start"]
                            )
                    counts.append(npnde_count)
                counts = np.array(counts)
            elif ntype == "pharm":
                counts = np.array(
                    [
                        row["pharm_end"] - row["pharm_start"]
                        for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                            "records"
                        )
                    ]
                )

        node_counts.append(counts)

        node_counts = np.stack(node_counts, axis=0).sum(axis=0)
        node_counts = torch.from_numpy(node_counts)
        return node_counts

    def get_num_edges(self, task: Task, start_idx: int, end_idx: int):
        pass

import dgl
import torch
import math
from omegaconf import DictConfig

from omtra.dataset.zarr_dataset import ZarrDataset
from omtra.constants import (
    lig_atom_type_map,
    charge_map,
    npnde_atom_type_map,
    ph_idx_to_type,
    aa_substitutions,
    residue_map,
    protein_element_map,
    protein_atom_map,
)
from omtra.data.graph import build_complex_graph
from omtra.data.graph import edge_builders, approx_n_edges
from omtra.data.xace_ligand import sparse_to_dense, add_k_hop_edges
from omtra.tasks.register import task_name_to_class
from omtra.tasks.tasks import Task
from omtra.utils.misc import classproperty
from omtra.priors.prior_factory import get_prior
from omtra.priors.sample import sample_priors
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
import biotite.structure as struc
from omtra.constants import _DEFAULT_DISTANCE_RANGE
import functools

import warnings

# Suppress the specific warning from vlen_utf8.py
warnings.filterwarnings(
    "ignore",
    message="The codec `vlen-utf8` is currently not part in the Zarr format 3 specification.*",
    module="zarr.codecs.vlen_utf8"
)


class PlinderDataset(ZarrDataset):
    def __init__(
        self,
        link_version: str,
        split: str,
        processed_data_dir: str,
        graph_config: Optional[DictConfig] = None,
        prior_config: Optional[DictConfig] = None,
    ):
        super().__init__(
            split,
            f"{processed_data_dir}/{link_version}"
            if link_version
            else f"{processed_data_dir}/no_links",
        )
        self.split = split
        self.link_version = link_version
        self.graph_config = graph_config
        self.prior_config = prior_config

        self.system_lookup = pd.DataFrame(self.root.attrs["system_lookup"])
        self.npnde_lookup = pd.DataFrame(self.root.attrs["npnde_lookup"])

        self.encode_element = {
            element: i for i, element in enumerate(protein_element_map)
        }
        self.encode_residue = {res: i for i, res in enumerate(residue_map)}
        self.encode_atom = {atom: i for i, atom in enumerate(protein_atom_map)}

    @classproperty
    def name(cls):
        return "plinder"
    
    @property
    def n_zarr_chunks(self):
        coords_arr = self.root["pocket/coords"]
        n_atoms = coords_arr.shape[0]
        n_chunks = math.ceil(n_atoms / coords_arr.chunks[0])
        return n_chunks
    
    @property
    def graphs_per_chunk(self):
        return len(self) // self.n_zarr_chunks


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
                sdf=npnde_info["sdf"],
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

    def get_system(self, index: int, include_pharmacophore: bool) -> SystemData:
        system_info = self.system_lookup[
            self.system_lookup["system_idx"] == index
        ].iloc[0]

        rec_start, rec_end = int(system_info["rec_start"]), int(system_info["rec_end"])
        backbone_start, backbone_end = (
            int(system_info["backbone_start"]),
            int(system_info["backbone_end"]),
        )

        lig_atom_start, lig_atom_end = (
            int(system_info["lig_atom_start"]),
            int(system_info["lig_atom_end"]),
        )
        lig_bond_start, lig_bond_end = (
            int(system_info["lig_bond_start"]),
            int(system_info["lig_bond_end"]),
        )

        pocket_start, pocket_end = (
            int(system_info["pocket_start"]),
            int(system_info["pocket_end"]),
        )
        pocket_bb_start, pocket_bb_end = (
            int(system_info["pocket_bb_start"]),
            int(system_info["pocket_bb_end"]),
        )

        link_type = system_info["link_type"]
        if link_type:
            link_start, link_end = (
                int(system_info["link_start"]),
                int(system_info["link_end"]),
            )
            link_bb_start, link_bb_end = (
                int(system_info["link_bb_start"]),
                int(system_info["link_bb_end"]),
            )

        backbone = BackboneData(
            coords=self.slice_array(
                "receptor/backbone_coords", backbone_start, backbone_end
            ),
            res_ids=self.slice_array(
                "receptor/backbone_res_ids", backbone_start, backbone_end
            ),
            res_names=self.slice_array(
                "receptor/backbone_res_names", backbone_start, backbone_end
            ),
            chain_ids=self.slice_array(
                "receptor/backbone_chain_ids", backbone_start, backbone_end
            ),
        )

        receptor = StructureData(
            coords=self.slice_array("receptor/coords", rec_start, rec_end),
            atom_names=self.slice_array("receptor/atom_names", rec_start, rec_end),
            elements=self.slice_array("receptor/elements", rec_start, rec_end),
            res_ids=self.slice_array("receptor/res_ids", rec_start, rec_end),
            res_names=self.slice_array("receptor/res_names", rec_start, rec_end),
            chain_ids=self.slice_array("receptor/chain_ids", rec_start, rec_end),
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

        if include_pharmacophore:
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
            ),
            chain_ids=self.slice_array(
                "pocket/backbone_chain_ids", pocket_bb_start, pocket_bb_end
            ),
        )

        pocket = StructureData(
            coords=self.slice_array("pocket/coords", pocket_start, pocket_end),
            atom_names=self.slice_array("pocket/atom_names", pocket_start, pocket_end),
            elements=self.slice_array("pocket/elements", pocket_start, pocket_end),
            res_ids=self.slice_array("pocket/res_ids", pocket_start, pocket_end),
            res_names=self.slice_array("pocket/res_names", pocket_start, pocket_end),
            chain_ids=self.slice_array("pocket/chain_ids", pocket_start, pocket_end),
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
                res_ids=None,
                res_names=None,
                chain_ids=None,
            )
            apo = StructureData(
                coords=self.slice_array("apo/coords", link_start, link_end),
                atom_names=None,
                elements=None,
                res_ids=None,
                res_names=None,
                chain_ids=None,
                cif=system_info["link_cif"],
                backbone_mask=None,
                backbone=apo_backbone,
            )
        elif link_type == "pred":
            pred_backbone = BackboneData(
                coords=self.slice_array(
                    "pred/backbone_coords", link_bb_start, link_bb_end
                ),
                res_ids=None,
                res_names=None,
                chain_ids=None,
            )
            pred = StructureData(
                coords=self.slice_array("pred/coords", link_start, link_end),
                atom_names=None,
                elements=None,
                res_ids=None,
                res_names=None,
                chain_ids=None,
                cif=system_info["link_cif"],
                backbone_mask=None,
                backbone=pred_backbone,
            )

        system = SystemData(
            system_id=system_info["system_id"],
            ligand_id=system_info["ligand_id"],
            receptor=receptor,
            ligand=ligand,
            pharmacophore=pharmacophore
            if include_pharmacophore
            else PharmacophoreData(
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
        pocket_mask: torch.Tensor,
        bb_pocket_mask: torch.Tensor,
        modality_name: str,
    ) -> torch.Tensor:
        if modality_name == "prot_atom_x":
            x_0 = torch.from_numpy(link.coords[pocket_mask]).float()
        elif modality_name == "prot_res":
            x_0 = torch.from_numpy(link.backbone.coords[bb_pocket_mask]).float()
        else:
            raise NotImplementedError(
                f"{modality_name} does not have linked structure coords"
            )
        return x_0

    def convert_protein(
        self,
        holo: StructureData,
        pocket: StructureData,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
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
        
        unique_chains = sorted(set(holo.chain_ids))
        chain_to_idx = {chain: idx for idx, chain in enumerate(unique_chains)}
        prot_chain_ids = torch.tensor([chain_to_idx[chain_id] for chain_id in holo.chain_ids], dtype=torch.long)

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
            "x_1_true": prot_coords[pocket_mask],
            "a_1_true": prot_atom_names[pocket_mask],
            "e_1_true": prot_elements[pocket_mask],
            "res_id": prot_res_ids[pocket_mask],
            "res_names": prot_res_names[pocket_mask],
            "chain_id": prot_chain_ids[pocket_mask],
            "backbone_mask": prot_backbone_mask[pocket_mask],
        }

        backbone_coords = torch.from_numpy(holo.backbone.coords).float()
        backbone_res_ids = torch.from_numpy(holo.backbone.res_ids).long()
        backbone_res_names = torch.from_numpy(
            self.encode_res_names(holo.backbone.res_names)
        ).long()

        # TODO: figure out how to store chain ids
        backbone_chain_ids = torch.tensor(
            [chain_to_idx[chain_id] for chain_id in holo.backbone.chain_ids],
            dtype=torch.long,
        )

        backbone_pocket_mask = torch.zeros_like(backbone_res_ids, dtype=torch.bool)
        for i in range(len(backbone_res_ids)):
            chain_id = holo.backbone.chain_ids[i]
            res_id = backbone_res_ids[i].item()
            if (chain_id, res_id) in pocket_res_identifiers:
                backbone_pocket_mask[i] = True

        node_data["prot_res"] = {
            "x_1_true": backbone_coords[backbone_pocket_mask],
            "res_id": backbone_res_ids[backbone_pocket_mask],
            "a_1_true": backbone_res_names[backbone_pocket_mask],
            "chain_id": backbone_chain_ids[backbone_pocket_mask],
        }

        return node_data, edge_idxs, edge_data, pocket_mask, backbone_pocket_mask

    def encode_charges(self, charges: torch.Tensor) -> torch.Tensor:
        charge_type_map = {charge: i for i, charge in enumerate(charge_map)}
        encoded_charges = []
        for charge in charges:
            charge = int(charge.item())
            if charge not in charge_type_map:
                raise ValueError(f"{charge} not in charge map")
            else:
                encoded_charges.append(charge_type_map[charge])
        return torch.Tensor(encoded_charges).long()

    def infer_covalent_bonds(
        self,
        ligand: LigandData,
        pocket: StructureData,
        atom_type_map: List[str],
        )   -> Tuple[torch.Tensor, torch.Tensor]:
        ligand_arr = ligand.to_atom_array(atom_type_map)
        pocket_arr = pocket.to_atom_array()
        
        prot_atom_covalent_lig = []
        prot_res_covalent_lig = []
        dists = struc.distance(
            ligand_arr.coord[:, np.newaxis, :], pocket_arr.coord[np.newaxis, :, :]
        )
        for i, lig_atom in enumerate(ligand_arr):
            for j, rec_atom in enumerate(pocket_arr):
                dist_range = _DEFAULT_DISTANCE_RANGE.get(
                    (lig_atom.element, rec_atom.element)
                ) or _DEFAULT_DISTANCE_RANGE.get((rec_atom.element, lig_atom.element))
                if dist_range is None:
                    continue
                else:
                    min_dist, max_dist = dist_range
                dist = dists[i, j]
                if dist >= min_dist and dist <= max_dist:
                    prot_atom_covalent_lig.append([j, i])
                    res_id = pocket_arr.get_annotation("res_id")[j]
                    chain_id = pocket_arr.get_annotation("chain_id")[j]
                    
                    res_ids = pocket.backbone.res_ids
                    chain_ids = pocket.backbone.chain_ids
                    
                    res_id_mask = (res_ids == res_id)
                    chain_id_mask = (chain_ids == chain_id)
                    combined_mask = res_id_mask & chain_id_mask
                    
                    res_idx = np.where(combined_mask)[0]
                    if len(res_idx > 0):
                        prot_res_covalent_lig.append([res_idx[0], i])
                        
                   
        if prot_atom_covalent_lig:
            prot_atom_covalent_lig = torch.tensor(prot_atom_covalent_lig, dtype=torch.long).t()
        else:
            prot_atom_covalent_lig = torch.zeros((2, 0), dtype=torch.long)
        
        if prot_res_covalent_lig:
            prot_res_covalent_lig = torch.tensor(prot_res_covalent_lig, dtype=torch.long).t()
        else:
            prot_res_covalent_lig = torch.zeros((2, 0), dtype=torch.long)
                
        return prot_atom_covalent_lig, prot_res_covalent_lig
        
        
    def convert_ligand(
        self,
        ligand: LigandData,
        ligand_id: str,
        pocket: Optional[StructureData] = None,
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

        lig_c = self.encode_charges(lig_c)
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
        if ligand.is_covalent and ligand.linkages and pocket is not None:
            prot_atom_to_lig_tensor, prot_res_to_lig_tensor = self.infer_covalent_bonds(
                ligand, pocket, lig_atom_type_map
            )
            if prot_atom_to_lig_tensor.shape[1] > 0:
                edge_idxs["prot_atom_covalent_lig"] = prot_atom_to_lig_tensor
                lig_to_prot_atom_tensor = prot_atom_to_lig_tensor[[1, 0]]
                edge_idxs["lig_covalent_prot_atom"] = lig_to_prot_atom_tensor

            if prot_res_to_lig_tensor.shape[1] > 0:
                edge_idxs["prot_res_covalent_lig"] = prot_res_to_lig_tensor
                lig_to_prot_res_tensor = prot_res_to_lig_tensor[[1, 0]]
                edge_idxs["lig_covalent_prot_res"] = lig_to_prot_res_tensor

        return node_data, edge_idxs, edge_data

    def convert_npndes(
        self,
        npndes: Dict[str, LigandData],
        pocket: Optional[StructureData] = None,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        node_data, edge_data, edge_idxs = {}, {}, {}
        node_data["npnde"] = {
            "x_1_true": torch.empty(0),
            "a_1_true": torch.empty(0),
            "c_1_true": torch.empty(0),
        }
        edge_data["npnde_to_npnde"] = {"e_1_true": torch.empty(0)}
        edge_idxs["npnde_to_npnde"] = torch.empty((2, 0), dtype=torch.long)
        edge_idxs["prot_atom_covalent_npnde"] = torch.empty((2, 0), dtype=torch.long)
        edge_idxs["npnde_covalent_prot_atom"] = torch.empty((2, 0), dtype=torch.long)
        edge_idxs["prot_res_covalent_npnde"] = torch.empty((2, 0), dtype=torch.long)
        edge_idxs["npnde_covalent_prot_res"] = torch.empty((2, 0), dtype=torch.long)

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

            # check if the npnde has bonds
            has_bonds = ligand_data.bond_types is not None and ligand_data.bond_indices is not None
            if has_bonds and ligand_data.bond_types.shape[0] == 0:
                has_bonds = False

            if has_bonds:
                bond_types = torch.from_numpy(ligand_data.bond_types).long()
                bond_indices = torch.from_numpy(ligand_data.bond_indices).long()
                
                adjusted_indices = bond_indices.clone()
                adjusted_indices[:, 0] += node_offset
                adjusted_indices[:, 1] += node_offset

                all_bond_types.append(bond_types)
                all_bond_indices.append(adjusted_indices)

            if ligand_data.is_covalent and ligand_data.linkages and pocket is not None:
                prot_atom_to_npnde_tensor, prot_res_to_npnde_tensor = self.infer_covalent_bonds(
                    ligand_data, pocket, npnde_atom_type_map
                )
                if prot_atom_to_npnde_tensor.shape[1] > 0:
                    prot_atom_to_npnde_tensor[1, :] += node_offset
                    all_prot_atom_to_npnde_idxs.append(prot_atom_to_npnde_tensor)
                if prot_res_to_npnde_tensor.shape[1] > 0:
                    prot_res_to_npnde_tensor[1, :] += node_offset
                    all_prot_res_to_npnde_idxs.append(prot_res_to_npnde_tensor)

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
            combined_bond_indices = torch.cat(all_bond_indices, dim=0)

            k = self.graph_config["edges"]["npnde_to_npnde"]["params"]["k"]
            npnde_x, npnde_a, npnde_c, npnde_e, npnde_edge_idxs = (
                add_k_hop_edges(
                    combined_coords,
                    combined_atom_types,
                    combined_atom_charges,
                    combined_bond_types,
                    combined_bond_indices,
                    k=k
                )
            )
            npnde_c = self.encode_charges(npnde_c)

            node_data["npnde"] = {
                "x_1_true": npnde_x,
                "a_1_true": npnde_a,
                "c_1_true": npnde_c,
            }

            edge_data["npnde_to_npnde"] = {"e_1_true": npnde_e}

            edge_idxs["npnde_to_npnde"] = npnde_edge_idxs
        else:
            combined_atom_charges = self.encode_charges(combined_atom_charges)
            node_data["npnde"] = {
                "x_1_true": combined_coords,
                "a_1_true": combined_atom_types,
                "c_1_true": combined_atom_charges,
            }

        if all_prot_atom_to_npnde_idxs:
            prot_atom_to_npnde_tensor = torch.cat(all_prot_atom_to_npnde_idxs, dim=1)
            edge_idxs["prot_atom_covalent_npnde"] = prot_atom_to_npnde_tensor
            npnde_to_prot_atom_tensor = prot_atom_to_npnde_tensor[[1, 0]]
            edge_idxs["npnde_covalent_prot_atom"] = npnde_to_prot_atom_tensor

        if all_prot_res_to_npnde_idxs:
            prot_res_to_npnde_tensor = torch.cat(
                all_prot_res_to_npnde_idxs, dim=1
            )
            edge_idxs["prot_res_covalent_npnde"] = prot_res_to_npnde_tensor
            npnde_to_prot_res_tensor = prot_res_to_npnde_tensor[[1, 0]]
            edge_idxs["npnde_covalent_prot_res"] = npnde_to_prot_res_tensor

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

        node_data["pharm"] = {
            "x_1_true": coords,
            "a_1_true": types,
            "v_1_true": vectors,
            "i_1_true": interactions,
        }

        return node_data, edge_idxs, edge_data

    def convert_system(
        self,
        system: SystemData,
        include_pharmacophore: bool,
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, torch.Tensor],
        Dict[str, Dict[str, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        node_data = {}
        edge_idxs = {}
        edge_data = {}

        # read protein data
        prot_node_data, prot_edge_idxs, prot_edge_data, pocket_mask, bb_pocket_mask = (
            self.convert_protein(system.receptor, system.pocket)
        )
        node_data.update(prot_node_data)
        edge_idxs.update(prot_edge_idxs)
        edge_data.update(prot_edge_data)

        # read ligand data
        lig_node_data, lig_edge_idxs, lig_edge_data = self.convert_ligand(
            system.ligand, system.ligand_id, system.pocket
        )
        node_data.update(lig_node_data)
        edge_idxs.update(lig_edge_idxs)
        edge_data.update(lig_edge_data)

        # read npnde data
        npnde_node_data, npnde_edge_idxs, npnde_edge_data = self.convert_npndes(
            system.npndes if system.npndes is not None else {}, system.pocket
        )
        node_data.update(npnde_node_data)
        edge_idxs.update(npnde_edge_idxs)
        edge_data.update(npnde_edge_data)

        if include_pharmacophore:
            pharm_node_data, pharm_edge_idxs, pharm_edge_data = (
                self.convert_pharmacophore(system.pharmacophore)
            )
            node_data.update(pharm_node_data)
            edge_idxs.update(pharm_edge_idxs)
            edge_data.update(pharm_edge_data)

        return node_data, edge_idxs, edge_data, pocket_mask, bb_pocket_mask

    def __getitem__(self, index) -> dgl.DGLHeteroGraph:
        task_name, idx = index
        task_class: Task = task_name_to_class(task_name)

        include_pharmacophore = 'pharmacophore' in task_class.groups_present

        system = self.get_system(idx, include_pharmacophore=include_pharmacophore)

        node_data, edge_idxs, edge_data, pocket_mask, bb_pocket_mask = (
            self.convert_system(system, include_pharmacophore=include_pharmacophore)
        )

        g = build_complex_graph(node_data, edge_idxs, edge_data)

        # get prior functions
        prior_fns = get_prior(task_class, self.prior_config, training=True)

        # first, if the task requires a linked structure for the prior,
        # manually add this to the graph

        if 'apo' in prior_fns.get("prot_atom_x", ("", None))[0]:

            if system.link is None:
                raise ValueError("system.link is None, cannot retrieve link coordinates.")

            g.nodes['prot_atom'].data['x_0'] = self.get_link_coords(
                system.link, 
                pocket_mask, 
                bb_pocket_mask, 
                'prot_atom_x'
            )
            
        # sample priors
        g = sample_priors(g, 
                          task_class=task_class,
                          prior_fns=prior_fns, 
                          training=True)

        return g

    def retrieve_graph_chunks(self, frac_start, frac_end, apo_systems: bool = False):
        """
        This dataset contains len(self) examples. We divide all samples (or, graphs) into separate chunk.
        We call these "graph chunks"; this is not the same thing as chunks defined in zarr arrays.
        I know we need better terminology; but they're chunks! they're totally chunks. just a different kind of chunk.
        """
        n_graphs = len(self)  # this is wrong! n_graphs depends on apo_systems!!!!
        n_even_chunks, n_graphs_in_last_chunk = divmod(n_graphs, self.graphs_per_chunk)

        n_chunks = n_even_chunks + int(n_graphs_in_last_chunk > 0)

        # raise NotImplementedError(
        #     "need to build capability to modify chunks based on whether or not the task uses the apo state"
        # )

        # construct a tensor containing the index ranges for each chunk
        chunk_index = torch.zeros(n_chunks, 2, dtype=torch.int64)
        chunk_index[:, 0] = self.graphs_per_chunk * torch.arange(n_chunks)
        chunk_index[:-1, 1] = chunk_index[1:, 0]
        chunk_index[-1, 1] = n_graphs

        return chunk_index

    def get_num_nodes(self, task: Task, start_idx, end_idx, per_ntype=False):
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
                        row["pocket_end"] - row["pocket_start"]
                        for row in self.system_lookup.iloc[start_idx:end_idx].to_dict(
                            "records"
                        )
                    ]
                )
            elif ntype == "prot_res":
                counts = np.array(
                    [
                        row["pocket_bb_end"] - row["pocket_bb_start"]
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

        if per_ntype:
            num_nodes_dict = {
                ntype: ncount for ntype, ncount in zip(node_types, node_counts)
            }
            return num_nodes_dict

        node_counts = np.stack(node_counts, axis=0).sum(axis=0)
        node_counts = torch.from_numpy(node_counts)
        return node_counts



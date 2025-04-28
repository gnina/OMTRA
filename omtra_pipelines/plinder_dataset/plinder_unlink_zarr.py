import logging
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import zarr
from numcodecs import VLenUTF8
from omtra_pipelines.plinder_dataset.utils import setup_logger
from omtra_pipelines.plinder_dataset.plinder_pipeline import (
    LigandData,
    StructureData,
    PharmacophoreData,
    SystemData,
    SystemProcessor,
)
from omtra.constants import lig_atom_type_map, npnde_atom_type_map
from tqdm import tqdm

logger = setup_logger(
    __name__,
)


class PlinderNoLinksZarrConverter:
    def __init__(
        self,
        output_path: str,
        struc_chunk_size: int = 1500000,
        lig_atom_chunk_size: int = 10000,
        lig_bond_chunk_size: int = 10000,
        pharmacophore_chunk_size: int = 10000,
        pocket_chunk_size: int = 150000,
        backbone_chunk_size: int = 150000,
        category: str = None,
        num_workers: int = 1,
        batch_size: int = 200,
        embeddings: Optional[bool] = False,
    ):
        self.output_path = Path(output_path)
        self.struc_chunk_size = struc_chunk_size
        self.lig_atom_chunk_size = lig_atom_chunk_size
        self.lig_bond_chunk_size = lig_bond_chunk_size
        self.pharmacophore_chunk_size = pharmacophore_chunk_size
        self.pocket_chunk_size = pocket_chunk_size
        self.category = category
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.embeddings = embeddings

        if not self.output_path.exists():
            self.store = zarr.storage.LocalStore(str(self.output_path))
            self.root = zarr.group(store=self.store)

            self.receptor = self.root.create_group("receptor")

            self.pocket = self.root.create_group("pocket")
            self.pharmacophore = self.root.create_group("pharmacophore")
            self.ligand = self.root.create_group("ligand")
            self.npnde = self.root.create_group("npnde")

            for group in [self.receptor, self.pocket]:
                chunk = self.struc_chunk_size
                if group == self.pocket:
                    chunk = self.pocket_chunk_size
                    if self.embeddings:
                        embedding_dim = 1536 #specific to ESM3
                        group.create_array(
                        "embeddings",
                        shape=(0, embedding_dim),    
                        chunks=(chunk, embedding_dim),
                        dtype=np.float32,
                        compressors=None,
                    )

                group.create_array(
                    "coords", shape=(0, 3), chunks=(chunk, 3), dtype=np.float32
                )
                group.create_array(
                    "atom_names",
                    shape=(0,),
                    chunks=(chunk,),
                    dtype=str,
                    compressors=None,
                )
                group.create_array(
                    "elements",
                    shape=(0,),
                    chunks=(chunk,),
                    dtype=str,
                    compressors=None,
                )
                group.create_array(
                    "res_ids", shape=(0,), chunks=(chunk,), dtype=np.int32
                )
                group.create_array(
                    "res_names",
                    shape=(0,),
                    chunks=(chunk,),
                    dtype=str,
                    compressors=None,
                )
                group.create_array(
                    "chain_ids",
                    shape=(0,),
                    chunks=(chunk,),
                    dtype=str,
                    compressors=None,
                )
                group.create_array(
                    "backbone_mask", shape=(0,), chunks=(chunk,), dtype=bool
                )
                group.create_array(
                    "backbone_coords",
                    shape=(0, 3, 3),
                    chunks=(chunk, 3, 3),
                    dtype=np.float32,
                )
                group.create_array(
                    "backbone_res_ids",
                    shape=(0,),
                    chunks=(chunk,),
                    dtype=np.int32,
                )
                group.create_array(
                    "backbone_res_names",
                    shape=(0,),
                    chunks=(chunk,),
                    dtype=str,
                    compressors=None,
                )
                group.create_array(
                    "backbone_chain_ids",
                    shape=(0,),
                    chunks=(chunk,),
                    dtype=str,
                    compressors=None,
                )

            self.pharmacophore.create_array(
                "coords",
                shape=(0, 3),
                chunks=(self.pharmacophore_chunk_size, 3),
                dtype=np.float32,
            )

            self.pharmacophore.create_array(
                "types",
                shape=(0,),
                chunks=(self.pharmacophore_chunk_size,),
                dtype=np.int32,
            )

            self.pharmacophore.create_array(
                "vectors",
                shape=(0, 4, 3),
                chunks=(self.pharmacophore_chunk_size, 4, 3),
                dtype=np.float32,
            )

            self.pharmacophore.create_array(
                "interactions",
                shape=(0,),
                chunks=(self.pharmacophore_chunk_size,),
                dtype=bool,
            )

            for group in [self.ligand, self.npnde]:
                group.create_array(
                    "coords",
                    shape=(0, 3),
                    chunks=(self.lig_atom_chunk_size, 3),
                    dtype=np.float32,
                )
                group.create_array(
                    "atom_types",
                    shape=(0,),
                    chunks=(self.lig_atom_chunk_size,),
                    dtype=np.int32,
                )
                group.create_array(
                    "atom_charges",
                    shape=(0,),
                    chunks=(self.lig_atom_chunk_size,),
                    dtype=np.float32,
                )
                group.create_array(
                    "bond_types",
                    shape=(0,),
                    chunks=(self.lig_bond_chunk_size,),
                    dtype=np.int32,
                )
                group.create_array(
                    "bond_indices",
                    shape=(0, 2),
                    chunks=(self.lig_bond_chunk_size, 2),
                    dtype=np.int32,
                )

            # Initialize lookup tables/attrs
            self.root.attrs["system_lookup"] = []
            self.root.attrs["npnde_lookup"] = []
            self.root.attrs["chunk_sizes"] = [
                {
                    "struc": self.struc_chunk_size,
                    "lig_atom": self.lig_atom_chunk_size,
                    "lig_bond": self.lig_bond_chunk_size,
                    "pocket": self.pocket_chunk_size,
                    "pharmacophore": self.pharmacophore_chunk_size,
                }
            ]
            self.root.attrs["system_type_idxs"] = []

            self.system_lookup = self.root.attrs[
                "system_lookup"
            ]  # [{system_id, ligand_id, system_idx, ligand_idx, rec_start, rec_end, backbone_start, backbone_end, lig_atom_start, lig_atom_end, lig_bond_start, lig_bond_end, pharmacophore_idx, pharm_start, pharm_end, npnde_idxs, pocket_idx, pocket_start, pocket_end, pocket_bb_start, pocket_bb_end, apo_idx, pred_idx, link_start, link_end, cif}]
            self.npnde_lookup = self.root.attrs[
                "npnde_lookup"
            ]  # [{system_id, npnde_id, receptor_idx, npnde_idx, ccd, linkages, atom_start, atom_end, bond_start, bond_end, sdf}]

        else:
            self.root = zarr.open_group(store=str(self.output_path), mode="r+")

            self.receptor = self.root["receptor"]
            self.pocket = self.root["pocket"]
            self.ligand = self.root["ligand"]
            self.npnde = self.root["npnde"]
            self.pharmacophore = self.root["pharmacophore"]

            self.system_lookup = list(self.root.attrs["system_lookup"])
            self.npnde_lookup = list(self.root.attrs["npnde_lookup"])
    def _append_embedding_data_batch(
        self, group: zarr.Group, data_batch: List[StructureData]
    ) -> List[Tuple[int, int, int, int]]:
        
        embeddings_indices = []

        current_len = group["embeddings"].shape[0]
        emb_counts = [data.pocket_embedding.shape[0] for data in data_batch]
        emb_offsets = [current_len]

        for i in range(len(emb_counts)):
            emb_offsets.append(emb_offsets[-1] + emb_counts[i])

        for i in range(len(data_batch)):
            embeddings_indices.append(
                (emb_offsets[i], emb_offsets[i + 1])
            )

        # double check we are generating embeddings then we'll return "atom_indices" with embedding indices for each pocket
        if self.embeddings and "embeddings" in group.array_keys():
            try:
                all_embeddings = np.vstack(
                [
                    data.pocket_embedding
                    for data in data_batch
                    if getattr(data, "pocket_embedding", None) is not None 
                ]
            )
                group["embeddings"].append(all_embeddings)
            except Exception as e:
                logger.warning(f"No embeddings found in batch, skipping. Error: {e}")
        
        return embeddings_indices
        
    def _append_structure_data_batch(
        self, group: zarr.Group, data_batch: List[StructureData]
    ) -> List[Tuple[int, int, int, int]]:
        if not data_batch:
            return []

        atom_indices = []
        bb_indices = []

        current_len = group["coords"].shape[0]
        bb_current_len = group["backbone_coords"].shape[0]

        atom_counts = [len(data.coords) for data in data_batch]
        bb_counts = [len(data.backbone.coords) for data in data_batch]

        atom_offsets = [current_len]
        bb_offsets = [bb_current_len]

        for i in range(len(atom_counts)):
            atom_offsets.append(atom_offsets[-1] + atom_counts[i])
            bb_offsets.append(bb_offsets[-1] + bb_counts[i])

        for i in range(len(data_batch)):
            atom_indices.append(
                (atom_offsets[i], atom_offsets[i + 1], bb_offsets[i], bb_offsets[i + 1])
            )

        all_coords = np.vstack(
            [data.coords for data in data_batch if len(data.coords) > 0]
        )
        all_atom_names = np.concatenate(
            [data.atom_names for data in data_batch if len(data.atom_names) > 0]
        )
        all_elements = np.concatenate(
            [data.elements for data in data_batch if len(data.elements) > 0]
        )
        all_res_ids = np.concatenate(
            [data.res_ids for data in data_batch if len(data.res_ids) > 0]
        )
        all_res_names = np.concatenate(
            [data.res_names for data in data_batch if len(data.res_names) > 0]
        )
        all_chain_ids = np.concatenate(
            [data.chain_ids for data in data_batch if len(data.chain_ids) > 0]
        )
        all_backbone_masks = np.concatenate(
            [data.backbone_mask for data in data_batch if len(data.backbone_mask) > 0]
        )

        all_bb_coords = np.vstack(
            [
                data.backbone.coords
                for data in data_batch
                if len(data.backbone.coords) > 0
            ]
        )
        all_bb_res_ids = np.concatenate(
            [
                data.backbone.res_ids
                for data in data_batch
                if len(data.backbone.res_ids) > 0
            ]
        )
        all_bb_res_names = np.concatenate(
            [
                data.backbone.res_names
                for data in data_batch
                if len(data.backbone.res_names) > 0
            ]
        )
        all_bb_chain_ids = np.concatenate(
            [
                data.backbone.chain_ids
                for data in data_batch
                if len(data.backbone.chain_ids) > 0
            ]
        )

        group["coords"].append(all_coords)
        group["atom_names"].append(all_atom_names)
        group["elements"].append(all_elements)
        group["res_ids"].append(all_res_ids)
        group["res_names"].append(all_res_names)
        group["chain_ids"].append(all_chain_ids)
        group["backbone_mask"].append(all_backbone_masks)

        group["backbone_coords"].append(all_bb_coords)
        group["backbone_res_ids"].append(all_bb_res_ids)
        group["backbone_res_names"].append(all_bb_res_names)
        group["backbone_chain_ids"].append(all_bb_chain_ids)

        return atom_indices

    def _append_pharmacophore_data_batch(
        self, group: zarr.Group, data_batch: List[PharmacophoreData]
    ) -> List[Tuple[Optional[int], Optional[int]]]:
        valid_data = []
        valid_indices = []

        for i, data in enumerate(data_batch):
            if len(data.coords) > 0:
                valid_data.append(data)
                valid_indices.append(i)

        if not valid_data:
            return [(None, None)] * len(data_batch)

        current_len = group["coords"].shape[0]

        center_counts = [len(data.coords) for data in valid_data]

        offsets = [current_len]
        for count in center_counts:
            offsets.append(offsets[-1] + count)

        raw_indices = [(offsets[i], offsets[i + 1]) for i in range(len(valid_data))]

        index_to_range = {
            valid_indices[i]: raw_indices[i] for i in range(len(valid_indices))
        }

        all_coords = np.vstack([data.coords for data in valid_data])
        all_types = np.concatenate([data.types for data in valid_data])
        all_vectors = np.vstack([data.vectors for data in valid_data])
        all_interactions = np.concatenate([data.interactions for data in valid_data])

        group["coords"].append(all_coords)
        group["types"].append(all_types)
        group["vectors"].append(all_vectors)
        group["interactions"].append(all_interactions)

        result_indices = []
        for i in range(len(data_batch)):
            if i in index_to_range:
                result_indices.append(index_to_range[i])
            else:
                result_indices.append((None, None))

        return result_indices

    def _append_ligand_data_batch(
        self, group: zarr.Group, data_batch: List[LigandData]
    ) -> List[Tuple[int, int, Optional[int], Optional[int]]]:
        if not data_batch:
            return []

        current_atom_len = group["coords"].shape[0]
        current_bond_len = group["bond_types"].shape[0]

        atom_counts = [len(data.coords) for data in data_batch]
        bond_counts = [len(data.bond_types) for data in data_batch]

        atom_offsets = [current_atom_len]
        for count in atom_counts:
            atom_offsets.append(atom_offsets[-1] + count)

        bond_offsets = [current_bond_len]
        for count in bond_counts:
            bond_offsets.append(bond_offsets[-1] + count)

        indices = []
        for i, data in enumerate(data_batch):
            bond_start = bond_offsets[i] if bond_counts[i] > 0 else None
            bond_end = bond_offsets[i + 1] if bond_counts[i] > 0 else None
            indices.append((atom_offsets[i], atom_offsets[i + 1], bond_start, bond_end))

        all_coords = np.vstack(
            [data.coords for data in data_batch if len(data.coords) > 0]
        )
        all_atom_types = np.concatenate(
            [data.atom_types for data in data_batch if len(data.atom_types) > 0]
        )
        all_atom_charges = np.concatenate(
            [data.atom_charges for data in data_batch if len(data.atom_charges) > 0]
        )

        group["coords"].append(all_coords)
        group["atom_types"].append(all_atom_types)
        group["atom_charges"].append(all_atom_charges)

        if sum(bond_counts) > 0:
            data_with_bonds = [data for data in data_batch if len(data.bond_types) > 0]

            all_bond_types = np.concatenate(
                [data.bond_types for data in data_with_bonds]
            )
            all_bond_indices = np.vstack(
                [data.bond_indices for data in data_with_bonds]
            )

            group["bond_types"].append(all_bond_types)
            group["bond_indices"].append(all_bond_indices)

        return indices

    def _process_system(self, system_id: str):
        try:
            system_processor = SystemProcessor(system_id=system_id, link_type=None, embeddings=self.embeddings)
            return system_processor.process_system()

        except Exception as e:
            logging.exception(f"Error processing system {system_id}: {e}")
            return None

    def _collect_batch_results(self, results_list):
        batch_results = []

        for system_data in results_list:
            if system_data and system_data.get("systems_list"):
                batch_results.extend(system_data["systems_list"])

        return batch_results

    def _process_system_batch(self, system_ids: List[str]):
        results = []

        for system_id in system_ids:
            try:
                system_data = self._process_system(system_id)
                if system_data and system_data.get("systems_list"):
                    results.extend(system_data["systems_list"])
            except Exception as e:
                logger.exception(f"Error processing system batch {system_id}: {e}")

        return results

    def _write_system_batch(self, system_data_batch: List[SystemData]):
        if not system_data_batch:
            return

        receptor_data = []
        ligand_data = []
        pocket_data = []
        pharm_data = []
        npnde_data = []

        system_info = []
        for system_data in system_data_batch:
            link_type = system_data.link_type
            link_cif = None

            system_idx = len(self.system_lookup) + len(system_info)
            system_entry = {
                "system_id": system_data.system_id,
                "ligand_id": system_data.ligand_id,
                "system_idx": system_idx,
                "linkages": system_data.ligand.linkages,
                "ccd": system_data.ligand.ccd,
                "link_type": link_type,
                "lig_sdf": system_data.ligand.sdf,
                "rec_cif": system_data.receptor.cif,  # TODO: fix this issue when receptor is None
                "npnde_idxs": None,
            }

            receptor_data.append(system_data.receptor)
            ligand_data.append(system_data.ligand)
            pocket_data.append(system_data.pocket)
            pharm_data.append(system_data.pharmacophore)

            if system_data.npndes:
                npnde_idxs = []

                for i, (npnde_id, npnde_data_item) in enumerate(
                    system_data.npndes.items()
                ):
                    npnde_idx = len(self.npnde_lookup) + len(npnde_data)
                    npnde_data.append(
                        (
                            npnde_data_item,
                            system_idx,
                            npnde_id,
                            system_data.system_id,
                            npnde_idx,
                        )
                    )
                    npnde_idxs.append(npnde_idx)

                system_entry["npnde_idxs"] = npnde_idxs

            system_info.append(system_entry)

        receptor_indices = self._append_structure_data_batch(
            self.receptor, receptor_data
        )
        ligand_indices = self._append_ligand_data_batch(self.ligand, ligand_data)
        pocket_indices = self._append_structure_data_batch(self.pocket, pocket_data)
        pharm_indices = self._append_pharmacophore_data_batch(
            self.pharmacophore, pharm_data
        )
        #embeddings are associated with pocket data so we use a similar function to track their indices
        if self.embeddings:
            embeddings_indices = self._append_embedding_data_batch(self.pocket, pocket_data)

        npnde_entries = []
        if npnde_data:
            npnde_data_objects = [item[0] for item in npnde_data]
            npnde_indices = self._append_ligand_data_batch(
                self.npnde, npnde_data_objects
            )

            for i, (atom_start, atom_end, bond_start, bond_end) in enumerate(
                npnde_indices
            ):
                _, system_idx, npnde_id, system_id, npnde_idx = npnde_data[i]

                npnde_entries.append(
                    {
                        "system_id": system_id,
                        "npnde_id": npnde_id,
                        "system_idx": system_idx,
                        "npnde_idx": npnde_idx,
                        "atom_start": atom_start,
                        "atom_end": atom_end,
                        "bond_start": bond_start,
                        "bond_end": bond_end,
                        "linkages": npnde_data[i][0].linkages,
                        "ccd": npnde_data[i][0].ccd,
                        "sdf": npnde_data[i][0].sdf,
                    }
                )

        for i, entry in enumerate(system_info):
            (
                entry["rec_start"],
                entry["rec_end"],
                entry["backbone_start"],
                entry["backbone_end"],
            ) = receptor_indices[i]
            (
                entry["lig_atom_start"],
                entry["lig_atom_end"],
                entry["lig_bond_start"],
                entry["lig_bond_end"],
            ) = ligand_indices[i]
            (
                entry["pocket_start"],
                entry["pocket_end"],
                entry["pocket_bb_start"],
                entry["pocket_bb_end"],
            ) = pocket_indices[i]
            if self.embeddings: 
                (
                    entry["embeddings_start"],
                    entry["embeddings_end"],
                ) = embeddings_indices[i]

            entry["pharm_start"], entry["pharm_end"] = pharm_indices[i]

        self.system_lookup.extend(system_info)
        self.root.attrs["system_lookup"] = self.system_lookup

        if npnde_entries:
            self.npnde_lookup.extend(npnde_entries)
            self.root.attrs["npnde_lookup"] = self.npnde_lookup

        logger.info(f"Wrote batch of {len(system_info)} systems to zarr store")

    def process_dataset(self, system_ids: List[str], max_pending=None):
        if max_pending is None:
            max_pending = self.num_workers * 2

        start = len(self.system_lookup)
        logger.info(
            f"Processing {len(system_ids)} systems with {self.num_workers} workers"
        )

        batches = [
            system_ids[i : i + self.batch_size]
            for i in range(0, len(system_ids), self.batch_size)
        ]

        pbar = tqdm(
            total=len(batches), desc="Processing system batches", unit="batches"
        )
        successful_count = 0
        failed_count = 0

        for batch_idx, batch in enumerate(batches):
            logger.info(
                f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} systems"
            )

            batch_results = []
            with mp.Pool(processes=self.num_workers) as pool:
                system_results = pool.map(self._process_system, batch)

                valid_results = [r for r in system_results if r is not None]
                failed_count += len(system_results) - len(valid_results)
                successful_count += len(valid_results)

                if valid_results:
                    batch_data = self._collect_batch_results(valid_results)
                    if batch_data:
                        self._write_system_batch(batch_data)

            pbar.set_postfix({"success": successful_count, "failed": failed_count})
            pbar.update(1)

        pbar.close()

        end = len(self.system_lookup)
        logger.info(
            f"Processing complete. Success: {successful_count}, Failed: {failed_count}"
        )

        if self.category:
            self.root.attrs["system_type_idxs"].append((self.category, start, end))

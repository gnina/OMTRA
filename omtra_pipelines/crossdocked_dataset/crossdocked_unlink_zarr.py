import logging
import os
import psutil
import multiprocessing as mp
import sys
from multiprocessing import Manager, Queue, Pool
import queue
from queue import Empty
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import functools
import numpy as np
import pandas as pd
import zarr
from numcodecs import VLenUTF8
from omtra_pipelines.plinder_dataset.utils import setup_logger
from omtra.data.plinder import (
    LigandData,
    PharmacophoreData,
    StructureData,
    SystemData,
    BackboneData,
)
from omtra_pipelines.crossdocked_dataset.pipeline_components import SystemProcessor
from omtra.constants import lig_atom_type_map, npnde_atom_type_map
from tqdm import tqdm

logger = setup_logger(
    __name__,
)
logging.getLogger().setLevel(logging.CRITICAL)
class CrossdockedNoLinksZarrConverter:
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

                #Add extra ligand features
                if group == self.ligand:
                    lig_node_group = self.root['ligand']
                    n_atoms = lig_node_group['coords'].shape[0]
                    nodes_per_chunk = lig_node_group['coords'].chunks[0]

                    lig_node_group.create_array(
                        "extra_feats", 
                        shape=(n_atoms, 6), 
                        chunks=(nodes_per_chunk, 6), 
                        dtype=np.int8, 
                        overwrite=False,
                        attributes=['impl_H', 'aro', 'hyb', 'ring', 'chiral', 'frag']
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
        

        # Add extra features for ligands (not npnde)- concatenate each feature type separately, then stack
        if group == self.ligand:
            if sum(atom_counts) > 0:
                data_with_atoms = [data for data in data_batch if len(data.coords) > 0]
                all_impl_H = np.concatenate([data.atom_impl_H for data in data_with_atoms])
                all_aro = np.concatenate([data.atom_aro for data in data_with_atoms])
                all_hyb = np.concatenate([data.atom_hyb for data in data_with_atoms])
                all_ring = np.concatenate([data.atom_ring for data in data_with_atoms])
                all_chiral = np.concatenate([data.atom_chiral for data in data_with_atoms])
                all_frag = np.concatenate([data.fragments for data in data_with_atoms])
                
                # Single column_stack operation to create (total_atoms, 6) array
                all_extra_feats = np.column_stack([
                    all_impl_H, all_aro, all_hyb, all_ring, all_chiral, all_frag
                ])
                
                group["extra_feats"].append(all_extra_feats)

        return indices
    
    @staticmethod
    #each batch will contain a list of tuples with 2 entries: (receptor_path, ligand_path)
    def get_ligand_receptor_batches_types(types_file: str, root_dir: str, batch_size: int, max_num_batches: Optional[int] = None) -> List[List[Tuple[str, str]]]:
        batches = []
        current_batch = []
        valid_line_count = 0
        total_lines = sum(1 for _ in open(types_file, 'r'))
        with open(types_file, 'r') as f:
            for line in tqdm(f, desc="Reading types file & preparing lig-rec batches", total=total_lines, unit="line", leave=True, file=sys.stderr):
                parts = line.strip().split()
                if len(parts) < 5 or parts[0] != "1":
                    continue
                rec_path = os.path.join(root_dir, parts[3])
                lig_path = os.path.join(root_dir, parts[4])
                if os.path.exists(rec_path) and os.path.exists(lig_path):
                    current_batch.append((rec_path, lig_path))
                    if len(current_batch) == batch_size:
                        batches.append(current_batch)
                        current_batch = []

                        # these 2 lines only for pipeline testing!
                        if max_num_batches is not None and len(batches) >= max_num_batches:
                            break
        # Add any remaining pairs in the last batch
        if current_batch:
            batches.append(current_batch)

        logger.info(f"Finished preparing {len(batches)} batches from {valid_line_count} valid lines in {types_file}")
        return batches
    
    @staticmethod
    # Input should be the value of one key (either train or test) in the data dictionary
    def get_ligand_receptor_batches_external(data: List[Tuple[str, str]], root_dir: str, batch_size: int, max_num_batches: Optional[int] = None) -> List[List[Tuple[str, str]]]:        
        batches = []
        current_batch = []

        for i in range(0, len(data)):
            rec_path = os.path.join(root_dir, data[i][0])
            lig_path = os.path.join(root_dir, data[i][1])

            current_batch.append((rec_path, lig_path))
            if len(current_batch) == batch_size:
                batches.append(current_batch)
                current_batch = []

                # these 2 lines only for pipeline testing!
                if max_num_batches is not None and len(batches) >= max_num_batches:
                    break

        if current_batch:
            batches.append(current_batch)
        return batches

    # modified inputs for compatibility with new SystemProcessor for crossdocked
    def _process_system(self, receptor_path: str, ligand_path: str, pocket_cutoff: float, n_cpus: int):
        try:
            #modified inputs to SystemProcessor here
            system_processor = SystemProcessor(receptor_path, ligand_path, pocket_cutoff, n_cpus)
            return system_processor.process_system()

        except Exception as e:
            logging.exception(f"Error processing system: {e}")
            return None

    # Process a batch of receptor-ligand pairs and return a list of SystemData objects
    def _process_batch(self, batch: List[Tuple[str, str]], pocket_cutoff: float, n_cpus: int) -> List[SystemData]:
        system_data_list = []
        for rec_path, lig_path in batch:
            try:
                processor = SystemProcessor(
                    receptor_path=rec_path, 
                    ligand_path=lig_path, 
                    pocket_cutoff=pocket_cutoff, 
                    n_cpus=n_cpus)
                
                result = processor.process_system(save_pockets=False)
                if result:
                    sysdata = result["systems_list"]
                    system_data_list.append(sysdata)
            except Exception as e:
                logger.critical(f"Error processing system {rec_path}, {lig_path}: {e}")
                logger.critical(traceback.format_exc())

        return system_data_list

    #modified to not include npnde, link_types. list of systemdata objects should be of length 1
    def _write_system_batch(self, system_data_batch: List[SystemData]):
        if not system_data_batch:
            return

        receptor_data = []
        ligand_data = []
        pocket_data = []
        pharm_data = []
        npnde_data = []

        system_info = []
        #loop will only run once (list of 1 System data)
        for i, system_data in enumerate(system_data_batch):
            #link_type = system_data.link_type
            #link_cif = None

            system_idx = len(self.system_lookup) + len(system_info)
            system_entry = {
                #"ligand_id": system_data.ligand_id,
                #"system_idx": system_idx,
                #"linkages": system_data.ligand.linkages, #do we need this?
                #"ccd": system_data.ligand.ccd,
                "lig_sdf": str(system_data.ligand.sdf) if isinstance(system_data.ligand.sdf, Path) else system_data.ligand.sdf,
                "rec_pdb": str(system_data.receptor.cif) if isinstance(system_data.receptor.cif, Path) else system_data.receptor.cif,
                "npnde_idxs": None,
            }

            receptor_data.append(system_data.receptor)
            ligand_data.append(system_data.ligand)
            pocket_data.append(system_data.pocket)
            pharm_data.append(system_data.pharmacophore)
            
            if system_data.npndes:

                npnde_idxs = []

                for j, (npnde_id, npnde_data_item) in enumerate(
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
            entry["pharm_start"], entry["pharm_end"] = pharm_indices[i]

        self.system_lookup.extend(system_info)
        self.root.attrs["system_lookup"] = self.system_lookup

        if npnde_entries:
            self.npnde_lookup.extend(npnde_entries)
            self.root.attrs["npnde_lookup"] = self.npnde_lookup

        logger.info(f"Wrote batch of {len(system_info)} systems to zarr store")


    def process_one_pair(self, receptor_path: str, ligand_path: str, pocket_cutoff: float, max_pending=None):
        #process a single receptor-ligand pair and write to zarr store
        logger.info(f"Processing system with receptor: {receptor_path}, ligand: {ligand_path}")
        start = len(self.system_lookup)

        try:
            processor = SystemProcessor(
                receptor_path=receptor_path, 
                ligand_path=ligand_path, 
                pocket_cutoff=pocket_cutoff, 
                n_cpus=1
            )

            result = processor.process_system(save_pockets=False)

            if result: 
                system_data = result["systems_list"] #will be a single SystemData object
                self._write_system_batch([system_data])
                logger.info("System written successfully.")
            else:
                logger.warning("System processing returned None. Skipping.")

        except Exception as e:
            logger.error(f"Failed to process system: {e}")
            return
        
        end = len(self.system_lookup)
        if self.category:
            self.root.attrs["system_type_idxs"].append((self.category, start, end))

        logger.info("Processing complete.")

    #support parallel processing functions s
    def callback(self, results, progressBar):
        progressBar.update(1)
        self.result_queue.extend(results)
        if len(self.result_queue) > 5: #batch size (write to zarr store as soon as we get 50 results
            self.write_queue()
    
    def write_queue(self): 
        self._write_system_batch(self.result_queue)
        self.result_queue = []

    def error_callback(self, error, progressBar, errorCounter):
        traceback.print_exception(type(error), error, error.__traceback__)
        errorCounter[0] += 1
        progressBar.set_postfix(errors=errorCounter[0])
        progressBar.update(1)

    # periodically pulls from the queue and writes to zarr store
    # flush_interval is the time to wait before flushing the buffer to zarr store
    # queue holds lists of SystemData objects
    # zarr_converter is the instance of CrossdockedNoLinksZarrConverter
    # stop_flag is a threading.Event that signals when to stop the thread
    def writer_thread_func(self, queue, zarr_converter, stop_flag, flush_interval=10):
        buffer = [] #list of SystemData objects
        while not stop_flag.is_set() or not queue.empty():
            try: 
                # waits flush_interval seconds for an item to be available in the queue
                item = queue.get(timeout=flush_interval)
                if item:
                    buffer.extend(item)
            except Empty:
                pass #no items available in flush_interval window

            if buffer:
                #write the buffer to zarr store
                zarr_converter._write_system_batch(buffer)
                buffer.clear()

    #dataset is a types file
    # max_pending is the maximum number of pending tasks in the pool (batches processed at once)
    def process_dataset_parallel(self, batches: List[List[Tuple[str, str]]], pocket_cutoff: float, n_cpus: int, max_pending: int = 1000):    
    
        progressBar = tqdm(total=len(batches), desc="Processing batches", unit="batch")
        errorCounter = [0]
   
        callback_func = functools.partial(self.callback, progressBar=progressBar)
        error_callback_func = functools.partial(self.error_callback, progressBar=progressBar, errorCounter=errorCounter)
        self.result_queue = []
        #launch workers
        with Pool(processes=n_cpus, maxtasksperchild=2) as pool:
            pending = []
            for i, batch in enumerate(batches):
                
                #if the number of pending tasks is greater than or equal to max_pending, wait for some to complete
                while len(pending) >= max_pending:
                    pending = [p for p in pending if not p.ready()]
                
                    if len(pending) >= max_pending:
                        time.sleep(0.1)

                result = pool.apply_async(self._process_batch,
                        args=(batch, pocket_cutoff, n_cpus),
                        callback=callback_func, 
                        error_callback=error_callback_func)
                pending.append(result)
            #wait for all pending tasks to complete
            for r in pending:
                r.wait()
        
        #final flush
        pool.join()
        pool.close()
        
        if len(self.result_queue) > 0:
            self.write_queue()

        progressBar.close()

    def process_dataset_serial(self, batches: List[List[Tuple[str, str]]], pocket_cutoff: float, n_cpus: int = 1):
        progressBar = tqdm(total=len(batches), desc="Processing batches", unit="batch")
        errorCounter = [0]

        self.result_queue = []

        for batch in batches:
            try:
                results = self._process_batch(batch, pocket_cutoff, n_cpus)
                self.callback(results, progressBar)
            except Exception as e:
                self.error_callback(e, progressBar, errorCounter)

        # Flush remaining results
        if self.result_queue:
            self.write_queue()

        progressBar.close()
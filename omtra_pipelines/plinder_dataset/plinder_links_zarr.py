import logging
import multiprocessing as mp
import queue
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import zarr
from numcodecs import VLenUTF8
from omtra_pipelines.plinder_dataset.plinder_pipeline import (
    LigandData,
    StructureData,
    PharmacophoreData,
    SystemData,
    SystemProcessor,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PlinderLinksZarrConverter:
    def __init__(
        self,
        output_path: str,
        system_processor: SystemProcessor,
        struc_chunk_size: int = 235000,
        lig_chunk_size: int = 2000,
        category: str = None,
        num_workers: int = 1,
    ):
        self.output_path = Path(output_path)
        self.system_processor = system_processor
        self.struc_chunk_size = struc_chunk_size
        self.lig_atom_chunk_size = lig_atom_chunk_size
        self.lig_bond_chunk_size = lig_bond_chunk_size
        self.pharmacophore_chunk_size = pharmacophore_chunk_size
        self.pocket_chunk_size = pocket_chunk_size
        self.category = category
        self.num_workers = num_workers

        if not self.output_path.exists():
            self.store = zarr.storage.LocalStore(str(self.output_path))
            self.root = zarr.group(store=self.store)

            self.receptor = self.root.create_group("receptor")
            self.apo = self.root.create_group("apo")
            self.pred = self.root.create_group("pred")

            self.pocket = self.root.create_group("pocket")
            self.pharmacophore = self.root.create_group("pharmacophore")
            self.ligand = self.root.create_group("ligand")
            self.npnde = self.root.create_group("npnde")

            for group in [self.receptor, self.apo, self.pred, self.pocket]:
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
                    chunks=(self.lig_chunk_size, 3),
                    dtype=np.float32,
                )
                group.create_array(
                    "atom_types",
                    shape=(0,),
                    chunks=(self.lig_chunk_size,),
                    dtype=np.int32,
                )
                group.create_array(
                    "atom_charges",
                    shape=(0,),
                    chunks=(self.lig_chunk_size,),
                    dtype=np.float32,
                )
                group.create_array(
                    "bond_types",
                    shape=(0,),
                    chunks=(self.lig_chunk_size,),
                    dtype=np.int32,
                )
                group.create_array(
                    "bond_indices",
                    shape=(0, 2),
                    chunks=(self.lig_chunk_size, 2),
                    dtype=np.int32,
                )

            # Initialize lookup tables
            self.root.attrs["system_lookup"] = []
            self.root.attrs["npnde_lookup"] = []
            self.root.attrs["chunk_sizes"] = []
            self.root.attrs["system_type_idxs"] = []

            self.system_lookup = self.root.attrs[
                "system_lookup"
            ]  # [{system_id, ligand_id, receptor_idx, ligand_idx, rec_start, rec_end, lig_atom_start, lig_atom_end, lig_bond_start, lig_bond_end, pharmacophore_idx, pharm_start, pharm_end, npnde_idxs, pocket_idx, pocket_start, pocket_end apo_idx, pred_idx, link_start, link_end, cif}]
            self.npnde_lookup = self.root.attrs[
                "npnde_lookup"
            ]  # [{system_id, npnde_id, receptor_idx, npnde_idx, ccd, linkages, atom_start, atom_end, bond_start, bond_end, sdf}]

        else:
            self.root = zarr.open_group(store=str(self.output_path), mode="r+")

            self.receptor = self.root["receptor"]
            self.apo = self.root["apo"]
            self.pred = self.root["pred"]
            self.pocket = self.root["pocket"]
            self.ligand = self.root["ligand"]
            self.npnde = self.root["npnde"]
            self.pharmacophore = self.root["pharmacophore"]

            self.system_lookup = self.root.attrs["system_lookup"]
            self.npnde_lookup = self.root.attrs["npnde_lookup"]

    def _append_structure_data(
        self, group: zarr.Group, data: StructureData
    ) -> Tuple[int, int]:
        """
        Append structure data to arrays and return start and end indices.

        Args:
            group: zarr group to append to
            data: structure data to append

        Returns:
            tuple[int, int]: (start_idx, end_idx) of the appended structure
        """
        current_len = group["coords"].shape[0]
        num_atoms = len(data.coords)
        new_len = current_len + num_atoms

        # Resize and append atomic data
        group["coords"].resize((new_len, 3))
        group["coords"][current_len:] = data.coords

        group["atom_names"].resize((new_len,))
        group["atom_names"][current_len:] = data.atom_names

        group["res_ids"].resize((new_len,))
        group["res_ids"][current_len:] = data.res_ids

        group["res_names"].resize((new_len,))
        group["res_names"][current_len:] = data.res_names

        group["chain_ids"].resize((new_len,))
        group["chain_ids"][current_len:] = data.chain_ids

        return current_len, new_len

    def _append_pharmacophore_data(
        self, group: zarr.Group, data: PharmacophoreData
    ) -> Tuple[int, int]:
        current_len = group["coords"].shape[0]
        num_centers = len(data.coords)
        if num_centers < 1:
            return None, None
        new_len = current_len + num_centers

        group["coords"].resize((new_len, 3))
        group["coords"][current_len:] = data.coords

        group["types"].resize((new_len,))
        group["types"][current_len:] = data.types

        group["vectors"].resize((new_len, 4, 3))
        group["vectors"][current_len:] = data.vectors

        group["interactions"].resize((new_len,))
        group["interactions"][current_len:] = data.interactions

        return current_len, new_len

    def _append_ligand_data(
        self, group: zarr.Group, data: LigandData
    ) -> Tuple[int, int, int, int]:
        """
        Append ligand data to arrays

        Args:
            group: zarr group to append to
            data: ligand data to append

        Returns:
            tuple[int, int, int, int]: (atom_start, atom_end, bond_start, bond_end) of the appended structure

        """
        current_len = group["coords"].shape[0]
        num_atoms = len(data.coords)
        new_len = current_len + num_atoms

        # Resize and append atomic data
        group["coords"].resize((new_len, 3))
        group["coords"][current_len:] = data.coords

        group["atom_types"].resize((new_len,))
        group["atom_types"][current_len:] = data.atom_types

        group["atom_charges"].resize((new_len,))
        group["atom_charges"][current_len:] = data.atom_charges

        num_bonds = len(data.bond_indices)

        bond_current_len, new_bond_len = None, None
        if num_bonds > 0:
            bond_current_len = group["bond_types"].shape[0]
            new_bond_len = bond_current_len + num_bonds
            group["bond_types"].resize((new_bond_len,))
            group["bond_types"][-num_bonds:] = data.bond_types

            group["bond_indices"].resize((new_bond_len, 2))
            group["bond_indices"][-num_bonds:] = data.bond_indices

        return current_len, new_len, bond_current_len, new_bond_len

    def _process_system(self, system_id: str):
        try:
            return self.system_processor.process_system(system_id)
        except Exception as e:
            logging.exception(f"Error processing system {system_id}: {e}")
            return None

    def _write_system(self, system_data: SystemData):
        """Process a single pair"""

        if not system_data:
            return
        link_type = None
        link_cif = None
        if system_data.apo:
            link_type = "apo"
            link_cif = system_data.apo.cif
        elif system_data.pred:
            link_type = "pred"
            link_cif = system_data.pred.cif
        else:
            return

        # Process holo structure
        system_idx = len(self.system_lookup)
        receptor_start, receptor_end = self._append_structure_data(
            self.receptor, system_data.receptor
        )
        # [{system_id, ligand_id, receptor_idx, ligand_idx, rec_start, rec_end, lig_atom_start, lig_atom_end, lig_bond_start, lig_bond_end, pharmacophore_idx, pharm_start, pharm_end, npnde_idxs, pocket_idx, pocket_start, pocket_end apo_idx, pred_idx, link_start, link_end, cif, sdf}]
        system_entry = {
            "system_id": system_data.system_id,
            "ligand_id": system_data.ligand_id,
            "system_idx": system_idx,
            "rec_start": receptor_start,
            "rec_end": receptor_end,
            "lig_atom_start": None,
            "lig_atom_end": None,
            "lig_bond_start": None,
            "lig_bond_end": None,
            "linkages": None,
            "ccd": None,
            "pocket_start": None,
            "pocket_end": None,
            "pharm_start": None,
            "pharm_end": None,
            "npnde_idxs": None,
            "link_type": link_type,
            "link_start": None,
            "link_end": None,
            "link_cif": link_cif,
            "lig_sdf": system_data.ligand.sdf,
            "rec_cif": system_data.receptor.cif,
        }

        # Process ligand
        lig_atom_start, lig_atom_end, lig_bond_start, lig_bond_end = (
            self._append_ligand_data(self.ligand, system_data.ligand)
        )
        system_entry["lig_atom_start"] = lig_atom_start
        system_entry["lig_atom_end"] = lig_atom_end
        system_entry["lig_bond_start"] = lig_bond_start
        system_entry["lig_bond_end"] = lig_bond_end
        system_entry["linkages"] = system_data.ligand.linkages
        system_entry["ccd"] = system_data.ccd

        # Process corresponding pocket
        pocket_start, pocket_end = self._append_structure_data(
            self.pocket, system_data.pocket
        )
        system_entry["pocket_start"] = pocket_start
        system_entry["pocket_end"] = pocket_end

        pharm_start, pharm_end = self._append_pharmacophore_data(
            self.pharmacophore, system_data.pharmacophore
        )
        system_entry["pharm_start"] = pharm_start
        system_entry["pharm_end"] = pharm_end

        # process npndes
        if system_data.npndes:
            npnde_idxs = []
            for npnde_id, npnde_data in system_data.npndes.items():
                npnde_idx = len(self.npnde_lookup)
                npnde_idxs.append(npnde_idx)

                atom_start, atom_end, bond_start, bond_end = self._append_ligand_data(
                    self.npnde, npnde_data
                )
                self.npnde_lookup.append(
                    {
                        "system_id": system_data.system_id,
                        "npnde_id": npnde_id,
                        "receptor_idx": receptor_idx,
                        "npnde_idx": npnde_idx,
                        "atom_start": atom_start,
                        "atom_end": atom_end,
                        "bond_start": bond_start,
                        "bond_end": bond_end,
                        "linkages": npnde_data.linkages,
                        "ccd": npnde_data.ccd,
                        "sdf": npnde_data.sdf,
                    }
                )
            system_entry["npnde_idxs"] = npnde_idxs

        # Process apo structure
        if link_type == "apo":
            apo_start, apo_end = self._append_structure_data(self.apo, system_data.apo)
            system_entry["link_start"] = apo_start
            system_entry["link_end"] = apo_end

        # Process pred structure
        if link_type == "pred":
            pred_start, pred_end = self._append_structure_data(
                self.pred, system_data.pred
            )
            system_entry["link_start"] = pred_start
            system_entry["link_end"] = pred_end

        self.system_lookup.append(system_entry)

    def process_dataset(self, system_ids: List[str]):
        """Process list of systems"""
        start = len(self.receptor_lookup)
        with mp.Manager() as manager:
            lock = manager.Lock()
            result_queue = manager.Queue()

            def _write_results_worker():
                while True:
                    try:
                        result = result_queue.get(timeout=1)
                        if result is None:
                            break
                        apo = result.get("apo")
                        pred = result.get("pred")

                        if apo:
                            for system in apo:
                                with lock:
                                    self._write_system(system)
                        if pred:
                            for system in pred:
                                with lock:
                                    self._write_system(system)

                    except queue.Empty:
                        continue

            writer_thread = threading.Thread(target=_write_results_worker)
            writer_thread.start()

            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                for sid in system_ids:
                    futures.append(executor.submit(self._process_system, sid))

                for future in tqdm(futures, desc="Processing systems"):
                    try:
                        result = future.result()
                        if result:
                            result_queue.put(result)
                    except Exception as e:
                        logging.exception(f"Error in future: {e}")
                        continue

            result_queue.put(None)
            writer_thread.join()

            end = len(self.receptor_lookup)
            if self.category:
                self.root.attrs["system_type_idxs"].append((category, start, end))

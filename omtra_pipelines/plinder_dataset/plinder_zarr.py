import zarr
import numpy as np
import pandas as pd
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import queue
import threading
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from omtra_pipelines.plinder_dataset.plinder_pipeline import (
    SystemProcessor,
    StructureData,
    LigandData,
)

logger = logging.getLogger(__name__)


class PlinderZarrConverter:
    def __init__(
        self,
        output_path: str,
        system_processor: SystemProcessor,
        struc_chunk_size: int = 235000,
        lig_chunk_size: int = 2000,
        num_workers: int = None,
    ):
        self.output_path = Path(output_path)
        self.system_processor = system_processor
        self.struc_chunk_size = struc_chunk_size
        self.lig_chunk_size = lig_chunk_size
        self.num_workers = num_workers or mp.cpu_count()

        self.store = zarr.storage.LocalStore(str(self.output_path))
        self.root = zarr.group(store=self.store)

        self.receptor = self.root.create_group("receptor")
        self.apo = self.root.create_group("apo")
        self.pred = self.root.create_group("pred")
        self.pocket = self.root.create_group("pocket")

        for group in [self.receptor, self.apo, self.pred, self.pocket]:
            chunk = self.struc_chunk_size
            if group == self.pocket:
                chunk = self.lig_chunk_size

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
            group.create_array("res_ids", shape=(0,), chunks=(chunk,), dtype=np.int32)
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

        self.ligand = self.root.create_group("ligand")
        self.npnde = self.root.create_group("npnde")

        for group in [self.ligand, self.npnde]:
            group.create_array(
                "coords",
                shape=(0, 3),
                chunks=(self.lig_chunk_size, 3),
                dtype=np.float32,
            )
            group.create_array(
                "atom_types", shape=(0,), chunks=(self.lig_chunk_size,), dtype=np.int32
            )
            group.create_array(
                "atom_charges",
                shape=(0,),
                chunks=(self.lig_chunk_size,),
                dtype=np.float32,
            )
            group.create_array(
                "bond_types", shape=(0,), chunks=(self.lig_chunk_size,), dtype=np.int32
            )
            group.create_array(
                "bond_indices",
                shape=(0, 2),
                chunks=(self.lig_chunk_size, 2),
                dtype=np.int32,
            )

        # Initialize lookup tables
        self.receptor_lookup = []  # [{system_id, receptor_idx, start, end, ligand_idxs, npnde_idxs, pocket_idxs, apo_idxs, pred_idxs, cif}]
        self.apo_lookup = []  # [{system_id, apo_id, receptor_idx, apo_idx, start, end, cif}]
        self.pred_lookup = []  # [{system_id, pred_id, receptor_idx, pred_idx, start, end, cif}]
        self.ligand_lookup = []  # [{system_id, ligand_id, receptor_idx, ligand_idx, ligand_num, atom_start, atom_end, bond_start, bond_end, sdf}]
        self.pocket_lookup = []  # [{system_id, pocket_id, receptor_idx, pocket_idx, pocket_num, start, end}]
        self.npnde_lookup = []  # [{system_id, npnde_id, receptor_idx, npnde_idx, atom_start, atom_end, bond_start, bond_end, sdf}]

    def _append_structure_data(
        self, group: zarr.Group, data: StructureData
    ) -> tuple[int, int]:
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

    def _append_ligand_data(
        self, group: zarr.Group, data: LigandData
    ) -> tuple[int, int, int, int]:
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

    def _write_system(self, system_data: Dict[str, Any]):
        """Process a single system"""

        if not system_data:
            return

        # Process holo structure
        receptor_idx = len(self.receptor_lookup)
        receptor_start, receptor_end = self._append_structure_data(
            self.receptor, system_data["receptor"]
        )

        receptor_entry = {
            "system_id": system_data["system_id"],
            "receptor_idx": receptor_idx,
            "start": receptor_start,
            "end": receptor_end,
            "ligand_idxs": [],
            "pocket_idxs": [],
            "npnde_idxs": None,
            "apo_idxs": None,
            "pred_idxs": None,
            "cif": system_data["receptor"].cif,
        }

        # Process ligands and their corresponding pockets
        ligand_count = 0
        ligand_idxs, pocket_idxs = [], []
        for ligand_id, ligand_data in system_data["ligands"].items():
            # Process ligand
            ligand_idx = len(self.ligand_lookup)
            ligand_idxs.append(ligand_idx)
            atom_start, atom_end, bond_start, bond_end = self._append_ligand_data(
                self.ligand, ligand_data
            )
            self.ligand_lookup.append(
                {
                    "system_id": system_data["system_id"],
                    "ligand_id": ligand_id,
                    "receptor_idx": receptor_idx,
                    "ligand_idx": ligand_idx,
                    "ligand_num": ligand_count,
                    "atom_start": atom_start,
                    "atom_end": atom_end,
                    "bond_start": bond_start,
                    "bond_end": bond_end,
                    "sdf": ligand_data.sdf,
                }
            )

            # Process corresponding pocket
            pocket_data = system_data["pockets"][ligand_id]
            pocket_idx = len(self.pocket_lookup)
            pocket_idxs.append(pocket_idx)
            pocket_start, pocket_end = self._append_structure_data(
                self.pocket, pocket_data
            )
            self.pocket_lookup.append(
                {
                    "system_id": system_data["system_id"],
                    "pocket_id": ligand_id,
                    "receptor_idx": receptor_idx,
                    "pocket_idx": pocket_idx,  # should be 1:1 ligand to pocket correspondance, but just in case
                    "pocket_count": ligand_count,
                    "start": pocket_start,
                    "end": pocket_end,
                }
            )

            ligand_count += 1

        receptor_entry["ligand_idxs"] = ligand_idxs
        receptor_entry["pocket_idxs"] = pocket_idxs

        # process npndes
        if system_data["npndes"]:
            npnde_idxs = []
            for npnde_id, npnde_data in system_data["npndes"].items():
                npnde_idx = len(self.npnde_lookup)
                npnde_idxs.append(npnde_idx)

                atom_start, atom_end, bond_start, bond_end = self._append_ligand_data(
                    self.npnde, npnde_data
                )
                self.npnde_lookup.append(
                    {
                        "system_id": system_data["system_id"],
                        "npnde_id": npnde_id,
                        "receptor_idx": receptor_idx,
                        "npnde_idx": npnde_idx,
                        "atom_start": atom_start,
                        "atom_end": atom_end,
                        "bond_start": bond_start,
                        "bond_end": bond_end,
                        "sdf": npnde_data.sdf,
                    }
                )
            receptor_entry["npnde_idxs"] = npnde_idxs

        # Process apo structures
        if system_data["apo_structures"]:
            apo_idxs = []
            for apo_id, apo_data in system_data["apo_structures"].items():
                apo_idx = len(self.apo_lookup)
                apo_idxs.append(apo_idx)
                apo_start, apo_end = self._append_structure_data(self.apo, apo_data)
                self.apo_lookup.append(
                    {
                        "system_id": system_data["system_id"],
                        "apo_id": apo_id,
                        "receptor_idx": receptor_idx,
                        "apo_idx": apo_idx,
                        "start": apo_start,
                        "end": apo_end,
                        "cif": apo_data.cif,
                    }
                )
            receptor_entry["apo_idxs"] = apo_idxs

        # Process pred structures
        if system_data["pred_structures"]:
            pred_idxs = []
            for pred_id, pred_data in system_data["pred_structures"].items():
                pred_idx = len(self.pred_lookup)
                pred_idxs.append(pred_idx)
                pred_start, pred_end = self._append_structure_data(self.pred, pred_data)
                self.pred_lookup.append(
                    {
                        "system_id": system_data["system_id"],
                        "pred_id": pred_id,
                        "receptor_idx": receptor_idx,
                        "pred_idx": pred_idx,
                        "start": pred_start,
                        "end": pred_end,
                        "cif": pred_data.cif,
                    }
                )
            receptor_entry["pred_idxs"] = pred_idxs
        self.receptor_lookup.append(receptor_entry)

    def process_dataset(self, system_ids: List[str]):
        """Process list of systems"""
        with mp.Manager() as manager:
            lock = manager.Lock()
            result_queue = manager.Queue()

            def _write_results_worker():
                while True:
                    try:
                        result = result_queue.get(timeout=1)
                        if result is None:
                            break

                        with lock:
                            self._write_system(result)

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

        # Store lookup tables as attributes
        self.root.attrs["receptor_lookup"] = self.receptor_lookup
        self.root.attrs["apo_lookup"] = self.apo_lookup
        self.root.attrs["pred_lookup"] = self.pred_lookup
        self.root.attrs["ligand_lookup"] = self.ligand_lookup
        self.root.attrs["pocket_lookup"] = self.pocket_lookup
        self.root.attrs["npnde_lookup"] = self.npnde_lookup


def load_lookups(
    zarr_path: str = None, root: zarr.Group = None
) -> Dict[str, pd.DataFrame]:
    """Helper function to load lookup tables from a zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    lookups = {}
    for key in [
        "receptor_lookup",
        "apo_lookup",
        "pred_lookup",
        "ligand_lookup",
        "pocket_lookup",
    ]:
        if key in root.attrs:
            data_type = key[:-7]
            lookups[data_type] = pd.DataFrame(root.attrs[key])

    return lookups


def get_receptor(
    idx: int, zarr_path: str = None, root: zarr.Group = None
) -> StructureData:
    """Helper function to load receptor from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    receptor_df = pd.DataFrame(root.attrs["receptor_lookup"])
    receptor_info = receptor_df[receptor_df["receptor_idx"] == idx].iloc[0]

    start, end = receptor_info["start"], receptor_info["end"]

    receptor = StructureData(
        coords=root["receptor"]["coords"][start:end],
        atom_names=root["receptor"]["atom_names"][start:end].astype(str),
        res_ids=root["receptor"]["res_ids"][start:end],
        res_names=root["receptor"]["res_names"][start:end].astype(str),
        chain_ids=root["receptor"]["chain_ids"][start:end].astype(str),
        cif=receptor_info["cif"],
    )
    return receptor


def get_ligand(
    lig_idx: int, zarr_path: str = None, root: zarr.Group = None
) -> LigandData:
    """Helper function to load ligand from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    ligand_df = pd.DataFrame(root.attrs["ligand_lookup"])
    ligand_info = ligand_df[ligand_df["ligand_idx"] == lig_idx].iloc[0]

    atom_start, atom_end = ligand_info["atom_start"], ligand_info["atom_end"]
    bond_start, bond_end = ligand_info["bond_start"], ligand_info["bond_end"]

    ligand = LigandData(
        sdf=ligand_info["sdf"],
        coords=root["ligand"]["coords"][atom_start:atom_end],
        atom_types=root["ligand"]["atom_types"][atom_start:atom_end],
        atom_charges=root["ligand"]["atom_charges"][atom_start:atom_end],
        bond_types=root["ligand"]["bond_types"][bond_start:bond_end],
        bond_indices=root["ligand"]["bond_indices"][bond_start:bond_end],
    )

    return ligand_info["ligand_id"], ligand


def get_npnde(
    npnde_idx: int, zarr_path: str = None, root: zarr.Group = None
) -> LigandData:
    """Helper function to load npnde from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    npnde_df = pd.DataFrame(root.attrs["npnde_lookup"])
    npnde_info = npnde_df[npnde_df["npnde_idx"] == npnde_idx].iloc[0]

    atom_start, atom_end = npnde_info["atom_start"], npnde_info["atom_end"]
    bond_start, bond_end = npnde_info["bond_start"], npnde_info["bond_end"]

    npnde = LigandData(
        sdf=npnde_info["sdf"],
        coords=root["npnde"]["coords"][atom_start:atom_end],
        atom_types=root["npnde"]["atom_types"][atom_start:atom_end],
        atom_charges=root["npnde"]["atom_charges"][atom_start:atom_end],
        bond_types=root["npnde"]["bond_types"][bond_start:bond_end],
        bond_indices=root["npnde"]["bond_indices"][bond_start:bond_end],
    )

    return npnde_info["npnde_id"], npnde


def get_pocket(
    pocket_idx: int, zarr_path: str = None, root: zarr.Group = None
) -> StructureData:
    """Helper function to load pocket from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    pocket_df = pd.DataFrame(root.attrs["pocket_lookup"])
    pocket_info = pocket_df[pocket_df["pocket_idx"] == pocket_idx].iloc[0]

    start, end = pocket_info["start"], pocket_info["end"]

    pocket = StructureData(
        coords=root["pocket"]["coords"][start:end],
        atom_names=root["pocket"]["atom_names"][start:end].astype(str),
        res_ids=root["pocket"]["res_ids"][start:end],
        res_names=root["pocket"]["res_names"][start:end].astype(str),
        chain_ids=root["pocket"]["chain_ids"][start:end].astype(str),
    )

    return pocket_info["pocket_id"], pocket


def get_apo(
    apo_idx: int, zarr_path: str = None, root: zarr.Group = None
) -> StructureData:
    """Helper function to load apo structure from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    apo_df = pd.DataFrame(root.attrs["apo_lookup"])
    apo_info = apo_df[apo_df["apo_idx"] == apo_idx].iloc[0]

    start, end = apo_info["start"], apo_info["end"]

    apo = StructureData(
        coords=root["apo"]["coords"][start:end],
        atom_names=root["apo"]["atom_names"][start:end].astype(str),
        res_ids=root["apo"]["res_ids"][start:end],
        res_names=root["apo"]["res_names"][start:end].astype(str),
        chain_ids=root["apo"]["chain_ids"][start:end].astype(str),
        cif=apo_info["cif"],
    )

    return apo_info["apo_id"], apo


def get_pred(
    pred_idx: int, zarr_path: str = None, root: zarr.Group = None
) -> StructureData:
    """Helper function to load predicted structure from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    pred_df = pd.DataFrame(root.attrs["pred_lookup"])
    pred_info = pred_df[pred_df["pred_idx"] == pred_idx].iloc[0]

    start, end = pred_info["start"], pred_info["end"]

    pred = StructureData(
        coords=root["pred"]["coords"][start:end],
        atom_names=root["pred"]["atom_names"][start:end].astype(str),
        res_ids=root["pred"]["res_ids"][start:end],
        res_names=root["pred"]["res_names"][start:end].astype(str),
        chain_ids=root["pred"]["chain_ids"][start:end].astype(str),
        cif=pred_info["cif"],
    )

    return pred_info["pred_id"], pred


def get_system(zarr_path: str, receptor_idx: int) -> Dict:
    """Helper function to load system from zarr store"""
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.group(store=store)

    receptor_df = pd.DataFrame(root.attrs["receptor_lookup"])
    receptor_info = receptor_df[receptor_df["receptor_idx"] == receptor_idx].iloc[0]

    receptor = get_receptor(receptor_idx, root=root)

    ligands = {}
    for lig_idx in receptor_info["ligand_idxs"]:
        lig_id, ligand = get_ligand(lig_idx, root=root)
        ligands[lig_id] = ligand

    pockets = {}
    for pocket_idx in receptor_info["pocket_idxs"]:
        pocket_id, pocket = get_pocket(pocket_idx, root=root)
        pockets[pocket_id] = pocket

    npndes = None
    if receptor_info["npnde_idxs"] is not None:
        npndes = {}
        for npnde_idx in receptor_info["npnde_idxs"]:
            npnde_id, npnde = get_npnde(npnde_idx, root=root)
            npndes[npnde_id] = npnde

    apos = None
    if receptor_info["apo_idxs"] is not None:
        apos = {}
        for apo_idx in receptor_info["apo_idxs"]:
            apo_id, apo = get_apo(apo_idx, root=root)
            apos[apo_id] = apo

    preds = None
    if receptor_info["pred_idxs"] is not None:
        preds = {}
        for pred_idx in receptor_info["pred_idxs"]:
            pred_id, pred = get_pred(pred_idx, root=root)
            preds[pred_id] = pred

    return {
        "system_id": receptor_info["system_id"],
        "receptor": receptor,
        "ligands": ligands,
        "pockets": pockets,
        "npndes": npndes,
        "apo_structures": apos,
        "pred_structures": preds,
    }

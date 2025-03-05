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
) -> (str, Dict[str, StructureData]):
    """Helper function to load apo structure from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    apo_df = pd.DataFrame(root.attrs["apo_lookup"])
    apo_info = apo_df[apo_df["apo_idx"] == apo_idx].iloc[0]

    receptor_df = pd.DataFrame(root.attrs["receptor_lookup"])
    receptor_info = receptor_df[
        receptor_df["receptor_idx"] == apo_info["receptor_idx"]
    ].iloc[0]

    start, end = apo_info["start"], apo_info["end"]
    link_start, link_end = apo_info["link_start"], apo_info["link_end"]

    apo = StructureData(
        coords=root["apo"]["coords"][start:end],
        atom_names=root["apo"]["atom_names"][start:end].astype(str),
        res_ids=root["apo"]["res_ids"][start:end],
        res_names=root["apo"]["res_names"][start:end].astype(str),
        chain_ids=root["apo"]["chain_ids"][start:end].astype(str),
        cif=apo_info["cif"],
    )
    holo = StructureData(
        coords=root["link_receptor"]["coords"][link_start:link_end],
        atom_names=root["link_receptor"]["atom_names"][link_start:link_end].astype(str),
        res_ids=root["link_receptor"]["res_ids"][link_start:link_end],
        res_names=root["link_receptor"]["res_names"][link_start:link_end].astype(str),
        chain_ids=root["link_receptor"]["chain_ids"][link_start:link_end].astype(str),
        cif=receptor_info["cif"],
    )

    return apo_info["apo_id"], {apo_info["apo_id"]: apo, "holo": holo}


def get_pred(
    pred_idx: int, zarr_path: str = None, root: zarr.Group = None
) -> StructureData:
    """Helper function to load predicted structure from zarr store"""
    if not root:
        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.group(store=store)

    pred_df = pd.DataFrame(root.attrs["pred_lookup"])
    pred_info = pred_df[pred_df["pred_idx"] == pred_idx].iloc[0]

    receptor_df = pd.DataFrame(root.attrs["receptor_lookup"])
    receptor_info = receptor_df[
        receptor_df["receptor_idx"] == pred_info["receptor_idx"]
    ].iloc[0]

    start, end = pred_info["start"], pred_info["end"]
    link_start, link_end = pred_info["link_start"], pred_info["link_end"]

    pred = StructureData(
        coords=root["pred"]["coords"][start:end],
        atom_names=root["pred"]["atom_names"][start:end].astype(str),
        res_ids=root["pred"]["res_ids"][start:end],
        res_names=root["pred"]["res_names"][start:end].astype(str),
        chain_ids=root["pred"]["chain_ids"][start:end].astype(str),
        cif=pred_info["cif"],
    )
    holo = StructureData(
        coords=root["link_receptor"]["coords"][link_start:link_end],
        atom_names=root["link_receptor"]["atom_names"][link_start:link_end].astype(str),
        res_ids=root["link_receptor"]["res_ids"][link_start:link_end],
        res_names=root["link_receptor"]["res_names"][link_start:link_end].astype(str),
        chain_ids=root["link_receptor"]["chain_ids"][link_start:link_end].astype(str),
        cif=receptor_info["cif"],
    )

    return pred_info["pred_id"], {pred_info["pred_id"]: pred, "holo": holo}


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

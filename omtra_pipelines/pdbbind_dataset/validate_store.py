"""Spot-check PDBBind zarr store against reprocessed SystemData."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from omtra_pipelines.pdbbind_dataset.pdbbind_pipeline import PdbbindSystemProcessor
from omtra_pipelines.plinder_dataset.utils import setup_logger

logger = setup_logger(__name__)


def load_zarr_coords(root, group: str, start: int, end: int) -> np.ndarray:
    return np.array(root[group]["coords"][start:end])


def compare_structure_coords(
    name: str,
    zarr_coords: np.ndarray,
    zarr_atom_names: np.ndarray,
    zarr_res_ids: np.ndarray,
    zarr_chain_ids: np.ndarray,
    struct_coords: np.ndarray,
    struct_atom_names: np.ndarray,
    struct_res_ids: np.ndarray,
    struct_chain_ids: np.ndarray,
    atol: float = 1e-4,
) -> bool:
    def sort_key(coords, atom_names, res_ids, chain_ids):
        keys = [
            (str(chain_ids[i]), int(res_ids[i]), str(atom_names[i]))
            for i in range(len(coords))
        ]
        order = sorted(range(len(keys)), key=lambda i: keys[i])
        return coords[order], [keys[i] for i in order]

    z_sorted, z_keys = sort_key(
        zarr_coords, zarr_atom_names, zarr_res_ids, zarr_chain_ids
    )
    s_sorted, s_keys = sort_key(
        struct_coords, struct_atom_names, struct_res_ids, struct_chain_ids
    )
    if z_keys != s_keys:
        logger.error(f"{name}: atom key mismatch in stored vs reprocessed pocket")
        return False
    if not np.allclose(z_sorted, s_sorted, atol=atol, rtol=0):
        max_diff = np.max(np.abs(z_sorted - s_sorted))
        logger.error(f"{name}: max coord diff {max_diff}")
        return False
    logger.info(f"{name}: OK ({len(z_sorted)} atoms)")
    return True


def validate_system(
    system_id: str,
    row: pd.Series,
    zarr_path: Path,
    raw_root: str,
    pocket_cutoff: float,
    atol: float,
) -> bool:
    store = zarr.open(str(zarr_path), mode="r")
    lookup = store.attrs["system_lookup"]
    entry = next(e for e in lookup if e["system_id"] == system_id)

    processor = PdbbindSystemProcessor(
        system_id=system_id,
        holo_pdb=row["holo_pdb"],
        ligand_sdf=row["ligand_sdf"],
        pred_pdb=row["pred_pdb"],
        link_id=row["uid"],
        pocket_cutoff=pocket_cutoff,
        raw_data_root=raw_root,
    )
    result = processor.process_system()
    if result is None:
        logger.error(f"Reprocessing failed for {system_id}")
        return False

    system_data = result["pred"][row["uid"]][0]
    ok = True
    ok &= compare_structure_coords(
        "receptor",
        load_zarr_coords(store, "receptor", entry["rec_start"], entry["rec_end"]),
        np.array(store["receptor"]["atom_names"][entry["rec_start"] : entry["rec_end"]]),
        np.array(store["receptor"]["res_ids"][entry["rec_start"] : entry["rec_end"]]),
        np.array(store["receptor"]["chain_ids"][entry["rec_start"] : entry["rec_end"]]),
        system_data.receptor.coords,
        system_data.receptor.atom_names,
        system_data.receptor.res_ids,
        system_data.receptor.chain_ids,
        atol,
    )
    ok &= compare_structure_coords(
        "pred",
        load_zarr_coords(store, "pred", entry["link_start"], entry["link_end"]),
        np.array(store["pred"]["atom_names"][entry["link_start"] : entry["link_end"]]),
        np.array(store["pred"]["res_ids"][entry["link_start"] : entry["link_end"]]),
        np.array(store["pred"]["chain_ids"][entry["link_start"] : entry["link_end"]]),
        system_data.link.coords,
        system_data.link.atom_names,
        system_data.link.res_ids,
        system_data.link.chain_ids,
        atol,
    )
    ok &= compare_structure_coords(
        "pocket",
        load_zarr_coords(store, "pocket", entry["pocket_start"], entry["pocket_end"]),
        np.array(store["pocket"]["atom_names"][entry["pocket_start"] : entry["pocket_end"]]),
        np.array(store["pocket"]["res_ids"][entry["pocket_start"] : entry["pocket_end"]]),
        np.array(store["pocket"]["chain_ids"][entry["pocket_start"] : entry["pocket_end"]]),
        system_data.pocket.coords,
        system_data.pocket.atom_names,
        system_data.pocket.res_ids,
        system_data.pocket.chain_ids,
        atol,
    )
    lig_z = load_zarr_coords(store, "ligand", entry["lig_atom_start"], entry["lig_atom_end"])
    if not np.allclose(lig_z, system_data.ligand.coords, atol=atol, rtol=0):
        logger.error("ligand: coord mismatch")
        ok = False
    else:
        logger.info(f"ligand: OK ({len(lig_z)} atoms)")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Validate PDBBind zarr store")
    parser.add_argument("--manifest", default="data/pdbbind_12A/manifest.parquet")
    parser.add_argument("--zarr", required=True, help="Path to split zarr store")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-systems", type=int, default=5)
    parser.add_argument("--pocket-cutoff", type=float, default=12.0)
    parser.add_argument(
        "--raw-root",
        default=os.environ.get(
            "PDBBIND_RAW_ROOT",
            "~/.local/share/dynamicbind_data",
        ),
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    manifest = manifest[manifest["split"] == args.split]
    if "files_ok" in manifest.columns:
        manifest = manifest[manifest["files_ok"]]

    store = zarr.open(args.zarr, mode="r")
    stored_ids = {e["system_id"] for e in store.attrs["system_lookup"]}
    candidates = manifest[manifest["system_id"].isin(stored_ids)].head(args.n_systems)

    n_pass = 0
    for _, row in candidates.iterrows():
        sid = row["system_id"]
        logger.info(f"Validating {sid}...")
        if validate_system(
            sid, row, Path(args.zarr), args.raw_root, args.pocket_cutoff, args.atol
        ):
            n_pass += 1

    print(f"Validated {n_pass}/{len(candidates)} systems successfully")
    if n_pass < len(candidates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

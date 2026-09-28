#!/usr/bin/env python3
"""Re-extract protein pockets at a larger cutoff from an existing pred zarr.

Copies non-pocket groups from the source store and rebuilds the pocket group
(+ pocket backbone offsets in system_lookup) at ``--pocket_cutoff`` using the
already-aligned receptor coordinates. Avoids reloading Plinder/AF2.

Writes pockets in batched appends for acceptable throughput.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=str, required=True, help="Source pred zarr path")
    p.add_argument("--dst", type=str, required=True, help="Destination pred zarr path")
    p.add_argument("--pocket_cutoff", type=float, default=12.0)
    p.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Systems per pocket append batch",
    )
    p.add_argument(
        "--num_systems",
        type=int,
        default=None,
        help="Limit systems (debug)",
    )
    return p.parse_args()


def extract_pocket_indices(
    rec_coords: np.ndarray,
    rec_res_ids: np.ndarray,
    rec_chain_ids: np.ndarray,
    lig_coords: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Return atom indices of whole residues within cutoff of any ligand atom."""
    if len(rec_coords) == 0 or len(lig_coords) == 0:
        return np.array([], dtype=np.int64)

    tree = cKDTree(rec_coords)
    # Query all ligand atoms at once when possible
    hits = tree.query_ball_point(lig_coords, r=cutoff)
    close: set[int] = set()
    for h in hits:
        close.update(h)
    if not close:
        return np.array([], dtype=np.int64)

    keep_res = {(str(rec_chain_ids[i]), int(rec_res_ids[i])) for i in close}
    mask = np.fromiter(
        ((str(c), int(r)) in keep_res for c, r in zip(rec_chain_ids, rec_res_ids)),
        dtype=bool,
        count=len(rec_res_ids),
    )
    return np.nonzero(mask)[0]


def build_backbone_from_atoms(
    coords: np.ndarray,
    atom_names: np.ndarray,
    res_ids: np.ndarray,
    res_names: np.ndarray,
    chain_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (N,3,3) backbone coords for residues that have N/CA/C."""
    order: List[Tuple[str, int]] = []
    seen = set()
    for c, r in zip(chain_ids, res_ids):
        key = (str(c), int(r))
        if key not in seen:
            seen.add(key)
            order.append(key)

    atom_map: Dict[Tuple[str, int, str], int] = {}
    for i, (c, r, a) in enumerate(zip(chain_ids, res_ids, atom_names)):
        atom_map[(str(c), int(r), str(a))] = i

    bb_coords = []
    bb_res_ids = []
    bb_res_names = []
    bb_chain_ids = []
    res_name_lookup = {
        (str(c), int(r)): str(n) for c, r, n in zip(chain_ids, res_ids, res_names)
    }

    for chain, rid in order:
        idxs = []
        ok = True
        for aname in ("N", "CA", "C"):
            key = (chain, rid, aname)
            if key not in atom_map:
                ok = False
                break
            idxs.append(atom_map[key])
        if not ok:
            continue
        bb_coords.append(coords[idxs])
        bb_res_ids.append(rid)
        bb_res_names.append(res_name_lookup[(chain, rid)])
        bb_chain_ids.append(chain)

    if not bb_coords:
        return (
            np.zeros((0, 3, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.array([], dtype=object),
            np.array([], dtype=object),
        )

    return (
        np.asarray(bb_coords, dtype=np.float32),
        np.asarray(bb_res_ids, dtype=np.int32),
        np.asarray(bb_res_names, dtype=object),
        np.asarray(bb_chain_ids, dtype=object),
    )


def reset_pocket_group(root: zarr.Group, chunk: int = 150000) -> None:
    if "pocket" in root:
        del root["pocket"]
    pocket = root.create_group("pocket")
    pocket.create_array("coords", shape=(0, 3), chunks=(chunk, 3), dtype=np.float32)
    for name, dtype in [
        ("atom_names", str),
        ("elements", str),
        ("res_names", str),
        ("chain_ids", str),
        ("backbone_res_names", str),
        ("backbone_chain_ids", str),
    ]:
        pocket.create_array(
            name, shape=(0,), chunks=(chunk,), dtype=dtype, compressors=None
        )
    pocket.create_array("res_ids", shape=(0,), chunks=(chunk,), dtype=np.int32)
    pocket.create_array("backbone_mask", shape=(0,), chunks=(chunk,), dtype=bool)
    pocket.create_array(
        "backbone_coords", shape=(0, 3, 3), chunks=(chunk, 3, 3), dtype=np.float32
    )
    pocket.create_array("backbone_res_ids", shape=(0,), chunks=(chunk,), dtype=np.int32)


def backbone_atom_mask(atom_names: np.ndarray) -> np.ndarray:
    return np.isin(np.asarray(atom_names, dtype=str), ["N", "CA", "C"])


def flush_batch(pocket: zarr.Group, batch: Dict[str, list]) -> None:
    if not batch["coords"]:
        return
    pocket["coords"].append(np.concatenate(batch["coords"], axis=0).astype(np.float32))
    pocket["atom_names"].append(np.concatenate(batch["atom_names"]).astype(str))
    pocket["elements"].append(np.concatenate(batch["elements"]).astype(str))
    pocket["res_ids"].append(np.concatenate(batch["res_ids"]).astype(np.int32))
    pocket["res_names"].append(np.concatenate(batch["res_names"]).astype(str))
    pocket["chain_ids"].append(np.concatenate(batch["chain_ids"]).astype(str))
    pocket["backbone_mask"].append(np.concatenate(batch["backbone_mask"]))
    pocket["backbone_coords"].append(
        np.concatenate(batch["backbone_coords"], axis=0).astype(np.float32)
        if batch["backbone_coords"]
        else np.zeros((0, 3, 3), dtype=np.float32)
    )
    pocket["backbone_res_ids"].append(
        np.concatenate(batch["backbone_res_ids"]).astype(np.int32)
        if batch["backbone_res_ids"]
        else np.zeros((0,), dtype=np.int32)
    )
    pocket["backbone_res_names"].append(
        np.concatenate(batch["backbone_res_names"]).astype(str)
        if batch["backbone_res_names"]
        else np.array([], dtype=str)
    )
    pocket["backbone_chain_ids"].append(
        np.concatenate(batch["backbone_chain_ids"]).astype(str)
        if batch["backbone_chain_ids"]
        else np.array([], dtype=str)
    )
    for k in batch:
        batch[k].clear()


def empty_batch() -> Dict[str, list]:
    return {
        "coords": [],
        "atom_names": [],
        "elements": [],
        "res_ids": [],
        "res_names": [],
        "chain_ids": [],
        "backbone_mask": [],
        "backbone_coords": [],
        "backbone_res_ids": [],
        "backbone_res_names": [],
        "backbone_chain_ids": [],
    }


def repocket(
    src: Path, dst: Path, cutoff: float, num_systems: int | None, batch_size: int
) -> None:
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    print(f"Copying {src} -> {dst}")
    shutil.copytree(src, dst)

    root = zarr.open_group(str(dst), mode="r+")
    lookup = list(root.attrs["system_lookup"])
    if num_systems is not None:
        lookup = lookup[:num_systems]

    reset_pocket_group(root)
    pocket = root["pocket"]

    atom_ptr = 0
    bb_ptr = 0
    new_lookup = []
    batch = empty_batch()
    n_in_batch = 0

    for entry in tqdm(lookup, desc=f"repocket@{cutoff}A"):
        e = dict(entry)
        lig = np.asarray(
            root["ligand"]["coords"][e["lig_atom_start"] : e["lig_atom_end"]]
        )
        rs, re = int(e["rec_start"]), int(e["rec_end"])
        rec_coords = np.asarray(root["receptor"]["coords"][rs:re])
        rec_res = np.asarray(root["receptor"]["res_ids"][rs:re])
        rec_chain = np.asarray(root["receptor"]["chain_ids"][rs:re])
        rec_names = np.asarray(root["receptor"]["atom_names"][rs:re])
        rec_elements = np.asarray(root["receptor"]["elements"][rs:re])
        rec_res_names = np.asarray(root["receptor"]["res_names"][rs:re])

        idxs = extract_pocket_indices(rec_coords, rec_res, rec_chain, lig, cutoff)
        if len(idxs) == 0:
            raise RuntimeError(
                f"Empty pocket for {e.get('system_id')} after {cutoff}A extract"
            )

        p_coords = rec_coords[idxs]
        p_atom_names = rec_names[idxs]
        p_elements = rec_elements[idxs]
        p_res_ids = rec_res[idxs]
        p_res_names = rec_res_names[idxs]
        p_chain_ids = rec_chain[idxs]
        p_bb_mask = backbone_atom_mask(p_atom_names)

        bb_coords, bb_res_ids, bb_res_names, bb_chain_ids = build_backbone_from_atoms(
            p_coords, p_atom_names, p_res_ids, p_res_names, p_chain_ids
        )

        n_atoms = len(p_coords)
        n_bb = len(bb_coords)

        batch["coords"].append(p_coords.astype(np.float32, copy=False))
        batch["atom_names"].append(np.asarray(p_atom_names, dtype=object))
        batch["elements"].append(np.asarray(p_elements, dtype=object))
        batch["res_ids"].append(p_res_ids.astype(np.int32, copy=False))
        batch["res_names"].append(np.asarray(p_res_names, dtype=object))
        batch["chain_ids"].append(np.asarray(p_chain_ids, dtype=object))
        batch["backbone_mask"].append(p_bb_mask)
        if n_bb:
            batch["backbone_coords"].append(bb_coords.astype(np.float32, copy=False))
            batch["backbone_res_ids"].append(bb_res_ids.astype(np.int32, copy=False))
            batch["backbone_res_names"].append(np.asarray(bb_res_names, dtype=object))
            batch["backbone_chain_ids"].append(np.asarray(bb_chain_ids, dtype=object))

        e["pocket_start"] = atom_ptr
        e["pocket_end"] = atom_ptr + n_atoms
        e["pocket_bb_start"] = bb_ptr
        e["pocket_bb_end"] = bb_ptr + n_bb
        atom_ptr += n_atoms
        bb_ptr += n_bb
        new_lookup.append(e)

        n_in_batch += 1
        if n_in_batch >= batch_size:
            flush_batch(pocket, batch)
            n_in_batch = 0

    flush_batch(pocket, batch)
    root.attrs["system_lookup"] = new_lookup

    # Preserve cluster_id presence check
    has_cluster = sum(1 for e in new_lookup if e.get("cluster_id") is not None)
    print(
        f"Done. systems={len(new_lookup)} pocket_atoms={atom_ptr} "
        f"pocket_bb={bb_ptr} with_cluster_id={has_cluster}"
    )
    print(f"Wrote {dst}")


def main():
    args = parse_args()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    repocket(src, dst, args.pocket_cutoff, args.num_systems, args.batch_size)


if __name__ == "__main__":
    main()

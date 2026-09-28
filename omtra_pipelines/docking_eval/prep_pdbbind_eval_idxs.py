#!/usr/bin/env python3
"""Build PDBBind eval system index file (DynamicBind 303-filtered test set)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import zarr

from omtra.utils import omtra_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(omtra_root()) / "local" / "d3_with_clash_info.csv",
        help="DynamicBind d3_with_clash_info.csv",
    )
    parser.add_argument(
        "--zarr",
        type=Path,
        default=Path(omtra_root()) / "data" / "pdbbind_12A" / "pred" / "test.zarr",
        help="PDBBind pred test zarr store",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "eval_idx_files"
        / "pdbbind_dynamicbind_test_sys_idxs.csv",
        help="Output CSV (one comma-separated line of system_idx values)",
    )
    parser.add_argument(
        "--fraction-threshold",
        type=float,
        default=0.8,
        help="DynamicBind fraction_of_this_chain filter (default: 0.8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv)
    filtered = df[
        (df["group"] == "test")
        & (df["fraction_of_this_chain"] > args.fraction_threshold)
    ].copy()
    filtered["pdb"] = filtered["pdb"].str.lower()

    root = zarr.open(args.zarr, mode="r")
    lookup = pd.DataFrame(root.attrs["system_lookup"])
    pdb_to_idx = dict(zip(lookup["system_id"].str.lower(), lookup["system_idx"]))

    idxs = []
    missing = []
    for pdb in filtered["pdb"]:
        if pdb in pdb_to_idx:
            idxs.append(int(pdb_to_idx[pdb]))
        else:
            missing.append(pdb)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(",".join(str(i) for i in idxs))
        f.write("\n")

    print(f"CSV test rows with fraction > {args.fraction_threshold}: {len(filtered)}")
    print(f"Mapped to zarr system_idx: {len(idxs)}")
    if missing:
        print(f"Missing from zarr ({len(missing)}): {', '.join(sorted(missing))}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

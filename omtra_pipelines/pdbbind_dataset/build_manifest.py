"""Build PDBBind manifest parquet from DynamicBind d3_with_clash_info.csv."""

import argparse
import os
from pathlib import Path

import pandas as pd

NAS_PREFIX = "/mnt/nas/research-data/luwei/dynamicbind_data"
DEFAULT_RAW_ROOT = os.path.expanduser("~/.local/share/dynamicbind_data")


def remap_path(path: str, raw_root: Path) -> Path | None:
    if pd.isna(path) or not path:
        return None
    path = path.replace("//", "/")
    if path.startswith(NAS_PREFIX):
        rel = path[len(NAS_PREFIX) :].lstrip("/")
        return raw_root / rel
    return Path(path)


def main():
    parser = argparse.ArgumentParser(description="Build PDBBind manifest from DynamicBind CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default="local/d3_with_clash_info.csv",
        help="Path to d3_with_clash_info.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pdbbind_12A/manifest.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default=os.environ.get("PDBBIND_RAW_ROOT", DEFAULT_RAW_ROOT),
        help="Local root replacing NAS prefix in CSV paths",
    )
    parser.add_argument(
        "--require-files",
        action="store_true",
        help="Drop rows with missing holo/ligand/pred files (default: keep all, mark missing)",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    df = pd.read_csv(args.csv)

    split_map = {"train": "train", "valid": "val", "test": "test", "remove": "remove"}
    df = df[df["group"] != "remove"].copy()
    df["split"] = df["group"].map(split_map)

    records = []
    n_missing = 0
    for _, row in df.iterrows():
        holo = remap_path(row["pdbFile"], raw_root)
        pred = remap_path(row["af2File"], raw_root)
        ligand = remap_path(row["ligandFile"], raw_root)
        raw_af2 = (
            remap_path(row["raw_af2File"], raw_root)
            if "raw_af2File" in row.index
            else None
        )

        files_ok = (
            holo is not None
            and pred is not None
            and ligand is not None
            and holo.exists()
            and pred.exists()
            and ligand.exists()
        )
        if args.require_files and not files_ok:
            n_missing += 1
            continue

        records.append(
            {
                "system_id": str(row["pdb"]).lower(),
                "uid": row["uid"],
                "split": row["split"],
                "holo_pdb": str(holo) if holo else None,
                "pred_pdb": str(pred) if pred else None,
                "ligand_sdf": str(ligand) if ligand else None,
                "raw_af2_pdb": str(raw_af2)
                if raw_af2 is not None and raw_af2.exists()
                else None,
                "files_ok": files_ok,
                "pred_ids": "1",
            }
        )

    out_df = pd.DataFrame(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    n_ok = out_df["files_ok"].sum() if "files_ok" in out_df.columns else len(out_df)
    print(
        f"Wrote {len(out_df)} rows to {out_path} "
        f"({n_ok} with all files present, {n_missing} skipped missing)"
    )


if __name__ == "__main__":
    main()

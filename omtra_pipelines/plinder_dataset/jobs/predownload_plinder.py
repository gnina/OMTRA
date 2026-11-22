#!/usr/bin/env python3
"""
Pre-download all Plinder 2024-04/v1 system zip files and extract them.

Usage:
    python predownload_plinder.py [--workers 8]
"""
import argparse
import os
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import gcsfs


GCS_PREFIX = "plinder/2024-04/v1/systems"
LOCAL_DIR = Path.home() / ".local/share/plinder/2024-04/v1/systems"


def get_all_gcs_zips():
    fs = gcsfs.GCSFileSystem(token="anon")
    entries = fs.ls(GCS_PREFIX)
    return sorted([e.split("/")[-1] for e in entries if e.endswith(".zip")])


def download_and_extract(fs, zip_name, local_dir):
    """Download a zip from GCS, validate it, and extract."""
    remote = f"{GCS_PREFIX}/{zip_name}"
    local = local_dir / zip_name
    try:
        fs.get(remote, str(local))
        # Validate
        with zipfile.ZipFile(local, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                return False, zip_name, f"corrupt entry: {bad}"
            zf.extractall(local_dir)
        return True, zip_name, None
    except Exception as e:
        # Clean up partial download
        if local.exists():
            local.unlink()
        return False, zip_name, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Listing GCS zip files...")
    gcs_zips = get_all_gcs_zips()
    print(f"  Total on GCS: {len(gcs_zips)}")

    local_zips = set(p.name for p in LOCAL_DIR.glob("*.zip"))
    missing = [z for z in gcs_zips if z not in local_zips]
    print(f"  Already local: {len(local_zips)}")
    print(f"  To download: {len(missing)}")

    if not missing:
        print("Nothing to download!")
        return

    print(f"\nDownloading {len(missing)} zips with {args.workers} threads...")
    fs = gcsfs.GCSFileSystem(token="anon")
    ok_count = 0
    fail_count = 0
    failed_zips = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_and_extract, fs, z, LOCAL_DIR): z for z in missing
        }
        for fut in as_completed(futures):
            success, name, err = fut.result()
            if success:
                ok_count += 1
                if ok_count % 20 == 0 or ok_count == len(missing):
                    print(f"  Progress: {ok_count}/{len(missing)} downloaded & extracted")
            else:
                fail_count += 1
                failed_zips.append(name)
                print(f"  FAILED: {name}: {err}")

    print(f"\nDone: {ok_count} ok, {fail_count} failed")
    if failed_zips:
        print("Failed zips:")
        for z in sorted(failed_zips):
            print(f"  {z}")


if __name__ == "__main__":
    main()

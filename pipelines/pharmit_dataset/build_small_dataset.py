import argparse
from pathlib import Path
import re
import shutil


def parse_args():
    p = argparse.ArgumentParser(description="This script can be run on jabba (where the raw pharmit dataset lives) to build a small dataset for dev purporses.")
    p.add_argument('--src_dir', type=Path, default='/')
    p.add_argument('--dst_dir', type=Path, default='./pharmit_small/')
    p.add_argument('--n_data_dirs', type=int, default=3)
    p.add_argument('--n_conformer_dirs', type=int, default=3)
    p.add_argument('--n_conformer_files', type=int, default=3)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Add your main code logic here

    pattern = re.compile(r'data\d{2}')
    data_dirs = [d for d in args.src_dir.iterdir() if d.is_dir() and pattern.match(d.name)]
    data_dirs = data_dirs[:args.n_data_dirs]

    files_to_copy = []

    for data_dir in data_dirs:

        conformer_parent_dir = data_dir / "conformers"
        conformer_dir_iterator = conformer_parent_dir.iterdir()
        for _ in range(args.n_conformer_dirs):
            conformer_dir = next(conformer_dir_iterator)
            conformer_iterator = conformer_dir.iterdir()
            conformer_files = [ next(conformer_iterator) for _ in range(args.n_conformer_files) ]
            files_to_copy.append((data_dir.name, conformer_dir.name, conformer_files))

    
    dst_dir = args.dst_dir
    dst_dir.mkdir(exist_ok=True)

    for data_dir_name, conformer_dir_name, conformer_files in files_to_copy:
        data_dir = dst_dir / data_dir_name
        data_dir.mkdir(exist_ok=True)

        conformer_dir = data_dir / 'conformers' / conformer_dir_name
        conformer_dir.mkdir(exist_ok=True, parents=True)

        for conformer_file in conformer_files:
            shutil.copy(conformer_file, conformer_dir)
            # print(f"Copying {conformer_file} to {conformer_dir}")

    
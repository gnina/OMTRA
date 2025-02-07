import argparse
import logging
from pathlib import Path
from omtra_pipelines.plinder_dataset.plinder_zarr import *


def setup_logger(log_output):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_output)],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process all of plinder and store in zarr"
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to systems directory"
    )
    parser.add_argument("--output", type=str, required=True, help="Path to zarr store")
    parser.add_argument(
        "--pocket_cutoff",
        type=int,
        required=False,
        default=8,
        help="Angstrom cutoff for pocket extraction",
    )
    parser.add_argument(
        "--num_systems", type=int, required=False, help="Number of systems to process, "
    )

    return parser.parse_args()


def main():
    args = parse_args()
    log_path = str(Path(args.output).parent / "plinder_storage.log")
    setup_logger(log_path)
    processor = SystemProcessor(
        atom_map=["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"],
        pocket_cutoff=args.pocket_cutoff,
        raw_data=Path(args.data).parent,
    )
    converter = PlinderZarrConverter(
        output_path=args.output, system_processor=processor
    )

    systems = Path(args.data)
    count = 0
    for system in systems.iterdir():
        if system.is_dir():
            if args.num_dirs and count == args.num_dirs:
                break
            try:
                system_id = str(system.name)
                converter.process_system(system_id)
                count += 1
            except Exception as e:
                logging.exception("Unexpected error: %s", e)


if __name__ == "__main__":
    main()

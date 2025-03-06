import argparse
import logging
from pathlib import Path
from omtra_pipelines.plinder_dataset.plinder_zarr import PlinderZarrConverter
from omtra_pipelines.plinder_dataset.plinder_pipeline import SystemProcessor

LIGAND_MAP = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
NPNDE_MAP = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "I",
    "Si",
    "Pd",
    "Ca",
    "Sc",
    "Al",
    "Pt",
    "Cr",
    "Rh",
    "Mo",
    "Y",
    "Cu",
    "Sm",
    "Bi",
    "Ir",
    "Be",
    "Pb",
    "Kr",
    "As",
    "Sn",
    "Lu",
    "Co",
    "Ni",
    "K",
    "Am",
    "Rb",
    "Gd",
    "Ce",
    "Ba",
    "Au",
    "Na",
    "Th",
    "Pr",
    "Yb",
    "Ti",
    "Mg",
    "Sr",
    "Cs",
    "Tl",
    "B",
    "Xe",
    "Cm",
    "Li",
    "Zr",
    "Cf",
    "Ag",
    "Cd",
    "Mn",
    "Eu",
    "In",
    "Hg",
    "W",
    "Tb",
    "La",
    "Sb",
    "Te",
    "Se",
    "Zn",
    "Os",
    "Ga",
    "Fe",
    "Nd",
    "Pu",
    "V",
    "U",
    "Ru",
    "Re",
]


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
    parser.add_argument(
        "--num_cpus", type=int, required=False, default=1, help="Number of cpus"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    log_path = str(Path(args.output).parent / "plinder_storage.log")
    setup_logger(log_path)
    processor = SystemProcessor(
        ligand_atom_map=LIGAND_MAP,
        npnde_atom_map=NPNDE_MAP,
        pocket_cutoff=args.pocket_cutoff,
        raw_data=Path(args.data).parent,
    )
    converter = PlinderZarrConverter(
        output_path=args.output, system_processor=processor, num_workers=args.num_cpus
    )

    systems_dir = Path(args.data)
    system_ids = [d.name for d in systems_dir.iterdir() if d.is_dir()]

    if args.num_systems:
        system_ids = system_ids[: args.num_systems]

    converter.process_dataset(system_ids)


if __name__ == "__main__":
    main()

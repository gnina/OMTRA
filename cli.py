import argparse
import sys
from pathlib import Path
import omtra.tasks
from omtra.utils import omtra_root
from omtra.tasks.register import TASK_REGISTER

def create_parser():
    """Create the argument parser for sampling."""
    parser = argparse.ArgumentParser(
        prog='omtra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    available_tasks = sorted([name for name in TASK_REGISTER.keys() if "_condensed" in name])
    
    # sampling args
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the model checkpoint (inferred from --task if not provided)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=available_tasks,
        help=f"Task to sample for.",
        required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pharmit",
        help="Dataset to sample from (e.g. pharmit)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help=(
            "Number of samples to draw. "
            "When using input files (--protein_file, etc.), this is the number of samples generated from that single input. "
            "When using datasets, this is the number of systems to sample from the dataset."
        )
    )
    parser.add_argument(
        "--n_replicates",
        type=int,
        default=1,
        help=(
            "Number of replicates per system. "
            "When using input files (--protein_file, etc.), this is ignored (set --n_samples instead). "
            "When using datasets, this is the number of replicates per sampled system from the dataset."
        )
    )
    parser.add_argument(
        "--dataset_start_idx",
        type=int,
        default=0,
        help="Index in the dataset to start sampling from"
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=250,
        help="Number of integration steps to take when sampling"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, visualize the sampling process"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to write outputs to"
    )
    parser.add_argument(
        "--pharmit_path",
        type=str,
        default=None,
        help="Path to the Pharmit dataset (required for conditional tasks without input files)"
    )
    parser.add_argument(
        "--plinder_path",
        type=str,
        default=None,
        help="Path to the Plinder dataset (required for conditional tasks without input files)"
    )
    parser.add_argument('--split', type=str, default='val', help='Which data split to use')
    
    parser.add_argument(
        "--stochastic_sampling",
        action="store_true",
        help="If set, perform stochastic sampling."
    )
    parser.add_argument(
        "--noise_scaler",
        type=float,
        default=1.0,
        help="Scaling factor for noise (stochasticity)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Scaling factor for noise (stochasticity)"
    )
    parser.add_argument("--use_gt_n_lig_atoms", action="store_true", help="When enabled, use the number of ground truth ligand atoms for de novo design.")
    parser.add_argument(
        '--n_lig_atom_margin',
        type=float,
        default=0.15,
        help='number of atoms in the ligand will be +/- this margin from number of atoms in the ground truth ligand, only if --use_gt_n_lig_atoms is set (default: 0.15, i.e. +/- 15 percent)'
    )
    parser.add_argument(
        '--n_lig_atoms_mean',
        type=float,
        default=None,
        help='Mean number of atoms for ligand samples (if provided with --n_lig_atoms_std, uses normal distribution instead of dataset distribution)'
    )
    parser.add_argument(
        '--n_lig_atoms_std',
        type=float,
        default=None,
        help='Standard deviation for number of atoms (required if --n_lig_atoms_mean is provided)'
    )
    parser.add_argument("--metrics", action="store_true", help="If set, compute metrics for the samples")

    parser.add_argument(
        "--protein_file",
        type=Path,
        default=None,
        help="Path to protein structure file (PDB or CIF) for protein-conditioned tasks"
    )
    parser.add_argument(
        "--ligand_file", 
        type=Path,
        default=None,
        help="Path to ligand structure file (SDF) for ligand-conditioned tasks"
    )
    parser.add_argument(
        "--pharmacophore_file",
        type=Path, 
        default=None,
        help="Path to pharmacophore file (XYZ format) for pharmacophore-conditioned tasks"
    )
    return parser


def _check_available_files(args):
    """Check what input files are available."""
    has_protein = args.protein_file is not None
    has_ligand = args.ligand_file is not None
    has_pharmacophore = args.pharmacophore_file is not None
    
    return has_protein, has_ligand, has_pharmacophore


def _validate_task_inputs(args, task, has_protein, has_ligand, has_pharmacophore):
    """Validate that required inputs for the task are provided."""
    required = set(task.groups_fixed)
    missing = []
    
    # Map groups_fixed to file types
    if 'protein_identity' in required and not has_protein:
        missing.append("protein file (--protein_file)")
    if 'ligand_identity' in required and not has_ligand:
        missing.append("ligand file (--ligand_file)")
    if 'ligand_identity_condensed' in required and not has_ligand:
        missing.append("ligand file (--ligand_file)")
    if 'pharmacophore' in required and not has_pharmacophore:
        missing.append("pharmacophore file (--pharmacophore_file)")
    
    has_dataset_path = args.pharmit_path is not None or args.plinder_path is not None
    
    if missing:
        if has_dataset_path:
            # Warn but continue using dataset
            print(f"Warning: Task '{args.task}' requires the following inputs that were not provided:")
            for item in missing:
                print(f"  - {item}")
            print("Using dataset path to sample from instead.")
        else:
            print(f"Error: Task '{args.task}' requires the following inputs that were not provided:")
            for item in missing:
                print(f"  - {item}")
            print("\nEither provide the required input files or specify a dataset path (--pharmit_path or --plinder_path).")
            sys.exit(1)
    
    if 'protein_identity' in task.groups_fixed and not has_ligand:
        print("Warning: Protein-conditioned task detected but no reference ligand provided.")
        print("The system will use the protein center of mass instead of the reference ligand center of mass.")
        print("Consider providing a reference ligand file (--ligand_file) for better results.")


def run_sample(args):
    from routines.sample import main as sample_main
    import torch
    from omtra.tasks.register import task_name_to_class
    from omtra.utils.checkpoints import get_checkpoint_path_for_task, TASK_TO_CHECKPOINT
    
    task = task_name_to_class(args.task)
    
    if args.checkpoint is None:
        checkpoint_dir = Path("./checkpoints")
        checkpoint_dir = Path(omtra_root()) / "omtra/trained_models/"
        checkpoint_path = get_checkpoint_path_for_task(
            args.task,
            checkpoint_dir=checkpoint_dir
        )
        if checkpoint_path is None:
            expected_ckpt = TASK_TO_CHECKPOINT.get(args.task, "unknown")
            print(f"Error: No checkpoint found for task '{args.task}'")
            print(f"expected checkpoint: {expected_ckpt} at {checkpoint_dir.absolute()}")
            sys.exit(1)
        args.checkpoint = checkpoint_path
    
    has_protein, has_ligand, has_pharmacophore = _check_available_files(args)
    
    # Validate inputs
    if not task.unconditional:
        _validate_task_inputs(args, task, has_protein, has_ligand, has_pharmacophore)
    
    # create graphs from files
    if args.protein_file or args.ligand_file or args.pharmacophore_file:
        from omtra.utils.file_to_graph import create_conditional_graphs_from_files
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        g_list = create_conditional_graphs_from_files(
            protein_file=args.protein_file,
            ligand_file=args.ligand_file, 
            pharmacophore_file=args.pharmacophore_file,
            task=task,
            n_samples=1,  # 1 graph from the input file
            device=device
        )
        
        # When using input files: 1 system, n_samples is the number of replicates
        args.n_replicates = args.n_samples
        args.n_samples = 1
        
        args.g_list_from_files = g_list
    
    if hasattr(args, 'checkpoint') and args.checkpoint:
        args.checkpoint = Path(args.checkpoint)
    
    sample_main(args)


def main():
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        run_sample(args)
    except Exception as e:
        import os, traceback
        if os.environ.get("OMTRA_DEBUG") == "1":
            traceback.print_exc()
            raise
        else:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()

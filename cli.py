import argparse
import sys
from pathlib import Path

def create_parser():
    """Create the argument parser for sampling."""
    parser = argparse.ArgumentParser(
        prog='omtra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # sampling args
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the model checkpoint (required)"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task to sample for (e.g. denovo_ligand)",
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
        help="Number of samples to draw"
    )
    parser.add_argument(
        "--n_replicates",
        type=int,
        default=1,
        help=(
            "For conditional sampling: number of replicates per input sample"
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
        default=15,
        help='number of atoms in the ligand will be +/- this margin from number of atoms in the ground truth ligand, only if --use_gt_n_lig_atoms is set (default: 0.15, i.e. +/- 15 percent)'
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
        help="Path to pharmacophore file (XYZ or json) for pharmacophore-conditioned tasks"
    )
    parser.add_argument(
        "--input_files_dir",
        type=Path,
        default=None,
        help="Directory containing input files (any .pdb/.cif for protein, .sdf for ligand, .xyz for pharmacophore)"
    )
    return parser


def run_sample(args):
    from routines.sample import main as sample_main
    import torch
    
    if args.protein_file or args.ligand_file or args.pharmacophore_file or args.input_files_dir:
        from omtra.tasks.register import task_name_to_class
        from omtra.utils.file_to_graph import create_conditional_graphs_from_files
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        task = task_name_to_class(args.task)
        
        g_list = create_conditional_graphs_from_files(
            protein_file=args.protein_file,
            ligand_file=args.ligand_file, 
            pharmacophore_file=args.pharmacophore_file,
            input_files_dir=args.input_files_dir,
            task=task,
            n_samples=1,  # 1 graph per input file
            device=device
        )
        
        args.n_replicates = args.n_samples
        args.n_samples = 1
        
        args.g_list_from_files = g_list
    else:
        # check if task requires conditioning
        from omtra.tasks.register import task_name_to_class
        task = task_name_to_class(args.task)
        
        if not task.unconditional and not (args.protein_file or args.ligand_file or args.pharmacophore_file or args.input_files_dir):
            if not args.pharmit_path and not args.plinder_path:
                print(f"Error: Task '{args.task}' is conditional but no input files provided and no dataset path specified.")
                print("Either provide input files (--protein_file, --ligand_file, --pharmacophore_file) or dataset path (--pharmit_path, --plinder_path).")
                sys.exit(1)
            else:
                print(f"Warning: Task '{args.task}' is conditional task but no input files provided. Using dataset path to sample from.")
    
    if hasattr(args, 'checkpoint') and args.checkpoint:
        from pathlib import Path
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

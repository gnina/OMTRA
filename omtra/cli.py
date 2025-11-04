import argparse
import sys
from pathlib import Path

def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='omtra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True
    )
    
    train_parser = subparsers.add_parser(
        'train',
        help='Train an OMTRA model',
        description='Train an OMTRA model using Hydra configuration'
    )
    train_parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file (default: configs/config.yaml)'
    )
    train_parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint to resume training from'
    )

    train_parser.add_argument(
        'hydra_args',
        nargs='*',
        help='Additional arguments passed to Hydra'
    )
    
    sample_parser = subparsers.add_parser(
        'sample',
        help='Sample from a trained OMTRA model',
        description='Generate samples using a trained OMTRA model'
    )
    
    # sampling args
    sample_parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to the model checkpoint (required)"
    )
    sample_parser.add_argument(
        "--task",
        type=str,
        help="Task to sample for (e.g. denovo_ligand)",
        required=True
    )
    sample_parser.add_argument(
        "--dataset",
        type=str,
        default="pharmit",
        help="Dataset to sample from (e.g. pharmit)"
    )
    sample_parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to draw"
    )
    sample_parser.add_argument(
        "--n_replicates",
        type=int,
        default=1,
        help=(
            "For conditional sampling: number of replicates per input sample"
        )
    )
    sample_parser.add_argument(
        "--dataset_start_idx",
        type=int,
        default=0,
        help="Index in the dataset to start sampling from"
    )
    sample_parser.add_argument(
        "--n_timesteps",
        type=int,
        default=250,
        help="Number of integration steps to take when sampling"
    )
    sample_parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, visualize the sampling process"
    )
    sample_parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to write outputs to"
    )
    sample_parser.add_argument(
        "--pharmit_path",
        type=str,
        default=None,
        help="Path to the Pharmit dataset (required for conditional tasks without input files)"
    )
    sample_parser.add_argument(
        "--plinder_path",
        type=str,
        default=None,
        help="Path to the Plinder dataset (required for conditional tasks without input files)"
    )
    sample_parser.add_argument('--split', type=str, default='val', help='Which data split to use')
    
    sample_parser.add_argument(
        "--stochastic_sampling",
        action="store_true",
        help="If set, perform stochastic sampling."
    )
    sample_parser.add_argument(
        "--noise_scaler",
        type=float,
        default=1.0,
        help="Scaling factor for noise (stochasticity)"
    )
    sample_parser.add_argument(
        "--eps",
        type=float,
        default=0.01,
        help="Scaling factor for noise (stochasticity)"
    )
    sample_parser.add_argument("--use_gt_n_lig_atoms", action="store_true", help="When enabled, use the number of ground truth ligand atoms for de novo design.")
    sample_parser.add_argument(
        '--n_lig_atom_margin',
        type=float,
        default=15,
        help='number of atoms in the ligand will be +/- this margin from number of atoms in the ground truth ligand, only if --use_gt_n_lig_atoms is set (default: 0.15, i.e. +/- 15 percent)'
    )
    sample_parser.add_argument("--metrics", action="store_true", help="If set, compute metrics for the samples")

    sample_parser.add_argument(
        "--protein_file",
        type=Path,
        default=None,
        help="Path to protein structure file (PDB or CIF) for protein-conditioned tasks"
    )
    sample_parser.add_argument(
        "--ligand_file", 
        type=Path,
        default=None,
        help="Path to ligand structure file (SDF) for ligand-conditioned tasks"
    )
    sample_parser.add_argument(
        "--pharmacophore_file",
        type=Path, 
        default=None,
        help="Path to pharmacophore file (XYZ or json) for pharmacophore-conditioned tasks"
    )
    sample_parser.add_argument(
        "--input_files_dir",
        type=Path,
        default=None,
        help="Directory containing input files (protein.pdb, ligand.sdf, pharmacophore.xyz)"
    )

    sample_parser.add_argument(
        "--anchor1",
        type=str,
        default=None,
        help="First protein anchor in format RESID:ATOMNAME (e.g., '20:CB')"
    )
    sample_parser.add_argument(
        "--anchor2",
        type=str,
        default=None,
        help="Second protein anchor in format RESID:ATOMNAME (e.g., '27:CB')"
    )

    return parser


def run_train(args):
    """Run the training command."""
    from routines.train import main as train_main
    from omegaconf import DictConfig, OmegaConf
    from hydra.core.hydra_config import HydraConfig
    from hydra import initialize, compose
    from pathlib import Path
    import sys
    
    # Set up Hydra configuration
    config_path = Path(args.config).parent
    config_name = Path(args.config).stem
    
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        run_dir = ckpt_path.parent.parent
        original_cfg_path = run_dir / ".hydra/config.yaml"
        original_cfg = OmegaConf.load(original_cfg_path)
        
        overrides = args.hydra_args if args.hydra_args else []
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(original_cfg, cli_cfg)
        cfg.og_run_dir = str(run_dir)
    else:
        with initialize(config_path=str(config_path)):
            cfg = compose(config_name=config_name, overrides=args.hydra_args)
    
    train_main(cfg)


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
        if args.command == 'train':
            run_train(args)
        elif args.command == 'sample':
            run_sample(args)
        else:
            parser.error(f"Unknown command: {args.command}")
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

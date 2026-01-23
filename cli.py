import argparse
import sys
from pathlib import Path
import omtra.tasks
from omtra.utils import omtra_root
from omtra.tasks.register import TASK_REGISTER
from omtra.utils.checkpoints import TASK_TO_CHECKPOINT

def create_parser():
    """Create the argument parser for sampling."""
    parser = argparse.ArgumentParser(
        prog='omtra',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    available_tasks = sorted(TASK_TO_CHECKPOINT.keys())
    
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
    pocket_group = parser.add_mutually_exclusive_group()
    pocket_group.add_argument(
        "--pocket_ligand",
        type=Path,
        default=None,
        help="Path to reference ligand file (SDF) to define pocket around ligand atoms"
    )
    pocket_group.add_argument(
        "--pocket_center",
        type=str,
        default=None,
        help="Pocket center coordinates as 'x,y,z' (also set --bbox_length, default 23.0)"
    )
    pocket_group.add_argument(
        "--pocket_residues",
        type=str,
        default=None,
        help="Pocket residues as 'CHAIN:RESID,CHAIN:START-END' (e.g., 'A:123-125,B:200')"
    )
    parser.add_argument(
        "--pharmacophore_file",
        type=Path, 
        default=None,
        help="Path to pharmacophore file (JSON from Pharmit, XYZ, or ligand SDF) for pharmacophore-conditioned tasks."
    )
    parser.add_argument(
        "--bbox_length",
        type=float,
        default=23.0,
        help="Bounding box length (Angstroms) when using --pocket_center. Default: 23.0"
    )
    return parser


def _build_pocket_definition(pocket_ligand, pocket_center, pocket_residues, bbox_length):
    """Build pocket_definition dict from separate CLI arguments."""
    # Argparse enforces mutual exclusivity, so at most one will be provided
    if pocket_ligand is not None:
        return {'type': 'file', 'value': Path(pocket_ligand)}
    
    if pocket_center is not None:
        parts = [x.strip() for x in pocket_center.split(',')]
        if len(parts) != 3:
            raise ValueError(f"--pocket_center expects exactly 3 values (x,y,z), got {len(parts)}")
        try:
            center_coords = [float(p) for p in parts]
            pocket_def = {'type': 'center', 'value': center_coords}
            if bbox_length is not None:
                pocket_def['bbox_length'] = bbox_length
            return pocket_def
        except ValueError:
            raise ValueError(f"--pocket_center coordinates must be numbers")
    
    if pocket_residues is not None:
        if not pocket_residues:
            raise ValueError("--pocket_residues requires residue specifications")
        
        residue_specs = []
        for spec in pocket_residues.split(','):
            spec = spec.strip()
            if ':' not in spec:
                raise ValueError(f"Expected 'CHAIN:RESID', got '{spec}'")
            chain, res_part = spec.split(':', 1)
            try:
                if '-' in res_part:
                    start, end = map(int, res_part.split('-'))
                    residue_specs.extend((chain, r) for r in range(start, end + 1))
                else:
                    residue_specs.append((chain, int(res_part)))
            except ValueError:
                raise ValueError(f"Invalid residue: '{spec}'")
        
        if not residue_specs:
            raise ValueError("No valid residues specified")
        return {'type': 'residues', 'value': residue_specs}
    
    return None


def _validate_task_inputs(args, task):
    has_protein = args.protein_file is not None
    has_ligand = args.ligand_file is not None
    has_pharmacophore = args.pharmacophore_file is not None
    has_dataset_path = args.pharmit_path is not None or args.plinder_path is not None
    
    pocket_definition = None
    if args.pocket_ligand is not None or args.pocket_center is not None or args.pocket_residues is not None:
        try:
            pocket_definition = _build_pocket_definition(
                args.pocket_ligand,
                args.pocket_center,
                args.pocket_residues,
                args.bbox_length
            )
        except ValueError as e:
            print(f"Error parsing pocket arguments: {e}")
            sys.exit(1)
    
    if task.unconditional:
        return has_protein, has_ligand, has_pharmacophore, False, pocket_definition
    required = set(task.groups_fixed)
    
    has_pocket_definition = pocket_definition is not None
    has_pocket_ligand_file = has_pocket_definition and pocket_definition.get('type') == 'file'
    
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
    if 'protein_identity' in required and not has_dataset_path and not has_pocket_definition:
        missing.append(
            "pocket definition (--pocket_ligand, --pocket_center, or --pocket_residues)"
        )
    
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
            print("Either provide the required input files or specify a dataset path (--pharmit_path or --plinder_path).")
            sys.exit(1)
    
    return has_protein, has_ligand, has_pharmacophore, has_pocket_ligand_file, pocket_definition


def run_sample(args):
    from omtra.tasks.register import task_name_to_class
    from omtra.utils.checkpoints import get_checkpoint_path_for_task, TASK_TO_CHECKPOINT
    
    task = task_name_to_class(args.task)
    
    if args.checkpoint is None:
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
    
    has_protein, has_ligand, has_pharmacophore, has_pocket_ligand_file, pocket_definition = _validate_task_inputs(args, task)
    
    # validate --use_gt_n_lig_atoms: requires pocket ligand 
    if args.use_gt_n_lig_atoms:
        if not has_pocket_ligand_file:
            print("Error: --use_gt_n_lig_atoms requires a pocket ligand, but no pocket ligand was passed.")
            sys.exit(1)
    
    # create graphs from files
    has_pocket_arg = args.pocket_ligand is not None or args.pocket_center is not None or args.pocket_residues is not None
    if args.protein_file or args.ligand_file or has_pocket_arg or args.pharmacophore_file:
        import torch
        from omtra.utils.file_to_graph import create_conditional_graphs_from_files
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        if pocket_definition is not None and pocket_definition.get('type') == 'center':
            if 'bbox_length' not in pocket_definition:
                pocket_definition['bbox_length'] = args.bbox_length
            args.pocket_center = pocket_definition['value']
        else:
            args.pocket_center = None
        
        g_list = create_conditional_graphs_from_files(
            protein_file=args.protein_file,
            ligand_file=args.ligand_file,
            pocket_definition=pocket_definition,
            pharmacophore_file=args.pharmacophore_file,
            task=task,
            n_samples=1,  # 1 graph from the input file
            device=device,
        )
        
        if pocket_definition and not args.pocket_center:
            import numpy as np
            pocket_type = pocket_definition.get('type')
            if pocket_type == 'file':
                from omtra.utils.file_to_graph import load_ligand_rdkit
                coords = load_ligand_rdkit(pocket_definition['value'], compute_condensed=False).coords
                if isinstance(coords, torch.Tensor):
                    coords = coords.cpu().numpy()
                args.pocket_center = np.mean(coords, axis=0).tolist()
            elif pocket_type == 'residues' and g_list:
                coords = g_list[0].nodes['prot_atom'].data.get('x_1_true')
                if coords is not None:
                    args.pocket_center = coords.mean(dim=0).cpu().numpy().tolist()
        
        # When using input files: 1 system, n_samples is the number of replicates
        args.n_replicates = args.n_samples
        args.n_samples = 1
        
        args.g_list_from_files = g_list
    
    from routines.sample import main as sample_main
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

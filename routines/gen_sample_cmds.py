#!/usr/bin/env python3

import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from omtra.utils import omtra_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sample.py commands from a YAML configuration file"
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--default_n_samples",
        type=int,
        default=10,
        help="Default number of samples if not specified in config"
    )
    parser.add_argument(
        "--default_n_replicates", 
        type=int,
        default=1,
        help="Default number of replicates if not specified in config"
    )
    parser.add_argument(
        "--default_dataset",
        type=str,
        default="pharmit",
        help="Default dataset if not specified in config"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=None,
        help="File to write commands to (default: print to stdout)"
    )
    parser.add_argument(
        "--sample_script_path",
        type=Path,
        default=Path(__file__).parent / "sample.py",
        help="Path to the sample.py script"
    )
    parser.add_argument(
        '--plinder_path',
        type=Path,
        default = Path(omtra_root()) / 'data' / 'plinder',
        help='Path to plinder dataset'
    )
    parser.add_argument(
        '--pharmit_path',
        type=Path,
        default = Path(omtra_root()) / 'data' / 'pharmit',
        help='Path to pharmit dataset'
    )
    parser.add_argument(
        '--crossdocked_path',
        type=Path,
        default=Path(omtra_root()) / 'data' / 'crossdocked' / 'external_split',
        help='Path to crossdocked dataset'
    )
    parser.add_argument('--dataset_start_idx', type=int, default=0, help='Index to start sampling from')
    
    return parser.parse_args()


def find_best_checkpoint(model_dir: Path) -> Path:
    """
    Find the best checkpoint in a model directory.
    Prefers 'last.ckpt' if it exists, otherwise the batch_[integer].ckpt with highest integer.
    
    Args:
        model_dir: Path to model directory containing checkpoints/ subdirectory
        
    Returns:
        Path to the best checkpoint
        
    Raises:
        FileNotFoundError: If no valid checkpoints are found
    """
    checkpoints_dir = model_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found in {model_dir}")
    
    # Check for last.ckpt first
    last_ckpt = checkpoints_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt
    
    # Find batch_[integer].ckpt files
    batch_ckpts = []
    for ckpt_file in checkpoints_dir.glob("batch_*.ckpt"):
        try:
            # Extract integer from batch_[integer].ckpt
            stem = ckpt_file.stem  # removes .ckpt
            if stem.startswith("batch_"):
                batch_num = int(stem.split("_")[1])
                batch_ckpts.append((batch_num, ckpt_file))
        except (ValueError, IndexError):
            continue
    
    if not batch_ckpts:
        raise FileNotFoundError(f"No valid checkpoints found in {checkpoints_dir}")
    
    # Return the checkpoint with the highest batch number
    batch_ckpts.sort(key=lambda x: x[0])
    return batch_ckpts[-1][1]


def load_config(config_file: Path) -> List[Dict[str, Any]]:
    """Load and validate the YAML configuration file."""
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, list):
        raise ValueError("Configuration must be a list of dictionaries")
    
    for i, item in enumerate(config):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} in configuration is not a dictionary")
        if 'model_dir' not in item:
            raise ValueError(f"Item {i} missing required 'model_dir' key")
        if 'tasks' not in item:
            raise ValueError(f"Item {i} missing required 'tasks' key")
        if not isinstance(item['tasks'], list):
            raise ValueError(f"Item {i} 'tasks' must be a list")
    
    return config


def generate_commands(
    config: List[Dict[str, Any]], 
    args: argparse.Namespace
) -> List[str]:
    """Generate sample.py commands from configuration."""
    commands = []
    
    for item in config:
        model_dir = Path(item['model_dir'])
        tasks = item['tasks']
        
        # Get optional parameters with defaults
        n_samples = item.get('n_samples', args.default_n_samples)
        n_replicates = item.get('n_replicates', args.default_n_replicates)
        dataset = item.get('dataset', args.default_dataset)
        dataset_start_idx = item.get('dataset_start_idx', args.dataset_start_idx)
        
        try:
            checkpoint_path = find_best_checkpoint(model_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        
        # Generate one command per task for this model
        for task in tasks:
            cmd_parts = [
                "python", str(args.sample_script_path),
                str(checkpoint_path),
                "--task", task,
                "--dataset", dataset,
                "--n_samples", str(n_samples),
                "--n_replicates", str(n_replicates),
                "--pharmit_path", str(args.pharmit_path),
                "--plinder_path", str(args.plinder_path),
                "--crossdocked_path", str(args.crossdocked_path),
                "--dataset_start_idx", str(dataset_start_idx)
            ]
            
            commands.append(" ".join(cmd_parts))
    
    return commands


def main():
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config_file)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Generate commands
    commands = generate_commands(config, args)
    
    if not commands:
        print("No valid commands generated")
        return 1
    
    # Output commands
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for cmd in commands:
                f.write(cmd + '\n')
        print(f"Generated {len(commands)} commands and wrote to {args.output_file}")
    else:
        for cmd in commands:
            print(cmd)
    
    return 0


if __name__ == "__main__":
    exit(main())
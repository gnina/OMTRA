#!/usr/bin/env python3
"""
Script to generate chunked CSV files and corresponding docking_eval.py commands.

This script takes a CSV file with system indices (like plinder_eval_sys_idxs.csv),
chunks it into smaller files, and generates commands to run docking_eval.py on each chunk
with multiple replicates.
"""

import argparse
import csv
from pathlib import Path
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate chunked CSV files and docking_eval.py commands'
    )
    
    parser.add_argument(
        '--input_csv', 
        type=Path, 
        required=True,
        help='Path to input CSV file containing system indices (e.g., plinder_eval_sys_idxs.csv)'
    )
    
    parser.add_argument(
        '--chunk_size', 
        type=int, 
        required=True,
        help='Number of system indices per chunk'
    )
    
    parser.add_argument(
        '--n_replicates', 
        type=int, 
        required=True,
        help='Number of replicate commands per chunk'
    )
    
    parser.add_argument(
        '--ckpt_path', 
        type=Path, 
        required=True,
        help='Path to model checkpoint for docking_eval.py'
    )
    parser.add_argument(
        '--reps_per_cmd', 
        type=int, 
        default=1,
        help='Number of replicates per command (default: 1)'
    )
    
    parser.add_argument(
        '--task', 
        type=str, 
        default='fixed_protein_pharmacophore_ligand_denovo_condensed',
        help='Task name for docking_eval.py (default: fixed_protein_pharmacophore_ligand_denovo_condensed)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=Path, 
        default=None,
        help='Output directory for chunked CSV files and commands (default: same as input CSV directory)'
    )
    
    parser.add_argument(
        '--commands_file', 
        type=str, 
        default='eval_commands.txt',
        help='Name of output file containing commands (default: eval_commands.txt)'
    )
    
    parser.add_argument(
        '--additional_args', 
        type=str, 
        default='--disable_gnina --disable_rmsd --disable_strain',
        help='Additional arguments to pass to docking_eval.py (default: --disable_gnina --disable_pb_valid --disable_rmsd)'
    )
    
    return parser.parse_args()


def read_system_indices(csv_file: Path):
    """Read system indices from CSV file."""
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        # Read the first (and only) line
        line = next(reader)
        # Convert to integers
        indices = [int(idx.strip()) for idx in line]
    return indices


def chunk_indices(indices, chunk_size):
    """Split indices into chunks of specified size."""
    chunks = []
    for i in range(0, len(indices), chunk_size):
        chunk = indices[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


def write_chunk_csv(chunk, output_file: Path):
    """Write a chunk of indices to a CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(chunk)


def generate_commands(chunks, chunk_files, args):
    """Generate docking_eval.py commands for each chunk and replicate."""
    commands = []

    docking_eval_output_dir = args.output_dir / f'samples_{args.task}'
    docking_eval_output_dir = docking_eval_output_dir.resolve()
    docking_eval_output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_idx, chunk_file in enumerate(chunk_files):
        for replicate in range(args.n_replicates):

            # determine output file
            cmd_output_dir = docking_eval_output_dir / f'chunk_{chunk_idx}_rep_{replicate}'

            # Build the command
            cmd_parts = [
                'python',
                str(Path(__file__).parent / 'docking_eval.py'),
                f'--ckpt_path={args.ckpt_path}',
                f'--task={args.task}',
                f'--sys_idx_file={chunk_file}',
                f'--n_replicates={args.reps_per_cmd}',  # Each command handles 1 replicate
                f'--n_samples={len(chunks[chunk_idx])}',  # Number of systems in this chunk
                f'--bs_per_gbmem=5',  # Example fixed argument; adjust as needed
                f'--output_dir={cmd_output_dir}',  # Output directory for this chunk and replicate
                f'--plinder_path=/net/galaxy/home/koes/icd3/moldiff/OMTRA/data/plinder',
                f'--split=test'
            ]
            
            # Add additional arguments
            if args.additional_args:
                cmd_parts.extend(args.additional_args.split())
            
            # Add replicate-specific output directory if multiple replicates
            # if args.n_replicates > 1:
            #     output_suffix = f'_chunk{chunk_idx}_rep{replicate}'
            #     cmd_parts.append(f'--output_dir=eval_output{output_suffix}')
            
            command = ' '.join(cmd_parts)
            commands.append(command)
    
    return commands


def main():
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(__file__).parent / 'sampleblaster_output'
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read system indices from input CSV
    print(f"Reading system indices from {args.input_csv}")
    indices = read_system_indices(args.input_csv)
    print(f"Found {len(indices)} system indices")
    
    # Chunk the indices
    chunks = chunk_indices(indices, args.chunk_size)
    n_chunks = len(chunks)
    print(f"Created {n_chunks} chunks of size {args.chunk_size}")
    
    # Write chunk CSV files
    chunk_files = []
    chunks_dir = args.output_dir / 'chunks'
    chunks_dir.mkdir(exist_ok=True)
    
    for i, chunk in enumerate(chunks):
        chunk_file = chunks_dir / f'chunk_{i:03d}.csv'
        write_chunk_csv(chunk, chunk_file)
        chunk_files.append(chunk_file)
        print(f"Wrote chunk {i} with {len(chunk)} indices to {chunk_file}")
    
    # Generate commands
    print(f"Generating commands with {args.n_replicates} replicates per chunk")
    commands = generate_commands(chunks, chunk_files, args)
    
    # Write commands to file
    commands_file = args.output_dir / args.commands_file
    with open(commands_file, 'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')
    
    print(f"Wrote {len(commands)} commands to {commands_file}")
    print(f"Total jobs: {n_chunks} chunks × {args.n_replicates} replicates = {len(commands)} commands")
    
    # Print summary
    print("\nSummary:")
    print(f"  Input CSV: {args.input_csv}")
    print(f"  Total system indices: {len(indices)}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Number of chunks: {n_chunks}")
    print(f"  Replicates per chunk: {args.n_replicates}")
    print(f"  Total commands: {len(commands)}")
    print(f"  Chunks directory: {chunks_dir}")
    print(f"  Commands file: {commands_file}")


if __name__ == '__main__':
    main()
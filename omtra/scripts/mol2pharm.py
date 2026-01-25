#!/usr/bin/env python3
"""
Convert ligand SDF files to pharmacophore JSON format.

This utility extracts pharmacophore features from ligand molecules and saves them
in the JSON format compatible with Pharmit and OMTRA.

Usage:
    omtra mol2pharm input.sdf -o output.json
    omtra mol2pharm input.sdf --output output.json --pretty
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rdkit import Chem

from omtra.data.pharmacophores import get_pharmacophores
from omtra.constants import ph_idx_to_type


def extract_pharmacophore_json(mol: Chem.Mol, all_enabled: bool = True) -> Dict[str, Any]:
    """
    Extract pharmacophore features from a molecule and return as JSON-compatible dict.
    
    Args:
        mol: RDKit molecule object
        all_enabled: If True, all features are enabled. If False, all disabled.
    
    Returns:
        Dictionary with pharmacophore data in Pharmit JSON format
    """
    # Get pharmacophore features
    P, X, V, I = get_pharmacophores(mol)
    
    if len(P) == 0:
        return {"points": []}
    
    # Convert to JSON format
    points = []
    for i in range(len(P)):
        point = {
            "name": ph_idx_to_type[X[i]],
            "x": float(P[i, 0]),
            "y": float(P[i, 1]),
            "z": float(P[i, 2]),
            "enabled": all_enabled
        }
        points.append(point)
    
    return {"points": points}


def main():
    parser = argparse.ArgumentParser(
        prog='omtra mol2pharm',
        description="Convert ligand SDF files to pharmacophore JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  omtra mol2pharm ligand.sdf -o pharmacophore.json
  
  # Process first molecule only
  omtra mol2pharm multi_ligand.sdf -o pharm.json --first-only
  
  # Disable all features by default
  omtra mol2pharm ligand.sdf -o pharm.json --all-disabled
  
  # Pretty-print JSON output
  omtra mol2pharm ligand.sdf -o pharm.json --pretty
        """
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input SDF file containing ligand structure(s)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output JSON file for pharmacophore features"
    )
    
    parser.add_argument(
        "--first-only",
        action="store_true",
        help="Process only the first molecule in multi-molecule SDF files"
    )
    
    parser.add_argument(
        "--all-disabled",
        action="store_true",
        help="Set all pharmacophore features to disabled (enabled=false)"
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output with indentation"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    
    if input_path.suffix.lower() not in ['.sdf', '.mol', '.mol2']:
        print(f"Warning: Input file '{args.input}' may not be an SDF file", file=sys.stderr)
    
    # Read molecule(s)
    try:
        supplier = Chem.SDMolSupplier(str(input_path))
    except Exception as e:
        print(f"Error: Failed to read SDF file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get first valid molecule
    mol = None
    mol_count = 0
    for m in supplier:
        mol_count += 1
        if m is not None:
            mol = m
            if args.verbose:
                print(f"Processing molecule {mol_count}: {mol.GetNumAtoms()} atoms")
            break
        elif args.verbose:
            print(f"Skipping invalid molecule {mol_count}")
        
        if args.first_only and mol is not None:
            break
    
    if mol is None:
        print("Error: No valid molecules found in SDF file", file=sys.stderr)
        sys.exit(1)
    
    # Check for 3D coordinates
    if not mol.GetNumConformers():
        print("Error: Molecule has no 3D conformer", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"Extracting pharmacophore features from molecule with {mol.GetNumAtoms()} atoms...")
    
    # Extract pharmacophore
    try:
        all_enabled = not args.all_disabled
        pharm_data = extract_pharmacophore_json(mol, all_enabled=all_enabled)
    except Exception as e:
        print(f"Error: Failed to extract pharmacophore: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Report results
    n_features = len(pharm_data["points"])
    if args.verbose:
        print(f"Extracted {n_features} pharmacophore features")
        if n_features > 0:
            feature_counts = {}
            for point in pharm_data["points"]:
                feature_type = point["name"]
                feature_counts[feature_type] = feature_counts.get(feature_type, 0) + 1
            print("Feature breakdown:")
            for feat_type, count in sorted(feature_counts.items()):
                print(f"  {feat_type}: {count}")
    
    if n_features == 0:
        print("Warning: No pharmacophore features extracted from molecule", file=sys.stderr)
    
    # Write output
    output_path = Path(args.output)
    try:
        with open(output_path, 'w') as f:
            if args.pretty:
                json.dump(pharm_data, f, indent=2)
            else:
                json.dump(pharm_data, f)
        
        if args.verbose or n_features > 0:
            print(f"Pharmacophore JSON written to: {args.output}")
    except Exception as e:
        print(f"Error: Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

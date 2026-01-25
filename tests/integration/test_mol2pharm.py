"""
Integration tests for mol2pharm utility.
"""

import json
import pytest
import subprocess
import sys
from pathlib import Path
from rdkit import Chem

from omtra.scripts.mol2pharm import extract_pharmacophore_json, main


def test_extract_pharmacophore_json():
    """Test extracting pharmacophore from a molecule."""
    # Load test ligand
    test_lig_path = Path(__file__).parent / "test_lig.sdf"
    supplier = Chem.SDMolSupplier(str(test_lig_path))
    mol = next(supplier)
    
    assert mol is not None, "Failed to load test ligand"
    assert mol.GetNumConformers() > 0, "Test ligand has no conformer"
    
    # Extract pharmacophore
    pharm_data = extract_pharmacophore_json(mol, all_enabled=True)
    
    # Validate structure
    assert "points" in pharm_data
    assert isinstance(pharm_data["points"], list)
    assert len(pharm_data["points"]) > 0, "No pharmacophore features extracted"
    
    # Validate each point
    for point in pharm_data["points"]:
        assert "name" in point
        assert "x" in point
        assert "y" in point
        assert "z" in point
        assert "enabled" in point
        assert isinstance(point["name"], str)
        assert isinstance(point["x"], float)
        assert isinstance(point["y"], float)
        assert isinstance(point["z"], float)
        assert isinstance(point["enabled"], bool)


def test_extract_pharmacophore_all_disabled():
    """Test extracting pharmacophore with all features disabled."""
    test_lig_path = Path(__file__).parent / "test_lig.sdf"
    supplier = Chem.SDMolSupplier(str(test_lig_path))
    mol = next(supplier)
    
    pharm_data = extract_pharmacophore_json(mol, all_enabled=False)
    
    # All features should be disabled
    for point in pharm_data["points"]:
        assert point["enabled"] is False


def test_cli_help(tmp_path):
    """Test CLI help message."""
    result = subprocess.run(
        [sys.executable, "-m", "omtra.scripts.mol2pharm", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Convert ligand SDF files to pharmacophore JSON format" in result.stdout


def test_cli_basic_usage(tmp_path):
    """Test basic CLI usage."""
    test_lig_path = Path(__file__).parent / "test_lig.sdf"
    output_path = tmp_path / "test_output.json"
    
    result = subprocess.run(
        [
            sys.executable, "-m", "omtra.scripts.mol2pharm",
            str(test_lig_path),
            "-o", str(output_path)
        ],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    assert output_path.exists(), "Output file was not created"
    
    # Validate JSON structure
    with open(output_path) as f:
        data = json.load(f)
    
    assert "points" in data
    assert len(data["points"]) > 0


def test_cli_pretty_output(tmp_path):
    """Test CLI with pretty-print option."""
    test_lig_path = Path(__file__).parent / "test_lig.sdf"
    output_path = tmp_path / "test_pretty.json"
    
    result = subprocess.run(
        [
            sys.executable, "-m", "omtra.scripts.mol2pharm",
            str(test_lig_path),
            "-o", str(output_path),
            "--pretty"
        ],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    
    # Check that output is formatted with indentation
    with open(output_path) as f:
        content = f.read()
    
    assert "\n  " in content, "Output is not pretty-printed"


def test_cli_verbose_output(tmp_path):
    """Test CLI with verbose option."""
    test_lig_path = Path(__file__).parent / "test_lig.sdf"
    output_path = tmp_path / "test_verbose.json"
    
    result = subprocess.run(
        [
            sys.executable, "-m", "omtra.scripts.mol2pharm",
            str(test_lig_path),
            "-o", str(output_path),
            "--verbose"
        ],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Extracting pharmacophore features" in result.stdout
    assert "Feature breakdown:" in result.stdout


def test_cli_all_disabled(tmp_path):
    """Test CLI with all-disabled option."""
    test_lig_path = Path(__file__).parent / "test_lig.sdf"
    output_path = tmp_path / "test_disabled.json"
    
    result = subprocess.run(
        [
            sys.executable, "-m", "omtra.scripts.mol2pharm",
            str(test_lig_path),
            "-o", str(output_path),
            "--all-disabled"
        ],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    
    with open(output_path) as f:
        data = json.load(f)
    
    # All features should be disabled
    for point in data["points"]:
        assert point["enabled"] is False


def test_cli_invalid_input():
    """Test CLI with non-existent input file."""
    result = subprocess.run(
        [
            sys.executable, "-m", "omtra.scripts.mol2pharm",
            "nonexistent_file.sdf",
            "-o", "/tmp/output.json"
        ],
        capture_output=True,
        text=True
    )
    
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_omtra_subcommand(tmp_path):
    """Test that mol2pharm works as an omtra subcommand."""
    test_lig_path = Path(__file__).parent / "test_lig.sdf"
    output_path = tmp_path / "test_subcommand.json"
    
    # Test using 'omtra mol2pharm' command
    result = subprocess.run(
        [
            "omtra", "mol2pharm",
            str(test_lig_path),
            "-o", str(output_path),
            "--pretty"
        ],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Subcommand failed with stderr: {result.stderr}"
    assert output_path.exists(), "Output file was not created"
    
    # Validate JSON structure
    with open(output_path) as f:
        data = json.load(f)
    
    assert "points" in data
    assert len(data["points"]) > 0

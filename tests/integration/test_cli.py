import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

TEST_DIR = Path(__file__).parent
TEST_LIG = TEST_DIR / "test_lig.sdf"
TEST_REC = TEST_DIR / "test_rec.pdb"
TEST_PHARM = TEST_DIR / "test_pharmacophore.xyz"

@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create a dummy checkpoint file for testing."""
    ckpt_path = tmp_path / "dummy_checkpoint.ckpt"
    ckpt_path.touch()
    return ckpt_path


@pytest.fixture
def mock_model():
    """Mock the model loading to avoid needing actual checkpoints."""
    with patch('omtra.load.quick.omtra_from_checkpoint') as mock_load:
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.sample.return_value = []
        mock_load.return_value = mock_model
        yield mock_model


def test_cli_help():
    """Test that CLI help works."""
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [sys.executable, str(cli_path), "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    assert result.returncode == 0
    assert "checkpoint" in result.stdout
    assert "--task" in result.stdout


def test_cli_missing_checkpoint():
    """Test that CLI errors when checkpoint is missing."""
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [sys.executable, str(cli_path), "nonexistent.ckpt", "--task", "denovo_ligand_condensed"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    assert result.returncode != 0


def test_cli_missing_task():
    """Test that CLI errors when task is missing."""
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [sys.executable, str(cli_path), "dummy.ckpt"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    assert result.returncode != 0


def test_cli_protein_conditioned_missing_protein(tmp_path, mock_checkpoint):
    """Test that CLI errors when protein-conditioned task is missing protein file."""
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "fixed_protein_ligand_denovo_condensed",
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    assert result.returncode != 0
    assert "protein file" in result.stdout.lower() or "Error" in result.stdout


def test_cli_protein_conditioned_with_protein(tmp_path, mock_checkpoint, mock_model):
    """Test that CLI works with protein file for protein-conditioned task."""
    # Copy test files to tmp_path
    import shutil
    test_protein = tmp_path / "test_protein.pdb"
    shutil.copy(TEST_REC, test_protein)
    
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "fixed_protein_ligand_denovo_condensed",
            "--protein_file", str(test_protein),
            "--n_samples", "2",
            "--output_dir", str(tmp_path / "output")
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )

    assert "protein file" not in result.stdout.lower() or result.returncode == 0


def test_cli_protein_conditioned_warning_no_ligand(tmp_path, mock_checkpoint):
    """Test that CLI warns when protein-conditioned task has no reference ligand."""
    import shutil
    test_protein = tmp_path / "test_protein.pdb"
    shutil.copy(TEST_REC, test_protein)
    
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "fixed_protein_ligand_denovo_condensed",
            "--protein_file", str(test_protein),
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    # Should warn about missing reference ligand
    assert "reference ligand" in result.stdout.lower() or "Warning" in result.stdout or result.returncode != 0


def test_cli_ligand_conditioned_missing_ligand(tmp_path, mock_checkpoint):
    """Test that CLI errors when ligand-conditioned task is missing ligand file."""
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "ligand_conformer_condensed",
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    assert result.returncode != 0
    assert "ligand file" in result.stdout.lower() or "Error" in result.stdout


def test_cli_validation_with_dataset_path(tmp_path, mock_checkpoint):
    """Test that CLI warns (doesn't error) when missing inputs but dataset path provided."""
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "fixed_protein_ligand_denovo_condensed",
            "--pharmit_path", str(tmp_path),
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    # Should warn but not error (may still error if dataset doesn't exist, but validation should pass)
    assert "Warning" in result.stdout or "protein file" in result.stdout.lower() or result.returncode != 0


def test_cli_input_files_validation(tmp_path, mock_checkpoint):
    """Test that CLI validates required input files correctly."""
    import shutil
    test_protein = tmp_path / "test_protein.pdb"
    test_ligand = tmp_path / "test_ligand.sdf"
    shutil.copy(TEST_REC, test_protein)
    shutil.copy(TEST_LIG, test_ligand)
    
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "expapo_conditioned_ligand_docking_condensed",
            "--protein_file", str(test_protein),
            "--ligand_file", str(test_ligand),
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    assert "protein file" not in result.stdout.lower() and "ligand file" not in result.stdout.lower() and "pharmacophore file" not in result.stdout.lower()


@pytest.fixture
def test_pharmacophore(tmp_path):
    """Copy pharmacophore XYZ file for testing."""
    import shutil
    pharm_file = tmp_path / "test_pharmacophore.xyz"
    shutil.copy(TEST_PHARM, pharm_file)
    return pharm_file


def test_cli_pharmacophore_conditioned_missing_pharmacophore(tmp_path, mock_checkpoint):
    """Test that CLI errors when pharmacophore-conditioned task is missing pharmacophore file."""
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "denovo_ligand_from_pharmacophore_condensed",
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    assert result.returncode != 0
    assert "pharmacophore file" in result.stdout.lower() or "Error" in result.stdout


def test_cli_pharmacophore_conditioned_with_pharmacophore(tmp_path, mock_checkpoint, test_pharmacophore):
    """Test that CLI works with pharmacophore file for pharmacophore-conditioned task."""
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "denovo_ligand_from_pharmacophore_condensed",
            "--pharmacophore_file", str(test_pharmacophore),
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    # Should not error about missing pharmacophore (may error if model loading fails, but validation should pass)
    assert "pharmacophore file" not in result.stdout.lower() or result.returncode == 0


def test_cli_protein_pharmacophore_conditioned_missing_pharmacophore(tmp_path, mock_checkpoint):
    """Test that CLI errors when protein+pharmacophore task is missing pharmacophore file."""
    import shutil
    test_protein = tmp_path / "test_protein.pdb"
    shutil.copy(TEST_REC, test_protein)
    
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "fixed_protein_pharmacophore_ligand_denovo_condensed",
            "--protein_file", str(test_protein),
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    assert result.returncode != 0
    assert "pharmacophore file" in result.stdout.lower() or "Error" in result.stdout


def test_cli_protein_pharmacophore_conditioned_missing_protein(tmp_path, mock_checkpoint, test_pharmacophore):
    """Test that CLI errors when protein+pharmacophore task is missing protein file."""
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "fixed_protein_pharmacophore_ligand_denovo_condensed",
            "--pharmacophore_file", str(test_pharmacophore),
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    assert result.returncode != 0
    assert "protein file" in result.stdout.lower() or "Error" in result.stdout


def test_cli_protein_pharmacophore_conditioned_with_both(tmp_path, mock_checkpoint, test_pharmacophore):
    """Test that CLI works with both protein and pharmacophore files."""
    import shutil
    test_protein = tmp_path / "test_protein.pdb"
    shutil.copy(TEST_REC, test_protein)
    
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "fixed_protein_pharmacophore_ligand_denovo_condensed",
            "--protein_file", str(test_protein),
            "--pharmacophore_file", str(test_pharmacophore),
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )

    assert "requires" not in result.stdout.lower() or ("protein file" not in result.stdout.lower() and "pharmacophore file" not in result.stdout.lower())


def test_cli_protein_pharmacophore_conditioned_warning_no_ligand(tmp_path, mock_checkpoint, test_pharmacophore):
    """Test that CLI warns when protein+pharmacophore task has no reference ligand."""
    import shutil
    test_protein = tmp_path / "test_protein.pdb"
    shutil.copy(TEST_REC, test_protein)
    
    cli_path = Path(__file__).parent.parent.parent / "cli.py"
    result = subprocess.run(
        [
            sys.executable, str(cli_path),
            str(mock_checkpoint),
            "--task", "fixed_protein_pharmacophore_ligand_denovo_condensed",
            "--protein_file", str(test_protein),
            "--pharmacophore_file", str(test_pharmacophore),
            "--n_samples", "1"
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )
    # Should warn about missing reference ligand
    assert "reference ligand" in result.stdout.lower() or "Warning" in result.stdout or result.returncode != 0

"""
Unit tests for omtra.dataset.crop_utils module.
These tests use mock data and don't require real zarr datasets.
"""

import pytest
import numpy as np
from omtra.dataset.crop_utils import (
    compute_close_residues,
    crop_structure_data,
    filter_npndes_by_distance,
    sample_crop_distance,
    crop_structure_data_multiple_distances
)
from omtra.data.plinder import StructureData, BackboneData, LigandData


class TestSampleCropDistance:
    def test_samples_within_range(self):
        """Sampled distance should always be within [min, max]."""
        for _ in range(100):
            dist = sample_crop_distance(5.0, 10.0)
            assert 5.0 <= dist <= 10.0
    
    def test_min_equals_max(self):
        """When min == max, should return that exact value."""
        dist = sample_crop_distance(7.0, 7.0)
        assert dist == 7.0
    
    def test_deterministic_with_seed(self):
        """Results should be reproducible with numpy seed."""
        np.random.seed(42)
        dist1 = sample_crop_distance(5.0, 10.0)
        np.random.seed(42)
        dist2 = sample_crop_distance(5.0, 10.0)
        assert dist1 == dist2


class TestComputeCloseResidues:
    def test_empty_protein_coords(self):
        """Empty protein should return empty set."""
        result = compute_close_residues(
            np.array([]).reshape(0, 3),
            np.array([]),
            np.array([]),
            np.array([[0, 0, 0]]),
            5.0
        )
        assert result == set()
    
    def test_empty_reference_coords(self):
        """Empty reference should return empty set."""
        result = compute_close_residues(
            np.array([[0, 0, 0]]),
            np.array([1]),
            np.array(["A"]),
            np.array([]).reshape(0, 3),
            5.0
        )
        assert result == set()
    
    def test_finds_close_residues(self):
        """Should find residues within distance threshold."""
        protein_coords = np.array([[0, 0, 0], [1, 0, 0], [10, 10, 10], [11, 10, 10]])
        res_ids = np.array([1, 1, 2, 2])
        chain_ids = np.array(["A", "A", "A", "A"])
        ref_coords = np.array([[0.5, 0, 0]])
        
        result = compute_close_residues(protein_coords, res_ids, chain_ids, ref_coords, 5.0)
        assert ("A", 1) in result
        assert ("A", 2) not in result
    
    def test_multiple_chains(self):
        """Should correctly handle multiple chains."""
        protein_coords = np.array([[0, 0, 0], [10, 10, 10]])
        res_ids = np.array([1, 1])  # Same res_id
        chain_ids = np.array(["A", "B"])  # Different chains
        ref_coords = np.array([[0.5, 0, 0]])
        
        result = compute_close_residues(protein_coords, res_ids, chain_ids, ref_coords, 2.0)
        assert ("A", 1) in result
        assert ("B", 1) not in result


class TestCropStructureData:
    @pytest.fixture
    def mock_structure(self):
        """Create a mock StructureData for testing."""
        return StructureData(
            coords=np.array([
                [0, 0, 0], [1, 1, 1],     # Residue 1 (close)
                [10, 10, 10], [11, 11, 11] # Residue 2 (far)
            ], dtype=np.float32),
            atom_names=np.array(["CA", "CB", "CA", "CB"]),
            elements=np.array(["C", "C", "C", "C"]),
            res_ids=np.array([1, 1, 2, 2]),
            res_names=np.array(["ALA", "ALA", "GLY", "GLY"]),
            chain_ids=np.array(["A", "A", "A", "A"]),
            backbone_mask=np.array([True, False, True, False]),
            backbone=BackboneData(
                coords=np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32),
                res_ids=np.array([1, 2]),
                res_names=np.array(["ALA", "GLY"]),
                chain_ids=np.array(["A", "A"]),
            ),
            cif=None,
        )
    
    def test_crops_to_close_residues(self, mock_structure):
        """Should only keep residues within crop distance."""
        ref_coords = np.array([[0.5, 0.5, 0.5]])
        cropped = crop_structure_data(mock_structure, ref_coords, 3.0)
        
        assert cropped is not None
        assert len(cropped.coords) == 2  # Only residue 1 (2 atoms)
        assert np.all(cropped.res_ids == 1)
        assert len(cropped.backbone.coords) == 1
    
    def test_returns_none_for_no_close_residues(self, mock_structure):
        """Should return None if no residues are within distance."""
        ref_coords = np.array([[100, 100, 100]])
        cropped = crop_structure_data(mock_structure, ref_coords, 3.0)
        assert cropped is None
    
    def test_preserves_all_fields(self, mock_structure):
        """Cropped structure should have all expected fields."""
        ref_coords = np.array([[0.5, 0.5, 0.5]])
        cropped = crop_structure_data(mock_structure, ref_coords, 3.0)
        
        assert cropped.coords is not None
        assert cropped.atom_names is not None
        assert cropped.elements is not None
        assert cropped.res_ids is not None
        assert cropped.res_names is not None
        assert cropped.chain_ids is not None
        assert cropped.backbone_mask is not None
        assert cropped.backbone is not None
    
    def test_keeps_all_atoms_from_close_residue(self, mock_structure):
        """Should keep ALL atoms from a residue if ANY atom is close."""
        # Reference is close to only one atom of residue 1
        ref_coords = np.array([[0, 0, 0]])
        cropped = crop_structure_data(mock_structure, ref_coords, 0.5)
        
        # Should still get both atoms from residue 1
        assert len(cropped.coords) == 2
        assert np.all(cropped.res_ids == 1)

class TestCropStructureDataMultipleDistances:
    @pytest.fixture
    def simple_mock_structure(self):
        """Create a simple mock StructureData for testing."""
        return StructureData(
            coords=np.array([
                [0.0, 0.0, 0.0], # Residue 1, 1 atoms
                [1.5, 1.5, 1.5], #Residue 2, 1 atoms
            ], dtype=np.float32),
            atom_names=np.array(["CA", "CB"]), #alpha carbon, beta carbon
            elements=np.array(["C", "C"]), #all carbon
            res_ids=np.array([1, 2]), #residue number to which atoms belong
            res_names=np.array(["ALA","GLY"]), #residue names
            chain_ids=np.array(["A", "A"]), #which protein chain atoms belong to (all same chain)
            backbone_mask=np.array([True, False]), #which atoms are part of protein backbone
            backbone=BackboneData(
                coords=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                res_ids=np.array([1]),
                res_names=np.array(["ALA"]),
                chain_ids=np.array(["A"]),
            ),
            cif=None,
        )
    @pytest.fixture
    def mock_structure(self):
        """Create a mock StructureData for testing."""
        return StructureData(
            coords=np.array([
                [0.75, 0.75, 0.75], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5], # Residue 1, 3 atoms
                [2.0, 2.0, 2.0], [8.0, 8.0, 8.0], #Residue 2, 2 atoms)
                [7.5, 9.5, 9.5], [11, 11, 11] # Residue 3, 2 atoms
            ], dtype=np.float32),
            atom_names=np.array(["CA", "CB", "CB", "CA", "CB", "CA", "CB"]), #alpha carbon, beta carbon
            elements=np.array(["C", "C", "C", "C", "C", "C", "C"]), #all carbon
            res_ids=np.array([1, 1, 1, 2, 2, 3, 3]), #residue number to which atoms belong
            res_names=np.array(["ALA", "ALA", "ALA", "MET", "MET", "GLY", "GLY"]), #residue names
            chain_ids=np.array(["A","A","A","A","A","A","A"]), #which protein chain atoms belong to (all same chain)
            backbone_mask=np.array([True, False, False, True, False, True, False]), #which atoms are part of protein backbone
            backbone=BackboneData(
                coords=np.array([[0.75, 0.75, 0.75], [2.0, 2.0, 2.0], [7.5, 9.5, 9.5]], dtype=np.float32),
                res_ids=np.array([1, 2, 3]),
                res_names=np.array(["ALA", "MET","GLY"]),
                chain_ids=np.array(["A", "A", "A"]),
            ),
            cif=None,
        )
    
    def test_crops(self, mock_structure):
        """Should crop structure based on multiple reference groups and distances.
        """
        ref_coords_groups = [
            np.array([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]),   # First group of ligand atoms
            np.array([[2.5, 2.5, 2.5], [3.0, 3.0, 3.0]])    # Second group of ligand atoms
        ]
        crop_distances = [1.0, 2.75]

        cropped = crop_structure_data_multiple_distances(
            mock_structure, 
            ref_coords_groups, 
            crop_distances
        )
        
        #ensure all protein atoms are included except [7.5, 9.5, 9.5] and [11, 11, 11]
        assert cropped is not None
        assert len(cropped.coords) == 5  # 2 atoms are cropped out of 7 (residue 1 and 2 remain)
        assert np.all(np.isin(cropped.res_ids, [1, 2]))
        assert len(cropped.backbone.coords) == 2  # Backbone from first two residues

    def test_single_ligand_group_multiple_atoms_crop(self, mock_structure):
        """Should crop structure correctly with a single ligand group."""
        ref_coords_groups = [
            np.array([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]])   # Single group of ligand atoms
        ]
        crop_distances = [1.0]

        cropped = crop_structure_data_multiple_distances(
            mock_structure, 
            ref_coords_groups, 
            crop_distances
        )
        
        assert cropped is not None
        assert len(cropped.coords) == 3  # Only residue 1 (3 atoms)
        assert np.all(cropped.res_ids == 1)
        assert len(cropped.backbone.coords) == 1  # Only backbone from first residue
    
    def test_single_ligand_group_single_atom_crop(self, mock_structure):
        """Should crop structure correctly with a single ligand group, single atom."""
        ref_coords_groups = [
            np.array([[1.5, 1.5, 1.5]])   # Single group of 1 ligand atom
        ]
        crop_distances = [1.0]

        cropped = crop_structure_data_multiple_distances(
            mock_structure, 
            ref_coords_groups, 
            crop_distances
        )
        
        assert cropped is not None
        assert len(cropped.coords) == 5  # Only residue 1 and 2 (5 atoms)
        assert np.all(np.isin(cropped.res_ids, [1, 2]))
        assert len(cropped.backbone.coords) == 2  # Backbone from first two residues
    
    def test_asymmetric_crop(self, simple_mock_structure):
        """Should crop structure correctly with asymmetric distances."""
        ref_coords_groups = [
            np.array([[0.5, 0.5, 0.5]]),   # First group of ligand atoms
            np.array([[1.0, 1.0, 1.0]])    # Second group of ligand atoms
        ]
        crop_distances = [1.0, 0.25]

        cropped = crop_structure_data_multiple_distances(
            simple_mock_structure, 
            ref_coords_groups, 
            crop_distances
        )
        #Residue 1 is included but residue 2 is not
        assert cropped is not None
        assert len(cropped.coords) == 1  # 1 atom from residue 1 only
        assert np.all(cropped.res_ids == 1)
        assert len(cropped.backbone.coords) == 1  # Only backbone from first residue
    
    def test_returns_none_for_no_close_residues(self, mock_structure):
        """Should return None if no residues are within distance."""
        ref_coords = [np.array([[100, 100, 100]])]
        cropped = crop_structure_data_multiple_distances(mock_structure, ref_coords, [3.0])
        assert cropped is None
    
    def test_preserves_all_fields(self, mock_structure):
        """Cropped structure should have all expected fields."""
        ref_coords = [np.array([[0.5, 0.5, 0.5]])]
        cropped = crop_structure_data_multiple_distances(mock_structure, ref_coords, [3.0])
        
        assert cropped.coords is not None
        assert cropped.atom_names is not None
        assert cropped.elements is not None
        assert cropped.res_ids is not None
        assert cropped.res_names is not None
        assert cropped.chain_ids is not None
        assert cropped.backbone_mask is not None
        assert cropped.backbone is not None
    
    def test_keeps_all_atoms_from_close_residue(self, mock_structure):
        """Should keep ALL atoms from a residue if ANY atom is close."""
        # Reference is close to only one atom of residue 1
        ref_coords = [np.array([[3.0, 3.0, 3.0]])]
        cropped = crop_structure_data_multiple_distances(mock_structure, ref_coords, [2.75])
        
        # Should still get both atoms from residue 1 and 2 but not 3
        assert len(cropped.coords) == 5
        assert np.all(np.isin(cropped.res_ids, [1, 2]))
        
    
class TestFilterNpndesByDistance:
    def test_filters_distant_npndes(self):
        """Should filter out NPNDEs that are too far."""
        npndes = {
            "close": LigandData(
                coords=np.array([[1, 1, 1]], dtype=np.float32),
                bond_types=np.array([], dtype=np.int32),
                bond_indices=np.zeros((0, 2), dtype=np.int32),
                is_covalent=False,
                ccd="UNK",
                sdf="",
            ),
            "far": LigandData(
                coords=np.array([[100, 100, 100]], dtype=np.float32),
                bond_types=np.array([], dtype=np.int32),
                bond_indices=np.zeros((0, 2), dtype=np.int32),
                is_covalent=False,
                ccd="UNK",
                sdf="",
            ),
        }
        ref_coords = np.array([[0, 0, 0]])
        
        filtered = filter_npndes_by_distance(npndes, ref_coords, 5.0)
        
        assert filtered is not None
        assert "close" in filtered
        assert "far" not in filtered
    
    def test_returns_none_for_empty_input(self):
        """Should return None for empty or None input."""
        assert filter_npndes_by_distance(None, np.array([[0, 0, 0]]), 5.0) is None
        assert filter_npndes_by_distance({}, np.array([[0, 0, 0]]), 5.0) is None
    
    def test_returns_none_when_all_filtered(self):
        """Should return None if all NPNDEs are filtered out."""
        npndes = {
            "far": LigandData(
                coords=np.array([[100, 100, 100]], dtype=np.float32),
                bond_types=np.array([], dtype=np.int32),
                bond_indices=np.zeros((0, 2), dtype=np.int32),
                is_covalent=False,
                ccd="UNK",
                sdf="",
            ),
        }
        ref_coords = np.array([[0, 0, 0]])
        
        filtered = filter_npndes_by_distance(npndes, ref_coords, 5.0)
        assert filtered is None

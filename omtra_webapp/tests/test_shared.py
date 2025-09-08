import pytest
import sys
import os
from pathlib import Path

# Add shared module to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.models import (
    SamplingParams, JobSubmission, validate_filename, 
    validate_file_extension, calculate_file_hash
)
from shared.file_utils import validate_file_safety, FileValidationError


class TestModels:
    """Test shared models and validation"""
    
    def test_sampling_params_valid(self):
        """Test valid sampling parameters"""
        params = SamplingParams(
            seed=42,
            n_samples=10,
            steps=100,
            temperature=1.0,
            guidance_scale=1.0,
            conditioning_strength=1.0
        )
        assert params.seed == 42
        assert params.n_samples == 10
        assert params.steps == 100
        assert params.temperature == 1.0
    
    def test_sampling_params_bounds(self):
        """Test parameter bounds validation"""
        with pytest.raises(ValueError):
            SamplingParams(n_samples=0)  # Below minimum
        
        with pytest.raises(ValueError):
            SamplingParams(n_samples=101)  # Above maximum
        
        with pytest.raises(ValueError):
            SamplingParams(temperature=0.05)  # Below minimum
    
    def test_job_submission_valid(self):
        """Test valid job submission"""
        params = SamplingParams(n_samples=5)
        submission = JobSubmission(
            params=params,
            uploads=['token1', 'token2']
        )
        assert len(submission.uploads) == 2
    
    def test_job_submission_too_many_files(self):
        """Test job submission with too many files"""
        params = SamplingParams(n_samples=5)
        with pytest.raises(ValueError, match="Maximum 3 files"):
            JobSubmission(
                params=params,
                uploads=['token1', 'token2', 'token3', 'token4']
            )
    
    def test_filename_validation(self):
        """Test filename sanitization"""
        assert validate_filename("test.sdf") == "test.sdf"
        assert validate_filename("../../../etc/passwd") == "passwd"
        assert validate_filename("file with spaces.sdf") == "file_with_spaces.sdf"
        assert validate_filename("test@#$%.sdf") == "test____.sdf"
    
    def test_file_extension_validation(self):
        """Test file extension validation"""
        allowed = ['.sdf', '.mol2', '.pdb', '.cif']
        
        assert validate_file_extension("test.sdf", allowed) == True
        assert validate_file_extension("test.SDF", allowed) == True  # Case insensitive
        assert validate_file_extension("test.txt", allowed) == False
        assert validate_file_extension("test", allowed) == False
    
    def test_file_hash(self):
        """Test file hash calculation"""
        content = b"test content"
        hash1 = calculate_file_hash(content)
        hash2 = calculate_file_hash(content)
        
        assert hash1 == hash2  # Same content, same hash
        assert len(hash1) == 64  # SHA256 hex string length
        
        different_content = b"different content"
        hash3 = calculate_file_hash(different_content)
        assert hash1 != hash3  # Different content, different hash


class TestFileUtils:
    """Test file utilities"""
    
    def test_file_validation_valid_sdf(self):
        """Test validation of valid SDF file"""
        sdf_content = b"""Test Molecule
  Generated for testing
  
  6  6  0  0  0  0  0  0  0  0999 V2000
    0.0000    1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
M  END
$$$$
"""
        
        result = validate_file_safety("test.sdf", sdf_content)
        assert result['safe_filename'] == "test.sdf"
        assert result['size'] == len(sdf_content)
        assert 'sha256' in result
        assert result['mime_type'] == 'chemical/x-mdl-sdfile'
    
    def test_file_validation_invalid_extension(self):
        """Test validation with invalid extension"""
        with pytest.raises(FileValidationError, match="File extension not allowed"):
            validate_file_safety("test.txt", b"some content")
    
    def test_file_validation_too_large(self):
        """Test validation with file too large"""
        large_content = b"x" * (26 * 1024 * 1024 + 1)  # Just over 26MB
        
        with pytest.raises(FileValidationError, match="File too large"):
            validate_file_safety("test.sdf", large_content)
    
    def test_file_validation_empty_file(self):
        """Test validation with empty file"""
        with pytest.raises(FileValidationError, match="Empty file not allowed"):
            validate_file_safety("test.sdf", b"")
    
    def test_file_validation_invalid_sdf(self):
        """Test validation with invalid SDF content"""
        invalid_sdf = b"This is not a valid SDF file"
        
        with pytest.raises(FileValidationError, match="Invalid molecular file format"):
            validate_file_safety("test.sdf", invalid_sdf)

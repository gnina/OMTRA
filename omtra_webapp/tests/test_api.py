import pytest
import requests
import time
import json
import io
from pathlib import Path


class TestAPI:
    """End-to-end API tests"""
    
    @pytest.fixture
    def api_base_url(self):
        """API base URL for testing"""
        return "http://localhost:8000"
    
    @pytest.fixture
    def sample_sdf_content(self):
        """Sample SDF content for testing"""
        return b"""Test Molecule
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
    
    def test_health_check(self, api_base_url):
        """Test health check endpoint"""
        response = requests.get(f"{api_base_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_upload_init(self, api_base_url):
        """Test upload initialization"""
        response = requests.post(f"{api_base_url}/upload/init")
        assert response.status_code == 200
        
        data = response.json()
        assert "upload_token" in data
        assert "max_file_size" in data
        assert data["max_file_size"] > 0
    
    def test_file_upload(self, api_base_url, sample_sdf_content):
        """Test file upload"""
        # Initialize upload
        init_response = requests.post(f"{api_base_url}/upload/init")
        assert init_response.status_code == 200
        
        upload_token = init_response.json()["upload_token"]
        
        # Upload file
        files = {'file': ('test.sdf', io.BytesIO(sample_sdf_content), 'application/octet-stream')}
        response = requests.post(f"{api_base_url}/upload/{upload_token}", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert data["filename"] == "test.sdf"
        assert data["size"] == len(sample_sdf_content)
        assert data["upload_token"] == upload_token
    
    def test_job_submission_and_completion(self, api_base_url, sample_sdf_content):
        """Test complete job workflow"""
        # Upload file
        init_response = requests.post(f"{api_base_url}/upload/init")
        upload_token = init_response.json()["upload_token"]
        
        files = {'file': ('test.sdf', io.BytesIO(sample_sdf_content), 'application/octet-stream')}
        upload_response = requests.post(f"{api_base_url}/upload/{upload_token}", files=files)
        assert upload_response.status_code == 200
        
        # Submit job
        job_data = {
            "params": {
                "seed": 42,
                "n_samples": 3,
                "steps": 10,
                "temperature": 1.0,
                "guidance_scale": 1.0,
                "conditioning_strength": 1.0
            },
            "uploads": [upload_token]
        }
        
        job_response = requests.post(f"{api_base_url}/sample", json=job_data)
        assert job_response.status_code == 200
        
        job_id = job_response.json()["job_id"]
        assert job_id is not None
        
        # Poll job status until completion
        max_wait_time = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(f"{api_base_url}/status/{job_id}")
            assert status_response.status_code == 200
            
            status_data = status_response.json()
            state = status_data["state"]
            
            if state in ["SUCCEEDED", "FAILED"]:
                break
            
            time.sleep(1)
        
        # Job should have completed
        assert state == "SUCCEEDED", f"Job failed or timed out. Final state: {state}"
        
        # Get results
        result_response = requests.get(f"{api_base_url}/result/{job_id}")
        assert result_response.status_code == 200
        
        result_data = result_response.json()
        assert result_data["state"] == "SUCCEEDED"
        assert len(result_data["artifacts"]) > 0
        
        # Test file download
        first_artifact = result_data["artifacts"][0]
        download_response = requests.get(
            f"{api_base_url}/download/{job_id}/{first_artifact['filename']}"
        )
        assert download_response.status_code == 200
        assert len(download_response.content) > 0
    
    def test_job_submission_no_uploads(self, api_base_url):
        """Test job submission without uploads"""
        job_data = {
            "params": {
                "seed": 42,
                "n_samples": 2,
                "steps": 10,
                "temperature": 1.0,
                "guidance_scale": 1.0,
                "conditioning_strength": 1.0
            },
            "uploads": []
        }
        
        job_response = requests.post(f"{api_base_url}/sample", json=job_data)
        assert job_response.status_code == 200
        
        job_id = job_response.json()["job_id"]
        
        # Wait for completion
        max_wait_time = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = requests.get(f"{api_base_url}/status/{job_id}")
            status_data = status_response.json()
            
            if status_data["state"] in ["SUCCEEDED", "FAILED"]:
                break
            
            time.sleep(1)
        
        assert status_data["state"] == "SUCCEEDED"
    
    def test_invalid_upload_token(self, api_base_url):
        """Test job submission with invalid upload token"""
        job_data = {
            "params": {
                "seed": 42,
                "n_samples": 1,
                "steps": 10,
                "temperature": 1.0,
                "guidance_scale": 1.0,
                "conditioning_strength": 1.0
            },
            "uploads": ["invalid-token"]
        }
        
        job_response = requests.post(f"{api_base_url}/sample", json=job_data)
        assert job_response.status_code == 400
    
    def test_invalid_file_extension(self, api_base_url):
        """Test upload with invalid file extension"""
        # Initialize upload
        init_response = requests.post(f"{api_base_url}/upload/init")
        upload_token = init_response.json()["upload_token"]
        
        # Try to upload invalid file
        invalid_content = b"This is not a molecular structure file"
        files = {'file': ('test.txt', io.BytesIO(invalid_content), 'text/plain')}
        
        response = requests.post(f"{api_base_url}/upload/{upload_token}", files=files)
        assert response.status_code == 400
    
    def test_job_not_found(self, api_base_url):
        """Test accessing non-existent job"""
        fake_job_id = "00000000-0000-0000-0000-000000000000"
        
        response = requests.get(f"{api_base_url}/status/{fake_job_id}")
        assert response.status_code == 404

import streamlit as st
import requests
import time
import json
import io
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Import molecular visualization
try:
    import stmol
    import py3Dmol
    VISUALIZATION_AVAILABLE = True
except ImportError:
    st.warning("Molecular visualization not available. Install stmol and py3Dmol for 3D viewing.")
    VISUALIZATION_AVAILABLE = False

# Configuration
API_URL = "http://api:8000"  # Docker service name
if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"]

# Page configuration
st.set_page_config(
    page_title="OMTRA Molecule Sampler",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .molecule-viewer {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .job-status {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-weight: bold;
    }
    .status-queued { background-color: #fff3cd; color: #856404; }
    .status-running { background-color: #d1ecf1; color: #0c5460; }
    .status-succeeded { background-color: #d4edda; color: #155724; }
    .status-failed { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


class APIClient:
    """Client for interacting with the OMTRA API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def init_upload(self) -> Optional[Dict]:
        """Initialize file upload"""
        try:
            response = requests.post(f"{self.base_url}/upload/init")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to initialize upload: {e}")
            return None
    
    def upload_file(self, upload_token: str, file_content: bytes, filename: str) -> Optional[Dict]:
        """Upload a file"""
        try:
            files = {'file': (filename, io.BytesIO(file_content), 'application/octet-stream')}
            response = requests.post(
                f"{self.base_url}/upload/{upload_token}",
                files=files
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to upload file {filename}: {e}")
            return None
    
    def submit_job(self, params: Dict, upload_tokens: List[str]) -> Optional[Dict]:
        """Submit a sampling job"""
        try:
            data = {
                "params": params,
                "uploads": upload_tokens
            }
            response = requests.post(f"{self.base_url}/sample", json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to submit job: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status"""
        try:
            response = requests.get(f"{self.base_url}/status/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get job status: {e}")
            return None
    
    def get_job_result(self, job_id: str) -> Optional[Dict]:
        """Get job results"""
        try:
            response = requests.get(f"{self.base_url}/result/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to get job result: {e}")
            return None
    
    def download_file(self, job_id: str, filename: str) -> Optional[bytes]:
        """Download a file"""
        try:
            response = requests.get(f"{self.base_url}/download/{job_id}/{filename}")
            response.raise_for_status()
            return response.content
        except Exception as e:
            st.error(f"Failed to download file: {e}")
            return None
    
    def download_all_outputs(self, job_id: str) -> Optional[bytes]:
        """Download all outputs as ZIP"""
        try:
            response = requests.get(f"{self.base_url}/download/{job_id}/all")
            response.raise_for_status()
            return response.content
        except Exception as e:
            st.error(f"Failed to download outputs: {e}")
            return None


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient(API_URL)
    
    if 'jobs' not in st.session_state:
        st.session_state.jobs = {}
    
    if 'upload_tokens' not in st.session_state:
        st.session_state.upload_tokens = []
    
    if 'selected_job' not in st.session_state:
        st.session_state.selected_job = None


def render_sidebar():
    """Render the sidebar with parameters and file upload"""
    st.sidebar.title("🧪 OMTRA Sampler")
    
    # API health check
    if st.session_state.api_client.health_check():
        st.sidebar.success("API Connected")
    else:
        st.sidebar.error("API Unavailable")
        return None
    
    st.sidebar.markdown("---")
    
    # Sampling parameters
    st.sidebar.subheader("Sampling Parameters")
    
    params = {}
    params['seed'] = st.sidebar.number_input(
        "Random Seed", 
        min_value=0, 
        max_value=2**31-1,
        value=42,
        help="Set to None for random seed"
    )
    
    params['n_samples'] = st.sidebar.number_input(
        "Number of Samples",
        min_value=1,
        max_value=100,
        value=10,
        help="Number of molecules to generate"
    )
    
    params['steps'] = st.sidebar.number_input(
        "Sampling Steps",
        min_value=10,
        max_value=1000,
        value=100,
        help="Number of diffusion steps"
    )
    
    params['temperature'] = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Sampling temperature (higher = more diverse)"
    )
    
    params['guidance_scale'] = st.sidebar.slider(
        "Guidance Scale",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Strength of conditional guidance"
    )
    
    params['conditioning_strength'] = st.sidebar.slider(
        "Conditioning Strength",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Strength of conditioning signal"
    )
    
    # File upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("Input Files")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload molecular structures",
        type=['sdf', 'cif', 'mol2', 'pdb'],
        accept_multiple_files=True,
        help="Upload up to 3 molecular structure files"
    )
    
    # Process uploaded files
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.sidebar.error("Maximum 3 files allowed")
            uploaded_files = uploaded_files[:3]
        
        # Upload files to API
        upload_tokens = []
        for uploaded_file in uploaded_files:
            if uploaded_file.size > 25 * 1024 * 1024:  # 25MB
                st.sidebar.error(f"File {uploaded_file.name} too large (max 25MB)")
                continue
            
            # Initialize upload
            upload_init = st.session_state.api_client.init_upload()
            if upload_init:
                upload_token = upload_init['upload_token']
                
                # Upload file
                file_content = uploaded_file.read()
                upload_result = st.session_state.api_client.upload_file(
                    upload_token, file_content, uploaded_file.name
                )
                
                if upload_result:
                    upload_tokens.append(upload_token)
                    st.sidebar.success(f"✓ {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        st.session_state.upload_tokens = upload_tokens
    
    # Submit job button
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Run Sampling", type="primary", use_container_width=True):
        if not st.session_state.upload_tokens and params['n_samples'] > 0:
            st.sidebar.warning("Consider uploading input files for better results")
        
        # Submit job
        job_response = st.session_state.api_client.submit_job(params, st.session_state.upload_tokens)
        
        if job_response:
            job_id = job_response['job_id']
            st.session_state.jobs[job_id] = {
                'job_id': job_id,
                'params': params,
                'submitted_at': time.time(),
                'status': 'QUEUED'
            }
            st.session_state.selected_job = job_id
            st.sidebar.success(f"Job submitted: {job_id[:8]}...")
            st.rerun()
    
    return params


def render_job_list():
    """Render the list of jobs"""
    st.subheader("Jobs")
    
    if not st.session_state.jobs:
        st.info("No jobs submitted yet. Use the sidebar to start sampling!")
        return
    
    # Create job status table
    job_data = []
    for job_id, job_info in st.session_state.jobs.items():
        # Update job status
        status_response = st.session_state.api_client.get_job_status(job_id)
        if status_response:
            job_info.update(status_response)
        
        job_data.append({
            'Job ID': job_id[:8] + '...',
            'Status': job_info.get('state', 'Unknown'),
            'Progress': f"{job_info.get('progress', 0)}%",
            'Samples': job_info.get('params', {}).get('n_samples', 'N/A'),
            'Elapsed': format_duration(job_info.get('elapsed_seconds', 0)),
            'Select': job_id
        })
    
    if job_data:
        df = pd.DataFrame(job_data)
        
        # Display table with selection
        selected_rows = st.data_editor(
            df,
            column_config={
                'Select': st.column_config.CheckboxColumn(
                    'Select',
                    help='Select job to view details',
                    default=False
                )
            },
            disabled=['Job ID', 'Status', 'Progress', 'Samples', 'Elapsed'],
            use_container_width=True,
            hide_index=True
        )
        
        # Handle job selection
        for i, row in selected_rows.iterrows():
            if row['Select']:
                full_job_id = list(st.session_state.jobs.keys())[i]
                st.session_state.selected_job = full_job_id
                break


def render_job_details():
    """Render details for selected job"""
    if not st.session_state.selected_job:
        return
    
    job_id = st.session_state.selected_job
    job_info = st.session_state.jobs.get(job_id, {})
    
    st.subheader(f"Job Details: {job_id[:8]}...")
    
    # Get current status
    status_response = st.session_state.api_client.get_job_status(job_id)
    if status_response:
        job_info.update(status_response)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status = job_info.get('state', 'Unknown')
        status_class = f"status-{status.lower()}"
        st.markdown(f'<div class="job-status {status_class}">Status: {status}</div>', 
                   unsafe_allow_html=True)
    
    with col2:
        progress = job_info.get('progress', 0)
        st.progress(progress / 100)
        st.write(f"Progress: {progress}%")
    
    with col3:
        elapsed = job_info.get('elapsed_seconds', 0)
        st.write(f"Elapsed: {format_duration(elapsed)}")
    
    # Show parameters
    if 'params' in job_info:
        with st.expander("Parameters"):
            st.json(job_info['params'])
    
    # Show results if completed
    if status_response and status_response['state'] == 'SUCCEEDED':
        render_job_results(job_id)
    elif status_response and status_response['state'] == 'FAILED':
        st.error(f"Job failed: {status_response.get('message', 'Unknown error')}")


def render_job_results(job_id: str):
    """Render job results"""
    st.subheader("Results")
    
    # Get results
    result_response = st.session_state.api_client.get_job_result(job_id)
    if not result_response:
        return
    
    artifacts = result_response.get('artifacts', [])
    
    if not artifacts:
        st.info("No output files generated")
        return
    
    # Download all button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📦 Download All", use_container_width=True):
            zip_content = st.session_state.api_client.download_all_outputs(job_id)
            if zip_content:
                st.download_button(
                    label="💾 Save ZIP",
                    data=zip_content,
                    file_name=f"{job_id}_outputs.zip",
                    mime="application/zip",
                    use_container_width=True
                )
    
    # Display artifacts
    st.write(f"Generated {len(artifacts)} files:")
    
    for i, artifact in enumerate(artifacts):
        with st.expander(f"📄 {artifact['filename']} ({format_file_size(artifact['size'])})"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Download individual file
                if st.button(f"💾 Download", key=f"download_{i}"):
                    file_content = st.session_state.api_client.download_file(
                        job_id, artifact['filename']
                    )
                    if file_content:
                        st.download_button(
                            label="Save File",
                            data=file_content,
                            file_name=artifact['filename'],
                            mime="application/octet-stream"
                        )
            
            with col2:
                # Display molecular viewer for structure files
                if VISUALIZATION_AVAILABLE and artifact['format'].lower() in ['sdf', 'mol2', 'pdb']:
                    render_molecular_viewer(job_id, artifact['filename'])


def render_molecular_viewer(job_id: str, filename: str):
    """Render 3D molecular viewer"""
    if not VISUALIZATION_AVAILABLE:
        st.info("Install stmol and py3Dmol for 3D molecular visualization")
        return
    
    try:
        # Download file content
        file_content = st.session_state.api_client.download_file(job_id, filename)
        if not file_content:
            return
        
        # Determine file format
        file_format = filename.split('.')[-1].lower()
        if file_format not in ['sdf', 'mol2', 'pdb']:
            return
        
        # Create viewer
        content_str = file_content.decode('utf-8', errors='ignore')
        
        # Use stmol for visualization
        stmol.showmol(
            content_str,
            format=file_format,
            height=400,
            width=600
        )
        
    except Exception as e:
        st.error(f"Failed to render molecule: {e}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable form"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"


def main():
    """Main application"""
    initialize_session_state()
    
    # Render sidebar
    params = render_sidebar()
    
    # Main content area
    st.title("🧪 OMTRA Molecule Sampler")
    st.markdown("Generate novel molecular structures using deep learning")
    
    # Auto-refresh for running jobs
    if any(job.get('state') in ['QUEUED', 'RUNNING'] for job in st.session_state.jobs.values()):
        time.sleep(2)
        st.rerun()
    
    # Render main content
    tab1, tab2 = st.tabs(["Jobs", "Help"])
    
    with tab1:
        render_job_list()
        if st.session_state.selected_job:
            st.markdown("---")
            render_job_details()
    
    with tab2:
        render_help()


def render_help():
    """Render help documentation"""
    st.markdown("""
    ## How to Use OMTRA Molecule Sampler
    
    ### 1. Set Parameters
    Use the sidebar to configure sampling parameters:
    - **Random Seed**: Set for reproducible results
    - **Number of Samples**: How many molecules to generate (1-100)
    - **Sampling Steps**: More steps = higher quality but slower (10-1000)
    - **Temperature**: Higher values increase diversity
    - **Guidance Scale**: Controls conditional guidance strength
    - **Conditioning Strength**: Controls conditioning signal strength
    
    ### 2. Upload Input Files (Optional)
    - Supported formats: SDF, CIF, MOL2, PDB
    - Maximum 3 files per job
    - Maximum 25MB per file
    - Files are used for conditioning/guidance
    
    ### 3. Submit Job
    Click "🚀 Run Sampling" to submit your job. Jobs typically complete in under 2 minutes.
    
    ### 4. View Results
    - Monitor job progress in the Jobs tab
    - Download individual files or all outputs as ZIP
    - View 3D molecular structures in the browser
    
    ### Supported File Formats
    - **SDF**: Structure Data Format (most common)
    - **PDB**: Protein Data Bank format
    - **MOL2**: Tripos MOL2 format
    - **CIF**: Crystallographic Information File
    
    ### Tips
    - Start with default parameters for best results
    - Use seed values for reproducible experiments
    - Higher temperature increases diversity but may reduce quality
    - Upload reference structures for guided generation
    """)


if __name__ == "__main__":
    main()

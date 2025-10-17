import streamlit as st
import requests
import time
import json
import io
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import sys
import os
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

# Import molecular visualization
try:
    import stmol
    import py3Dmol
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    st.warning(f"Molecular visualization not available. Error: {e}")
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
    
    def submit_job(self, params: Dict, upload_tokens: List[str], custom_job_id: Optional[str] = None) -> Optional[Dict]:
        """Submit a sampling job"""
        try:
            data = {
                "params": params,
                "uploads": upload_tokens
            }
            if custom_job_id:
                data["job_id"] = custom_job_id
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
    
    def download_input_file(self, job_id: str, filename: str) -> Optional[bytes]:
        """Download an input file"""
        try:
            response = requests.get(f"{self.base_url}/download/{job_id}/inputs/{filename}")
            response.raise_for_status()
            return response.content
        except Exception as e:
            return None
    
    def list_input_files(self, job_id: str) -> Optional[Dict]:
        """List all input files for a job"""
        try:
            response = requests.get(f"{self.base_url}/inputs/{job_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to list input files: {e}")
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

    def list_all_jobs(self) -> Optional[dict]:
        """List all jobs"""
        try:
            response = requests.get(f"{self.base_url}/jobs")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to list jobs: {e}")
            return None

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID"""
        try:
            response = requests.delete(f"{self.base_url}/jobs/{job_id}")
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Failed to delete job {job_id}: {e}")
            return False


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient(API_URL)
    
    if 'jobs' not in st.session_state:
        st.session_state.jobs = {}
        # Load existing jobs from backend on first initialization
        try:
            jobs_response = st.session_state.api_client.list_all_jobs()
            if jobs_response and 'jobs' in jobs_response:
                for job in jobs_response['jobs']:
                    job_id = job['job_id']
                    st.session_state.jobs[job_id] = job
        except Exception as e:
            # Silently fail if we can't load jobs (e.g., API not available)
            pass
    
    if 'upload_tokens' not in st.session_state:
        st.session_state.upload_tokens = []
    
    if 'selected_job' not in st.session_state:
        st.session_state.selected_job = None
    
    if 'selected_artifact' not in st.session_state:
        # Map job_id -> selected filename
        st.session_state.selected_artifact = {}


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
    
    # Sampling mode selection
    sampling_mode = st.sidebar.selectbox(
        "Sampling Mode",
        options=["Unconditional", "Pharmacophore-conditioned", "Protein-conditioned"],
        index=0,
        help="Choose the type of molecular generation"
    )
    
    params = {}
    params['sampling_mode'] = sampling_mode
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
    
    # File upload - conditional based on sampling mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("Input Files")
    
    if sampling_mode == "Unconditional":
        st.sidebar.info("No input files needed for unconditional generation")
        uploaded_files = []
    elif sampling_mode == "Pharmacophore-conditioned":
        uploaded_files = st.sidebar.file_uploader(
            "Upload pharmacophore files",
            type=['xyz'],
            accept_multiple_files=True,
            help="Upload pharmacophore files (XYZ format)"
        )
    elif sampling_mode == "Protein-conditioned":
        uploaded_files = st.sidebar.file_uploader(
            "Upload protein structures",
            type=['pdb', 'cif'],
            accept_multiple_files=True,
            help="Upload protein structure files (PDB/CIF format)"
        )
    else:
        uploaded_files = st.sidebar.file_uploader(
            "Upload molecular structures",
            type=['sdf', 'cif', 'mol2', 'pdb'],
            accept_multiple_files=True,
            help="Upload molecular structure files"
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
    
    # Custom job ID option
    st.sidebar.markdown("---")
    st.sidebar.subheader("Job Settings")
    
    use_custom_job_id = st.sidebar.checkbox(
        "Use custom job ID",
        value=False,
        help="Set a custom job ID instead of auto-generated one"
    )
    
    custom_job_id = None
    if use_custom_job_id:
        custom_job_id = st.sidebar.text_input(
            "Custom Job ID",
            value="",
            help="Enter a custom job ID (leave empty for auto-generated)",
            placeholder="my-custom-job-123"
        )
        if custom_job_id and not custom_job_id.strip():
            custom_job_id = None
    
    # Submit job button
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🚀 Run Sampling", type="primary", use_container_width=True):
        
        # Frontend guard: require input files for conditional modes
        if params.get('sampling_mode') in ["Pharmacophore-conditioned", "Protein-conditioned"] and not st.session_state.upload_tokens:
            st.sidebar.error("Please upload required input files before submitting this job.")
        else:
            # Submit job
            job_response = st.session_state.api_client.submit_job(params, st.session_state.upload_tokens, custom_job_id)
            
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
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Jobs")
    with col2:
        if st.button("🔄 Refresh", help="Reload jobs from server"):
            try:
                jobs_response = st.session_state.api_client.list_all_jobs()
                if jobs_response and 'jobs' in jobs_response:
                    st.session_state.jobs = {}
                    for job in jobs_response['jobs']:
                        job_id = job['job_id']
                        st.session_state.jobs[job_id] = job
                    st.success(f"Loaded {len(jobs_response['jobs'])} jobs")
                    st.rerun()
                else:
                    st.warning("No jobs found on server")
            except Exception as e:
                st.error(f"Failed to refresh jobs: {e}")
    
    if not st.session_state.jobs:
        st.info("No jobs submitted yet. Use the sidebar to start sampling!")
        return
    
    # Create job status table
    job_data = []
    # Build selector options (labels mapped to job ids)
    option_labels: List[str] = []
    option_job_ids: List[str] = []
    selector_index = 0
    for job_id, job_info in st.session_state.jobs.items():
        # Update job status
        status_response = st.session_state.api_client.get_job_status(job_id)
        if status_response:
            job_info.update(status_response)
        
        # Get elapsed seconds, ensuring it's not None
        elapsed_seconds = job_info.get('elapsed_seconds', 0)
        if elapsed_seconds is None:
            elapsed_seconds = 0
        
        # Populate selector option
        label = f"{job_id[:8]}... | {job_info.get('state', 'Unknown')}"
        option_labels.append(label)
        option_job_ids.append(job_id)
        if st.session_state.get('selected_job') == job_id:
            selector_index = len(option_labels) - 1

        job_data.append({
            'Job ID': job_id[:8] + '...',
            'Status': job_info.get('state', 'Unknown'),
            'Progress': f"{job_info.get('progress', 0)}%",
            'Samples': job_info.get('params', {}).get('n_samples', 'N/A'),
            'Elapsed': format_duration(elapsed_seconds),
            'Select': job_id
        })
    
    # Single-select dropdown to switch active job
    if True:
        choice = st.selectbox(
            "Select a job to view details",
            option_labels,
            index=min(selector_index, max(0, len(option_labels)-1)),
            key="job_select_dropdown",
        )
        # Map the chosen label back to its job id using the same index
        chosen_idx = option_labels.index(choice)
        new_selected_job = option_job_ids[chosen_idx]
        
        if st.session_state.get('selected_job') != new_selected_job:
            st.session_state.selected_job = new_selected_job
            st.rerun()

    # Bulk selection UI for deletion
    st.markdown("---")
    st.markdown("#### Manage Jobs")

    # Initialize state for bulk deletion confirmation
    if 'bulk_delete_pending_ids' not in st.session_state:
        st.session_state.bulk_delete_pending_ids = []
    if 'show_bulk_delete_confirm' not in st.session_state:
        st.session_state.show_bulk_delete_confirm = False

    selected_labels = st.multiselect(
        "Select jobs to delete",
        option_labels,
        default=[],
        key="job_bulk_select",
        help="Pick one or more jobs"
    )
    selected_ids = [option_job_ids[option_labels.index(lbl)] for lbl in selected_labels] if selected_labels else []

    def delete_jobs(job_ids: list[str]) -> int:
        deleted = 0
        for jid in job_ids:
            ok = st.session_state.api_client.delete_job(jid)
            if ok:
                st.session_state.jobs.pop(jid, None)
                if st.session_state.get('selected_job') == jid:
                    st.session_state.selected_job = None
                deleted += 1
        return deleted

    # Trigger confirmation step and persist selection
    if st.button("🗑️ Delete selected", disabled=len(selected_ids) == 0, help="Delete chosen jobs"):
        if len(selected_ids) == 0:
            st.warning("No jobs selected")
        else:
            st.session_state.bulk_delete_pending_ids = selected_ids
            st.session_state.show_bulk_delete_confirm = True
            st.rerun()

    # Render confirmation step if active
    if st.session_state.show_bulk_delete_confirm:
        pending = st.session_state.bulk_delete_pending_ids
        st.warning(f"Are you sure you want to delete {len(pending)} job(s)? This cannot be undone.")
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            if st.button("Confirm delete", type="primary", key="confirm_bulk_delete"):
                deleted = delete_jobs(pending)
                # Clear state and refresh list from server to stay in sync
                st.session_state.bulk_delete_pending_ids = []
                st.session_state.show_bulk_delete_confirm = False
                try:
                    jobs_response = st.session_state.api_client.list_all_jobs()
                    if jobs_response and 'jobs' in jobs_response:
                        st.session_state.jobs = {job['job_id']: job for job in jobs_response['jobs']}
                except Exception:
                    pass
                st.success(f"Deleted {deleted} job(s)")
                st.rerun()
        with col_c2:
            if st.button("Cancel", key="cancel_bulk_delete"):
                st.session_state.bulk_delete_pending_ids = []
                st.session_state.show_bulk_delete_confirm = False
                st.rerun()

    if job_data:
        df = pd.DataFrame(job_data)
        
        # Display table (read-only, selection via dropdown above)
        st.dataframe(
            df[['Job ID', 'Status', 'Progress', 'Samples', 'Elapsed']],
            use_container_width=True,
            hide_index=True
        )


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
        # Download ZIP content and show download button in one go
        zip_content = st.session_state.api_client.download_all_outputs(job_id)
        if zip_content:
            st.download_button(
                label="📦 Download All",
                data=zip_content,
                file_name=f"{job_id}_outputs.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"zip_download_{job_id}"
            )
        else:
            st.error("Failed to prepare download")
    
    # Sort artifacts: summary.json first, then SDF files in numerical order
    def sort_key(artifact):
        filename = artifact['filename']
        if filename == 'summary.json':
            return (0, '')  # First priority
        elif filename.startswith('sample_') and filename.endswith('.sdf'):
            # Extract number from sample filename
            try:
                num = int(filename.split('_')[1].split('.')[0])
                return (1, num)  # Second priority, sorted by number
            except (ValueError, IndexError):
                return (2, filename)  # Fallback to filename
        else:
            return (2, filename)  # Other files last
    
    sorted_artifacts = sorted(artifacts, key=sort_key)
    
    # Unified viewer layout: list on the left, big viewer on the right
    left, right = st.columns([1, 3], gap="large")

    with left:
        labels = []
        filenames = []
        for art in sorted_artifacts:
            label = f"{art['filename']} ({format_file_size(art['size'])})"
            labels.append(label)
            filenames.append(art['filename'])

        # Default to first item if not set for this job
        current = st.session_state.selected_artifact.get(job_id, filenames[0]) if filenames else None
        default_index = filenames.index(current) if current in filenames else 0

        choice = st.selectbox(
            "Files",
            options=labels,
            index=default_index,
            key=f"files_select_{job_id}"
        )
        chosen_idx = labels.index(choice)
        chosen_filename = filenames[chosen_idx]
        
        # Update selection and trigger rerun if changed
        if st.session_state.selected_artifact.get(job_id) != chosen_filename:
            st.session_state.selected_artifact[job_id] = chosen_filename
            st.rerun()

        # Download button for the selected file
        file_content = st.session_state.api_client.download_file(job_id, chosen_filename)
        if file_content:
            st.download_button(
                label="💾 Download Selected",
                data=file_content,
                file_name=chosen_filename,
                mime="application/octet-stream",
                use_container_width=True,
                key=f"file_download_{job_id}"
            )
        else:
            st.error("Failed to download file")

    with right:
        # Render selected item: JSON pretty view or 3D molecule
        if chosen_filename == 'summary.json':
            content = st.session_state.api_client.download_file(job_id, chosen_filename)
            if content:
                try:
                    st.json(json.loads(content.decode('utf-8', errors='ignore')))
                except Exception:
                    st.code(content.decode('utf-8', errors='ignore'))
        else:
            if VISUALIZATION_AVAILABLE and chosen_filename.split('.')[-1].lower() in ['sdf', 'mol2', 'pdb']:
                render_molecular_viewer(job_id, chosen_filename)
            else:
                st.info("Select an SDF/MOL2/PDB file to view in 3D.")


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
        
        # Build py3Dmol view and render via stmol
        view = py3Dmol.view(width=600, height=400)
        if file_format == 'sdf':
            view.addModel(content_str, 'sdf')
        elif file_format == 'mol2':
            view.addModel(content_str, 'mol2')
        elif file_format == 'pdb':
            view.addModel(content_str, 'pdb')
        view.setStyle({'stick': {}})
        
        # Add pharmacophore spheres if this is a pharmacophore-conditioned job
        job_info = st.session_state.jobs.get(job_id, {})
        sampling_mode = job_info.get('params', {}).get('sampling_mode')
        
        if sampling_mode == 'Pharmacophore-conditioned':
            add_pharmacophore_spheres(view, job_id)
        
        # Add protein surface overlay for Protein-conditioned jobs
        if sampling_mode == 'Protein-conditioned':
            add_protein_surface(view, job_id)
        
        view.zoomTo()
        stmol.showmol(view, height=400, width=600)
        
        if sampling_mode == 'Pharmacophore-conditioned':
            
            # Show pharmacophore color legend
            with st.expander("Pharmacophore Color Legend"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("🟠 **Aromatic** - Orange")
                    st.markdown("🔵 **Hydrogen Donor** - Blue")
                    st.markdown("🔴 **Hydrogen Acceptor** - Red")
                with col2:
                    st.markdown("🟣 **Positive Ion** - Purple")
                    st.markdown("🟡 **Negative Ion** - Yellow")
                with col3:
                    st.markdown("🟢 **Hydrophobic** - Green")
                    st.markdown("🔷 **Halogen** - Cyan")
        
    except Exception as e:
        st.error(f"Failed to render molecule: {e}")


def add_pharmacophore_spheres(view, job_id: str):
    """Add pharmacophore spheres to the 3D viewer"""
    try:
        # Get list of input files for this job
        input_files_response = st.session_state.api_client.list_input_files(job_id)
        if not input_files_response:
            return
        
        input_files = input_files_response.get('files', [])
        
        # Look for XYZ files in the input files
        pharmacophore_data = None
        xyz_files = [f for f in input_files if f['extension'] == '.xyz']
        
        for xyz_file in xyz_files:
            filename = xyz_file['filename']
            try:
                # Try to download the XYZ file
                pharm_content = st.session_state.api_client.download_input_file(job_id, filename)
                if pharm_content:
                    pharmacophore_data = pharm_content.decode('utf-8', errors='ignore')
                    break
            except Exception as e:
                continue
        
        if not pharmacophore_data:
            return
        
        # Parse XYZ file and add spheres
        from shared.file_utils import parse_xyz_atom_lines
        atom_lines, _ = parse_xyz_atom_lines(pharmacophore_data)
        
        element_to_pharm_type = {
            'P': 'Aromatic',
            'S': 'HydrogenDonor',
            'F': 'HydrogenAcceptor',
            'N': 'PositiveIon',
            'O': 'NegativeIon',
            'C': 'Hydrophobic',
            'Cl': 'Halogen'
        }
        
        # Pharmacophore type to color mapping
        pharm_colors = {
            'Aromatic': 'orange',
            'HydrogenDonor': 'blue', 
            'HydrogenAcceptor': 'red',
            'PositiveIon': 'purple',
            'NegativeIon': 'yellow',
            'Hydrophobic': 'green',
            'Halogen': 'cyan'
        }
        
        for line in atom_lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            
            element_symbol = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            # Map element symbol to pharmacophore type
            pharm_type = element_to_pharm_type.get(element_symbol, 'Unknown')
            
            color = pharm_colors.get(pharm_type, 'gray')
            
            view.addSphere({
                'center': {'x': x, 'y': y, 'z': z},
                'radius': 1,
                'color': color,
                'alpha': 0.7
            })
            
            
    except Exception as e:
        st.warning(f"Could not load pharmacophore data: {e}")


def add_protein_surface(view, job_id: str):
    """Add a semi-transparent protein surface for Protein-conditioned jobs."""
    try:
        # Get list of input files for this job
        input_files_response = st.session_state.api_client.list_input_files(job_id)
        if not input_files_response:
            return
        
        input_files = input_files_response.get('files', [])
        
        # Look for protein structure files (PDB/CIF)
        prot_files = [f for f in input_files if f['extension'] in ['.pdb', '.cif']]
        if not prot_files:
            return
        
        # Use the first protein file
        protein_name = prot_files[0]['filename']
        protein_bytes = st.session_state.api_client.download_input_file(job_id, protein_name)
        if not protein_bytes:
            return
        
        protein_str = protein_bytes.decode('utf-8', errors='ignore')
        # Add protein model as second model (after ligand)
        file_ext = protein_name.split('.')[-1].lower()
        if file_ext == 'pdb':
            view.addModel(protein_str, 'pdb')
        elif file_ext == 'cif':
            view.addModel(protein_str, 'cif')
        else:
            return
        
        # Define a pocket around the ligand (model 0) within 6 Å on the protein model (model 1)
        pocket_selection = { 'model': 1, 'within': { 'distance': 6.0, 'sel': { 'model': 0 } } }
        
        # Add semi-transparent surface for only the pocket region
        try:
            view.addSurface(py3Dmol.VDW, { 'opacity': 0.6, 'color': 'white' }, pocket_selection)
        except Exception:
            # Fallback: cartoon for the pocket region
            view.setStyle(pocket_selection, { 'cartoon': { 'color': 'white', 'opacity': 0.4 } })
        
        # Optionally, hide the rest of the protein (no style applied)
        # This keeps the visualization focused on the pocket only
    except Exception:
        # Non-fatal; keep rendering ligand if pocket surface fails
        pass


def format_duration(seconds: float) -> str:
    """Format duration in human-readable form"""
    # Handle None values
    if seconds is None:
        return "0.0s"
    
    # Convert to float if it's not already
    try:
        seconds = float(seconds)
    except (ValueError, TypeError):
        return "0.0s"
    
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
    
    # Proactively refresh job statuses before deciding on auto-refresh
    if st.session_state.jobs:
        try:
            for _job_id in list(st.session_state.jobs.keys()):
                latest = st.session_state.api_client.get_job_status(_job_id)
                if latest:
                    st.session_state.jobs[_job_id].update(latest)
        except Exception:
            pass
    
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
    - **Random Seed**: Set for reproducible results (optional)
    - **Number of Samples**: How many molecules to generate (1-100)
    - **Sampling Steps**: More steps = higher quality but slower (10-1000)
    - **Sampling Mode**: Choose between Unconditional, Pharmacophore-conditioned, or Protein-conditioned generation
    
    ### 2. Upload Input Files (Optional)
    - **Pharmacophore-conditioned**: XYZ files
    - **Protein-conditioned**: PDB, CIF files
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
    - **XYZ**: XYZ coordinate format (for pharmacophores)
    - **SDF**: Structure Data Format (most common)
    - **PDB**: Protein Data Bank format
    - **MOL2**: Tripos MOL2 format
    - **CIF**: Crystallographic Information File
    
    ### Tips
    - Start with default parameters for best results
    - Use seed values for reproducible experiments
    - More sampling steps increase quality but take longer
    - Upload reference structures for guided generation
    """)


if __name__ == "__main__":
    main()

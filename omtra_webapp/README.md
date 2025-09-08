# OMTRA Molecule Sampler Web App

A lightweight web application for molecular generation using the OMTRA deep learning model. Upload molecular structures, configure sampling parameters, and generate novel molecules through an intuitive web interface.

## 🌟 Features

- **Interactive Web Interface**: Clean Streamlit UI with molecular visualization
- **File Upload & Validation**: Support for SDF, CIF, MOL2, and PDB formats
- **Real-time Job Monitoring**: Live progress tracking with automatic updates  
- **3D Molecular Visualization**: Interactive py3Dmol viewer for results
- **Batch Processing**: Generate multiple molecules per job
- **Secure File Handling**: Comprehensive validation and sanitization
- **Containerized Deployment**: Docker-based architecture for easy deployment

## 🏗️ Architecture

### Components

- **Frontend** (`frontend/`): Streamlit web application
- **API** (`api/`): FastAPI backend service  
- **Worker** (`worker/`): RQ-based job processing
- **Shared** (`shared/`): Common utilities and models
- **Infrastructure** (`infra/`): Deployment configurations

### Technology Stack

- **Frontend**: Streamlit, py3Dmol, stmol
- **Backend**: FastAPI, Redis, RQ (Redis Queue)
- **Worker**: PyTorch, RDKit (optional)
- **Containerization**: Docker, Docker Compose
- **Reverse Proxy**: Nginx (production)

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with Docker GPU support (for model inference)
- At least 8GB RAM and 2GB GPU memory

### Development Setup

1. **Clone and Navigate**
   ```bash
   cd omtra_webapp
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start Services**
   ```bash
   docker-compose up --build
   ```

4. **Access Application**
   - Web App: http://localhost:8501
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Production Deployment

1. **Update Environment**
   ```bash
   # Set production values in .env
   API_SECRET_KEY=your-secure-secret-key
   LOG_LEVEL=INFO
   ```

2. **Deploy with GPU Support**
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

3. **Setup Nginx (Optional)**
   ```bash
   # Copy nginx configuration
   cp infra/nginx.conf /etc/nginx/sites-available/omtra
   ln -s /etc/nginx/sites-available/omtra /etc/nginx/sites-enabled/
   systemctl reload nginx
   ```

4. **Setup Cleanup Cron Job**
   ```bash
   chmod +x infra/cleanup_jobs.sh
   # Add to crontab (see infra/crontab.example)
   ```

## 📖 User Guide

### Basic Workflow

1. **Configure Parameters**
   - Set random seed for reproducibility
   - Choose number of samples (1-100)
   - Adjust sampling steps and temperature
   - Configure guidance parameters

2. **Upload Input Files** (Optional)
   - Drag and drop molecular structure files
   - Supported: .sdf, .cif, .mol2, .pdb
   - Maximum 3 files, 25MB each

3. **Submit Job**
   - Click "🚀 Run Sampling"
   - Jobs complete in ~2 minutes
   - Monitor progress in real-time

4. **View Results**
   - Browse generated molecules
   - Interactive 3D visualization
   - Download individual files or ZIP archive

### Parameter Guide

| Parameter | Description | Range | Default |
|-----------|-------------|--------|---------|
| Seed | Random seed for reproducibility | 0-2³¹ | 42 |
| Samples | Number of molecules to generate | 1-100 | 10 |
| Steps | Sampling steps (quality vs speed) | 10-1000 | 100 |
| Temperature | Diversity control | 0.1-2.0 | 1.0 |
| Guidance Scale | Conditional guidance strength | 0.0-10.0 | 1.0 |
| Conditioning | Conditioning signal strength | 0.0-2.0 | 1.0 |

## 🔧 Development

### Project Structure

```
omtra_webapp/
├── frontend/           # Streamlit web app
│   ├── app.py         # Main application
│   ├── requirements.txt
│   └── Dockerfile
├── api/               # FastAPI backend
│   ├── main.py        # API endpoints
│   ├── requirements.txt
│   └── Dockerfile
├── worker/            # Job processing
│   ├── worker.py      # RQ worker
│   ├── requirements.txt
│   └── Dockerfile
├── shared/            # Common utilities
│   ├── models.py      # Pydantic models
│   ├── file_utils.py  # File handling
│   └── logging_utils.py
├── tests/             # Test suite
│   ├── test_shared.py
│   └── test_api.py
├── infra/             # Infrastructure
│   ├── nginx.conf
│   ├── cleanup_jobs.sh
│   └── crontab.example
├── docker-compose.yml
└── .env.example
```

### Running Tests

```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Run unit tests
pytest tests/test_shared.py -v

# Run API integration tests (requires running services)
pytest tests/test_api.py -v

# Run all tests
pytest tests/ -v
```

### Adding Real Model Integration

The current implementation includes a stub sampler for testing. To integrate your OMTRA model:

1. **Update Worker Dependencies**
   ```bash
   # Add your model dependencies to worker/requirements.txt
   echo "your-model-package>=1.0.0" >> worker/requirements.txt
   ```

2. **Implement Model Loading**
   ```python
   # In worker/worker.py, update load_omtra_model()
   def load_omtra_model():
       from omtra.models import YourModel
       model = YourModel.load_pretrained(MODEL_CHECKPOINT_PATH)
       return model
   ```

3. **Implement Sampling Logic**
   ```python
   # In worker/worker.py, update run_omtra_sampler()
   def run_omtra_sampler(job_id, params, input_files, model, job_logger):
       # Your sampling implementation
       results = model.sample(**params)
       # Save results to outputs directory
       return summary
   ```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/upload/init` | Initialize file upload |
| POST | `/upload/{token}` | Upload file |
| POST | `/sample` | Submit sampling job |
| GET | `/status/{job_id}` | Get job status |
| GET | `/result/{job_id}` | Get job results |
| GET | `/download/{job_id}/{file}` | Download output file |
| GET | `/download/{job_id}/all` | Download all outputs (ZIP) |
| GET | `/logs/{job_id}` | Get job logs |
| DELETE | `/job/{job_id}` | Cancel job |

## 🔒 Security

### File Safety Checklist

- ✅ Extension whitelist (`.sdf`, `.cif`, `.mol2`, `.pdb`)
- ✅ MIME type validation
- ✅ File size limits (25MB per file, 3 files max)
- ✅ Filename sanitization
- ✅ Content validation with RDKit
- ✅ Temporary file cleanup
- ✅ Secure file permissions
- ✅ SHA256 hash logging
- ✅ No execution of user content

### Production Security

1. **Use HTTPS**
   ```nginx
   # Add SSL configuration to nginx
   listen 443 ssl;
   ssl_certificate /path/to/cert.pem;
   ssl_certificate_key /path/to/key.pem;
   ```

2. **Set Strong Secrets**
   ```bash
   # Generate secure API key
   openssl rand -hex 32
   ```

3. **Enable Rate Limiting**
   ```nginx
   # Add to nginx configuration
   limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
   limit_req zone=api burst=5;
   ```

4. **Monitor Logs**
   ```bash
   # Set up log rotation
   logrotate -d /etc/logrotate.d/omtra
   ```

## 📊 Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Redis connection
redis-cli ping

# Worker status
docker logs omtra_worker
```

### Metrics Collection

Key metrics to monitor:
- Job completion rate
- Average processing time
- Queue depth
- Disk usage (`/srv/app/jobs`)
- GPU memory utilization
- Error rates

### Log Analysis

```bash
# View structured logs
docker logs omtra_api | jq '.message'

# Monitor job events
docker logs omtra_worker | grep "job_event"

# Check error logs
docker logs omtra_api | jq 'select(.level=="ERROR")'
```

## 🐛 Troubleshooting

### Common Issues

**"API Unavailable"**
- Check if API container is running: `docker ps`
- Verify Redis connection: `redis-cli ping`
- Check API logs: `docker logs omtra_api`

**Jobs Stuck in Queue**
- Check worker status: `docker logs omtra_worker`
- Verify GPU availability: `nvidia-smi`
- Restart worker: `docker restart omtra_worker`

**File Upload Fails**
- Check file size and format
- Verify disk space: `df -h`
- Check API logs for validation errors

**Out of Memory**
- Reduce batch size (`n_samples`)
- Monitor GPU memory: `nvidia-smi`
- Check container memory limits

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG docker-compose up

# Access container shells
docker exec -it omtra_api bash
docker exec -it omtra_worker bash
```

## 📝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature-name`
3. **Add tests** for new functionality
4. **Ensure tests pass**: `pytest tests/`
5. **Submit pull request**

### Code Style

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Format code
black .
ruff check . --fix
```

## 📄 License

This project is part of the OMTRA repository and follows the same license terms.

## 🙏 Acknowledgments

- OMTRA team for the underlying molecular generation model
- RDKit for molecular structure validation
- py3Dmol for 3D visualization capabilities
- Streamlit team for the excellent web framework

---

For issues and feature requests, please use the GitHub issue tracker.

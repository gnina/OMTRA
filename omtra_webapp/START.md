# Starting OMTRA Web Application

The webapp is configured to use pre-built Docker images. To start the services:

## Quick Start

```bash
cd omtra_webapp
docker-compose up -d
```

## Images

The docker-compose.yml is configured to pull these images from Docker Hub:
- `gnina/omtra_webapp-api:latest` - API service
- `gnina/omtra_webapp-worker:latest` - Worker service
- `gnina/omtra_webapp-frontend-react:latest` - Frontend service  
- `redis:7-alpine` - Redis service

## Environment Variables

Most variables don't need to be changed. See `.env.example` for all options.

**Variables you might want to customize:**

- `FRONTEND_PORT` - Port for frontend (default: 5900)
- `CUDA_VISIBLE_DEVICES` - GPU device IDs (default: 0)
- `WORKER_TIMEOUT` - Maximum job execution time in seconds (default: 600 = 10 minutes)
- `JOB_TTL_HOURS` - How long to keep job data before cleanup (default: 48 hours)

## Accessing the Webapp

Once started, access the webapp at:
- Frontend: http://localhost:5900 (or the port specified by FRONTEND_PORT)

## Stopping the Services

```bash
docker-compose down
```




# Starting OMTRA Web Application

The webapp is configured to use pre-built Docker images. To start the services:

## Quick Start

```bash
cd omtra_webapp
docker-compose up -d
```

## Images

The docker-compose.yml is configured to use these local images:
- `omtra_webapp-api:latest` - API service
- `omtra_webapp-worker:latest` - Worker service
- `omtra_webapp-frontend-react:latest` - Frontend service  
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

## Building from Source (if needed)

To build from source instead of using pre-built images:
1. Comment out the `image:` lines in docker-compose.yml
2. Uncomment the `build:` sections
3. Uncomment the source code volume mounts (e.g., `./api:/app`, `./worker:/app`, etc.)
4. Run `docker-compose build` then `docker-compose up -d`


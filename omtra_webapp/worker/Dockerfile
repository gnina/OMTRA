FROM ubuntu:22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    gnupg2 \
    software-properties-common \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set up conda environment (mounted from host)
ENV PATH="/opt/conda/bin:$PATH"
ENV CONDA_DEFAULT_ENV=omtra
ENV CONDA_PREFIX=/opt/conda/envs/omtra
ENV PATH="/opt/conda/envs/omtra/bin:$PATH"

# Copy application code
COPY . .

# Install webapp dependencies will be done at runtime

# Create directories
RUN mkdir -p /srv/app/jobs /srv/app/models

# Set permissions
RUN chmod -R 755 /app
RUN chmod -R 777 /srv/app/jobs

# Change ownership to appuser
RUN chown -R appuser:appuser /app /srv/app

# Switch to non-root user
USER appuser

# Run the worker
CMD ["/bin/bash", "start_worker.sh"]

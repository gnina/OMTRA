# OMTRA CLI Docker Image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Set working directory
WORKDIR /workspace

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy requirements file
COPY requirements-cuda.txt /workspace/

# Install CUDA-enabled dependencies
RUN uv pip install --system -r requirements-cuda.txt

# Copy project files
COPY pyproject.toml /workspace/
COPY omtra /workspace/omtra/
COPY omtra_pipelines /workspace/omtra_pipelines/
COPY configs /workspace/configs/
COPY routines /workspace/routines/
COPY cli.py /workspace/

RUN uv pip install --system -e .

# Create entrypoint script that runs cli.py
RUN echo '#!/bin/bash\npython /workspace/cli.py "$@"' > /usr/local/bin/omtra && \
    chmod +x /usr/local/bin/omtra

ENTRYPOINT ["python", "/workspace/cli.py"]




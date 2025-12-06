# Multi-stage build: builder stage with build tools
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

WORKDIR /build

# Install system dependencies and Python 3.11 with build tools
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
       python3.11 \
       python3.11-dev \
       python3-pip \
       wget \
       curl \
       git \
       gnupg2 \
       build-essential \
       libnetcdf-dev \
       libhdf5-dev \
       libxrender1 \
       libxext6 \
       libsm6 \
       libxrandr2 \
       libxi6 \
       libfontconfig1 \
       libxss1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install uv package manager
RUN pip install --no-cache-dir uv

# Copy OMTRA source code first (needed for installation)
COPY ./pyproject.toml ./requirements-cuda.txt /tmp/omtra/
COPY ./omtra/ /tmp/omtra/omtra/

# Install OMTRA dependencies first (CUDA packages - torch, dgl, etc.)
RUN cd /tmp/omtra && uv pip install --system --no-cache -r requirements-cuda.txt

# Install OMTRA package itself
RUN cd /tmp/omtra && uv pip install --system --no-cache . && \
    mkdir -p /usr/local/lib/python3.11/dist-packages/omtra/constants && \
    cp -f /tmp/omtra/omtra/constants/*.pkl /usr/local/lib/python3.11/dist-packages/omtra/constants/ || true

# Install Python runtime dependencies for the worker
RUN uv pip install --system --no-cache \
    redis async-timeout rq python-dotenv pyyaml hydra-core omegaconf pandas requests aiofiles pydantic \
    posebusters posecheck gemmi MDAnalysis netCDF4 natsort \
    jaraco.classes jaraco.context jaraco.functools jaraco.text

# Try to install optional interaction backends
RUN uv pip install --system --no-cache hydride || true
RUN uv pip install --system --no-cache hydrite || true

# Download and install GNINA v1.3.2 binaries
RUN mkdir -p /usr/local/bin && \
    wget -q https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2 -O /tmp/gnina.1.3.2 && \
    chmod +x /tmp/gnina.1.3.2 && \
    cp /tmp/gnina.1.3.2 /usr/local/bin/gnina.1.3.2 && \
    wget -q https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8 -O /tmp/gnina.1.3.2.cuda12.8 && \
    chmod +x /tmp/gnina.1.3.2.cuda12.8 && \
    cp /tmp/gnina.1.3.2.cuda12.8 /usr/local/bin/gnina.1.3.2.cuda12.8 && \
    rm -f /tmp/gnina.1.3.2 /tmp/gnina.1.3.2.cuda12.8

# remove dependencies not needed for webapp runtime
RUN uv pip uninstall --system --yes \
    jupyterlab jupyter jupyter-client jupyter-core jupyter-server jupyter-events jupyter-lsp \
    matplotlib seaborn plotly \
    wandb \
    py3Dmol \
    mols2grid \
    umap-learn pynndescent \
    duckdb \
    peppr plinder \
    pyarrow \
    ipython ipykernel ipywidgets \
    notebook \
    || true

# Clean up Python cache, test directories, and unnecessary files BEFORE copying
# This reduces the size of the COPY layer to stay under GitHub's 10GB per-layer limit
RUN find /usr/local -name '*.pyc' -delete && \
    find /usr/local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    # Keep all .dist-info - many packages (rdkit, py3Dmol, etc.) need metadata
    # find /usr/local/lib/python3.11/dist-packages -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "demos" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pyx" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pxd" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pxi" -delete 2>/dev/null || true && \
    find /usr/local -name '*.a' -delete 2>/dev/null || true && \
    find /usr/local -name "*.so.debug" -delete 2>/dev/null || true && \
    # Don't strip .so files - can corrupt scipy, torchvision, and other libraries
    # find /usr/local/lib/python3.11/dist-packages -name "*.so" -exec strip {} \; 2>/dev/null || true && \
    # Remove pip, setuptools, wheel, uv from builder (not needed in final image)
    rm -rf /usr/local/lib/python3.11/dist-packages/pip* \
    /usr/local/lib/python3.11/dist-packages/setuptools* \
    /usr/local/lib/python3.11/dist-packages/wheel* \
    /usr/local/lib/python3.11/dist-packages/uv* \
    /usr/local/bin/pip* \
    /usr/local/bin/easy_install* \
    /usr/local/bin/uv*

# Runtime stage: minimal image without build tools
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install only minimal runtime dependencies (no build tools, no pip)
# Use --no-install-recommends to avoid unnecessary packages
# Keep wget for GNINA binary downloads if needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-minimal \
    wget \
    libnetcdf-dev \
    libhdf5-dev \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxrandr2 \
    libxi6 \
    libfontconfig1 \
    libxss1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    rm -rf /var/lib/apt/lists/* \
    /usr/share/doc \
    /usr/share/man \
    /usr/share/locale \
    /usr/share/info \
    /usr/share/lintian \
    /usr/share/common-licenses \
    /var/cache/apt/archives/*.deb \
    /var/cache/apt/archives/partial/*.deb \
    /var/cache/apt/*.bin \
    /usr/lib/python3.11/test \
    /usr/lib/python3.11/idle_test \
    /usr/lib/python3.11/distutils/tests \
    && apt-get clean && \
    find /usr/lib/python3.11 -name "test_*.py" -delete 2>/dev/null || true && \
    find /usr/lib/python3.11 -name "*_test.py" -delete 2>/dev/null || true

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy packages directly from builder (tar compression doesn't help with Docker layer size)
# Split into separate COPYs to potentially reduce individual layer size
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Final cleanup and optimization in runtime stage
# Keep all .dist-info - many packages (rdkit, py3Dmol, etc.) need metadata
# RUN find /usr/local/lib/python3.11/dist-packages -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
RUN find /usr/local/lib/python3.11/dist-packages -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "demos" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pyx" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pxd" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pxi" -delete 2>/dev/null || true && \
    # Compile Python to bytecode (but don't remove .py files - some packages need them)
    python3 -m compileall -q /usr/local/lib/python3.11/dist-packages 2>/dev/null || true

# Copy application code (build context is repo root)
COPY ./omtra_webapp/worker/ /app/
COPY ./omtra_webapp/shared/ /app/shared/
# Copy cli.py and routines directory needed by worker for sampling
COPY ./cli.py /app/
COPY ./routines/ /app/routines/

# Create directories and set permissions
RUN mkdir -p /srv/app/jobs /srv/app/models && \
    chmod -R 755 /app && \
    chmod -R 777 /srv/app/jobs && \
    chown -R appuser:appuser /app /srv/app

# Switch to non-root user
USER appuser

# Run the worker
CMD ["/bin/bash", "start_worker.sh"]

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11 and build tools
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
       python3.11 \
       python3.11-dev \
       python3-pip \
       gcc \
       g++ \
       curl \
       wget \
       unzip \
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

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /build

COPY omtra_webapp/api/requirements.txt .

RUN pip install --no-cache-dir --force-reinstall --no-deps "numpy>=1.21.0,<2.0.0" && \
    pip install --no-cache-dir "scipy>=1.7.0,<1.14.0"

RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p /usr/local/lib/python3.11/dist-packages/omtra/utils
COPY omtra/utils/checkpoints.py /usr/local/lib/python3.11/dist-packages/omtra/utils/
COPY omtra/utils/__init__.py /usr/local/lib/python3.11/dist-packages/omtra/utils/
RUN touch /usr/local/lib/python3.11/dist-packages/omtra/__init__.py

# clean up any remaining unused packages
RUN pip uninstall --yes \
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

# Clean up Python cache, test directories, and unnecessary files
RUN find /usr/local -name '*.pyc' -delete && \
    find /usr/local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    # Keep .dist-info for rdkit and other packages that need metadata
    # find /usr/local/lib/python3.11/dist-packages -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "demos" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pyx" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pxd" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pxi" -delete 2>/dev/null || true && \
    find /usr/local -name "*.so.debug" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.so" ! -name "*openblas*" ! -name "*blas*" -exec strip {} \; 2>/dev/null || true && \
    # Remove pip, setuptools, wheel from builder (not needed in final image)
    rm -rf /usr/local/lib/python3.11/dist-packages/pip* \
    /usr/local/lib/python3.11/dist-packages/setuptools* \
    /usr/local/lib/python3.11/dist-packages/wheel* \
    /usr/local/bin/pip* \
    /usr/local/bin/easy_install*

# Runtime stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-minimal \
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

WORKDIR /app

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Copy packages directly from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Final cleanup and optimization in runtime stage
RUN find /usr/local/lib/python3.11/dist-packages -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -type d -name "demos" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pyx" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pxd" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.11/dist-packages -name "*.pxi" -delete 2>/dev/null || true && \
    # Compile Python to bytecode (but don't remove .py files - some packages need them)
    python3 -m compileall -q /usr/local/lib/python3.11/dist-packages 2>/dev/null || true

# Copy application code (paths relative to repo root build context)
COPY omtra_webapp/api/ /app/
COPY omtra_webapp/shared/ /app/shared/

# Create directories and set permissions
RUN mkdir -p /srv/app/jobs /srv/app/models /srv/app/checkpoints && \
    chmod -R 755 /app && \
    chmod -R 777 /srv/app/jobs && \
    chown -R appuser:appuser /app /srv/app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120", "--timeout-graceful-shutdown", "30"]

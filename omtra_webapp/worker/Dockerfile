FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System deps + Python 3.11 (add deadsnakes PPA manually to avoid Launchpad API timeout)
RUN apt-get update && apt-get install -y \
       software-properties-common ca-certificates gpg \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 \
    && echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y \
       python3.11 python3.11-dev python3-pip \
       wget curl git build-essential gnupg2 \
       libnetcdf-dev libhdf5-dev \
       libxrender1 libxext6 libsm6 libxrandr2 libxi6 libfontconfig1 libxss1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# CUDA deps (torch, dgl, etc.)
COPY ./pyproject.toml ./requirements-cuda.txt /tmp/omtra/
COPY ./omtra/ /tmp/omtra/omtra/
RUN cd /tmp/omtra && uv pip install --system --no-cache -r requirements-cuda.txt

# OMTRA package
RUN cd /tmp/omtra && uv pip install --system --no-cache . && \
    mkdir -p /usr/local/lib/python3.11/dist-packages/omtra/constants && \
    cp -f /tmp/omtra/omtra/constants/*.pkl /usr/local/lib/python3.11/dist-packages/omtra/constants/ 2>/dev/null || true

# Pin numpy<2 for pandas 2.0.0 compatibility (required by posecheck)
RUN uv pip install --system --no-cache "numpy<2"

# Worker runtime deps
RUN uv pip install --system --no-cache \
    redis async-timeout rq python-dotenv pyyaml hydra-core omegaconf \
    pandas requests aiofiles pydantic \
    posebusters posecheck gemmi MDAnalysis netCDF4 natsort \
    jaraco.classes jaraco.context jaraco.functools jaraco.text

# Optional interaction backends
RUN uv pip install --system --no-cache hydride 2>/dev/null || true
RUN uv pip install --system --no-cache hydrite 2>/dev/null || true

# GNINA binaries
RUN wget -q https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2 -O /usr/local/bin/gnina.1.3.2 && \
    chmod +x /usr/local/bin/gnina.1.3.2 && \
    wget -q https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8 -O /usr/local/bin/gnina.1.3.2.cuda12.8 && \
    chmod +x /usr/local/bin/gnina.1.3.2.cuda12.8

# Application code
COPY ./omtra_webapp/worker/ /app/
COPY ./omtra_webapp/shared/ /app/shared/
COPY ./cli.py /app/
COPY ./routines/ /app/routines/

RUN mkdir -p /srv/app/jobs /srv/app/models && rm -rf /tmp/omtra

CMD ["/bin/bash", "start_worker.sh"]

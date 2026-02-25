FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System deps + Python 3.11 (add deadsnakes PPA manually to avoid Launchpad API timeout)
RUN apt-get update && apt-get install -y \
       software-properties-common ca-certificates gpg \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 \
    && echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes.list \
    && apt-get update && apt-get install -y \
       python3.11 python3.11-dev python3-pip \
       gcc g++ \
       libxrender1 libxext6 libsm6 libxrandr2 libxi6 libfontconfig1 libxss1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin numpy<2 for rdkit compatibility, then install deps
COPY omtra_webapp/api/requirements.txt .
RUN pip install --no-cache-dir "numpy>=1.21,<2" && \
    pip install --no-cache-dir -r requirements.txt

# Copy just the omtra utils needed by the API (not the full package)
RUN mkdir -p /usr/local/lib/python3.11/dist-packages/omtra/utils
COPY omtra/utils/checkpoints.py /usr/local/lib/python3.11/dist-packages/omtra/utils/
COPY omtra/utils/__init__.py /usr/local/lib/python3.11/dist-packages/omtra/utils/
RUN touch /usr/local/lib/python3.11/dist-packages/omtra/__init__.py

# Application code
COPY omtra_webapp/api/ /app/
COPY omtra_webapp/shared/ /app/shared/

RUN mkdir -p /srv/app/jobs /srv/app/models /srv/app/checkpoints

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "120", "--timeout-graceful-shutdown", "30"]

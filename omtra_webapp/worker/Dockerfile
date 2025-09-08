FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including CUDA support
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /srv/app/jobs /srv/app/models

# Set permissions
RUN chmod -R 755 /app
RUN chmod -R 777 /srv/app/jobs

# Run the worker
CMD ["python", "worker.py"]

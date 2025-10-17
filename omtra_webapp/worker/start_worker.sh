#!/bin/bash

# Install webapp dependencies if not already installed
echo "Installing webapp dependencies..."
python -m pip install redis rq python-dotenv pyyaml hydra-core omegaconf pandas requests aiofiles pydantic

# Start the worker
echo "Starting OMTRA worker..."
python worker.py

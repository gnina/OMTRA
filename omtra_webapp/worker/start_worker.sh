#!/bin/bash
set -euo pipefail

if [ -d "/omtra" ]; then
    export PYTHONPATH="/omtra:${PYTHONPATH:-}"
    echo "Added /omtra to PYTHONPATH for development mode"
fi

# Start the worker
echo "Starting OMTRA worker..."
python worker.py

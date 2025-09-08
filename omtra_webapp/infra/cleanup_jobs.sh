#!/bin/bash

# Cleanup script for job directories
# Run this as a cron job to clean up old job files

set -e

LOG_FILE="/var/log/omtra_cleanup.log"
JOBS_DIR="/srv/app/jobs"
MAX_AGE_HOURS=${JOB_TTL_HOURS:-48}

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log_message "Starting job cleanup (max age: ${MAX_AGE_HOURS} hours)"

if [ ! -d "$JOBS_DIR" ]; then
    log_message "Jobs directory $JOBS_DIR does not exist"
    exit 0
fi

# Find and remove old job directories
DELETED_COUNT=0
for job_dir in "$JOBS_DIR"/*; do
    if [ -d "$job_dir" ]; then
        # Check if directory is older than MAX_AGE_HOURS
        if [ $(find "$job_dir" -type d -mtime +$(echo "scale=2; $MAX_AGE_HOURS/24" | bc) | wc -l) -gt 0 ]; then
            job_id=$(basename "$job_dir")
            log_message "Removing old job directory: $job_id"
            rm -rf "$job_dir"
            DELETED_COUNT=$((DELETED_COUNT + 1))
        fi
    fi
done

log_message "Cleanup completed. Removed $DELETED_COUNT job directories"

# Clean up temp upload directories
TEMP_UPLOADS_DIR="/tmp/uploads"
if [ -d "$TEMP_UPLOADS_DIR" ]; then
    # Remove upload directories older than 1 hour
    find "$TEMP_UPLOADS_DIR" -type d -mtime +0.04 -exec rm -rf {} + 2>/dev/null || true
    log_message "Cleaned up temporary upload directories"
fi

log_message "Job cleanup finished"

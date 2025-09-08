# Make cleanup script executable
chmod +x infra/cleanup_jobs.sh

# Start the application
echo "Starting OMTRA Molecule Sampler..."
docker-compose up --build

#!/bin/bash

# OMTRA Webapp Local Startup Script

set -e

echo "🧪 Starting OMTRA Webapp..."

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi


# Set environment variables for local development
export ENVIRONMENT=local
export OMTRA_MODEL_AVAILABLE=true
export CHECKPOINT_DIR=/srv/app/checkpoints

echo "🔧 Environment configured for local development"

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Clean up any problematic networks
echo "🧹 Cleaning up networks..."
docker network rm omtra_network 2>/dev/null || true

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
echo "🔍 Checking service health..."

# Check API
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding"
    echo "   Check logs with: docker-compose logs api"
fi

# Check Frontend
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ Frontend is running"
else
    echo "❌ Frontend is not responding"
    echo "   Check logs with: docker-compose logs frontend"
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not responding"
    echo "   Check logs with: docker-compose logs redis"
fi

# Check Worker
if docker-compose ps worker | grep -q "Up"; then
    echo "✅ Worker is running"
else
    echo "❌ Worker is not running"
    echo "   Check logs with: docker-compose logs worker"
fi

echo ""
echo "🎉 OMTRA Webapp is starting up!"
echo ""
echo "📱 Access the webapp at: http://localhost:8501"
echo "🔧 API endpoint: http://localhost:8000"
echo ""
echo "📊 Monitor logs with:"
echo "   docker-compose logs -f [service_name]"
echo ""
echo "🛑 Stop the webapp with:"
echo "   docker-compose down"
echo ""

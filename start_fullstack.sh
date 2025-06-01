#!/bin/bash
# Start script for AgEval React + FastAPI application

set -e

echo "🚀 Starting AgEval Full-Stack Application"
echo "========================================"

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down services..."
    jobs -p | xargs -r kill
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Install Node.js dependencies
echo "📥 Installing Node.js dependencies..."
cd frontend/ageval-dashboard
npm install
cd ../..

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs reports data/cache

# Check if configuration exists
if [ ! -f "config/judges_config.yaml" ]; then
    echo "⚙️  Creating configuration file..."
    cp config/judges_config.yaml.example config/judges_config.yaml
    echo "📝 Please edit config/judges_config.yaml and add your API keys"
fi

echo ""
echo "🌟 Starting services..."
echo "Backend API will be available at: http://localhost:8000"
echo "Frontend will be available at: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start FastAPI backend
echo "🔗 Starting FastAPI backend..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start React frontend
echo "⚛️  Starting React frontend..."
cd frontend/ageval-dashboard
npm start &
FRONTEND_PID=$!
cd ../..

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
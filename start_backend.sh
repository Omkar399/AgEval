#!/bin/bash
# Start script for AgEval FastAPI backend

set -e

echo "🔗 Starting AgEval FastAPI Backend"
echo "================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run ./start_fullstack.sh first."
    exit 1
fi

# Install backend dependencies
pip install -r backend/requirements.txt

# Start FastAPI server
echo "🚀 Starting FastAPI server on http://localhost:8000"
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
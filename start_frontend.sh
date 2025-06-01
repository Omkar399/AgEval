#!/bin/bash
# Start script for AgEval React frontend

set -e

echo "⚛️  Starting AgEval React Frontend"
echo "================================="

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Navigate to frontend directory
cd frontend/ageval-dashboard

# Install dependencies
echo "📥 Installing dependencies..."
npm install

# Start React development server
echo "🚀 Starting React server on http://localhost:3000"
npm start
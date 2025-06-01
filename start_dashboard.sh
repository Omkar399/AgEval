#!/bin/bash

# AgEval Dashboard Launcher Script
# Automatically activates virtual environment and runs the dashboard

echo "🤖 AgEval Dashboard Launcher"
echo "=============================="

# Check if we're in the right directory
if [ ! -f "dashboard.py" ]; then
    echo "❌ Error: dashboard.py not found"
    echo "Please run this script from the AgEval project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment 'venv' not found"
    echo "Please create a virtual environment first:"
    echo "python -m venv venv"
    echo "source venv/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

echo "✅ Found virtual environment"
echo "🔄 Activating virtual environment..."

# Activate virtual environment and run dashboard
source venv/bin/activate

echo "✅ Virtual environment activated"
echo "🚀 Starting dashboard..."

# Run the dashboard
python run_dashboard.py

echo "�� Dashboard stopped" 
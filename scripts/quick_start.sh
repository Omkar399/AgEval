#!/bin/bash
# Quick start script for AgEval - Three-Judge AI Evaluation System

set -e

echo "🚀 AgEval Quick Start Setup"
echo "=========================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs reports data/cache

# Check if configuration exists
if [ ! -f "config/judges_config.yaml" ]; then
    echo "⚙️  Creating configuration file..."
    cp config/judges_config.yaml.example config/judges_config.yaml
    echo "📝 Please edit config/judges_config.yaml and add your API keys:"
    echo "   - OpenAI API key for JudgeA and JudgeC"
    echo "   - Anthropic API key for JudgeB"
    echo "   - API key for the agent being evaluated"
else
    echo "✅ Configuration file already exists"
fi

# Test basic functionality
echo "🧪 Testing basic functionality..."
python run_evaluation.py --phase 1

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit config/judges_config.yaml with your API keys"
echo "2. Run a full evaluation: python run_evaluation.py"
echo "3. Generate reports: python generate_report.py"
echo ""
echo "For help: python run_evaluation.py --help"
echo "Documentation: See README.md and docs/ directory" 
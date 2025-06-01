#!/usr/bin/env python3
"""
Dashboard launcher for AgEval Multi-Agent Analysis.
Automatically handles virtual environment activation and dependency checking.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_virtual_env():
    """Check if we're in a virtual environment and activate if needed."""
    # Check if we're already in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is already active")
        return True
    
    # Check if venv directory exists
    venv_path = Path("venv")
    if venv_path.exists():
        print("🔄 Virtual environment found but not active")
        print("Please run: source venv/bin/activate && python run_dashboard.py")
        return False
    
    print("⚠️  No virtual environment found")
    return True  # Continue anyway

def check_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def check_dependencies():
    """Check if required packages are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'streamlit': '1.28.0',
        'plotly': '5.15.0', 
        'pandas': '1.5.0'
    }
    
    missing_packages = []
    
    for package, min_version in required_packages.items():
        if check_package_installed(package):
            print(f"✅ {package} is installed")
        else:
            print(f"❌ {package} is not installed")
            missing_packages.append(f"{package}>={min_version}")
    
    if missing_packages:
        print(f"\n🚨 Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        
        # If we're in a virtual environment, suggest the correct command
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("(Virtual environment is active - you can install directly)")
        elif Path("venv").exists():
            print("Or with virtual environment:")
            print(f"source venv/bin/activate && pip install {' '.join(missing_packages)}")
        
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_data_availability():
    """Check if evaluation data is available."""
    print("\n📊 Checking data availability...")
    
    data_files = [
        'data/enhanced_evaluation_results.json',
        'data/comprehensive_analysis.json', 
        'data/final_performance.json',
        'data/tasks.json'
    ]
    
    available_files = []
    missing_files = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            available_files.append(file_path)
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if not available_files:
        print("\n🚨 No evaluation data found!")
        print("Please run an evaluation first:")
        print("python run_enhanced_evaluation.py")
        return False
    
    if missing_files:
        print(f"\n⚠️  Some data files are missing, but dashboard can still run with available data")
    
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard."""
    print("\n🚀 Launching AgEval Dashboard...")
    print("📱 Dashboard will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
        return True
    
    return True

def main():
    """Main launcher function."""
    print("🤖 AgEval Multi-Agent Dashboard Launcher")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists("dashboard.py"):
        print("❌ dashboard.py not found in current directory")
        print("Please run this script from the AgEval project root")
        sys.exit(1)
    
    print("📁 Current directory: OK")
    
    # Check virtual environment
    if not check_virtual_env():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data availability
    data_available = check_data_availability()
    
    if not data_available:
        print("\n❓ Continue anyway? The dashboard will show instructions to run evaluation.")
        response = input("Continue? (y/N): ").lower().strip()
        if response not in ['y', 'yes']:
            print("👋 Exiting...")
            sys.exit(0)
    
    # Launch dashboard
    success = launch_dashboard()
    
    if success:
        print("\n✅ Dashboard session completed successfully!")
    else:
        print("\n❌ Dashboard failed to launch")
        sys.exit(1)

if __name__ == "__main__":
    main() 
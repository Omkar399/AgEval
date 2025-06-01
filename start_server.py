#!/usr/bin/env python3
"""
AgEval - FastAPI Server Launcher
Starts the modern FastAPI-based evaluation dashboard
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting AgEval FastAPI Dashboard...")
    print("📊 Dashboard URL: http://localhost:8001")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Ensure we're in the correct directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # Start the FastAPI server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "fastapi_app:app", 
            "--host", "0.0.0.0", 
            "--port", "8001", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n✅ Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()